"""Same-data, same-scale optimized BPE transformer training."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from layercake.models.baseline_transformer import (
    BytePairTokenizer,
    ModernBPETransformer,
    TransformerConfig,
)
from .data import ByteCorpus, sha256_file
from .foundation import _config


def _token_batch(tokenizer, rows: torch.Tensor, *, device, max_tokens: int):
    encoded = [tokenizer.encode(bytes(row.tolist())) for row in rows.cpu()]
    length = min(min(map(len, encoded)), max_tokens)
    tokens = torch.tensor([row[:length] for row in encoded], dtype=torch.long, device=device)
    covered = sum(len(tokenizer.decode(row[1:length])) for row in encoded)
    return tokens, covered


@torch.inference_mode()
def evaluate_transformer(model, tokenizer, corpus, *, config, device):
    model.eval()
    loss_sum = 0.0
    raw_bytes = 0
    token_count = 0
    correct = 0
    for rows in corpus.fixed_batches(
        batch_size=int(config["batch_size"]), sequence_bytes=int(config["sequence_bytes"]),
        batches=int(config["batches"]), device="cpu",
    ):
        tokens, covered = _token_batch(
            tokenizer, rows, device=device, max_tokens=model.config.max_tokens
        )
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        loss_sum += float(F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="sum"
        ))
        correct += int((logits.argmax(-1) == targets).sum())
        token_count += targets.numel()
        raw_bytes += covered
    model.train()
    return {
        "bits_per_byte": loss_sum / max(raw_bytes, 1) / 0.6931471805599453,
        "token_accuracy": correct / max(token_count, 1),
        "evaluated_tokens": token_count,
        "covered_raw_bytes": raw_bytes,
    }


def train_bpe_transformer(config_path: str | Path, output_dir: str | Path) -> dict:
    config_path = Path(config_path)
    config = _config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    train = ByteCorpus(config["data"]["train"])
    validation = ByteCorpus(config["data"]["validation"])
    test = ByteCorpus(config["data"]["test"])
    architecture_selection = (
        ByteCorpus(config["data"]["architecture_selection"])
        if config["data"].get("architecture_selection") else None
    )
    tokenizer_started = time.perf_counter()
    tokenizer = BytePairTokenizer.train(
        bytes(train.data[:int(config["tokenizer"]["training_bytes"])]),
        merge_count=int(config["tokenizer"]["merges"]),
    )
    tokenizer_seconds = time.perf_counter() - tokenizer_started
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        **config["model"],
    )
    model = ModernBPETransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    precision = config.get("precision", "fp32")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and precision == "fp16" else nullcontext()
    )
    steps = int(config["training"]["steps"])
    batch_size = int(config["training"]["batch_size"])
    sequence_bytes = int(config["training"]["sequence_bytes"])
    resume_metadata_path = output / "resume.json"
    resume_model_path = output / "resume-model.safetensors"
    resume_optimizer_path = output / "resume-optimizer.pt"
    resume_enabled = bool(config["training"].get("resumable", True))
    start_step = 0
    previous_wall_seconds = 0.0
    curves = []
    preprocessing_seconds = tokenizer_seconds
    if resume_enabled and resume_metadata_path.is_file():
        resume = json.loads(resume_metadata_path.read_text(encoding="utf-8"))
        if resume.get("config_sha256") != sha256_file(config_path):
            raise ValueError("resume checkpoint belongs to a different transformer configuration")
        model.load_state_dict(load_file(str(resume_model_path), device=str(device)), strict=True)
        training_state = torch.load(resume_optimizer_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(training_state["optimizer"])
        scaler.load_state_dict(training_state["scaler"])
        start_step = int(resume["step"])
        previous_wall_seconds = float(resume["wall_seconds"])
        curves = list(resume.get("curves", []))
        preprocessing_seconds = float(resume.get("preprocessing_seconds", preprocessing_seconds))
        peak_memory = int(resume.get("peak_memory", 0))
    else:
        peak_memory = 0
    started = time.perf_counter()
    for step, rows in enumerate(train.batches(
        batch_size=batch_size, sequence_bytes=sequence_bytes, seed=seed,
        steps=steps, device="cpu",
    ), start=1):
        if step <= start_step:
            continue
        token_started = time.perf_counter()
        tokens, covered = _token_batch(
            tokenizer, rows, device=device, max_tokens=model.config.max_tokens
        )
        preprocessing_seconds += time.perf_counter() - token_started
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
        if step == 1 or step % int(config["training"]["evaluation_interval"]) == 0 or step == steps:
            validation_checkpoint = evaluate_transformer(
                model, tokenizer, validation, config=config["evaluation"], device=device
            )
            curves.append({
                "step": step, "loss": float(loss.detach()), "covered_bytes_in_batch": covered,
                "raw_bytes_seen": step * batch_size * sequence_bytes,
                "wall_seconds": previous_wall_seconds + time.perf_counter() - started,
                "validation": validation_checkpoint,
            })
            if resume_enabled:
                model_temp = output / "resume-model.tmp.safetensors"
                optimizer_temp = output / "resume-optimizer.tmp.pt"
                metadata_temp = output / "resume.tmp.json"
                save_file({
                    name: value.detach().cpu().contiguous()
                    for name, value in model.state_dict().items()
                }, str(model_temp))
                torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, optimizer_temp)
                metadata_temp.write_text(json.dumps({
                    "format": "layercake-transformer-resume/1",
                    "config_sha256": sha256_file(config_path),
                    "step": step,
                    "wall_seconds": curves[-1]["wall_seconds"],
                    "preprocessing_seconds": preprocessing_seconds,
                    "curves": curves,
                    "peak_memory": peak_memory,
                }, indent=2, sort_keys=True), encoding="utf-8")
                model_temp.replace(resume_model_path)
                optimizer_temp.replace(resume_optimizer_path)
                metadata_temp.replace(resume_metadata_path)
    training_seconds = previous_wall_seconds + time.perf_counter() - started
    evaluation = config["evaluation"]
    validation_score = evaluate_transformer(
        model, tokenizer, validation, config=evaluation, device=device
    )
    architecture_score = evaluate_transformer(
        model, tokenizer, architecture_selection, config=evaluation, device=device
    ) if architecture_selection is not None else None
    evaluate_test = bool(evaluation.get("evaluate_test", True))
    test_score = evaluate_transformer(
        model, tokenizer, test, config=evaluation, device=device
    ) if evaluate_test else None
    tensor_path = output / "model.safetensors"
    tokenizer_path = output / "tokenizer.json"
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(tensor_path))
    tokenizer_path.write_text(json.dumps(tokenizer.canonical_dict(), indent=2), encoding="utf-8")
    evidence = {
        "format": "layercake-transformer-baseline/2",
        "status": "PASS",
        "seed": seed,
        "architecture": model_config.canonical_dict(),
        "parameters": model.parameter_count(),
        "tokenizer": {
            "path": str(tokenizer_path.resolve()), "sha256": sha256_file(tokenizer_path),
            "vocab_size": tokenizer.vocab_size, "merges": len(tokenizer.merges),
            "training_bytes": int(config["tokenizer"]["training_bytes"]),
            "training_seconds": tokenizer_seconds,
        },
        "data": {
            name: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}
            for name, path in config["data"].items()
        },
        "training": {
            "steps": steps, "batch_size": batch_size, "sequence_bytes": sequence_bytes,
            "raw_bytes_seen": steps * batch_size * sequence_bytes,
            "wall_seconds": training_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "curves": curves,
            "resumable": resume_enabled,
            "resumed_from_step": start_step,
            "resume_metadata": str(resume_metadata_path.resolve()) if resume_enabled else None,
        },
        "quality": {
            "architecture_selection": architecture_score,
            "validation": validation_score,
            "test": test_score,
            "test_accessed": evaluate_test,
        },
        "checkpoint": {"path": str(tensor_path.resolve()), "sha256": sha256_file(tensor_path)},
        "memory": {"cuda_peak_allocated_bytes": int(peak_memory)},
        "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
        "optimizations": {
            "training_attention": "torch.scaled_dot_product_attention",
            "inference": "per-layer KV cache",
            "precision": precision,
            "pre_normalization": True,
            "swiglu": True,
            "tied_embeddings": True,
        },
    }
    (output / "metadata.json").write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def load_transformer_checkpoint(path: str | Path, *, device="cpu"):
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    tokenizer_data = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = BytePairTokenizer([tuple(pair) for pair in tokenizer_data["merges"]])
    model = ModernBPETransformer(TransformerConfig(**metadata["architecture"]))
    model.load_state_dict(load_file(str(root / "model.safetensors"), device=str(device)), strict=True)
    return model.to(device).eval(), tokenizer, metadata


def adapt_transformer_mixed_domain(
    baseline_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict:
    config = _config(config_path)
    device = torch.device("cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu")
    model, tokenizer, source_metadata = load_transformer_checkpoint(baseline_dir, device=device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    general = ByteCorpus(config["data"]["general_train"])
    domain = ByteCorpus(config["data"]["domain_train"])
    steps = int(config["training"]["steps"])
    batch_size = int(config["training"]["batch_size"])
    sequence_bytes = int(config["training"]["sequence_bytes"])
    general_batches = general.batches(
        batch_size=batch_size, sequence_bytes=sequence_bytes,
        seed=int(config["seed"]), steps=steps, device="cpu",
    )
    domain_batches = domain.batches(
        batch_size=batch_size, sequence_bytes=sequence_bytes,
        seed=int(config["seed"]) + 1, steps=steps, device="cpu",
    )
    started = time.perf_counter()
    curves = []
    for step, (general_rows, domain_rows) in enumerate(zip(general_batches, domain_batches), start=1):
        rows = general_rows if step % 2 else domain_rows
        tokens, covered = _token_batch(tokenizer, rows, device=device, max_tokens=model.config.max_tokens)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % int(config["training"]["evaluation_interval"]) == 0 or step == steps:
            curves.append({
                "step": step, "loss": float(loss.detach()), "source": "general" if step % 2 else "python",
                "covered_bytes": covered, "wall_seconds": time.perf_counter() - started,
            })
    training_seconds = time.perf_counter() - started
    evaluation = config["evaluation"]
    general_score = evaluate_transformer(
        model, tokenizer, ByteCorpus(config["data"]["general_test"]), config=evaluation, device=device
    )
    domain_score = evaluate_transformer(
        model, tokenizer, ByteCorpus(config["data"]["domain_test"]), config=evaluation, device=device
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "model.safetensors"
    tokenizer_path = output / "tokenizer.json"
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(tensor_path))
    tokenizer_path.write_text(json.dumps(tokenizer.canonical_dict(), indent=2), encoding="utf-8")
    evidence = {
        **source_metadata,
        "format": "layercake-domain-adapted-transformer/2",
        "source_checkpoint_sha256": source_metadata["checkpoint"]["sha256"],
        "checkpoint": {"path": str(tensor_path.resolve()), "sha256": sha256_file(tensor_path)},
        "training": {
            "strategy": "full-parameter monolithic 50/50 general-Python continuation",
            "steps": steps, "wall_seconds": training_seconds,
            "raw_bytes_seen": steps * batch_size * sequence_bytes,
            "curves": curves,
        },
        "quality": {"general_test": general_score, "python_test": domain_score},
        "data": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for name, path in config["data"].items()
        },
    }
    (output / "metadata.json").write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence
