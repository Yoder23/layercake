"""Training and loading for the integrated sparse BPE LayerCake core."""

from __future__ import annotations

from contextlib import nullcontext
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from layercake.models.baseline_transformer import BytePairTokenizer
from layercake.models.sparse_bpe_layercake import (
    LayerCakeSparseBPECore,
    SparseBPELayerCakeConfig,
)
from .baseline import _token_batch, evaluate_transformer
from .data import ByteCorpus, sha256_file


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tokenizer(path: str | Path) -> BytePairTokenizer:
    data = _read(path)
    return BytePairTokenizer([tuple(pair) for pair in data["merges"]])


def _cumulative_raw_bytes(metadata: dict, *, visited: set[Path] | None = None) -> int:
    """Recover cumulative pretraining exposure through an immutable checkpoint chain."""

    training = metadata.get("training", {})
    if "cumulative_raw_bytes_seen" in training:
        return int(training["cumulative_raw_bytes_seen"])
    stage = int(training.get("raw_bytes_seen", 0))
    parent_path = metadata.get("initialization", {}).get("parent_checkpoint_path")
    if not parent_path:
        return stage
    parent_metadata_path = Path(parent_path).resolve().parent / "metadata.json"
    seen = set() if visited is None else visited
    if parent_metadata_path in seen:
        raise RuntimeError("checkpoint initialization lineage contains a cycle")
    seen.add(parent_metadata_path)
    if not parent_metadata_path.is_file():
        raise RuntimeError(f"checkpoint lineage metadata is missing: {parent_metadata_path}")
    return stage + _cumulative_raw_bytes(_read(parent_metadata_path), visited=seen)


def load_sparse_bpe_checkpoint(path: str | Path, *, device="cpu"):
    root = Path(path)
    metadata = _read(root / "metadata.json")
    tokenizer = _tokenizer(root / "tokenizer.json")
    model = LayerCakeSparseBPECore(
        SparseBPELayerCakeConfig(**metadata["architecture"])
    )
    model.load_state_dict(
        load_file(str(root / "model.safetensors"), device=str(device)), strict=True
    )
    return model.to(device).eval(), tokenizer, metadata


def train_sparse_bpe_layercake(config_path: str | Path, output_dir: str | Path) -> dict:
    config_path = Path(config_path)
    config = _read(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = torch.device(
        "cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    )
    tokenizer_path = Path(config["tokenizer"]["path"])
    tokenizer = _tokenizer(tokenizer_path)
    model_config = SparseBPELayerCakeConfig(
        vocab_size=tokenizer.vocab_size,
        **config["model"],
    )
    model = LayerCakeSparseBPECore(model_config).to(device)
    initialization = {"method": "random", "parent_checkpoint_sha256": None}
    parent_cumulative_raw_bytes = 0
    initial_checkpoint = config.get("initial_checkpoint")
    if initial_checkpoint:
        parent_root = Path(initial_checkpoint)
        parent_metadata = _read(parent_root / "metadata.json")
        parent_cumulative_raw_bytes = _cumulative_raw_bytes(parent_metadata)
        parent_state = load_file(str(parent_root / "model.safetensors"), device=str(device))
        state = model.state_dict()
        for name, value in parent_state.items():
            if name == "embedding.weight":
                if value.shape[1] != state[name].shape[1] or value.shape[0] > state[name].shape[0]:
                    raise ValueError("parent embedding is incompatible with extended tokenizer")
                state[name][:value.shape[0]].copy_(value)
            elif value.shape != state[name].shape:
                raise ValueError(f"parent tensor shape changed: {name}")
            else:
                state[name].copy_(value)
        model.load_state_dict(state, strict=True)
        initialization = {
            "method": "exact architecture transfer with prefix-preserved embedding expansion",
            "parent_checkpoint_path": str((parent_root / "model.safetensors").resolve()),
            "parent_checkpoint_sha256": parent_metadata["checkpoint"]["sha256"],
            "parent_tokenizer_sha256": parent_metadata["tokenizer"]["sha256"],
            "transferred_embedding_rows": int(parent_state["embedding.weight"].shape[0]),
            "new_embedding_rows": model_config.vocab_size - int(parent_state["embedding.weight"].shape[0]),
        }
    train = ByteCorpus(config["data"]["train"])
    validation = ByteCorpus(config["data"]["validation"])
    selection = ByteCorpus(config["data"]["architecture_selection"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    precision = config.get("precision", "fp32")
    use_amp = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
        if use_amp else (lambda: nullcontext())
    )
    training = config["training"]
    steps = int(training["steps"])
    batch_size = int(training["batch_size"])
    sequence_bytes = int(training["sequence_bytes"])
    balance_weight = float(training.get("routing_balance_weight", 0.02))
    resume_path = output / "resume.json"
    resume_model = output / "resume-model.safetensors"
    resume_optimizer = output / "resume-optimizer.pt"
    config_sha = sha256_file(config_path)
    start_step = 0
    previous_wall = 0.0
    curves = []
    assignment_counts = torch.zeros(model_config.routed_experts, dtype=torch.long)
    if bool(training.get("resumable", True)) and resume_path.is_file():
        resume = _read(resume_path)
        if resume.get("config_sha256") != config_sha:
            raise ValueError("sparse LayerCake resume checkpoint has a different configuration")
        model.load_state_dict(load_file(str(resume_model), device=str(device)), strict=True)
        state = torch.load(resume_optimizer, map_location=device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(resume["step"])
        previous_wall = float(resume["wall_seconds"])
        curves = list(resume.get("curves", []))
        assignment_counts = torch.tensor(resume.get("assignment_counts", assignment_counts.tolist()))
    started = time.perf_counter()
    peak_memory = 0
    for step, rows in enumerate(train.batches(
        batch_size=batch_size,
        sequence_bytes=sequence_bytes,
        seed=seed,
        steps=steps,
        device="cpu",
    ), start=1):
        if step <= start_step:
            continue
        tokens, covered = _token_batch(
            tokenizer, rows, device=device, max_tokens=model.config.max_tokens
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(tokens[:, :-1])
            targets = tokens[:, 1:]
            language_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            routing = model.last_routing_aux
            if routing is None:
                raise RuntimeError("sparse LayerCake produced no routing evidence")
            loss = language_loss + balance_weight * routing["balance_loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        assignment_counts += routing["assignment_counts"].detach().cpu()
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
        evaluate = step == 1 or step % int(training["evaluation_interval"]) == 0 or step == steps
        if evaluate:
            score = evaluate_transformer(
                model, tokenizer, validation, config=config["evaluation"], device=device
            )
            curves.append({
                "step": step,
                "language_loss": float(language_loss.detach()),
                "training_loss": float(loss.detach()),
                "covered_bytes_in_batch": covered,
                "raw_bytes_seen": step * batch_size * sequence_bytes,
                "wall_seconds": previous_wall + time.perf_counter() - started,
                "validation": score,
            })
            if bool(training.get("resumable", True)):
                save_file(
                    {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
                    str(resume_model),
                )
                torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, resume_optimizer)
                resume_path.write_text(json.dumps({
                    "format": "layercake-sparse-bpe-resume/1",
                    "config_sha256": config_sha,
                    "step": step,
                    "wall_seconds": curves[-1]["wall_seconds"],
                    "curves": curves,
                    "assignment_counts": assignment_counts.tolist(),
                }, indent=2, sort_keys=True), encoding="utf-8")
    wall_seconds = previous_wall + time.perf_counter() - started
    validation_score = evaluate_transformer(
        model, tokenizer, validation, config=config["evaluation"], device=device
    )
    selection_score = evaluate_transformer(
        model, tokenizer, selection, config=config["evaluation"], device=device
    )
    checkpoint = output / "model.safetensors"
    saved_tokenizer = output / "tokenizer.json"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        str(checkpoint),
    )
    saved_tokenizer.write_text(
        json.dumps(tokenizer.canonical_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    total_assignments = int(assignment_counts.sum())
    utilization = (
        [float(value) / total_assignments for value in assignment_counts.tolist()]
        if total_assignments else [0.0] * model_config.routed_experts
    )
    metadata = {
        "format": "layercake-sparse-bpe-english-core/1",
        "status": "PASS",
        "seed": seed,
        "architecture": model_config.canonical_dict(),
        "parameters": {
            "total": model.parameter_count(),
            "active": model.active_parameter_count(),
            "active_fraction": model.active_parameter_count() / model.parameter_count(),
        },
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "tokenizer": {"path": str(saved_tokenizer.resolve()), "sha256": sha256_file(saved_tokenizer)},
        "config": {"path": str(config_path.resolve()), "sha256": config_sha},
        "data": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for name, path in config["data"].items()
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "sequence_bytes": sequence_bytes,
            "raw_bytes_seen": steps * batch_size * sequence_bytes,
            "stage_raw_bytes_seen": steps * batch_size * sequence_bytes,
            "cumulative_raw_bytes_seen": parent_cumulative_raw_bytes + steps * batch_size * sequence_bytes,
            "wall_seconds": wall_seconds,
            "curves": curves,
            "resumed_from_step": start_step,
        },
        "quality": {
            "architecture_selection": selection_score,
            "validation": validation_score,
            "test": None,
            "test_accessed": False,
        },
        "routing": {
            "mode": model_config.routing_mode,
            "physically_dispatched": True,
            "maximum_active_experts_per_token": 1,
            "assignment_counts": assignment_counts.tolist(),
            "utilization": utilization,
            "experts_used": sum(value > 0 for value in assignment_counts.tolist()),
            "maximum_load_fraction": max(utilization),
        },
        "memory": {"cuda_peak_allocated_bytes": peak_memory},
        "incremental_state": {"mechanism": "per-layer KV cache", "implemented": True},
        "english_planner": {
            "enabled": bool(model_config.constrained_english_planner),
            "checkpoint_buffer_sha256": model.planner_sha256(),
            "selection": "neural prefill logits choose lexical rotation; grammar constrains token path",
            "frozen_evaluation_content": False,
        },
        "initialization": initialization,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(train_sparse_bpe_layercake(args.config, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
