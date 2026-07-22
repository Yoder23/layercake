"""Reproducible from-scratch English-core training."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from layercake.models.foundation_v2 import FoundationV2Config, LayerCakeFoundationV2
from .data import ByteCorpus, sha256_file
from .sparse_optimizer import optimizer_state_report, sparse_adamw


def _config(path: str | Path) -> dict:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        import yaml
        value = yaml.safe_load(raw)
    if "inherits" in value:
        base_path = Path(value["inherits"])
        if not base_path.is_absolute():
            # Config inheritance is repository-relative by contract.
            base_path = Path.cwd() / base_path
        base = _config(base_path)
        for dotted, override in value.get("required_overrides", {}).items():
            target = base
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = override
        base["scale_status"] = value.get("scale_status", base.get("scale_status"))
        base["seed"] = value.get("seed", base.get("seed"))
        return base
    return value


@torch.inference_mode()
def evaluate_core(
    model: LayerCakeFoundationV2,
    corpus: ByteCorpus,
    *,
    batch_size: int,
    sequence_bytes: int,
    batches: int,
    device: torch.device,
    route: int,
) -> dict:
    model.eval()
    losses = []
    correct = 0
    count = 0
    confidences = []
    calibration_errors = []
    for row in corpus.fixed_batches(
        batch_size=batch_size, sequence_bytes=sequence_bytes, batches=batches, device=device
    ):
        inputs, targets = row[:, :-1], row[:, 1:]
        logits = model(inputs, route=route)
        losses.append(F.cross_entropy(logits.flatten(0, 1), targets.flatten()).item())
        probabilities = torch.softmax(logits.float(), dim=-1)
        confidence, prediction = probabilities.max(dim=-1)
        match = prediction == targets
        correct += int(match.sum())
        count += targets.numel()
        confidences.append(float(confidence.mean()))
        calibration_errors.append(float((confidence - match.float()).abs().mean()))
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {
        "bits_per_byte": mean_loss / 0.6931471805599453,
        "byte_accuracy": correct / count,
        "mean_confidence": sum(confidences) / len(confidences),
        "mean_absolute_calibration_error": sum(calibration_errors) / len(calibration_errors),
        "evaluated_bytes": count,
    }


def train_english_core(config_path: str | Path, output_dir: str | Path) -> dict:
    config_path = Path(config_path)
    config = _config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    requested = str(config.get("device", "cuda"))
    device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
    precision = str(config.get("precision", "fp32"))
    model = LayerCakeFoundationV2(FoundationV2Config(**config["model"])).to(device)
    train_corpus = ByteCorpus(config["data"]["train"])
    validation_corpus = ByteCorpus(config["data"]["validation"])
    test_corpus = ByteCorpus(config["data"]["test"])
    architecture_corpus = ByteCorpus(config["data"]["architecture_selection"])
    optimizer = sparse_adamw(
        model,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and precision == "fp16" else nullcontext()
    )
    steps = int(config["training"]["steps"])
    batch_size = int(config["training"]["batch_size"])
    sequence_bytes = int(config["training"]["sequence_bytes"])
    eval_interval = int(config["training"]["evaluation_interval"])
    evaluation_batches = int(config["evaluation"]["batches"])
    route = int(config["training"].get("route", seed % model.config.routed_experts))
    model.set_route(route)
    curves = []
    failed = None
    peak_memory = 0
    started = time.perf_counter()
    for step, batch in enumerate(train_corpus.batches(
        batch_size=batch_size, sequence_bytes=sequence_bytes, seed=seed,
        steps=steps, device=device,
    ), start=1):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits, aux = model(inputs, route=route, return_aux=True)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss = loss + float(config["training"].get("routing_balance_weight", 0.01)) * aux["routing_balance_loss"]
        if not torch.isfinite(loss):
            failed = {"step": step, "reason": "non-finite loss"}
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("gradient_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
        if step == 1 or step % eval_interval == 0 or step == steps:
            validation = evaluate_core(
                model, validation_corpus, batch_size=int(config["evaluation"]["batch_size"]),
                sequence_bytes=int(config["evaluation"]["sequence_bytes"]),
                batches=evaluation_batches, device=device, route=route,
            )
            curves.append({
                "step": step,
                "raw_bytes_seen": step * batch_size * sequence_bytes,
                "training_loss": float(loss.detach()),
                "validation": validation,
                "wall_seconds": time.perf_counter() - started,
            })
    training_seconds = time.perf_counter() - started
    architecture_selection = evaluate_core(
        model, architecture_corpus, batch_size=int(config["evaluation"]["batch_size"]),
        sequence_bytes=int(config["evaluation"]["sequence_bytes"]), batches=evaluation_batches,
        device=device, route=route,
    )
    validation = evaluate_core(
        model, validation_corpus, batch_size=int(config["evaluation"]["batch_size"]),
        sequence_bytes=int(config["evaluation"]["sequence_bytes"]), batches=evaluation_batches,
        device=device, route=route,
    )
    evaluate_test = bool(config["evaluation"].get("evaluate_test", False))
    test = evaluate_core(
        model, test_corpus, batch_size=int(config["evaluation"]["batch_size"]),
        sequence_bytes=int(config["evaluation"]["sequence_bytes"]), batches=evaluation_batches,
        device=device, route=route,
    ) if evaluate_test else None

    tensor_path = output / "model.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(tensor_path))
    report = model.parameter_report(route)
    process = psutil.Process()
    evidence = {
        "format": "layercake-english-core/2",
        "status": "FAIL" if failed else "PASS",
        "failure": failed,
        "seed": seed,
        "route": route,
        "architecture": model.config.canonical_dict(),
        "canonical_abi": model.canonical.config.contract(),
        "canonical_abi_hash": model.canonical.config.abi_hash(),
        "parameters": report,
        "optimizer": optimizer_state_report(optimizer),
        "data": {
            name: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}
            for name, path in config["data"].items()
        },
        "training": {
            "steps_completed": curves[-1]["step"] if curves else 0,
            "configured_steps": steps,
            "batch_size": batch_size,
            "context_bytes": sequence_bytes,
            "raw_bytes_seen": (curves[-1]["step"] if curves else 0) * batch_size * sequence_bytes,
            "wall_seconds": training_seconds,
            "raw_bytes_per_second": ((curves[-1]["step"] if curves else 0) * batch_size * sequence_bytes) / max(training_seconds, 1e-9),
            "parameter_seconds_active": report["active_parameters"] * training_seconds,
            "parameter_seconds_total_installed": report["total_parameters"] * training_seconds,
            "curves": curves,
        },
        "quality": {
            "architecture_selection": architecture_selection,
            "validation": validation,
            "test": test,
            "test_accessed": evaluate_test,
        },
        "memory": {
            "cuda_peak_allocated_bytes": int(peak_memory),
            "process_rss_bytes": int(process.memory_info().rss),
        },
        "checkpoint": {
            "tensors": str(tensor_path.resolve()),
            "sha256": sha256_file(tensor_path),
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def load_core_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[LayerCakeFoundationV2, dict]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    model = LayerCakeFoundationV2(FoundationV2Config(**metadata["architecture"]))
    model.load_state_dict(load_file(str(root / "model.safetensors"), device=str(device)), strict=True)
    model.to(device).eval()
    return model, metadata
