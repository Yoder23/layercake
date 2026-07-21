"""Calibrate a CountCake expert gate while freezing both language experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import (  # noqa: E402
    load_count_cake_bundle,
    save_count_cake_bundle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gate-hidden-width", type=int, default=0)
    parser.add_argument("--seed", type=int, default=24001)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.seq_len <= 0 or args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("sequence length, batch size, and steps must be positive")
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    source_path = Path(args.bundle)
    train_path = Path(args.train)
    train_bytes = train_path.read_bytes()
    train = torch.frombuffer(bytearray(train_bytes), dtype=torch.uint8)
    max_start = train.numel() - args.seq_len
    if max_start < 0:
        raise ValueError("training corpus is shorter than seq-len")

    model, manifest = load_count_cake_bundle(source_path, device=device)
    if args.gate_hidden_width > 0:
        model.enable_nonlinear_gate(args.gate_hidden_width)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gate_parameters = list(model.mixture_gate.parameters())
    if model.confidence_gate_enabled:
        gate_parameters.extend(model.confidence_gate.parameters())
    if model.expert_confidence_gate_enabled:
        gate_parameters.extend(model.expert_confidence_gate.parameters())
    if model.count_distribution_gate_enabled:
        gate_parameters.extend(model.count_distribution_gate.parameters())
    if model.count_order_routing_enabled:
        gate_parameters.extend(model.count_order_router.parameters())
    if model.gate_hidden_width:
        gate_parameters.extend(model.gate_mlp.parameters())
    for parameter in gate_parameters:
        parameter.requires_grad_(True)
    gate_parameter_count = sum(parameter.numel() for parameter in gate_parameters)
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.95),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(gate_parameters, **optimizer_kwargs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed + 17)
    offsets = torch.arange(args.seq_len, dtype=torch.long)
    model.train()
    started = time.perf_counter()
    history = []
    for step in range(1, args.steps + 1):
        starts = torch.randint(
            max_start + 1,
            (args.batch_size,),
            generator=generator,
        )
        rows = train[starts[:, None] + offsets].to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = model.loss(rows, neural_auxiliary_weight=0.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(gate_parameters, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 20 == 0 or step == args.steps:
            event = {
                "step": step,
                "loss": float(loss.detach()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    output = Path(args.output)
    saved = save_count_cake_bundle(
        model,
        output,
        metadata={
            "gate_calibration": {
                "source_bundle": str(source_path),
                "source_bundle_sha256": _sha256(source_path),
                "train": str(train_path),
                "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
                "source_bytes_exposed": args.steps * args.batch_size * args.seq_len,
                "config": vars(args),
            }
        },
    )
    report = {
        "format": "layercake-count-cake-gate-calibration/1",
        "status": "COMPLETE",
        "source_bundle": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "parameters": manifest["parameters"],
        },
        "output_bundle": {
            "path": str(output),
            "sha256": _sha256(output),
            "parameters": saved["parameters"],
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "source_bytes_exposed": args.steps * args.batch_size * args.seq_len,
            "gate_parameters": gate_parameter_count,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "seconds": elapsed,
            "history": history,
        },
        "corpus": {
            "path": str(train_path),
            "bytes": len(train_bytes),
            "sha256": hashlib.sha256(train_bytes).hexdigest(),
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
