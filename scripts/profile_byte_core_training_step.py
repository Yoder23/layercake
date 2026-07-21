"""Profile complete optimizer steps for one configured byte core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_byte_core_from_config import (  # noqa: E402
    _build_model,
    _load_config_with_extends,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, action="append", required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = _load_config_with_extends(Path(args.config))
    seq_len = int(config["training"]["seq_len"])
    results = []
    for batch_size in args.batch_size:
        torch.cuda.empty_cache()
        model = _build_model(config["model"], torch.device("cuda"))
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["lr"]),
            betas=(0.9, 0.95),
            weight_decay=float(config["training"]["weight_decay"]),
            fused=True,
        )
        scaler = torch.amp.GradScaler("cuda")
        rows = torch.randint(
            0, 256, (batch_size, seq_len), device="cuda", dtype=torch.long
        )
        durations = []
        peak_bytes = 0
        status = "COMPLETE"
        error = None
        try:
            for step in range(args.warmup_steps + args.steps):
                torch.cuda.reset_peak_memory_stats()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.amp.autocast("cuda"):
                    loss = model.domain_cake_patch_predictions(rows, loss_only=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated())
                if step >= args.warmup_steps:
                    durations.append(elapsed)
        except torch.OutOfMemoryError as exc:
            status = "OOM"
            error = str(exc)
        predicted = (
            (seq_len // int(config["model"]["patch_size"]) - 1)
            * int(config["model"]["patch_generation_bytes"])
            * batch_size
        )
        mean_seconds = sum(durations) / len(durations) if durations else None
        results.append(
            {
                "batch_size": batch_size,
                "status": status,
                "error": error,
                "mean_step_seconds": mean_seconds,
                "predicted_bytes_per_step": predicted,
                "bytes_per_second": predicted / mean_seconds if mean_seconds else None,
                "peak_cuda_allocated_bytes": peak_bytes,
            }
        )
        del rows, optimizer, model
        torch.cuda.empty_cache()
    report = {
        "format": "layercake-byte-core-step-profile/1",
        "status": "COMPLETE",
        "warning": "synthetic optimizer-step profile; not release evidence",
        "config": args.config,
        "parameters": sum(
            parameter.numel()
            for parameter in _build_model(config["model"], torch.device("cpu")).parameters()
        ),
        "seq_len": seq_len,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "results": results,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
