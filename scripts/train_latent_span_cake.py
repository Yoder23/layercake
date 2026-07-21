"""Train a tokenless latent-span LayerCake from raw bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.latent_span_cake import LatentSpanCakeLM  # noqa: E402


@torch.no_grad()
def _evaluate(
    model: LatentSpanCakeLM,
    payload: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    row_count = payload.numel() // seq_len
    rows = payload[: row_count * seq_len].reshape(row_count, seq_len)
    total_nll = 0.0
    total_bytes = 0
    for offset in range(0, row_count, batch_size):
        batch = rows[offset : offset + batch_size].to(device=device, dtype=torch.long)
        log_probability = model.span_log_probs(batch)
        total_nll -= float(log_probability.sum())
        total_bytes += log_probability.numel() * model.span_bytes
    model.train()
    return {
        "bytes": total_bytes,
        "bpb": total_nll / total_bytes / math.log(2.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--span-bytes", type=int, default=8)
    parser.add_argument("--d-byte", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--latent-states", type=int, default=128)
    parser.add_argument("--d-abi", type=int, default=64)
    parser.add_argument(
        "--emission-mode",
        choices=("product", "autoregressive"),
        default="product",
    )
    parser.add_argument("--local-width", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=896)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--min-lr", type=float, default=0.0002)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=24001)
    args = parser.parse_args()
    if args.seq_len % args.span_bytes:
        raise ValueError("seq-len must be divisible by span-bytes")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    train_bytes = Path(args.train).read_bytes()
    eval_bytes = Path(args.eval).read_bytes()
    train = torch.frombuffer(bytearray(train_bytes), dtype=torch.uint8)
    evaluation = torch.frombuffer(bytearray(eval_bytes), dtype=torch.uint8).long()
    model = LatentSpanCakeLM(
        span_bytes=args.span_bytes,
        d_byte=args.d_byte,
        d_model=args.d_model,
        layers=args.layers,
        latent_states=args.latent_states,
        d_abi=args.d_abi,
        emission_mode=args.emission_mode,
        local_width=args.local_width,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    offsets = torch.arange(args.seq_len)
    generator = torch.Generator().manual_seed(args.seed + 17)
    trace = []
    training_started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        starts = torch.randint(
            train.numel() - args.seq_len + 1,
            (args.batch_size,),
            generator=generator,
        )
        rows = train[starts[:, None] + offsets].to(device=device, dtype=torch.long)
        if step <= args.warmup_steps:
            lr = args.lr * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(
                args.steps - args.warmup_steps, 1
            )
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = model.loss(rows)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0 or step == args.steps:
            item = {
                "step": step,
                "train_bpb": float(loss.detach()) / math.log(2.0),
                "elapsed_seconds": time.perf_counter() - training_started,
                "lr": lr,
            }
            trace.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    eval_started = time.perf_counter()
    quality = _evaluate(
        model,
        evaluation,
        seq_len=args.seq_len,
        batch_size=args.eval_batch_size,
        device=device,
    )
    evaluation_seconds = time.perf_counter() - eval_started
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "model.pt"
    checkpoint = {
        "format": "layercake-latent-span-checkpoint/1",
        "model_config": {
            "span_bytes": args.span_bytes,
            "d_byte": args.d_byte,
            "d_model": args.d_model,
            "layers": args.layers,
            "latent_states": args.latent_states,
            "d_abi": args.d_abi,
            "emission_mode": args.emission_mode,
            "local_width": args.local_width,
        },
        "model": model.state_dict(),
        "parameters": model.logical_parameters,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "format": "layercake-latent-span-training/1",
        "status": "COMPLETE",
        "architecture": "raw-byte fixed spans with exactly marginalized latent cake states",
        "tokenizer": None,
        "parameters": model.logical_parameters,
        "config": vars(args),
        "corpus": {
            "train_bytes": len(train_bytes),
            "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
            "eval_bytes": len(eval_bytes),
            "eval_sha256": hashlib.sha256(eval_bytes).hexdigest(),
        },
        "training": {
            "source_bytes": args.steps * args.batch_size * args.seq_len,
            "predicted_bytes": args.steps
            * args.batch_size
            * (args.seq_len - args.span_bytes),
            "seconds": training_seconds,
            "trace": trace,
        },
        "quality": quality,
        "evaluation_seconds": evaluation_seconds,
        "artifact": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out_dir / "training_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
