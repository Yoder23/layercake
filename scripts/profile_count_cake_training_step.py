"""Measure synchronized CountCake optimizer steps without saving artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--seq-len", type=int, default=544)
    parser.add_argument("--batch-sizes", default="16,32,64,128")
    parser.add_argument("--lr", type=float, default=0.0012)
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="profile the neural-only phase used before mixture calibration",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    model, _ = load_count_cake_bundle(args.bundle, device=device)
    train = torch.frombuffer(bytearray(Path(args.train).read_bytes()), dtype=torch.uint8)
    offsets = torch.arange(args.seq_len, dtype=torch.long)
    maximum_start = train.numel() - args.seq_len
    generator = torch.Generator().manual_seed(args.seed + 17)
    results = []
    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        sparse_parameters = []
        dense_parameters = []
        for name, parameter in model.named_parameters():
            if (
                model.dynamic_hash_sparse
                and (
                    name == "dynamic_hash_embedding.weight"
                    or name.startswith("dynamic_hash_embeddings.")
                )
            ):
                sparse_parameters.append(parameter)
            else:
                dense_parameters.append(parameter)
        optimizer = torch.optim.AdamW(
            dense_parameters,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
            fused=True,
        )
        sparse_optimizer = (
            torch.optim.SparseAdam(
                sparse_parameters,
                lr=args.lr,
                betas=(0.9, 0.95),
            )
            if sparse_parameters
            else None
        )
        scaler = torch.amp.GradScaler("cuda")
        starts = torch.randint(maximum_start + 1, (batch_size,), generator=generator)
        rows = train[starts[:, None] + offsets].to(device=device, dtype=torch.long)
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if sparse_optimizer is not None:
            sparse_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = (
                model.neural_loss(rows)
                if args.neural_only
                else model.loss(rows, neural_auxiliary_weight=1.0)
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if sparse_optimizer is not None:
            scaler.unscale_(sparse_optimizer)
        torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0)
        scaler.step(optimizer)
        if sparse_optimizer is not None:
            scaler.step(sparse_optimizer)
        scaler.update()
        torch.cuda.synchronize()
        result = {
            "batch_size": batch_size,
            "seq_len": args.seq_len,
            "source_bytes": batch_size * args.seq_len,
            "loss": float(loss.detach()),
            "seconds": time.perf_counter() - started,
        }
        result["source_bytes_per_second"] = result["source_bytes"] / result["seconds"]
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del optimizer, sparse_optimizer, scaler, rows, loss
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    print(json.dumps({"status": "COMPLETE", "results": results}, indent=2))


if __name__ == "__main__":
    main()
