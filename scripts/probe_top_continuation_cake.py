"""Probe evidence-weighted top continuations from a frozen CountCake state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402


def _top_tables(cake, order: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = getattr(cake, f"keys_{order}")
    counts = getattr(cake, f"counts_{order}")
    dense_context = keys >> 8
    _, inverse = torch.unique_consecutive(dense_context, return_inverse=True)
    context_count = getattr(cake, f"context_keys_{order}").numel()
    maximum = torch.zeros(
        context_count, device=keys.device, dtype=counts.dtype
    )
    maximum.scatter_reduce_(0, inverse, counts, reduce="amax", include_self=True)
    targets = keys.bitwise_and(255)
    candidates = torch.where(
        counts == maximum[inverse], targets, torch.full_like(targets, 256)
    )
    top_target = torch.full(
        (context_count,), 256, device=keys.device, dtype=torch.int64
    )
    top_target.scatter_reduce_(
        0, inverse, candidates, reduce="amin", include_self=True
    )
    return top_target, maximum


def _context_query(cake, rows: torch.Tensor, start: int, order: int) -> torch.Tensor:
    targets = rows[:, start:]
    context = torch.zeros_like(targets)
    if cake.order_encodings[order - 1] == "packed":
        for lag in range(order):
            context.add_(
                rows[:, start - 1 - lag : rows.shape[1] - 1 - lag]
                << (8 * lag)
            )
    else:
        mask = (1 << cake.context_hash_bits[order - 1]) - 1
        for offset in range(order):
            context.mul_(257).add_(
                rows[
                    :,
                    start - order + offset : rows.shape[1] - order + offset,
                ]
                + 1
            ).bitwise_and_(mask)
    return context


@torch.inference_mode()
def _prepare(model, path: Path, seq_len: int, top_tables: list[tuple]) -> dict:
    payload = path.read_bytes()
    row_count = len(payload) // seq_len
    rows = torch.frombuffer(
        bytearray(payload[: row_count * seq_len]), dtype=torch.uint8
    ).reshape(row_count, seq_len).to(device="cuda", dtype=torch.long)
    base_chunks = []
    for offset in range(0, row_count, 128):
        base_chunks.append(
            model.target_log_probs(rows[offset : offset + 128]).exp().cpu()
        )
    base = torch.cat(base_chunks).numpy().astype(np.float64)
    targets = rows[:, model.prediction_start :]
    matches = []
    correct = []
    evidence = []
    for order, (top_target, top_count) in enumerate(top_tables, start=1):
        query = _context_query(
            model.count_cake, rows, model.prediction_start, order
        )
        context_keys = getattr(model.count_cake, f"context_keys_{order}")
        indices = torch.searchsorted(context_keys, query)
        safe = indices.clamp(max=context_keys.numel() - 1)
        found = (indices < context_keys.numel()) & (
            context_keys[safe] == query
        )
        matches.append(found.cpu().numpy())
        correct.append(
            (found & (top_target[safe] == targets)).cpu().numpy()
        )
        evidence.append(
            torch.where(found, top_count[safe], torch.zeros_like(query, dtype=top_count.dtype))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "scored_bytes": int(base.size),
        "base": base,
        "matches": matches,
        "correct": correct,
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the top-continuation probe requires CUDA")
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device="cuda")
    model.eval()
    top_tables = [
        _top_tables(model.count_cake, order)
        for order in range(1, model.count_cake.max_order + 1)
    ]
    splits = [
        _prepare(model, Path(path), args.seq_len, top_tables)
        for path in args.data
    ]
    recipes = []
    for minimum_order in (4, 6, 8, 10):
        for evidence_power in (0.0, 0.5, 1.0):
            for strength in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
                values = []
                for split in splits:
                    probability = split["base"].copy()
                    for order in range(minimum_order, model.count_cake.max_order + 1):
                        matched = split["matches"][order - 1]
                        correct = split["correct"][order - 1]
                        raw_evidence = split["evidence"][order - 1]
                        weighted_evidence = (
                            np.ones_like(raw_evidence)
                            if evidence_power == 0.0
                            else np.power(raw_evidence, evidence_power)
                        )
                        updated = (
                            weighted_evidence * correct + strength * probability
                        ) / (weighted_evidence + strength)
                        probability = np.where(matched, updated, probability)
                    values.append(
                        float(-np.log(probability).mean() / math.log(2.0))
                    )
                recipes.append(
                    {
                        "minimum_order": minimum_order,
                        "evidence_power": evidence_power,
                        "strength": strength,
                        "split_bpb": values,
                        "mean_bpb": float(np.mean(values)),
                        "worst_bpb": float(np.max(values)),
                    }
                )
    recipes.sort(key=lambda item: (item["mean_bpb"], item["worst_bpb"]))
    report = {
        "format": "layercake-top-continuation-cake-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection result; not a release certificate",
        "bundle": {
            "path": args.bundle,
            "parameters": manifest["parameters"],
            "max_order": model.count_cake.max_order,
        },
        "splits": [
            {
                key: split[key]
                for key in ("path", "bytes", "sha256", "scored_bytes")
            }
            | {
                "base_bpb": float(
                    -np.log(split["base"]).mean() / math.log(2.0)
                )
            }
            for split in splits
        ],
        "best_recipes": recipes[:20],
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
