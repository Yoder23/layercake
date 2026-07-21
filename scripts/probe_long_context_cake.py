"""Probe sparse long-context continuation tables over a frozen CountCake.

This is an architecture-selection tool.  It builds deterministic heavy-hitter
tables from the training stream and evaluates causal interpolation on a
development split.  The tables are deliberately kept outside the release
bundle: a successful probe must still be integrated under the exact logical
parameter budget and re-evaluated by the release protocol.
"""

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

from layercake.count_cake import load_count_cake_bundle  # noqa: E402


HASH_MASK = (1 << 55) - 1


def _hash_contexts(data: torch.Tensor, order: int) -> torch.Tensor:
    contexts = torch.zeros(
        data.numel() - order, device=data.device, dtype=torch.int64
    )
    for offset in range(order):
        contexts.mul_(257).add_(
            data[offset : data.numel() - order + offset].to(torch.int64) + 1
        ).bitwise_and_(HASH_MASK)
    return contexts


def _retain_heavy_hitters(
    keys: torch.Tensor,
    counts: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.numel() <= limit:
        return keys, counts
    chosen = torch.argsort(counts, descending=True, stable=True)[:limit]
    chosen = torch.sort(chosen).values
    return keys[chosen], counts[chosen]


@torch.inference_mode()
def _train_table(
    payload: torch.Tensor,
    *,
    order: int,
    budget: int,
    chunk_bytes: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int | float]:
    started = time.perf_counter()
    retained_keys = torch.empty(0, device=device, dtype=torch.int64)
    retained_counts = torch.empty(0, device=device, dtype=torch.int64)
    candidate_limit = budget * 2
    for start in range(0, payload.numel(), chunk_bytes):
        end = min(start + chunk_bytes, payload.numel())
        source_start = max(0, start - order)
        chunk = payload[source_start:end].to(device=device, dtype=torch.int64)
        if chunk.numel() <= order:
            continue
        targets = chunk[order:]
        contexts = _hash_contexts(chunk, order)
        joint = (contexts << 8) | targets
        chunk_keys, chunk_counts = torch.unique(
            joint, sorted=True, return_counts=True
        )
        chunk_keys, chunk_counts = _retain_heavy_hitters(
            chunk_keys, chunk_counts, candidate_limit
        )
        if retained_keys.numel():
            merged_keys = torch.cat((retained_keys, chunk_keys))
            merged_counts = torch.cat((retained_counts, chunk_counts))
            ordering = torch.argsort(merged_keys, stable=True)
            merged_keys = merged_keys[ordering]
            merged_counts = merged_counts[ordering]
            retained_keys, inverse = torch.unique_consecutive(
                merged_keys, return_inverse=True
            )
            retained_counts = torch.zeros(
                retained_keys.shape, device=device, dtype=torch.int64
            )
            retained_counts.scatter_add_(0, inverse, merged_counts)
        else:
            retained_keys, retained_counts = chunk_keys, chunk_counts
        retained_keys, retained_counts = _retain_heavy_hitters(
            retained_keys, retained_counts, candidate_limit
        )
    retained_keys, retained_counts = _retain_heavy_hitters(
        retained_keys, retained_counts, budget
    )
    context = retained_keys >> 8
    context_keys, inverse = torch.unique_consecutive(
        context, return_inverse=True
    )
    context_totals = torch.zeros(
        context_keys.shape, device=device, dtype=torch.float32
    )
    context_totals.scatter_add_(0, inverse, retained_counts.to(torch.float32))
    return {
        "keys": retained_keys,
        "counts": retained_counts.to(torch.float32),
        "context_keys": context_keys,
        "context_totals": context_totals,
        "entries": int(retained_keys.numel()),
        "contexts": int(context_keys.numel()),
        "seconds": time.perf_counter() - started,
    }


def _lookup(
    keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    positions = torch.searchsorted(keys, query)
    safe = positions.clamp(max=keys.numel() - 1)
    found = (positions < keys.numel()) & (keys[safe] == query)
    return torch.where(found, values[safe], torch.zeros_like(values[safe]))


@torch.inference_mode()
def _base_probability(model, rows: torch.Tensor, batch_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, rows.shape[0], batch_size):
        chunks.append(model.target_log_probs(rows[start : start + batch_size]).exp())
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def _observations(
    rows: torch.Tensor,
    table: dict[str, torch.Tensor | int | float],
    *,
    order: int,
    prediction_start: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_count = rows.shape[1] - prediction_start
    context = torch.zeros(
        (rows.shape[0], target_count), device=rows.device, dtype=torch.int64
    )
    for offset in range(order):
        context.mul_(257).add_(
            rows[
                :,
                prediction_start - order + offset : rows.shape[1]
                - order
                + offset,
            ]
            + 1
        ).bitwise_and_(HASH_MASK)
    targets = rows[:, prediction_start:]
    joint = (context << 8) | targets
    count = _lookup(table["keys"], table["counts"], joint)
    total = _lookup(
        table["context_keys"], table["context_totals"], context
    )
    return count, total, targets


def _bpb(probability: torch.Tensor) -> float:
    return float(-probability.clamp_min(1e-30).log().mean() / math.log(2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--orders", default="16,24,32")
    parser.add_argument("--budgets", default="2000000,2000000,2000000")
    parser.add_argument("--chunk-bytes", type=int, default=24_000_000)
    parser.add_argument("--seq-len", type=int, default=544)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    orders = [int(value) for value in args.orders.split(",")]
    budgets = [int(value) for value in args.budgets.split(",")]
    if len(orders) != len(budgets) or not orders:
        raise ValueError("orders and budgets must have equal nonzero lengths")
    device = torch.device("cuda")
    started = time.perf_counter()
    train_bytes = Path(args.train).read_bytes()
    train_tensor = torch.frombuffer(bytearray(train_bytes), dtype=torch.uint8)
    tables = []
    for order, budget in zip(orders, budgets):
        table = _train_table(
            train_tensor,
            order=order,
            budget=budget,
            chunk_bytes=args.chunk_bytes,
            device=device,
        )
        tables.append(table)
        print(
            json.dumps(
                {
                    "order": order,
                    "entries": table["entries"],
                    "contexts": table["contexts"],
                    "seconds": table["seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del train_tensor
    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    eval_bytes = Path(args.eval).read_bytes()
    rows_count = len(eval_bytes) // args.seq_len
    rows = torch.frombuffer(
        bytearray(eval_bytes[: rows_count * args.seq_len]), dtype=torch.uint8
    ).reshape(rows_count, args.seq_len).to(device=device, dtype=torch.int64)
    probability = _base_probability(model, rows, args.batch_size)
    base_bpb = _bpb(probability)
    stages = []
    strength_grid = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
    for order, table in zip(orders, tables):
        count, total, _ = _observations(
            rows,
            table,
            order=order,
            prediction_start=model.prediction_start,
        )
        candidates = []
        for strength in strength_grid:
            candidate = (count + strength * probability) / (total + strength)
            candidates.append((_bpb(candidate), strength, candidate))
        best_bpb, best_strength, probability = min(candidates, key=lambda item: item[0])
        stages.append(
            {
                "order": order,
                "budget": table["entries"],
                "contexts": table["contexts"],
                "match_fraction": float((total > 0).to(torch.float32).mean()),
                "observed_continuation_fraction": float(
                    (count > 0).to(torch.float32).mean()
                ),
                "selected_strength": best_strength,
                "bpb": best_bpb,
            }
        )
    report = {
        "format": "layercake-long-context-cake-probe/1",
        "status": "COMPLETE",
        "warning": (
            "architecture-selection result; strengths selected on the development "
            "split and long-context entries are not yet charged against the bundle"
        ),
        "bundle": {
            "path": args.bundle,
            "parameters": manifest["parameters"],
        },
        "train": {
            "path": args.train,
            "bytes": len(train_bytes),
            "sha256": hashlib.sha256(train_bytes).hexdigest(),
        },
        "evaluation": {
            "path": args.eval,
            "bytes": len(eval_bytes),
            "sha256": hashlib.sha256(eval_bytes).hexdigest(),
            "scored_bytes": int(probability.numel()),
            "base_bpb": base_bpb,
            "final_bpb": _bpb(probability),
        },
        "long_context_parameters": sum(int(table["entries"]) for table in tables),
        "stages": stages,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
