"""Validation-only probe for a non-contiguous byte word-context cake."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import (  # noqa: E402
    DEFAULT_ONLINE_CACHE_SPECS,
    apply_causal_online_cache_to_observed,
    load_count_cake_bundle,
)

FEATURE_MASK = (1 << 55) - 1


def _contexts(data: torch.Tensor) -> torch.Tensor:
    """Hash previous word suffix plus current word prefix for each target."""
    data = data.to(torch.int64)
    n = data.numel()
    index = torch.arange(n, device=data.device, dtype=torch.int64)
    delimiter = (data <= 32) | torch.isin(
        data,
        torch.tensor(list(b".,;:!?()[]{}\"'`/\\|-"), device=data.device),
    )
    last_delimiter = torch.cummax(
        torch.where(delimiter, index, torch.full_like(index, -1)), dim=0
    ).values
    last_non_delimiter = torch.cummax(
        torch.where(~delimiter, index, torch.full_like(index, -1)), dim=0
    ).values
    targets = index[1:]
    current_start = last_delimiter[:-1] + 1
    previous_lookup = (current_start - 2).clamp_min(0)
    previous_end = last_non_delimiter[previous_lookup]
    previous_end = torch.where(current_start >= 2, previous_end, -1)
    previous_start_lookup = (previous_end - 1).clamp_min(0)
    previous_start = last_delimiter[previous_start_lookup] + 1
    previous_start = torch.where(previous_end >= 0, previous_start, 0)

    previous_hash = torch.zeros_like(targets)
    for offset in range(16):
        position = previous_end - 15 + offset
        valid = (position >= previous_start) & (position <= previous_end)
        byte = data[position.clamp(0, n - 1)]
        previous_hash = torch.where(
            valid,
            (previous_hash * 257 + byte + 1) & FEATURE_MASK,
            previous_hash,
        )

    prefix_hash = torch.zeros_like(targets)
    for offset in range(8):
        position = targets - 8 + offset
        valid = (position >= current_start) & (position < targets)
        byte = data[position.clamp(0, n - 1)]
        prefix_hash = torch.where(
            valid,
            (prefix_hash * 257 + byte + 1) & FEATURE_MASK,
            prefix_hash,
        )
    return (
        previous_hash * 1_000_003 + prefix_hash * 257 + (targets - current_start)
    ) & FEATURE_MASK


def _lookup(keys: torch.Tensor, values: torch.Tensor, query: torch.Tensor):
    positions = torch.searchsorted(keys, query)
    safe = positions.clamp(max=keys.numel() - 1)
    found = (positions < keys.numel()) & (keys[safe] == query)
    return torch.where(found, values[safe], torch.zeros_like(values[safe]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--entries", type=int, default=2_000_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    started = time.perf_counter()
    train = torch.frombuffer(
        bytearray(Path(args.train).read_bytes()), dtype=torch.uint8
    ).to(device=device, dtype=torch.long)
    contexts = _contexts(train)
    joint = (contexts << 8) | train[1:]
    keys, counts = torch.unique(joint, sorted=True, return_counts=True)
    if keys.numel() > args.entries:
        selected = torch.argsort(counts, descending=True, stable=True)[: args.entries]
        selected = torch.sort(selected).values
        keys = keys[selected]
        counts = counts[selected]
    context_keys, inverse = torch.unique_consecutive(
        keys >> 8, return_inverse=True
    )
    totals = torch.zeros(context_keys.shape, device=device, dtype=torch.float32)
    totals.scatter_add_(0, inverse, counts.to(torch.float32))
    del joint, contexts, inverse

    model, _ = load_count_cake_bundle(args.bundle, device=device)
    payload = Path(args.eval).read_bytes()
    row_count = len(payload) // 1056
    rows_np = np.frombuffer(
        payload[: row_count * 1056], dtype=np.uint8
    ).reshape(row_count, 1056).copy()
    rows = torch.from_numpy(rows_np).to(device=device, dtype=torch.long)
    with torch.inference_mode():
        base = model.target_log_probs(rows).exp()
        flat = rows.flatten()
        flat_context = _contexts(flat)
        context_indices = (
            torch.arange(row_count, device=device)[:, None] * 1056
            + torch.arange(1, 1056, device=device)[None]
            - 1
        )
        feature_context = flat_context[context_indices]
        selected_context = feature_context[:, model.prediction_start - 1 :]
        targets = rows[:, model.prediction_start :]
        query = (selected_context << 8) | targets
        feature_count = _lookup(keys, counts.to(torch.float32), query)
        total = _lookup(context_keys, totals, selected_context)
    results = {}
    for strength in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
        probability = (feature_count + strength * base) / (total + strength)
        cached = apply_causal_online_cache_to_observed(
            probability.cpu().numpy(),
            rows_np,
            start=model.prediction_start,
            specs=DEFAULT_ONLINE_CACHE_SPECS,
        )
        results[str(strength)] = float(-np.log(cached).mean() / math.log(2.0))
    report = {
        "format": "layercake-word-feature-probe/1",
        "status": "COMPLETE",
        "logical_extra_entries": int(keys.numel()),
        "context_entries": int(context_keys.numel()),
        "quality_cached_bpb_by_strength": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
