"""Probe a bounded streaming byte-word feature cake on a frozen bundle.

This is an architecture-selection utility, not a certificate generator.  It
keeps the model interface byte-native: every scored event is the next raw byte.
The feature key is a deterministic hash of the previous delimiter-bounded byte
run, the current run prefix, and the current offset.  Training retains at most
``--entries`` continuation counts across streaming corpus chunks.
"""

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

from layercake.count_cake import apply_causal_online_cache_to_observed, load_count_cake_bundle  # noqa: E402
from scripts.probe_word_feature_cake import FEATURE_MASK, _contexts, _lookup  # noqa: E402


def _current_prefix_contexts(data: torch.Tensor, *, prefix_bytes: int) -> torch.Tensor:
    """Hash only the current delimiter-bounded byte prefix for each target."""
    data = data.to(torch.int64)
    count = data.numel()
    index = torch.arange(count, device=data.device, dtype=torch.int64)
    delimiter = (data <= 32) | torch.isin(
        data,
        torch.tensor(list(b".,;:!?()[]{}\"'`/\\|-"), device=data.device),
    )
    last_delimiter = torch.cummax(
        torch.where(delimiter, index, torch.full_like(index, -1)), dim=0
    ).values
    targets = index[1:]
    current_start = last_delimiter[:-1] + 1
    prefix_hash = torch.zeros_like(targets)
    for offset in range(prefix_bytes):
        position = targets - prefix_bytes + offset
        valid = (position >= current_start) & (position < targets)
        byte = data[position.clamp(0, count - 1)]
        prefix_hash = torch.where(
            valid,
            (prefix_hash * 257 + byte + 1) & FEATURE_MASK,
            prefix_hash,
        )
    prefix_length = (targets - current_start).clamp_max(prefix_bytes)
    return (prefix_hash * 257 + prefix_length) & FEATURE_MASK


def _retain_heavy_hitters(
    retained_keys: torch.Tensor,
    retained_counts: torch.Tensor,
    chunk_keys: torch.Tensor,
    chunk_counts: torch.Tensor,
    *,
    entries: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two sorted count tables and retain deterministic heavy hitters."""
    if retained_keys.numel():
        merged_keys = torch.cat((retained_keys, chunk_keys))
        merged_counts = torch.cat((retained_counts, chunk_counts))
        order = torch.argsort(merged_keys, stable=True)
        merged_keys = merged_keys[order]
        merged_counts = merged_counts[order]
        unique_keys, inverse = torch.unique_consecutive(
            merged_keys, return_inverse=True
        )
        unique_counts = torch.zeros(
            unique_keys.shape, device=unique_keys.device, dtype=torch.int64
        )
        unique_counts.scatter_add_(0, inverse, merged_counts)
        keys, counts = unique_keys, unique_counts
    else:
        keys, counts = chunk_keys, chunk_counts
    if keys.numel() > entries:
        selected = torch.argsort(counts, descending=True, stable=True)[:entries]
        selected = torch.sort(selected).values
        keys = keys[selected]
        counts = counts[selected]
    return keys, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--entries", type=int, default=20_000_000)
    parser.add_argument("--chunk-bytes", type=int, default=24_000_000)
    parser.add_argument("--overlap-bytes", type=int, default=64)
    parser.add_argument(
        "--feature-mode", choices=("previous_and_current", "current"), default="previous_and_current"
    )
    parser.add_argument("--prefix-bytes", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.entries <= 0 or args.chunk_bytes <= 0 or args.overlap_bytes < 32:
        raise ValueError("invalid bounded streaming settings")
    if not torch.cuda.is_available():
        raise RuntimeError("the architecture probe requires CUDA")

    device = torch.device("cuda")
    started = time.perf_counter()
    train_payload = Path(args.train).read_bytes()
    train_cpu = torch.frombuffer(bytearray(train_payload), dtype=torch.uint8)
    retained_keys = torch.empty(0, device=device, dtype=torch.int64)
    retained_counts = torch.empty(0, device=device, dtype=torch.int64)
    chunk_reports = []
    for start in range(0, train_cpu.numel(), args.chunk_bytes):
        end = min(start + args.chunk_bytes, train_cpu.numel())
        source_start = max(0, start - args.overlap_bytes)
        data = train_cpu[source_start:end].to(device=device, dtype=torch.long)
        contexts = (
            _contexts(data)
            if args.feature_mode == "previous_and_current"
            else _current_prefix_contexts(data, prefix_bytes=args.prefix_bytes)
        )
        first_target = 1 if start == 0 else start - source_start
        context_start = first_target - 1
        targets = data[first_target:]
        joint = (contexts[context_start:] << 8) | targets
        chunk_keys, chunk_counts = torch.unique(
            joint, sorted=True, return_counts=True
        )
        del data, contexts, targets, joint
        retained_keys, retained_counts = _retain_heavy_hitters(
            retained_keys,
            retained_counts,
            chunk_keys,
            chunk_counts,
            entries=args.entries,
        )
        del chunk_keys, chunk_counts
        chunk_reports.append(
            {
                "end_byte": end,
                "retained_entries": int(retained_keys.numel()),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps(chunk_reports[-1], sort_keys=True), flush=True)

    context_keys, inverse = torch.unique_consecutive(
        retained_keys >> 8, return_inverse=True
    )
    context_totals = torch.zeros(
        context_keys.shape, device=device, dtype=torch.float32
    )
    context_totals.scatter_add_(0, inverse, retained_counts.to(torch.float32))
    del inverse
    training_seconds = time.perf_counter() - started

    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    eval_payload = Path(args.eval).read_bytes()
    row_count = len(eval_payload) // 1056
    rows_np = np.frombuffer(
        eval_payload[: row_count * 1056], dtype=np.uint8
    ).reshape(row_count, 1056).copy()
    rows = torch.from_numpy(rows_np).to(device=device, dtype=torch.long)
    with torch.inference_mode():
        base = model.target_log_probs(rows).exp()
        flat = rows.flatten()
        flat_context = (
            _contexts(flat)
            if args.feature_mode == "previous_and_current"
            else _current_prefix_contexts(flat, prefix_bytes=args.prefix_bytes)
        )
        flat_positions = (
            torch.arange(row_count, device=device)[:, None] * 1056
            + torch.arange(model.prediction_start, 1056, device=device)[None]
        )
        selected_context = flat_context[flat_positions - 1]
        targets = rows[:, model.prediction_start:]
        query = (selected_context << 8) | targets
        feature_count = _lookup(
            retained_keys, retained_counts.to(torch.float32), query
        )
        total = _lookup(context_keys, context_totals, selected_context)

    base_np = base.cpu().numpy().astype(np.float64)
    results = {}
    for strength in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0):
        probability = (
            feature_count + strength * base
        ) / (total + strength)
        probability_np = probability.cpu().numpy().astype(np.float64)
        results[str(strength)] = {
            "raw_bpb": float(-np.log(probability_np).mean() / math.log(2.0)),
        }
    report = {
        "format": "layercake-streaming-word-feature-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection probe; extra state is not parameter-matched",
        "bundle": {
            "path": args.bundle,
            "logical_parameters": manifest["parameters"]["logical_total"],
        },
        "train": {
            "path": args.train,
            "bytes": len(train_payload),
            "sha256": hashlib.sha256(train_payload).hexdigest(),
            "chunk_bytes": args.chunk_bytes,
            "overlap_bytes": args.overlap_bytes,
        },
        "eval": {
            "path": args.eval,
            "bytes": len(eval_payload),
            "sha256": hashlib.sha256(eval_payload).hexdigest(),
            "scored_bytes": int(base.numel()),
        },
        "state": {
            "requested_entries": args.entries,
            "continuation_entries": int(retained_keys.numel()),
            "context_entries": int(context_keys.numel()),
        },
        "feature": {
            "mode": args.feature_mode,
            "prefix_bytes": args.prefix_bytes,
        },
        "quality": {
            "base_bpb": float(-np.log(base_np).mean() / math.log(2.0)),
            "by_strength": results,
        },
        "timing": {
            "training_seconds": training_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "chunks": chunk_reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
