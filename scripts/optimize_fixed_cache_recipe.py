"""Fit one frozen bounded-cache recipe across multiple development splits."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import ASCII_CLASS_TABLE, load_count_cake_bundle  # noqa: E402


def _base_probabilities(model, rows: np.ndarray, device: torch.device) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for offset in range(0, rows.shape[0], 128):
            batch = torch.from_numpy(rows[offset : offset + 128]).to(
                device=device, dtype=torch.long
            )
            chunks.append(model.target_log_probs(batch).exp().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64).reshape(-1)


def _cache_statistics(
    rows: np.ndarray,
    *,
    prediction_start: int,
    window: int,
    exact_orders: tuple[int, ...],
    recent_orders: tuple[int, ...],
    normalized_orders: tuple[int, ...],
) -> dict:
    size = rows.shape[0] * (rows.shape[1] - prediction_start)
    exact_counts = {order: np.empty(size, np.float32) for order in exact_orders}
    exact_totals = {order: np.empty(size, np.float32) for order in exact_orders}
    recent_matches = {order: np.empty(size, np.uint8) for order in recent_orders}
    recent_active = {order: np.empty(size, np.bool_) for order in recent_orders}
    normalized_counts = {
        order: np.empty(size, np.float32) for order in normalized_orders
    }
    normalized_totals = {
        order: np.empty(size, np.float32) for order in normalized_orders
    }
    exact_tables = {order: {} for order in exact_orders}
    exact_events = {order: deque() for order in exact_orders}
    recent_tables = {order: {} for order in recent_orders}
    normalized_tables = {order: {} for order in normalized_orders}
    normalized_events = {order: deque() for order in normalized_orders}
    max_history = max(exact_orders + recent_orders + normalized_orders)
    history = bytearray()
    scored = 0
    stream_position = 0
    for row in rows:
        for position, raw_target in enumerate(row):
            target = int(raw_target)
            if position >= prediction_start:
                for order in exact_orders:
                    context = bytes(history[-order:])
                    continuations = exact_tables[order].get(context)
                    if continuations:
                        exact_counts[order][scored] = continuations.get(target, 0)
                        exact_totals[order][scored] = sum(continuations.values())
                    else:
                        exact_counts[order][scored] = 0
                        exact_totals[order][scored] = 0
                for order in recent_orders:
                    match = recent_tables[order].get(bytes(history[-order:]))
                    active = (
                        match is not None
                        and stream_position - match[1] <= window
                    )
                    recent_active[order][scored] = active
                    recent_matches[order][scored] = int(
                        active and match[0] == target
                    )
                for order in normalized_orders:
                    context = bytes(history[-order:]).translate(ASCII_CLASS_TABLE)
                    continuations = normalized_tables[order].get(context)
                    if continuations:
                        normalized_counts[order][scored] = continuations.get(
                            target, 0
                        )
                        normalized_totals[order][scored] = sum(
                            continuations.values()
                        )
                    else:
                        normalized_counts[order][scored] = 0
                        normalized_totals[order][scored] = 0
                scored += 1

            for order in exact_orders:
                if len(history) < order:
                    continue
                context = bytes(history[-order:])
                continuations = exact_tables[order].setdefault(context, {})
                continuations[target] = continuations.get(target, 0) + 1
                events = exact_events[order]
                events.append((context, target))
                if len(events) > window:
                    old_context, old_target = events.popleft()
                    old = exact_tables[order][old_context]
                    remaining = old[old_target] - 1
                    if remaining:
                        old[old_target] = remaining
                    else:
                        del old[old_target]
                    if not old:
                        del exact_tables[order][old_context]
            for order in recent_orders:
                if len(history) >= order:
                    recent_tables[order][bytes(history[-order:])] = (
                        target,
                        stream_position,
                    )
            for order in normalized_orders:
                if len(history) < order:
                    continue
                context = bytes(history[-order:]).translate(ASCII_CLASS_TABLE)
                continuations = normalized_tables[order].setdefault(context, {})
                continuations[target] = continuations.get(target, 0) + 1
                events = normalized_events[order]
                events.append((context, target))
                if len(events) > window:
                    old_context, old_target = events.popleft()
                    old = normalized_tables[order][old_context]
                    remaining = old[old_target] - 1
                    if remaining:
                        old[old_target] = remaining
                    else:
                        del old[old_target]
                    if not old:
                        del normalized_tables[order][old_context]
            history.append(target)
            if len(history) > max_history:
                del history[: len(history) - max_history]
            stream_position += 1
    if scored != size:
        raise RuntimeError("cache statistics do not match scored byte count")
    return {
        "exact_counts": exact_counts,
        "exact_totals": exact_totals,
        "recent_matches": recent_matches,
        "recent_active": recent_active,
        "normalized_counts": normalized_counts,
        "normalized_totals": normalized_totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--window", type=int, default=768)
    parser.add_argument(
        "--fit-bytes-per-split",
        type=int,
        default=262_144,
        help="predeclared development prefix used for strength fitting",
    )
    parser.add_argument("--robust-weight", type=float, default=0.25)
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.robust_weight <= 1.0:
        raise ValueError("robust-weight must be in [0, 1]")
    if args.fit_bytes_per_split < 1056:
        raise ValueError("fit-bytes-per-split must contain one complete row")
    device = torch.device("cuda")
    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    exact_orders = tuple(order for order, _ in model.online_cache_specs)
    recent_orders = tuple(order for order, _ in model.recent_cache_specs)
    normalized_orders = tuple(order for order, _ in model.normalized_cache_specs)
    if not (exact_orders and recent_orders and normalized_orders):
        raise ValueError("bundle must contain the complete fixed cache recipe")

    splits = []
    for source in args.eval:
        full_payload = Path(source).read_bytes()
        payload = full_payload[: args.fit_bytes_per_split]
        row_count = len(payload) // 1056
        rows = np.frombuffer(
            payload[: row_count * 1056], dtype=np.uint8
        ).reshape(row_count, 1056).copy()
        base = _base_probabilities(model, rows, device)
        statistics = _cache_statistics(
            rows,
            prediction_start=model.prediction_start,
            window=args.window,
            exact_orders=exact_orders,
            recent_orders=recent_orders,
            normalized_orders=normalized_orders,
        )
        splits.append(
            {
                "path": source,
                "source_bytes": len(full_payload),
                "fit_bytes": len(payload),
                "scored_bytes": int(base.size),
                "base": base,
                "statistics": statistics,
            }
        )
        print(json.dumps({"prepared": source, "scored_bytes": int(base.size)}), flush=True)

    initial = np.array(
        [
            *[strength for _, strength in model.online_cache_specs],
            *[strength for _, strength in model.recent_cache_specs],
            *[strength for _, strength in model.normalized_cache_specs],
        ],
        dtype=np.float64,
    )
    calls = 0

    def evaluate(log_strengths: np.ndarray) -> tuple[float, list[float]]:
        strengths = np.exp(log_strengths)
        split_bpb = []
        for split in splits:
            probability = split["base"].copy()
            stats = split["statistics"]
            cursor = 0
            for order in exact_orders:
                strength = strengths[cursor]
                probability = (
                    stats["exact_counts"][order] + strength * probability
                ) / (stats["exact_totals"][order] + strength)
                cursor += 1
            for order in recent_orders:
                strength = strengths[cursor]
                active = stats["recent_active"][order]
                probability[active] = (
                    stats["recent_matches"][order][active]
                    + strength * probability[active]
                ) / (1.0 + strength)
                cursor += 1
            for order in normalized_orders:
                strength = strengths[cursor]
                probability = (
                    stats["normalized_counts"][order] + strength * probability
                ) / (stats["normalized_totals"][order] + strength)
                cursor += 1
            split_bpb.append(float(-np.log(probability).mean() / math.log(2.0)))
        mean = float(np.mean(split_bpb))
        robust = mean + args.robust_weight * (max(split_bpb) - mean)
        return robust, split_bpb

    initial_objective, initial_split_bpb = evaluate(np.log(initial))
    started = time.perf_counter()

    def objective(log_strengths: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        value, _ = evaluate(log_strengths)
        return value

    result = minimize(
        objective,
        np.log(initial),
        method="Powell",
        bounds=[(math.log(0.05), math.log(1e5))] * initial.size,
        options={
            "maxiter": args.max_iterations,
            "xtol": 0.002,
            "ftol": 1e-10,
        },
    )
    best = np.exp(result.x)
    best_objective, best_split_bpb = evaluate(result.x)
    exact_end = len(exact_orders)
    recent_end = exact_end + len(recent_orders)
    report = {
        "format": "layercake-fixed-cache-optimization/1",
        "status": "COMPLETE" if result.success else "LIMIT_REACHED",
        "bundle": args.bundle,
        "bundle_parameters": manifest["parameters"],
        "window": args.window,
        "robust_weight": args.robust_weight,
        "fit_bytes_per_split": args.fit_bytes_per_split,
        "initial": {
            "objective_bpb": initial_objective,
            "split_bpb": initial_split_bpb,
            "strengths": initial.tolist(),
        },
        "best": {
            "objective_bpb": best_objective,
            "split_bpb": best_split_bpb,
            "exact_specs": [
                [order, float(strength)]
                for order, strength in zip(exact_orders, best[:exact_end])
            ],
            "recent_specs": [
                [order, float(strength)]
                for order, strength in zip(
                    recent_orders, best[exact_end:recent_end]
                )
            ],
            "normalized_specs": [
                [order, float(strength)]
                for order, strength in zip(
                    normalized_orders, best[recent_end:]
                )
            ],
        },
        "splits": [
            {
                "path": split["path"],
                "source_bytes": split["source_bytes"],
                "fit_bytes": split["fit_bytes"],
                "scored_bytes": split["scored_bytes"],
            }
            for split in splits
        ],
        "objective_calls": calls,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_message": str(result.message),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
