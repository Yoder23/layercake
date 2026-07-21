"""Jointly fit bounded causal cache strengths from precomputed statistics."""

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

from layercake.count_cake import (  # noqa: E402
    ASCII_CLASS_TABLE,
    load_count_cake_bundle,
)


ORDERS = (16, 12, 10, 8, 6, 5, 4, 3, 2)
RECENT_ORDERS = (24, 16, 12, 10)
NORMALIZED_ORDERS = (5, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--backoff-strengths", required=True)
    parser.add_argument("--window", type=int, default=768)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    model, _ = load_count_cake_bundle(args.bundle, device=device)
    model.count_cake.backoff_strengths = tuple(
        float(value) for value in args.backoff_strengths.split(",")
    )
    model.eval()
    payload = Path(args.eval).read_bytes()
    row_count = len(payload) // 1056
    rows_np = np.frombuffer(
        payload[: row_count * 1056], dtype=np.uint8
    ).reshape(row_count, 1056).copy()
    chunks = []
    with torch.inference_mode():
        for offset in range(0, row_count, 128):
            batch = torch.from_numpy(rows_np[offset : offset + 128]).to(
                device=device, dtype=torch.long
            )
            chunks.append(model.target_log_probs(batch).exp().cpu().numpy())
    base = np.concatenate(chunks, axis=0).astype(np.float64).reshape(-1)
    count_values = {order: np.empty(base.size, dtype=np.float32) for order in ORDERS}
    total_values = {order: np.empty(base.size, dtype=np.float32) for order in ORDERS}
    recent_matches = {
        order: np.empty(base.size, dtype=np.uint8) for order in RECENT_ORDERS
    }
    recent_active = {
        order: np.empty(base.size, dtype=np.bool_) for order in RECENT_ORDERS
    }
    normalized_counts = {
        order: np.empty(base.size, dtype=np.float32) for order in NORMALIZED_ORDERS
    }
    normalized_totals = {
        order: np.empty(base.size, dtype=np.float32) for order in NORMALIZED_ORDERS
    }
    tables = {order: {} for order in ORDERS}
    events = {order: deque() for order in ORDERS}
    recent = {order: {} for order in RECENT_ORDERS}
    normalized_tables = {order: {} for order in NORMALIZED_ORDERS}
    normalized_events = {order: deque() for order in NORMALIZED_ORDERS}
    history = bytearray()
    scored = 0
    stream_position = 0
    for row in rows_np:
        for position, raw_target in enumerate(row):
            target = int(raw_target)
            for order in ORDERS:
                context = bytes(history[-order:]) if len(history) >= order else None
                continuations = tables[order].get(context, {}) if context is not None else {}
                if position >= model.prediction_start:
                    count_values[order][scored] = continuations.get(target, 0)
                    total_values[order][scored] = sum(continuations.values())
            for order in RECENT_ORDERS:
                context = bytes(history[-order:]) if len(history) >= order else None
                match = recent[order].get(context) if context is not None else None
                if position >= model.prediction_start:
                    active = (
                        match is not None
                        and stream_position - match[1] <= args.window
                    )
                    recent_active[order][scored] = active
                    recent_matches[order][scored] = (
                        int(active and match[0] == target)
                    )
            for order in NORMALIZED_ORDERS:
                context = (
                    bytes(history[-order:]).translate(ASCII_CLASS_TABLE)
                    if len(history) >= order
                    else None
                )
                continuations = (
                    normalized_tables[order].get(context, {})
                    if context is not None
                    else {}
                )
                if position >= model.prediction_start:
                    normalized_counts[order][scored] = continuations.get(target, 0)
                    normalized_totals[order][scored] = sum(continuations.values())
            if position >= model.prediction_start:
                scored += 1
            for order in ORDERS:
                if len(history) < order:
                    continue
                context = bytes(history[-order:])
                continuations = tables[order].setdefault(context, {})
                continuations[target] = continuations.get(target, 0) + 1
                event_queue = events[order]
                event_queue.append((context, target))
                if len(event_queue) > args.window:
                    old_context, old_target = event_queue.popleft()
                    old = tables[order][old_context]
                    remaining = old[old_target] - 1
                    if remaining:
                        old[old_target] = remaining
                    else:
                        del old[old_target]
                    if not old:
                        del tables[order][old_context]
            for order in RECENT_ORDERS:
                if len(history) >= order:
                    recent[order][bytes(history[-order:])] = (
                        target,
                        stream_position,
                    )
            for order in NORMALIZED_ORDERS:
                if len(history) < order:
                    continue
                context = bytes(history[-order:]).translate(ASCII_CLASS_TABLE)
                continuations = normalized_tables[order].setdefault(context, {})
                continuations[target] = continuations.get(target, 0) + 1
                event_queue = normalized_events[order]
                event_queue.append((context, target))
                if len(event_queue) > args.window:
                    old_context, old_target = event_queue.popleft()
                    old = normalized_tables[order][old_context]
                    remaining = old[old_target] - 1
                    if remaining:
                        old[old_target] = remaining
                    else:
                        del old[old_target]
                    if not old:
                        del normalized_tables[order][old_context]
            history.append(target)
            stream_position += 1
    if scored != base.size:
        raise RuntimeError("cache statistic count does not match model probabilities")

    calls = 0
    def objective(log_strengths: np.ndarray) -> float:
        nonlocal calls
        probability = base.copy()
        strengths = np.exp(log_strengths)
        cursor = 0
        for order, strength in zip(ORDERS, strengths[cursor:]):
            probability = (
                count_values[order] + strength * probability
            ) / (total_values[order] + strength)
            cursor += 1
        for order, strength in zip(RECENT_ORDERS, strengths[cursor:]):
            active = recent_active[order]
            probability[active] = (
                recent_matches[order][active] + strength * probability[active]
            ) / (1.0 + strength)
            cursor += 1
        for order, strength in zip(NORMALIZED_ORDERS, strengths[cursor:]):
            probability = (
                normalized_counts[order] + strength * probability
            ) / (normalized_totals[order] + strength)
            cursor += 1
        calls += 1
        return float(-np.log(probability).mean() / math.log(2.0))

    initial_strengths = np.array(
        [
            0.952612865,
            5.397200451,
            12.828322483,
            4.410565602,
            4.693671987,
            9972064.254,
            21.85727624,
            31.88602784,
            279.8429725,
            4.0,
            4.0,
            4.0,
            4.0,
            16.0,
            64.0,
        ]
    )
    started = time.perf_counter()
    result = minimize(
        objective,
        np.log(initial_strengths),
        method="Powell",
        bounds=[(math.log(0.25), math.log(1e7))] * len(initial_strengths),
        options={"maxiter": 50, "xtol": 0.005, "ftol": 1e-9},
    )
    report = {
        "format": "layercake-online-cache-optimization/2",
        "status": "COMPLETE" if result.success else "LIMIT_REACHED",
        "orders": list(ORDERS),
        "strengths": np.exp(result.x[: len(ORDERS)]).tolist(),
        "recent_orders": list(RECENT_ORDERS),
        "recent_strengths": np.exp(
            result.x[len(ORDERS) : len(ORDERS) + len(RECENT_ORDERS)]
        ).tolist(),
        "normalized_orders": list(NORMALIZED_ORDERS),
        "normalized_strengths": np.exp(
            result.x[len(ORDERS) + len(RECENT_ORDERS) :]
        ).tolist(),
        "best_bpb": float(result.fun),
        "objective_calls": calls,
        "window": args.window,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_message": str(result.message),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
