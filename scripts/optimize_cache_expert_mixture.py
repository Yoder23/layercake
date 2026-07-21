"""Fit a normalized convex mixture of frozen causal cache experts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from scripts.optimize_fixed_cache_recipe import (  # noqa: E402
    _base_probabilities,
    _cache_statistics,
)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--fit-bytes-per-split", type=int, default=65_536)
    parser.add_argument("--window", type=int, default=1344)
    parser.add_argument("--robust-weight", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    exact_specs = model.online_cache_specs
    recent_specs = model.recent_cache_specs
    normalized_specs = model.normalized_cache_specs
    if not (exact_specs and recent_specs and normalized_specs):
        raise ValueError("bundle requires exact, recent, and normalized experts")
    exact_orders = tuple(order for order, _ in exact_specs)
    recent_orders = tuple(order for order, _ in recent_specs)
    normalized_orders = tuple(order for order, _ in normalized_specs)
    names = [
        "base",
        *[f"exact_{order}" for order in exact_orders],
        *[f"recent_{order}" for order in recent_orders],
        *[f"normalized_{order}" for order in normalized_orders],
    ]
    matrices = []
    split_metadata = []
    for source in args.eval:
        full = Path(source).read_bytes()
        payload = full[: args.fit_bytes_per_split]
        row_count = len(payload) // 1056
        rows = np.frombuffer(
            payload[: row_count * 1056], dtype=np.uint8
        ).reshape(row_count, 1056).copy()
        base = _base_probabilities(model, rows, device).reshape(-1)
        stats = _cache_statistics(
            rows,
            prediction_start=model.prediction_start,
            window=args.window,
            exact_orders=exact_orders,
            recent_orders=recent_orders,
            normalized_orders=normalized_orders,
        )
        experts = [base]
        for order, strength in exact_specs:
            experts.append(
                (stats["exact_counts"][order] + strength * base)
                / (stats["exact_totals"][order] + strength)
            )
        for order, strength in recent_specs:
            probability = base.copy()
            active = stats["recent_active"][order]
            probability[active] = (
                stats["recent_matches"][order][active]
                + strength * base[active]
            ) / (1.0 + strength)
            experts.append(probability)
        for order, strength in normalized_specs:
            experts.append(
                (stats["normalized_counts"][order] + strength * base)
                / (stats["normalized_totals"][order] + strength)
            )
        matrix = np.stack(experts, axis=1).astype(np.float64)
        matrices.append(matrix)
        split_metadata.append(
            {
                "path": source,
                "source_bytes": len(full),
                "fit_bytes": row_count * 1056,
                "scored_bytes": int(matrix.shape[0]),
                "expert_bpb": {
                    name: float(-np.log(matrix[:, index]).mean() / math.log(2.0))
                    for index, name in enumerate(names)
                },
                "target_oracle_bpb": float(
                    -np.log(matrix.max(axis=1)).mean() / math.log(2.0)
                ),
            }
        )
        print(json.dumps({"prepared": source, "scored_bytes": matrix.shape[0]}), flush=True)

    calls = 0

    def evaluate(logits: np.ndarray) -> tuple[float, list[float]]:
        weights = _softmax(logits)
        values = [
            float(-np.log(matrix @ weights).mean() / math.log(2.0))
            for matrix in matrices
        ]
        mean = float(np.mean(values))
        return mean + args.robust_weight * (max(values) - mean), values

    def objective(logits: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return evaluate(logits)[0]

    initial = np.full(len(names), -4.0)
    initial[0] = 0.0
    started = time.perf_counter()
    result = minimize(
        objective,
        initial,
        method="BFGS",
        options={"maxiter": 500, "gtol": 1e-9},
    )
    best_objective, best_values = evaluate(result.x)
    weights = _softmax(result.x)
    report = {
        "format": "layercake-cache-expert-mixture-optimization/1",
        "status": "COMPLETE" if result.success else "LIMIT_REACHED",
        "bundle": args.bundle,
        "bundle_parameters": manifest["parameters"],
        "window": args.window,
        "fit_bytes_per_split": args.fit_bytes_per_split,
        "robust_weight": args.robust_weight,
        "experts": names,
        "weights": {name: float(weight) for name, weight in zip(names, weights)},
        "best_objective_bpb": best_objective,
        "best_split_bpb": best_values,
        "splits": split_metadata,
        "objective_calls": calls,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_message": str(result.message),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
