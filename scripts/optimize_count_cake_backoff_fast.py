"""Fit grouped CountCake backoff strengths from precomputed causal statistics."""

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


def _expand(values: np.ndarray, max_order: int) -> np.ndarray:
    grouped = np.exp(values)
    return np.asarray(
        [
            grouped[
                0
                if order == 1
                else 1
                if order == 2
                else 2
                if order == 3
                else 3
                if order <= 5
                else 4
                if order <= 8
                else 5
            ]
            for order in range(1, max_order + 1)
        ],
        dtype=np.float64,
    )


def _count_statistics(cake, rows: torch.Tensor, start: int) -> dict:
    """Materialize observed continuation counts and context totals per order."""
    rows = rows.to(torch.int64)
    targets = rows[:, start:]
    smoothed = cake.unigram_counts + 0.5
    unigram = (smoothed[targets] / smoothed.sum()).cpu().numpy().astype(np.float64)
    counts: list[np.ndarray] = []
    totals: list[np.ndarray] = []
    distincts: list[np.ndarray] = []
    for order in range(1, cake.max_order + 1):
        context = torch.zeros_like(targets)
        if cake.order_encodings[order - 1] == "packed":
            for lag in range(order):
                context.add_(
                    rows[:, start - 1 - lag : rows.shape[1] - 1 - lag]
                    << (8 * lag)
                )
            joint = targets + context.bitwise_left_shift(8)
            joint_counts = cake._lookup(
                getattr(cake, f"keys_{order}"),
                getattr(cake, f"counts_{order}"),
                joint,
            )
            context_totals = cake._lookup(
                getattr(cake, f"context_keys_{order}"),
                getattr(cake, f"context_totals_{order}"),
                context,
            )
            context_distinct = cake._lookup(
                getattr(cake, f"context_keys_{order}"),
                getattr(cake, f"context_distinct_{order}"),
                context,
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
            context_keys = getattr(cake, f"context_keys_{order}")
            indices = torch.searchsorted(context_keys, context)
            safe = indices.clamp(max=context_keys.numel() - 1)
            found = (indices < context_keys.numel()) & (
                context_keys[safe] == context
            )
            joint = targets + safe * 256
            joint_counts = torch.where(
                found,
                cake._lookup(
                    getattr(cake, f"keys_{order}"),
                    getattr(cake, f"counts_{order}"),
                    joint,
                ),
                torch.zeros_like(targets, dtype=torch.float32),
            )
            context_totals = torch.where(
                found,
                getattr(cake, f"context_totals_{order}")[safe],
                torch.zeros_like(targets, dtype=torch.float32),
            )
            context_distinct = torch.where(
                found,
                getattr(cake, f"context_distinct_{order}")[safe],
                torch.zeros_like(targets, dtype=torch.float32),
            )
        counts.append(joint_counts.cpu().numpy().astype(np.float64))
        totals.append(context_totals.cpu().numpy().astype(np.float64))
        distincts.append(context_distinct.cpu().numpy().astype(np.float64))
    return {
        "unigram": unigram,
        "counts": counts,
        "totals": totals,
        "distincts": distincts,
    }


@torch.inference_mode()
def _prepare_split(model, rows: torch.Tensor) -> dict:
    context = model._patch_context(rows)
    targets = rows[:, model.prediction_start :].reshape(
        rows.shape[0], -1, model.patch_size
    )
    neural_log_probability, neural_hidden = model._neural_log_probs(context, targets)
    _, count_features = model.count_cake.target_log_probs(
        rows, start=model.prediction_start, return_features=True
    )
    count_features = count_features.reshape(*neural_log_probability.shape, 3)
    gate_logit = model.mixture_gate(neural_hidden)
    if model.confidence_gate_enabled:
        gate_logit = gate_logit + model.confidence_gate(count_features)
    return {
        "neural": neural_log_probability.reshape(rows.shape[0], -1)
        .exp()
        .cpu()
        .numpy()
        .astype(np.float64),
        "gate": torch.sigmoid(gate_logit)
        .squeeze(-1)
        .reshape(rows.shape[0], -1)
        .cpu()
        .numpy()
        .astype(np.float64),
        **_count_statistics(model.count_cake, rows, model.prediction_start),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--fit-bytes-per-split", type=int, default=262_144)
    parser.add_argument("--robust-weight", type=float, default=0.25)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.robust_weight <= 1.0:
        raise ValueError("robust-weight must be in [0, 1]")
    device = torch.device("cuda")
    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    splits = []
    for source in args.eval:
        full = Path(source).read_bytes()
        payload = full[: args.fit_bytes_per_split]
        row_count = len(payload) // 1056
        rows = torch.frombuffer(
            bytearray(payload[: row_count * 1056]), dtype=torch.uint8
        ).reshape(row_count, 1056).to(device=device, dtype=torch.long)
        prepared = _prepare_split(model, rows)
        prepared.update(
            {"path": source, "source_bytes": len(full), "fit_bytes": row_count * 1056}
        )
        splits.append(prepared)
        print(json.dumps({"prepared": source, "scored_bytes": int(prepared["gate"].size)}), flush=True)

    current = model.count_cake.backoff_strengths
    representative_orders = [1, 2, 3, 4, 6, 9]
    active_representatives = [
        order for order in representative_orders if order <= model.count_cake.max_order
    ]
    initial = np.log(
        np.asarray([current[order - 1] for order in active_representatives])
    )
    calls = 0

    def evaluate(log_strengths: np.ndarray) -> tuple[float, list[float]]:
        strengths = _expand(log_strengths, model.count_cake.max_order)
        values = []
        for split in splits:
            probability = split["unigram"].copy()
            for count, total, strength in zip(
                split["counts"], split["totals"], strengths
            ):
                probability = (count + strength * probability) / (total + strength)
            mixture = (
                (1.0 - split["gate"]) * probability
                + split["gate"] * split["neural"]
            )
            values.append(float(-np.log(mixture).mean() / math.log(2.0)))
        mean = float(np.mean(values))
        return mean + args.robust_weight * (max(values) - mean), values

    initial_objective, initial_values = evaluate(initial)
    started = time.perf_counter()

    def objective(log_strengths: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return evaluate(log_strengths)[0]

    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=[(math.log(0.1), math.log(100_000.0))] * len(initial),
        options={"maxiter": args.max_iterations, "xtol": 0.002, "ftol": 1e-10},
    )
    best_objective, best_values = evaluate(result.x)
    report = {
        "format": "layercake-fast-backoff-optimization/1",
        "status": "COMPLETE" if result.success else "LIMIT_REACHED",
        "bundle": args.bundle,
        "bundle_parameters": manifest["parameters"],
        "fit_bytes_per_split": args.fit_bytes_per_split,
        "robust_weight": args.robust_weight,
        "initial": {
            "objective_bpb": initial_objective,
            "split_bpb": initial_values,
            "strengths": list(model.count_cake.backoff_strengths),
        },
        "best": {
            "objective_bpb": best_objective,
            "split_bpb": best_values,
            "strengths": _expand(result.x, model.count_cake.max_order).tolist(),
        },
        "splits": [
            {k: split[k] for k in ("path", "source_bytes", "fit_bytes")}
            for split in splits
        ],
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
