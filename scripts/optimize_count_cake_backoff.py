"""Fit a small grouped backoff recipe on validation bytes."""

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


def _expand(values: np.ndarray, max_order: int) -> tuple[float, ...]:
    strengths = np.exp(values)
    expanded = []
    for order in range(1, max_order + 1):
        group = 0 if order == 1 else 1 if order == 2 else 2 if order == 3 else 3 if order <= 5 else 4 if order <= 8 else 5
        expanded.append(float(strengths[group]))
    return tuple(expanded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--robust-weight", type=float, default=0.25)
    args = parser.parse_args()
    device = torch.device("cuda")
    model, _ = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    if not 0.0 <= args.robust_weight <= 1.0:
        raise ValueError("robust-weight must be in [0, 1]")
    splits = []
    for source in args.eval:
        payload = Path(source).read_bytes()
        row_count = len(payload) // 1056
        rows = torch.frombuffer(
            bytearray(payload[: row_count * 1056]), dtype=torch.uint8
        ).reshape(row_count, 1056)
        batches = [
            rows[offset : offset + 128].to(device=device, dtype=torch.long)
            for offset in range(0, row_count, 128)
        ]
        splits.append((source, batches))
    evaluations = []

    @torch.inference_mode()
    def objective(log_strengths: np.ndarray) -> float:
        strengths = _expand(log_strengths, model.count_cake.max_order)
        model.count_cake.backoff_strengths = strengths
        split_bpb = []
        for _, batches in splits:
            total_nll = 0.0
            total_bytes = 0
            for batch in batches:
                log_probability = model.target_log_probs(batch)
                total_nll -= float(log_probability.sum())
                total_bytes += log_probability.numel()
            split_bpb.append(total_nll / total_bytes / math.log(2.0))
        mean_bpb = float(np.mean(split_bpb))
        bpb = mean_bpb + args.robust_weight * (max(split_bpb) - mean_bpb)
        evaluations.append(
            {
                "objective_bpb": bpb,
                "split_bpb": split_bpb,
                "strengths": list(strengths),
            }
        )
        print(json.dumps(evaluations[-1]), flush=True)
        return bpb

    started = time.perf_counter()
    current = model.count_cake.backoff_strengths
    initial = np.log(
        np.array([current[0], current[1], current[2], current[3], current[5], current[8]])
    )
    result = minimize(
        objective,
        initial,
        method="Powell",
        bounds=[(math.log(0.1), math.log(100_000.0))] * 6,
        options={"maxiter": args.max_iterations, "xtol": 0.01, "ftol": 1e-7},
    )
    best_objective = objective(result.x)
    report = {
        "format": "layercake-backoff-optimization/1",
        "status": "COMPLETE" if result.success else "LIMIT_REACHED",
        "best_bpb": float(best_objective),
        "best_strengths": list(_expand(result.x, model.count_cake.max_order)),
        "best_split_bpb": evaluations[-1]["split_bpb"],
        "splits": [source for source, _ in splits],
        "robust_weight": args.robust_weight,
        "evaluations": len(evaluations),
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_message": str(result.message),
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
