"""Fit and cross-check normalized CountCake backoff families.

This is a development probe: it fits only on the named development split and
reports untouched comparison splits separately.  It never opens a sealed split.
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
from scipy.optimize import minimize
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from scripts.optimize_count_cake_backoff_fast import _count_statistics  # noqa: E402


def _rows(
    path: str,
    *,
    seq_len: int,
    device: torch.device,
    max_bytes: int | None = None,
) -> tuple[torch.Tensor, dict]:
    payload = Path(path).read_bytes()
    full_row_count = len(payload) // seq_len
    row_count = full_row_count
    selected_indices: np.ndarray | None = None
    if max_bytes is not None:
        requested_rows = max(1, max_bytes // seq_len)
        row_count = min(full_row_count, requested_rows)
        selected_indices = np.linspace(
            0, full_row_count - 1, row_count, dtype=np.int64
        )
    if row_count == 0:
        raise ValueError(f"{path} is shorter than one evaluation row")
    used = payload[: full_row_count * seq_len]
    rows = torch.frombuffer(bytearray(used), dtype=torch.uint8).reshape(
        full_row_count, seq_len
    )
    if selected_indices is not None:
        rows = rows[torch.from_numpy(selected_indices)]
    return rows.to(device=device, dtype=torch.long), {
        "path": path,
        "source_bytes": len(payload),
        "fit_bytes": row_count * seq_len if max_bytes is not None else None,
        "evaluated_rows": row_count,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _bpb(probability: np.ndarray) -> float:
    return float(-np.log(np.clip(probability, 1e-30, None)).mean() / math.log(2.0))


def _expand(parameters: np.ndarray, order_count: int) -> np.ndarray:
    """Share fit parameters over stable order bands to bound optimization."""
    return np.asarray(
        [
            parameters[
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
            for order in range(1, order_count + 1)
        ],
        dtype=np.float64,
    )


def _cascade(statistics: dict, mode: str, grouped: np.ndarray) -> np.ndarray:
    probability = statistics["unigram"].copy()
    parameters = _expand(grouped, len(statistics["counts"]))
    for index, (count, total, distinct) in enumerate(
        zip(statistics["counts"], statistics["totals"], statistics["distincts"])
    ):
        if mode == "fixed":
            strength = float(np.exp(parameters[index]))
            probability = (count + strength * probability) / (total + strength)
        elif mode == "distinct":
            strength = float(np.exp(parameters[index])) * distinct
            updated = (count + strength * probability) / np.maximum(
                total + strength, 1.0
            )
            probability = np.where(total > 0.0, updated, probability)
        elif mode == "discount":
            discount = float(1.0 / (1.0 + np.exp(-parameters[index])))
            discounted = np.maximum(count - discount, 0.0)
            escape = discount * distinct
            updated = (discounted + escape * probability) / np.maximum(total, 1.0)
            probability = np.where(total > 0.0, updated, probability)
        else:
            raise ValueError(mode)
    return probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--fit", required=True)
    parser.add_argument("--test", action="append", default=[])
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--fit-bytes", type=int, default=262_144)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    sources = [args.fit, *args.test]
    splits = []
    with torch.inference_mode():
        for split_index, path in enumerate(sources):
            rows, metadata = _rows(
                path,
                seq_len=args.seq_len,
                device=device,
                max_bytes=args.fit_bytes if split_index == 0 else None,
            )
            statistics = _count_statistics(
                model.count_cake, rows, model.prediction_start
            )
            splits.append({"metadata": metadata, "statistics": statistics})
            print(json.dumps({"prepared": path, "rows": rows.shape[0]}), flush=True)

    order_count = model.count_cake.max_order
    modes = {}
    group_count = 6
    strengths = model.count_cake.backoff_strengths
    initial_by_mode = {
        "fixed": np.log(
            np.asarray([strengths[0], strengths[1], strengths[2], strengths[3], strengths[5], strengths[8]])
        ),
        "distinct": np.zeros(group_count, dtype=np.float64),
        "discount": np.full(group_count, math.log(0.75 / 0.25), dtype=np.float64),
    }
    bounds_by_mode = {
        "fixed": [(math.log(0.01), math.log(100_000.0))] * group_count,
        "distinct": [(math.log(0.01), math.log(100.0))] * group_count,
        "discount": [(-8.0, 8.0)] * group_count,
    }
    for mode in ("fixed", "distinct", "discount"):
        initial = initial_by_mode[mode]

        def objective(parameters: np.ndarray) -> float:
            return _bpb(_cascade(splits[0]["statistics"], mode, parameters))

        initial_bpb = objective(initial)
        result = minimize(
            objective,
            initial,
            method="Powell",
            bounds=bounds_by_mode[mode],
            options={
                "maxiter": args.max_iterations,
                "xtol": 0.005,
                "ftol": 1e-10,
            },
        )
        fitted = np.asarray(result.x, dtype=np.float64)
        if mode == "discount":
            rendered_parameters = _expand(
                1.0 / (1.0 + np.exp(-fitted)), order_count
            ).tolist()
        else:
            rendered_parameters = _expand(np.exp(fitted), order_count).tolist()
        modes[mode] = {
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "initial_fit_bpb": initial_bpb,
            "parameters_by_order": rendered_parameters,
            "split_bpb": [
                _bpb(_cascade(split["statistics"], mode, fitted))
                for split in splits
            ],
        }
        print(json.dumps({"mode": mode, **modes[mode]}), flush=True)

    report = {
        "format": "layercake-count-cake-backoff-probe/1",
        "status": "COMPLETE",
        "warning": "parameters fit on the development split only; test splits are untouched cross-checks",
        "bundle": args.bundle,
        "logical_parameters": manifest["parameters"]["logical_total"],
        "max_order": order_count,
        "splits": [split["metadata"] for split in splits],
        "modes": modes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
