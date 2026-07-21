"""Measure the target-aware quality ceiling across CountCake order stages."""

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
from scripts.optimize_count_cake_backoff_fast import _count_statistics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the order-oracle analysis requires CUDA")
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device="cuda")
    payload = Path(args.data).read_bytes()
    row_count = len(payload) // args.seq_len
    if row_count == 0:
        raise ValueError("data is shorter than one sequence")
    rows = torch.frombuffer(
        bytearray(payload[: row_count * args.seq_len]), dtype=torch.uint8
    ).reshape(row_count, args.seq_len).to(device="cuda", dtype=torch.long)
    with torch.inference_mode():
        statistics = _count_statistics(
            model.count_cake, rows, model.prediction_start
        )
    probability = statistics["unigram"]
    oracle = probability.copy()
    winner = np.zeros(probability.shape, dtype=np.uint8)
    stages = [
        {
            "order": 0,
            "bpb": float(-np.log(probability).mean() / math.log(2.0)),
        }
    ]
    strengths = model.count_cake.backoff_strengths
    for order, (count, total, strength) in enumerate(
        zip(statistics["counts"], statistics["totals"], strengths),
        start=1,
    ):
        probability = (count + strength * probability) / (total + strength)
        better = probability > oracle
        oracle = np.where(better, probability, oracle)
        winner[better] = order
        stages.append(
            {
                "order": order,
                "bpb": float(-np.log(probability).mean() / math.log(2.0)),
                "matched_context_fraction": float(np.mean(total > 0)),
                "observed_continuation_fraction": float(np.mean(count > 0)),
            }
        )
    unique, counts = np.unique(winner, return_counts=True)
    report = {
        "format": "layercake-count-cake-order-oracle/1",
        "status": "COMPLETE",
        "warning": (
            "target-aware order selection is unattainable at inference and is "
            "reported only as a lower-bound architecture diagnostic"
        ),
        "bundle": {
            "path": args.bundle,
            "logical_parameters": manifest["parameters"]["logical_total"],
            "max_order": model.count_cake.max_order,
        },
        "data": {
            "path": args.data,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "scored_bytes": int(oracle.size),
        },
        "stages": stages,
        "target_aware_oracle_bpb": float(
            -np.log(oracle).mean() / math.log(2.0)
        ),
        "target_aware_winner_fraction": {
            str(int(order)): float(count / oracle.size)
            for order, count in zip(unique, counts)
        },
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
