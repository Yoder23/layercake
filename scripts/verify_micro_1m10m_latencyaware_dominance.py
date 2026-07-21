#!/usr/bin/env python
"""Verify latency-aware 1M-10M LayerCake dominance artifacts.

This verifier is intentionally stricter than the benchmark status field:
it combines focused per-scale artifacts, recomputes ratios, and only passes
when every requested scale beats the transformer baseline on loss, training
time, generation latency, quality heuristic, parameter count, and cost proxy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SCALES = ("1m", "2m", "5m", "10m")


def _load_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("scales", []):
            scale = str(row.get("scale", ""))
            if scale:
                rows[scale] = row
    return rows


def _ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def _verify_scale(
    row: dict[str, Any],
    *,
    min_cost_ratio: float,
    min_total_train_ratio: float,
    min_raw_train_ratio: float,
    min_generation_speed_ratio: float,
    min_quality_ratio: float,
) -> dict[str, Any]:
    lc = row["layercake"]
    bpe = row["baseline"]
    cost = row["cost_proxy_param_seconds"]

    lc_train_seconds = float(lc["train"]["elapsed_seconds"])
    bpe_raw_train_seconds = float(bpe["train"]["elapsed_seconds"])
    bpe_total_train_seconds = float(bpe["train"].get("elapsed_total_seconds", bpe_raw_train_seconds))
    lc_gen_bps = float(lc["generation"]["mean_bytes_per_second"])
    bpe_gen_bps = float(bpe["generation"]["mean_bytes_per_second"])
    lc_quality = float(lc["qa_quality_mean"])
    bpe_quality = float(bpe["qa_quality_mean"])
    lc_bpb = float(lc["general_bpb"])
    bpe_bpb = float(bpe["general_bpb"])
    lc_params = int(lc["params"])
    bpe_params = int(bpe["params"])

    ratios = {
        "raw_training_speed_ratio": _ratio(bpe_raw_train_seconds, lc_train_seconds),
        "total_training_speed_ratio": _ratio(bpe_total_train_seconds, lc_train_seconds),
        "cost_proxy_ratio": _ratio(float(cost["baseline"]), float(cost["layercake"])),
        "generation_speed_ratio": _ratio(lc_gen_bps, bpe_gen_bps),
        "quality_ratio": _ratio(lc_quality, bpe_quality),
        "bpb_ratio": _ratio(lc_bpb, bpe_bpb),
        "parameter_ratio_baseline_over_layercake": _ratio(bpe_params, lc_params),
    }
    gates = {
        "benchmark_status_pass": row.get("status") == "PASS",
        "bpb_lower": lc_bpb < bpe_bpb,
        "params_lower": lc_params < bpe_params,
        "raw_training_faster": ratios["raw_training_speed_ratio"] >= min_raw_train_ratio,
        "total_training_faster": ratios["total_training_speed_ratio"] >= min_total_train_ratio,
        "cost_proxy_ratio_met": ratios["cost_proxy_ratio"] >= min_cost_ratio,
        "generation_faster": ratios["generation_speed_ratio"] >= min_generation_speed_ratio,
        "quality_ratio_met": ratios["quality_ratio"] >= min_quality_ratio,
    }
    return {
        "scale": row["scale"],
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "ratios": ratios,
        "layercake": {
            "params": lc_params,
            "general_bpb": lc_bpb,
            "train_seconds": lc_train_seconds,
            "generation_bytes_per_second": lc_gen_bps,
            "quality": lc_quality,
            "selected_model": lc.get("selected_model"),
        },
        "baseline": {
            "params": bpe_params,
            "general_bpb": bpe_bpb,
            "raw_train_seconds": bpe_raw_train_seconds,
            "total_train_seconds": bpe_total_train_seconds,
            "generation_bytes_per_second": bpe_gen_bps,
            "quality": bpe_quality,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-cost-ratio", type=float, default=5.0)
    parser.add_argument("--min-total-train-ratio", type=float, default=1.0)
    parser.add_argument("--min-raw-train-ratio", type=float, default=1.0)
    parser.add_argument("--min-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    args = parser.parse_args()

    rows = _load_rows(args.artifacts)
    missing = [scale for scale in REQUIRED_SCALES if scale not in rows]
    scale_results = [
        _verify_scale(
            rows[scale],
            min_cost_ratio=args.min_cost_ratio,
            min_total_train_ratio=args.min_total_train_ratio,
            min_raw_train_ratio=args.min_raw_train_ratio,
            min_generation_speed_ratio=args.min_generation_speed_ratio,
            min_quality_ratio=args.min_quality_ratio,
        )
        for scale in REQUIRED_SCALES
        if scale in rows
    ]
    result = {
        "status": "PASS" if not missing and all(row["status"] == "PASS" for row in scale_results) else "FAIL",
        "scope": "Latency-aware micro dominance certificate for LayerCake vs BPE transformers at 1M/2M/5M/10M",
        "required_scales": list(REQUIRED_SCALES),
        "missing_scales": missing,
        "thresholds": {
            "min_cost_ratio": args.min_cost_ratio,
            "min_total_train_ratio": args.min_total_train_ratio,
            "min_raw_train_ratio": args.min_raw_train_ratio,
            "min_generation_speed_ratio": args.min_generation_speed_ratio,
            "min_quality_ratio": args.min_quality_ratio,
        },
        "scales": scale_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
