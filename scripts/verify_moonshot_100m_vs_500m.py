#!/usr/bin/env python
"""Verify the 100M LayerCake vs 500M transformer moonshot gate.

This script is deliberately strict. A PASS means the supplied artifacts prove a
100M-class LayerCake beats a 500M-class tokenizer transformer on the configured
training, quality, cost, and generation gates. Missing evidence is a FAIL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _latest(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("latest", data)


def _num(*values: Any, default: float = float("nan")) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _params(data: dict[str, Any]) -> int:
    latest = _latest(data)
    return int(
        _num(
            latest.get("trainable_params"),
            latest.get("parameters"),
            data.get("trainable_params"),
            data.get("parameters"),
            default=0,
        )
    )


def _bpb(data: dict[str, Any]) -> float:
    latest = _latest(data)
    general = data.get("general", {}) if isinstance(data.get("general"), dict) else {}
    return _num(latest.get("eval_bpb"), latest.get("bpb"), general.get("bpb"))


def _train_seconds(data: dict[str, Any]) -> float:
    latest = _latest(data)
    return _num(
        latest.get("elapsed_total_seconds"),
        latest.get("elapsed_seconds"),
        data.get("elapsed_seconds"),
    )


def _quality(data: dict[str, Any] | None) -> float:
    if not data:
        return float("nan")
    metrics = data.get("metrics", data)
    return _num(
        metrics.get("quality_score"),
        metrics.get("qa_quality_mean"),
        metrics.get("mean_quality"),
        metrics.get("quality"),
    )


def _generation_bps(data: dict[str, Any] | None) -> float:
    if not data:
        return float("nan")
    metrics = data.get("metrics", data)
    generation = metrics.get("generation", {}) if isinstance(metrics.get("generation"), dict) else {}
    return _num(
        metrics.get("generation_bytes_per_second"),
        metrics.get("bytes_per_second"),
        metrics.get("mean_bytes_per_second"),
        generation.get("mean_bytes_per_second"),
    )


def _ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("inf") if num > 0 else 0.0
    return num / den


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layercake-metrics", required=True, type=Path)
    parser.add_argument("--transformer-metrics", required=True, type=Path)
    parser.add_argument("--layercake-generation", type=Path)
    parser.add_argument("--transformer-generation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-layercake-params", type=float, default=75_000_000)
    parser.add_argument("--max-layercake-params", type=float, default=125_000_000)
    parser.add_argument("--min-transformer-params", type=float, default=500_000_000)
    parser.add_argument("--min-param-ratio", type=float, default=4.0)
    parser.add_argument("--min-training-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-cost-ratio", type=float, default=5.0)
    parser.add_argument("--min-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    parser.add_argument("--allow-missing-generation", action="store_true")
    args = parser.parse_args()

    lc = _load(args.layercake_metrics)
    tf = _load(args.transformer_metrics)
    lc_gen = _load(args.layercake_generation)
    tf_gen = _load(args.transformer_generation)

    lc_params = _params(lc)
    tf_params = _params(tf)
    lc_bpb = _bpb(lc)
    tf_bpb = _bpb(tf)
    lc_seconds = _train_seconds(lc)
    tf_seconds = _train_seconds(tf)
    lc_gen_bps = _generation_bps(lc_gen)
    tf_gen_bps = _generation_bps(tf_gen)
    lc_quality = _quality(lc_gen)
    tf_quality = _quality(tf_gen)

    ratios = {
        "parameter_ratio_transformer_over_layercake": _ratio(tf_params, lc_params),
        "bpb_ratio_layercake_over_transformer": _ratio(lc_bpb, tf_bpb),
        "training_speed_ratio_transformer_seconds_over_layercake_seconds": _ratio(tf_seconds, lc_seconds),
        "cost_proxy_ratio": _ratio(tf_seconds * tf_params, lc_seconds * lc_params),
        "generation_speed_ratio_layercake_over_transformer": _ratio(lc_gen_bps, tf_gen_bps),
        "quality_ratio_layercake_over_transformer": _ratio(lc_quality, tf_quality),
    }
    generation_present = lc_gen is not None and tf_gen is not None
    gates = {
        "layercake_is_100m_class": args.min_layercake_params <= lc_params <= args.max_layercake_params,
        "transformer_is_500m_class": tf_params >= args.min_transformer_params,
        "transformer_at_least_configured_multiple_larger": ratios["parameter_ratio_transformer_over_layercake"] >= args.min_param_ratio,
        "layercake_bpb_lower": lc_bpb < tf_bpb,
        "layercake_training_faster": ratios["training_speed_ratio_transformer_seconds_over_layercake_seconds"] >= args.min_training_speed_ratio,
        "layercake_cost_proxy_better": ratios["cost_proxy_ratio"] >= args.min_cost_ratio,
        "generation_evidence_present": generation_present or args.allow_missing_generation,
        "layercake_generation_faster": (
            args.allow_missing_generation and not generation_present
        )
        or ratios["generation_speed_ratio_layercake_over_transformer"] >= args.min_generation_speed_ratio,
        "layercake_generation_quality_noninferior": (
            args.allow_missing_generation and not generation_present
        )
        or ratios["quality_ratio_layercake_over_transformer"] >= args.min_quality_ratio,
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "100M-class LayerCake vs 500M-class BPE transformer moonshot certificate",
        "inputs": {
            "layercake_metrics": str(args.layercake_metrics),
            "transformer_metrics": str(args.transformer_metrics),
            "layercake_generation": str(args.layercake_generation) if args.layercake_generation else None,
            "transformer_generation": str(args.transformer_generation) if args.transformer_generation else None,
        },
        "thresholds": {
            "min_layercake_params": args.min_layercake_params,
            "max_layercake_params": args.max_layercake_params,
            "min_transformer_params": args.min_transformer_params,
            "min_param_ratio": args.min_param_ratio,
            "min_training_speed_ratio": args.min_training_speed_ratio,
            "min_cost_ratio": args.min_cost_ratio,
            "min_generation_speed_ratio": args.min_generation_speed_ratio,
            "min_quality_ratio": args.min_quality_ratio,
            "allow_missing_generation": args.allow_missing_generation,
        },
        "gates": gates,
        "metrics": {
            "layercake_params": lc_params,
            "transformer_params": tf_params,
            "layercake_bpb": lc_bpb,
            "transformer_bpb": tf_bpb,
            "layercake_train_seconds": lc_seconds,
            "transformer_train_seconds": tf_seconds,
            "layercake_generation_bytes_per_second": lc_gen_bps,
            "transformer_generation_bytes_per_second": tf_gen_bps,
            "layercake_quality": lc_quality,
            "transformer_quality": tf_quality,
        },
        "ratios": ratios,
        "interpretation": (
            "Moonshot gate passed for the supplied artifacts."
            if all(gates.values())
            else "Moonshot gate is not proven by the supplied artifacts. Do not promote a 100M-vs-500M dominance claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
