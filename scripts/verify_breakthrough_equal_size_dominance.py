#!/usr/bin/env python
"""Verify an equal-size LayerCake breakthrough dominance claim.

This gate is intentionally severe. A PASS means the supplied artifacts prove an
equal-size byte-level LayerCake beats a tokenizer transformer by configured
breakthrough margins on held-out modeling, training efficiency, CPU/GPU
inference, and generation/task quality evidence. Missing evidence is a FAIL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _latest(row: dict[str, Any]) -> dict[str, Any]:
    latest = row.get("latest")
    return latest if isinstance(latest, dict) else row


def _num(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _training_summary(row: dict[str, Any]) -> dict[str, float]:
    latest = _latest(row)
    general = row.get("general", {}) if isinstance(row.get("general"), dict) else {}
    return {
        "params": _num(
            latest.get("trainable_params"),
            latest.get("parameters"),
            row.get("trainable_params"),
            row.get("parameters"),
        ),
        "eval_bpb": _num(
            latest.get("eval_bpb"),
            latest.get("heldout_bpb"),
            latest.get("bpb"),
            row.get("eval_bpb"),
            row.get("heldout_bpb"),
            general.get("bpb"),
        ),
        "train_seconds": _num(
            latest.get("elapsed_total_seconds"),
            latest.get("elapsed_seconds"),
            row.get("elapsed_total_seconds"),
            row.get("elapsed_seconds"),
        ),
        "train_bytes": _num(
            latest.get("estimated_total_training_bytes"),
            row.get("estimated_total_training_bytes"),
            latest.get("train_bytes"),
            row.get("train_bytes"),
        ),
        "eval_bytes": _num(latest.get("eval_bytes"), row.get("eval_bytes")),
    }


def _generation_summary(row: dict[str, Any] | None) -> dict[str, float]:
    if not row:
        return {
            "bytes_per_second": 0.0,
            "quality_score": 0.0,
            "task_score": 0.0,
            "relevance_rate": 0.0,
            "sample_count": 0.0,
        }
    metrics = row.get("metrics", row)
    nested_generation = (
        metrics.get("generation", {}) if isinstance(metrics.get("generation"), dict) else {}
    )
    samples = row.get("samples", [])
    return {
        "bytes_per_second": _num(
            metrics.get("generation_bytes_per_second"),
            metrics.get("bytes_per_second"),
            metrics.get("mean_bytes_per_second"),
            nested_generation.get("mean_bytes_per_second"),
            row.get("speed_bytes_per_second"),
        ),
        "quality_score": _num(
            metrics.get("quality_score"),
            metrics.get("mean_quality"),
            metrics.get("quality"),
            row.get("quality_score"),
        ),
        "task_score": _num(
            metrics.get("task_score"),
            metrics.get("qa_quality_mean"),
            metrics.get("accuracy"),
            metrics.get("pass_rate"),
            row.get("task_score"),
        ),
        "relevance_rate": _num(metrics.get("relevance_rate"), row.get("relevance_rate")),
        "sample_count": float(len(samples)) if isinstance(samples, list) else 0.0,
    }


def _ratio(num: float, den: float) -> float:
    if den <= 0.0 and num > 0.0:
        return 1.0e300
    if den <= 0.0:
        return 0.0
    return num / den


def verify(
    *,
    layercake_training: dict[str, Any],
    transformer_training: dict[str, Any],
    layercake_cpu_generation: dict[str, Any] | None,
    transformer_cpu_generation: dict[str, Any] | None,
    layercake_gpu_generation: dict[str, Any] | None,
    transformer_gpu_generation: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    lc_train = _training_summary(layercake_training)
    tx_train = _training_summary(transformer_training)
    lc_cpu = _generation_summary(layercake_cpu_generation)
    tx_cpu = _generation_summary(transformer_cpu_generation)
    lc_gpu = _generation_summary(layercake_gpu_generation)
    tx_gpu = _generation_summary(transformer_gpu_generation)

    param_ratio = _ratio(lc_train["params"], tx_train["params"])
    max_param_ratio = 1.0 + float(args.param_tolerance)
    min_param_ratio = 1.0 - float(args.param_tolerance)
    cpu_quality_ratio = _ratio(lc_cpu["quality_score"], tx_cpu["quality_score"])
    gpu_quality_ratio = _ratio(lc_gpu["quality_score"], tx_gpu["quality_score"])
    cpu_task_ratio = _ratio(lc_cpu["task_score"], tx_cpu["task_score"])
    gpu_task_ratio = _ratio(lc_gpu["task_score"], tx_gpu["task_score"])

    ratios = {
        "parameter_ratio_layercake_over_transformer": param_ratio,
        "heldout_bpb_ratio_layercake_over_transformer": _ratio(
            lc_train["eval_bpb"], tx_train["eval_bpb"]
        ),
        "heldout_bpb_improvement_transformer_over_layercake": _ratio(
            tx_train["eval_bpb"], lc_train["eval_bpb"]
        ),
        "training_speed_ratio_transformer_seconds_over_layercake_seconds": _ratio(
            tx_train["train_seconds"], lc_train["train_seconds"]
        ),
        "training_cost_proxy_ratio_transformer_over_layercake": _ratio(
            tx_train["params"] * tx_train["train_seconds"],
            lc_train["params"] * lc_train["train_seconds"],
        ),
        "training_byte_ratio_transformer_over_layercake": _ratio(
            tx_train["train_bytes"], lc_train["train_bytes"]
        ),
        "cpu_generation_speed_ratio_layercake_over_transformer": _ratio(
            lc_cpu["bytes_per_second"], tx_cpu["bytes_per_second"]
        ),
        "gpu_generation_speed_ratio_layercake_over_transformer": _ratio(
            lc_gpu["bytes_per_second"], tx_gpu["bytes_per_second"]
        ),
        "cpu_generation_quality_ratio_layercake_over_transformer": cpu_quality_ratio,
        "gpu_generation_quality_ratio_layercake_over_transformer": gpu_quality_ratio,
        "cpu_task_score_ratio_layercake_over_transformer": cpu_task_ratio,
        "gpu_task_score_ratio_layercake_over_transformer": gpu_task_ratio,
        "cpu_relevance_ratio_layercake_over_transformer": _ratio(
            lc_cpu["relevance_rate"], tx_cpu["relevance_rate"]
        ),
        "gpu_relevance_ratio_layercake_over_transformer": _ratio(
            lc_gpu["relevance_rate"], tx_gpu["relevance_rate"]
        ),
    }

    cpu_generation_present = (
        layercake_cpu_generation is not None and transformer_cpu_generation is not None
    )
    gpu_generation_present = (
        layercake_gpu_generation is not None and transformer_gpu_generation is not None
    )
    cpu_quality_breakthrough = cpu_generation_present and (
        cpu_quality_ratio >= args.min_generation_quality_ratio
        or cpu_task_ratio >= args.min_task_score_ratio
    )
    gpu_quality_breakthrough = gpu_generation_present and (
        gpu_quality_ratio >= args.min_generation_quality_ratio
        or gpu_task_ratio >= args.min_task_score_ratio
    )
    gates = {
        "layercake_training_evidence_present": bool(layercake_training),
        "transformer_training_evidence_present": bool(transformer_training),
        "cpu_generation_evidence_present": cpu_generation_present,
        "gpu_generation_evidence_present": gpu_generation_present,
        "equal_size_parameter_window": min_param_ratio <= param_ratio <= max_param_ratio,
        "heldout_eval_bytes_met": (
            lc_train["eval_bytes"] >= args.min_eval_bytes
            and tx_train["eval_bytes"] >= args.min_eval_bytes
        ),
        "heldout_bpb_5x_better": (
            ratios["heldout_bpb_improvement_transformer_over_layercake"]
            >= args.min_quality_bpb_improvement_ratio
        ),
        "training_5x_faster": (
            ratios["training_speed_ratio_transformer_seconds_over_layercake_seconds"]
            >= args.min_training_speed_ratio
        ),
        "training_cost_5x_lower": (
            ratios["training_cost_proxy_ratio_transformer_over_layercake"]
            >= args.min_training_cost_ratio
        ),
        "no_more_training_bytes": (
            lc_train["train_bytes"] <= tx_train["train_bytes"] * args.max_train_byte_ratio
        ),
        "cpu_inference_5x_faster": (
            ratios["cpu_generation_speed_ratio_layercake_over_transformer"]
            >= args.min_inference_speed_ratio
        ),
        "gpu_inference_5x_faster": (
            ratios["gpu_generation_speed_ratio_layercake_over_transformer"]
            >= args.min_inference_speed_ratio
        ),
        "cpu_generation_or_task_quality_5x": cpu_quality_breakthrough,
        "gpu_generation_or_task_quality_5x": gpu_quality_breakthrough,
        "cpu_relevance_noninferior": (
            lc_cpu["relevance_rate"] >= args.min_layercake_relevance
            and ratios["cpu_relevance_ratio_layercake_over_transformer"]
            >= args.min_relevance_ratio
        ),
        "gpu_relevance_noninferior": (
            lc_gpu["relevance_rate"] >= args.min_layercake_relevance
            and ratios["gpu_relevance_ratio_layercake_over_transformer"]
            >= args.min_relevance_ratio
        ),
    }
    required = {
        "max_layercake_bpb_for_5x": (
            tx_train["eval_bpb"] / args.min_quality_bpb_improvement_ratio
            if args.min_quality_bpb_improvement_ratio > 0.0
            else 0.0
        ),
        "max_layercake_train_seconds_for_5x_speed": (
            tx_train["train_seconds"] / args.min_training_speed_ratio
            if args.min_training_speed_ratio > 0.0
            else 0.0
        ),
        "max_layercake_train_seconds_for_5x_cost_proxy": (
            (tx_train["params"] * tx_train["train_seconds"])
            / (lc_train["params"] * args.min_training_cost_ratio)
            if lc_train["params"] > 0.0 and args.min_training_cost_ratio > 0.0
            else 0.0
        ),
        "max_layercake_train_bytes": tx_train["train_bytes"] * args.max_train_byte_ratio,
    }
    shortfall = {
        "layercake_bpb_over_5x_target": _ratio(
            lc_train["eval_bpb"], required["max_layercake_bpb_for_5x"]
        ),
        "layercake_train_seconds_over_5x_speed_target": _ratio(
            lc_train["train_seconds"],
            required["max_layercake_train_seconds_for_5x_speed"],
        ),
        "layercake_train_seconds_over_5x_cost_target": _ratio(
            lc_train["train_seconds"],
            required["max_layercake_train_seconds_for_5x_cost_proxy"],
        ),
        "layercake_train_bytes_over_allowed": _ratio(
            lc_train["train_bytes"], required["max_layercake_train_bytes"]
        ),
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "status": "PASS" if not blockers else "FAIL",
        "scope": (
            "Equal-size breakthrough gate: byte-level LayerCake must beat a tokenizer "
            "transformer within the configured parameter window by 5x-class held-out "
            "BPB, training, cost, CPU/GPU inference, and generation/task-quality gates."
        ),
        "blockers": blockers,
        "thresholds": {
            "param_tolerance": args.param_tolerance,
            "min_eval_bytes": args.min_eval_bytes,
            "min_quality_bpb_improvement_ratio": args.min_quality_bpb_improvement_ratio,
            "min_training_speed_ratio": args.min_training_speed_ratio,
            "min_training_cost_ratio": args.min_training_cost_ratio,
            "max_train_byte_ratio": args.max_train_byte_ratio,
            "min_inference_speed_ratio": args.min_inference_speed_ratio,
            "min_generation_quality_ratio": args.min_generation_quality_ratio,
            "min_task_score_ratio": args.min_task_score_ratio,
            "min_relevance_ratio": args.min_relevance_ratio,
            "min_layercake_relevance": args.min_layercake_relevance,
        },
        "required_for_pass": required,
        "shortfall": shortfall,
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake_training": lc_train,
            "transformer_training": tx_train,
            "layercake_cpu_generation": lc_cpu,
            "transformer_cpu_generation": tx_cpu,
            "layercake_gpu_generation": lc_gpu,
            "transformer_gpu_generation": tx_gpu,
        },
        "interpretation": (
            "Breakthrough dominance is proven for the supplied artifacts."
            if not blockers
            else "Breakthrough dominance is not proven. Do not promote the claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layercake-training", required=True, type=Path)
    parser.add_argument("--transformer-training", required=True, type=Path)
    parser.add_argument("--layercake-cpu-generation", type=Path)
    parser.add_argument("--transformer-cpu-generation", type=Path)
    parser.add_argument("--layercake-gpu-generation", type=Path)
    parser.add_argument("--transformer-gpu-generation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--param-tolerance", type=float, default=0.05)
    parser.add_argument("--min-eval-bytes", type=float, default=1_000_000.0)
    parser.add_argument("--min-quality-bpb-improvement-ratio", type=float, default=5.0)
    parser.add_argument("--min-training-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-training-cost-ratio", type=float, default=5.0)
    parser.add_argument("--max-train-byte-ratio", type=float, default=1.0)
    parser.add_argument("--min-inference-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-generation-quality-ratio", type=float, default=5.0)
    parser.add_argument("--min-task-score-ratio", type=float, default=5.0)
    parser.add_argument("--min-relevance-ratio", type=float, default=1.0)
    parser.add_argument("--min-layercake-relevance", type=float, default=1.0)
    args = parser.parse_args()

    result = verify(
        layercake_training=_read(args.layercake_training) or {},
        transformer_training=_read(args.transformer_training) or {},
        layercake_cpu_generation=_read(args.layercake_cpu_generation),
        transformer_cpu_generation=_read(args.transformer_cpu_generation),
        layercake_gpu_generation=_read(args.layercake_gpu_generation),
        transformer_gpu_generation=_read(args.transformer_gpu_generation),
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
