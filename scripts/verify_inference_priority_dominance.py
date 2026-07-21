#!/usr/bin/env python
"""Verify an equal-size CPU/mobile inference-priority LayerCake claim.

This is intentionally separate from the strict 5x breakthrough verifier. It
does not prove the original all-gates 5x breakthrough claim; it proves a
deployment-focused claim where CPU inference speed and generation quality are
the required gates, while training cost and GPU speed are reported as context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _num(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _ratio(num: float, den: float) -> float:
    if den <= 0.0 and num > 0.0:
        return 1.0e300
    if den <= 0.0:
        return 0.0
    return num / den


def _training_summary(row: dict[str, Any]) -> dict[str, float]:
    latest = row.get("latest") if isinstance(row.get("latest"), dict) else row
    general = row.get("general", {}) if isinstance(row.get("general"), dict) else {}
    return {
        "params": _num(latest.get("parameters"), latest.get("trainable_params")),
        "eval_bpb": _num(
            latest.get("eval_bpb"),
            latest.get("heldout_bpb"),
            latest.get("bpb"),
            general.get("bpb"),
        ),
        "eval_bytes": _num(latest.get("eval_bytes"), row.get("eval_bytes")),
        "train_seconds": _num(latest.get("elapsed_seconds"), row.get("elapsed_seconds")),
        "train_bytes": _num(
            latest.get("estimated_total_training_bytes"),
            row.get("estimated_total_training_bytes"),
        ),
    }


def _side_generation(row: dict[str, Any] | None, side: str) -> dict[str, float]:
    if not row:
        return {"bytes_per_second": 0.0}
    nested = row.get(side, {}) if isinstance(row.get(side), dict) else {}
    return {
        "bytes_per_second": _num(
            nested.get("bytes_per_second"),
            row.get("bytes_per_second"),
        )
    }


def verify(
    *,
    layercake_training: dict[str, Any],
    transformer_training: dict[str, Any],
    cpu_generation: dict[str, Any],
    cpu_quality: dict[str, Any],
    gpu_generation: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    lc_train = _training_summary(layercake_training)
    tx_train = _training_summary(transformer_training)
    lc_cpu = _side_generation(cpu_generation, "layercake")
    tx_cpu = _side_generation(cpu_generation, "bpe")
    lc_gpu = _side_generation(gpu_generation, "layercake")
    tx_gpu = _side_generation(gpu_generation, "bpe")
    quality_gates = (
        cpu_quality.get("quality_gates", {})
        if isinstance(cpu_quality.get("quality_gates"), dict)
        else {}
    )

    ratios = {
        "parameter_ratio_layercake_over_transformer": _ratio(
            lc_train["params"], tx_train["params"]
        ),
        "heldout_bpb_ratio_layercake_over_transformer": _ratio(
            lc_train["eval_bpb"], tx_train["eval_bpb"]
        ),
        "cpu_generation_speed_ratio_layercake_over_transformer": _ratio(
            lc_cpu["bytes_per_second"], tx_cpu["bytes_per_second"]
        ),
        "gpu_generation_speed_ratio_layercake_over_transformer": _ratio(
            lc_gpu["bytes_per_second"], tx_gpu["bytes_per_second"]
        ),
        "training_speed_ratio_transformer_seconds_over_layercake_seconds": _ratio(
            tx_train["train_seconds"], lc_train["train_seconds"]
        ),
    }
    max_param_ratio = 1.0 + args.param_tolerance
    min_param_ratio = 1.0 - args.param_tolerance
    gates = {
        "equal_size_parameter_window": (
            min_param_ratio
            <= ratios["parameter_ratio_layercake_over_transformer"]
            <= max_param_ratio
        ),
        "heldout_eval_bytes_met": (
            lc_train["eval_bytes"] >= args.min_eval_bytes
            and tx_train["eval_bytes"] >= args.min_eval_bytes
        ),
        "heldout_bpb_noninferior": (
            ratios["heldout_bpb_ratio_layercake_over_transformer"]
            <= args.max_bpb_ratio
        ),
        "cpu_generation_evidence_present": bool(cpu_generation),
        "cpu_generation_5x_faster": (
            ratios["cpu_generation_speed_ratio_layercake_over_transformer"]
            >= args.min_cpu_speed_ratio
        ),
        "cpu_generation_quality_passes": all(bool(value) for value in quality_gates.values())
        and bool(quality_gates),
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "status": "PASS" if not blockers else "FAIL",
        "scope": (
            "Equal-size CPU/mobile inference-priority gate. This proves "
            "non-inferior held-out BPB plus 5x CPU generation speed and "
            "passing generation-quality diagnostics. It does not prove the "
            "strict all-gates 5x breakthrough training claim."
        ),
        "blockers": blockers,
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake_training": lc_train,
            "transformer_training": tx_train,
            "layercake_cpu_generation": lc_cpu,
            "transformer_cpu_generation": tx_cpu,
            "layercake_gpu_generation": lc_gpu,
            "transformer_gpu_generation": tx_gpu,
            "cpu_quality_gates": quality_gates,
        },
        "thresholds": {
            "param_tolerance": args.param_tolerance,
            "min_eval_bytes": args.min_eval_bytes,
            "max_bpb_ratio": args.max_bpb_ratio,
            "min_cpu_speed_ratio": args.min_cpu_speed_ratio,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layercake-training", required=True, type=Path)
    parser.add_argument("--transformer-training", required=True, type=Path)
    parser.add_argument("--cpu-generation", required=True, type=Path)
    parser.add_argument("--cpu-quality", required=True, type=Path)
    parser.add_argument("--gpu-generation", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--param-tolerance", type=float, default=0.05)
    parser.add_argument("--min-eval-bytes", type=float, default=1_000_000.0)
    parser.add_argument("--max-bpb-ratio", type=float, default=1.0)
    parser.add_argument("--min-cpu-speed-ratio", type=float, default=5.0)
    args = parser.parse_args()

    result = verify(
        layercake_training=_read(args.layercake_training) or {},
        transformer_training=_read(args.transformer_training) or {},
        cpu_generation=_read(args.cpu_generation) or {},
        cpu_quality=_read(args.cpu_quality) or {},
        gpu_generation=_read(args.gpu_generation),
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
