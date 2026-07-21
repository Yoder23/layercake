from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from verify_production_1m_vs_5m_dominance import _read, verify as _base_verify
except ModuleNotFoundError:  # pragma: no cover - exercised by package imports in tests
    from scripts.verify_production_1m_vs_5m_dominance import (
        _read,
        verify as _base_verify,
    )


def _ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def _scaled_cost_gates(result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metrics = result["metrics"]
    ratios = result["ratios"]
    lc = metrics["layercake"]
    tx = metrics["transformer"]
    training_cost_ratio = _ratio(
        float(tx["params"]) * float(tx["train_seconds"]),
        float(lc["params"]) * float(lc["train_seconds"]),
    )
    training_byte_ratio = _ratio(float(tx["train_bytes"]), float(lc["train_bytes"]))
    cpu_generation_cost_ratio = (
        ratios["parameter_ratio_transformer_over_layercake"]
        * ratios["cpu_generation_speed_ratio"]
    )
    gpu_generation_cost_ratio = (
        ratios["parameter_ratio_transformer_over_layercake"]
        * ratios["gpu_generation_speed_ratio"]
    )
    result["ratios"].update(
        {
            "training_cost_proxy_ratio": training_cost_ratio,
            "training_byte_efficiency_ratio": training_byte_ratio,
            "cpu_generation_cost_proxy_ratio": cpu_generation_cost_ratio,
            "gpu_generation_cost_proxy_ratio": gpu_generation_cost_ratio,
        }
    )
    result["gates"].update(
        {
            "training_cost_proxy_met": (
                training_cost_ratio >= args.min_training_cost_ratio
            ),
            "training_byte_efficiency_met": (
                training_byte_ratio >= args.min_training_byte_ratio
            ),
            "cpu_generation_cost_proxy_met": (
                cpu_generation_cost_ratio >= args.min_generation_cost_ratio
            ),
            "gpu_generation_cost_proxy_met": (
                gpu_generation_cost_ratio >= args.min_gpu_generation_cost_ratio
            ),
        }
    )
    result["status"] = "PASS" if all(result["gates"].values()) else "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify scaled LayerCake-vs-BPE production dominance with explicit cost gates."
    )
    parser.add_argument("--layercake-training", required=True, type=Path)
    parser.add_argument("--transformer-training", required=True, type=Path)
    parser.add_argument("--layercake-cpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-cpu-generation", required=True, type=Path)
    parser.add_argument("--layercake-gpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-gpu-generation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope-label", default="Scaled LayerCake vs tokenizer-transformer dominance certificate")
    parser.add_argument("--min-param-ratio", type=float, default=5.0)
    parser.add_argument("--min-training-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-training-cost-ratio", type=float, default=5.0)
    parser.add_argument("--min-training-byte-ratio", type=float, default=1.0)
    parser.add_argument("--min-cpu-generation-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-gpu-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-generation-cost-ratio", type=float, default=5.0)
    parser.add_argument("--min-gpu-generation-cost-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    args = parser.parse_args()
    result = _base_verify(
        layercake_training=_read(args.layercake_training),
        transformer_training=_read(args.transformer_training),
        layercake_cpu_generation=_read(args.layercake_cpu_generation),
        transformer_cpu_generation=_read(args.transformer_cpu_generation),
        layercake_gpu_generation=_read(args.layercake_gpu_generation),
        transformer_gpu_generation=_read(args.transformer_gpu_generation),
        args=args,
    )
    result = _scaled_cost_gates(result, args)
    result["scope"] = args.scope_label
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
