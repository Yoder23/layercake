from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0.0 else num / den


def _quality_gates(path: Path) -> dict[str, bool]:
    row = _read(path)
    return {key: bool(value) for key, value in row.get("quality_gates", {}).items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify measured equal-size LayerCake dominance from concrete artifacts."
    )
    parser.add_argument("--layercake-training", nargs="+", required=True, type=Path)
    parser.add_argument("--transformer-training", required=True, type=Path)
    parser.add_argument("--cpu-generation", required=True, type=Path)
    parser.add_argument("--gpu-generation", required=True, type=Path)
    parser.add_argument("--generation-quality", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--param-tolerance", type=float, default=0.05)
    parser.add_argument("--max-training-time-ratio", type=float, default=1.01)
    parser.add_argument("--min-cpu-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-gpu-speed-ratio", type=float, default=1.0)
    args = parser.parse_args()

    lc_parts = [_read(path) for path in args.layercake_training]
    tx = _read(args.transformer_training)
    cpu = _read(args.cpu_generation)
    gpu = _read(args.gpu_generation)
    quality_gates = _quality_gates(args.generation_quality)

    lc_params = float(lc_parts[-1]["parameters"])
    tx_params = float(tx["parameters"])
    lc_seconds = sum(float(row["elapsed_seconds"]) for row in lc_parts)
    tx_seconds = float(tx["elapsed_seconds"])
    lc_train_bytes = sum(float(row["estimated_total_training_bytes"]) for row in lc_parts)
    tx_train_bytes = float(tx["estimated_total_training_bytes"])
    lc_bpb = float(lc_parts[-1]["general"]["bpb"])
    tx_bpb = float(tx["general"]["bpb"])
    cpu_ratio = float(cpu["speed_ratio"])
    gpu_ratio = float(gpu["speed_ratio"])

    ratios = {
        "parameter_ratio_layercake_over_transformer": _ratio(lc_params, tx_params),
        "heldout_bpb_ratio_layercake_over_transformer": _ratio(lc_bpb, tx_bpb),
        "heldout_bpb_improvement_transformer_over_layercake": _ratio(tx_bpb, lc_bpb),
        "training_time_ratio_layercake_over_transformer": _ratio(lc_seconds, tx_seconds),
        "training_speed_ratio_transformer_over_layercake": _ratio(tx_seconds, lc_seconds),
        "training_cost_proxy_ratio_transformer_over_layercake": _ratio(
            tx_params * tx_seconds,
            lc_params * lc_seconds,
        ),
        "training_byte_ratio_transformer_over_layercake": _ratio(tx_train_bytes, lc_train_bytes),
        "cpu_generation_speed_ratio": cpu_ratio,
        "gpu_generation_speed_ratio": gpu_ratio,
    }
    gates = {
        "equal_size_parameter_window": (
            (1.0 - args.param_tolerance)
            <= ratios["parameter_ratio_layercake_over_transformer"]
            <= (1.0 + args.param_tolerance)
        ),
        "heldout_bpb_lower": lc_bpb < tx_bpb,
        "training_time_noninferior": (
            ratios["training_time_ratio_layercake_over_transformer"]
            <= args.max_training_time_ratio
        ),
        "training_cost_proxy_lower": (
            ratios["training_cost_proxy_ratio_transformer_over_layercake"] > 1.0
        ),
        "training_byte_efficiency": (
            ratios["training_byte_ratio_transformer_over_layercake"] > 1.0
        ),
        "cpu_generation_at_least_5x": cpu_ratio >= args.min_cpu_speed_ratio,
        "gpu_generation_noninferior": gpu_ratio >= args.min_gpu_speed_ratio,
        "generation_quality_gates_pass": bool(quality_gates)
        and all(quality_gates.values()),
    }
    failed = [name for name, passed in gates.items() if not passed]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "Measured equal-size LayerCake-vs-tokenizer-transformer dominance on "
            "the supplied trained artifacts. This is not the 5x breakthrough gate; "
            "it is the strongest measured certificate from the current run."
        ),
        "failed": failed,
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake_params": lc_params,
            "transformer_params": tx_params,
            "layercake_heldout_bpb": lc_bpb,
            "transformer_heldout_bpb": tx_bpb,
            "layercake_cumulative_train_seconds": lc_seconds,
            "transformer_train_seconds": tx_seconds,
            "layercake_cumulative_train_bytes": lc_train_bytes,
            "transformer_train_bytes": tx_train_bytes,
            "cpu_layercake_bytes_per_second": cpu["layercake"]["bytes_per_second"],
            "cpu_transformer_bytes_per_second": cpu["bpe"]["bytes_per_second"],
            "gpu_layercake_bytes_per_second": gpu["layercake"]["bytes_per_second"],
            "gpu_transformer_bytes_per_second": gpu["bpe"]["bytes_per_second"],
            "quality_gates": quality_gates,
        },
        "artifacts": {
            "layercake_training": [str(path) for path in args.layercake_training],
            "transformer_training": str(args.transformer_training),
            "cpu_generation": str(args.cpu_generation),
            "gpu_generation": str(args.gpu_generation),
            "generation_quality": str(args.generation_quality),
        },
        "limitations": {
            "five_x_training_speed": False,
            "five_x_heldout_bpb": False,
            "five_x_gpu_generation": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
