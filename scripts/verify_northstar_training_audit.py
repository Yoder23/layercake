"""Validate the v22 training measurements and report the 5x gate separately.

An OPEN North Star is not an invalid benchmark.  This verifier exits nonzero
only when measurement integrity fails; it records the performance target as
OPEN until every CPU/GPU and time-to-quality gate actually clears 5x.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "breakthrough_equal"
RECIPE = RESULTS / "northstar_v22_training_speed_recipe.json"
LOWER_BOUND = RESULTS / "northstar_v22_training_speed_favorable_lower_bound.json"
HISTORICAL = RESULTS / "measured_equal_size_dominance_transprior_certificate.json"
DEFAULT_OUTPUT = RESULTS / "northstar_v22_training_audit.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ratio_rows(device: dict[str, Any]) -> list[float]:
    layercake = device["repeat_details"]["layercake"]
    transformer = device["repeat_details"]["transformer"]
    if len(layercake) != len(transformer):
        raise ValueError("LayerCake/transformer repeat counts differ")
    return [
        float(layercake[index]["logical_bytes_per_second"])
        / float(transformer[index]["logical_bytes_per_second"])
        for index in range(len(layercake))
    ]


def _measurement_gates(document: dict[str, Any], expected_mode: str) -> dict[str, bool]:
    devices = document.get("devices", {})
    gates: dict[str, bool] = {
        "benchmark_schema": (
            document.get("schema_version") == 1
            and document.get("benchmark") == "northstar_v22_full_core_training_speed"
        ),
        "expected_layercake_mode": (
            document.get("protocol", {}).get("layercake_training_mode")
            == expected_mode
        ),
        "cpu_and_cuda_present": set(devices) == {"cpu", "cuda"},
        "three_repeats": document.get("protocol", {}).get("repeats") == 3,
        "positive_measurement_window": (
            int(document.get("protocol", {}).get("warmup_steps", 0)) > 0
            and int(document.get("protocol", {}).get("measured_steps", 0)) >= 10
        ),
    }
    for name in ("cpu", "cuda"):
        if name not in devices:
            continue
        device = devices[name]
        ratios = _ratio_rows(device)
        recorded = device["ratios"]
        workload = device["workload"]
        layercake = device["layercake"]
        transformer = device["transformer"]
        prefix = f"{name}_"
        gates[prefix + "logical_batch_matched"] = 0.99 <= float(
            workload["logical_batch_bytes_ratio_layercake_over_transformer"]
        ) <= 1.01
        gates[prefix + "parameters_matched"] = 0.95 <= float(
            recorded["parameter_count_layercake_over_transformer"]
        ) <= 1.05
        gates[prefix + "repeat_ratios_recomputed"] = all(
            math.isclose(actual, float(saved), rel_tol=1e-12, abs_tol=1e-12)
            for actual, saved in zip(
                ratios,
                recorded[
                    "training_throughput_layercake_over_transformer_per_repeat"
                ],
            )
        )
        gates[prefix + "median_recomputed"] = math.isclose(
            statistics.median(ratios),
            float(
                recorded[
                    "median_training_throughput_layercake_over_transformer"
                ]
            ),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        gates[prefix + "minimum_recomputed"] = math.isclose(
            min(ratios),
            float(
                recorded[
                    "minimum_training_throughput_layercake_over_transformer"
                ]
            ),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        gates[prefix + "finite_losses"] = bool(layercake["all_losses_finite"])
        gates[prefix + "finite_baseline_losses"] = bool(
            transformer["all_losses_finite"]
        )
        gates[prefix + "hardware_recorded"] = bool(
            device.get("environment", {}).get("cpu")
        ) and (
            name == "cpu"
            or bool(device.get("environment", {}).get("gpu"))
        )
    return gates


def _speed(document: dict[str, Any], device: str, statistic: str) -> float:
    return float(
        document["devices"][device]["ratios"][
            f"{statistic}_training_throughput_layercake_over_transformer"
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate v22 training evidence and its open 5x gate"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = {
        "recipe": RECIPE,
        "favorable_lower_bound": LOWER_BOUND,
        "historical_time_to_quality": HISTORICAL,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing training evidence: {missing}")
    recipe = _load(RECIPE)
    lower_bound = _load(LOWER_BOUND)
    historical = _load(HISTORICAL)
    integrity_gates = {
        **{
            f"recipe_{name}": passed
            for name, passed in _measurement_gates(recipe, "recipe").items()
        },
        **{
            f"lower_bound_{name}": passed
            for name, passed in _measurement_gates(
                lower_bound, "next_byte_only"
            ).items()
        },
        "historical_equal_size_certificate_passed": historical.get("status")
        == "PASS",
        "historical_quality_was_matched": bool(
            historical.get("gates", {}).get("heldout_bpb_lower")
        ),
    }
    failed_integrity = [
        name for name, passed in integrity_gates.items() if not passed
    ]

    historical_ratio = float(
        historical["ratios"]["training_speed_ratio_transformer_over_layercake"]
    )
    northstar_gates = {
        "recipe_cpu_minimum_at_least_5x": _speed(recipe, "cpu", "minimum")
        >= 5.0,
        "recipe_gpu_minimum_at_least_5x": _speed(recipe, "cuda", "minimum")
        >= 5.0,
        "favorable_lower_bound_cpu_minimum_at_least_5x": _speed(
            lower_bound, "cpu", "minimum"
        )
        >= 5.0,
        "favorable_lower_bound_gpu_minimum_at_least_5x": _speed(
            lower_bound, "cuda", "minimum"
        )
        >= 5.0,
        "historical_equal_quality_time_to_quality_at_least_5x": historical_ratio
        >= 5.0,
    }
    failed_northstar = [
        name for name, passed in northstar_gates.items() if not passed
    ]

    artifact = {
        "schema_version": 1,
        "measurement_status": "PASS" if not failed_integrity else "FAIL",
        "training_northstar_status": "PASS" if not failed_northstar else "OPEN",
        "claim_boundary": (
            "The v22 release proves bounded quality, generation speed, deployment, "
            "and transfer gates. It does not prove faster full-core training."
        ),
        "measurement_integrity_gates": integrity_gates,
        "failed_measurement_integrity": failed_integrity,
        "training_northstar_gates": northstar_gates,
        "failed_training_northstar": failed_northstar,
        "metrics": {
            "recipe": {
                "cpu_median_throughput_ratio": _speed(recipe, "cpu", "median"),
                "cpu_minimum_throughput_ratio": _speed(recipe, "cpu", "minimum"),
                "gpu_median_throughput_ratio": _speed(recipe, "cuda", "median"),
                "gpu_minimum_throughput_ratio": _speed(recipe, "cuda", "minimum"),
                "gpu_peak_memory_ratio_transformer_over_layercake": recipe[
                    "devices"
                ]["cuda"]["ratios"][
                    "cuda_peak_memory_transformer_over_layercake"
                ],
                "tensor_state_ratio_transformer_over_layercake": recipe[
                    "devices"
                ]["cuda"]["ratios"][
                    "tensor_state_bytes_transformer_over_layercake"
                ],
            },
            "favorable_lower_bound": {
                "cpu_median_throughput_ratio": _speed(
                    lower_bound, "cpu", "median"
                ),
                "cpu_minimum_throughput_ratio": _speed(
                    lower_bound, "cpu", "minimum"
                ),
                "gpu_median_throughput_ratio": _speed(
                    lower_bound, "cuda", "median"
                ),
                "gpu_minimum_throughput_ratio": _speed(
                    lower_bound, "cuda", "minimum"
                ),
                "gpu_peak_memory_ratio_transformer_over_layercake": lower_bound[
                    "devices"
                ]["cuda"]["ratios"][
                    "cuda_peak_memory_transformer_over_layercake"
                ],
                "tensor_state_ratio_transformer_over_layercake": lower_bound[
                    "devices"
                ]["cuda"]["ratios"][
                    "tensor_state_bytes_transformer_over_layercake"
                ],
            },
            "historical_equal_quality_time_to_quality_ratio": historical_ratio,
        },
        "evidence": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "required_architecture_change": (
            "A 5x dense full-core result cannot come from the current optimizer path, "
            "whose active parameters and optimizer state are nearly the same size as "
            "the transformer. It requires a quality-validated sparse/conditional update "
            "path, materially fewer active parameters, or at least 5x time-to-quality "
            "data efficiency under a locked convergence protocol."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not failed_integrity else 1


if __name__ == "__main__":
    raise SystemExit(main())
