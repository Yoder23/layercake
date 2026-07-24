"""Freeze the completed Phase 3 CPU-learning evidence and branch decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results/moonshot/phase3"


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha(relative: str) -> str:
    path = ROOT / relative
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    lc5_path = (
        "results/moonshot/phase3/learning_curves/"
        "layercake_seed9824_units5m_paired.json"
    )
    tf5_path = (
        "results/moonshot/phase3/learning_curves/"
        "transformer_seed9824_units5m_paired.json"
    )
    lc10_path = (
        "results/moonshot/phase3/learning_curves/"
        "layercake_seed9824_units10m_adaptive.json"
    )
    lc5 = read(lc5_path)
    tf5 = read(tf5_path)
    lc10 = read(lc10_path)
    junit_relative = "results/moonshot/phase3/pytest-continuation.xml"
    junit = ROOT / junit_relative
    junit_root = ET.parse(junit).getroot()
    suites = (
        [junit_root]
        if junit_root.tag == "testsuite"
        else list(junit_root.findall("testsuite"))
    )
    test_totals = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    test_results = {
        "format": "layercake-phase3-continuation-test-results/1",
        "status": (
            "PASS"
            if test_totals["tests"] > 0
            and test_totals["failures"] == 0
            and test_totals["errors"] == 0
            else "FAIL"
        ),
        **test_totals,
        "passed": (
            test_totals["tests"]
            - test_totals["failures"]
            - test_totals["errors"]
            - test_totals["skipped"]
        ),
        "duration_seconds": sum(
            float(suite.attrib.get("time", 0.0)) for suite in suites
        ),
        "command": (
            "pytest -q "
            "--junitxml=results/moonshot/phase3/pytest-continuation.xml"
        ),
        "junit_path": junit_relative,
        "junit_sha256": sha(junit_relative),
    }
    write(PHASE / "test_results.json", test_results)
    left = lc5["accounting"]
    right = tf5["accounting"]
    cumulative_layercake_seconds = (
        left["end_to_end_wall_time_seconds"]
        + lc10["accounting"]["end_to_end_wall_time_seconds"]
    )
    summary = {
        "format": "layercake-phase3-continuation-status/1",
        "status": "CONTINUATION_REQUIRED",
        "phase2_status": "SEALED",
        "phase2_tag": "layercake-moonshot-phase2-r3",
        "phase3_status": "OPEN",
        "protocol": {
            "path": "moonshot/phase3_training_efficiency_lock.json",
            "sha256": sha("moonshot/phase3_training_efficiency_lock.json"),
            "locked_before_runs": True,
        },
        "regression_suite": {
            "path": "results/moonshot/phase3/test_results.json",
            "status": test_results["status"],
            "tests": test_results["tests"],
            "passed": test_results["passed"],
            "failures": test_results["failures"],
            "errors": test_results["errors"],
            "junit_sha256": test_results["junit_sha256"],
        },
        "valid_paired_seed9824_5m": {
            "layercake": {
                "path": lc5_path,
                "sha256": sha(lc5_path),
                "checkpoint_sha256": lc5["checkpoint"]["model_sha256"],
                "model_visible_nonpadding_units": left[
                    "model_visible_nonpadding_units"
                ],
                "raw_utf8_bytes": left[
                    "raw_utf8_training_bytes_exposed"
                ],
                "end_to_end_seconds": left[
                    "end_to_end_wall_time_seconds"
                ],
                "validation_bpb": lc5["quality"]["bits_per_byte"],
                "peak_rss": left["peak_process_resident_memory_bytes"],
            },
            "transformer": {
                "path": tf5_path,
                "sha256": sha(tf5_path),
                "checkpoint_sha256": tf5["checkpoint"]["model_sha256"],
                "model_visible_nonpadding_units": right[
                    "model_visible_nonpadding_units"
                ],
                "raw_utf8_bytes": right[
                    "raw_utf8_training_bytes_exposed"
                ],
                "end_to_end_seconds": right[
                    "end_to_end_wall_time_seconds"
                ],
                "validation_bpb": tf5["quality"]["bits_per_byte"],
                "peak_rss": right["peak_process_resident_memory_bytes"],
            },
            "ratios_layercake_over_transformer": {
                "model_visible_nonpadding_units": (
                    left["model_visible_nonpadding_units"]
                    / right["model_visible_nonpadding_units"]
                ),
                "raw_utf8_bytes": (
                    left["raw_utf8_training_bytes_exposed"]
                    / right["raw_utf8_training_bytes_exposed"]
                ),
                "end_to_end_wall_time": (
                    left["end_to_end_wall_time_seconds"]
                    / right["end_to_end_wall_time_seconds"]
                ),
                "executed_cpu_operations": (
                    left["estimated_executed_cpu_multiply_accumulates"]
                    / right["estimated_executed_cpu_multiply_accumulates"]
                ),
                "active_parameter_seconds": (
                    left["active_parameter_seconds"]
                    / right["active_parameter_seconds"]
                ),
                "peak_rss": (
                    left["peak_process_resident_memory_bytes"]
                    / right["peak_process_resident_memory_bytes"]
                ),
            },
            "quality_gate_passed": False,
            "promotion_credit": False,
        },
        "layercake_adaptive_continuation_10m": {
            "path": lc10_path,
            "sha256": sha(lc10_path),
            "checkpoint_sha256": lc10["checkpoint"]["model_sha256"],
            "cumulative_model_visible_nonpadding_units": (
                left["model_visible_nonpadding_units"]
                + lc10["accounting"]["model_visible_nonpadding_units"]
            ),
            "cumulative_raw_utf8_bytes": (
                left["raw_utf8_training_bytes_exposed"]
                + lc10["accounting"]["raw_utf8_training_bytes_exposed"]
            ),
            "cumulative_end_to_end_seconds": cumulative_layercake_seconds,
            "cumulative_end_to_end_hours": (
                cumulative_layercake_seconds / 3600.0
            ),
            "validation_bpb": lc10["quality"]["bits_per_byte"],
            "quality_gate_passed": False,
            "paired_transformer_continuation_available": False,
            "promotion_credit": False,
        },
        "invalid_or_diagnostic_only_evidence": [
            {
                "glob": "results/moonshot/phase3/profiles/*_s128.json",
                "reason": "LayerCake and transformer profiles ran concurrently",
            },
            {
                "glob": (
                    "results/moonshot/phase3/learning_curves/"
                    "*_units0p8m.json"
                ),
                "reason": (
                    "model-dependent route choices broke exact paired "
                    "example order"
                ),
            },
            {
                "glob": "results/moonshot/phase3/profiles/*.json",
                "reason": "feasibility profiles are not quality promotion runs",
            },
        ],
        "closed_branches": [
            {
                "branch": "sampled_multihorizon_static_instruction_1_in_4",
                "reason": (
                    "both systems overfit instruction batches while missing "
                    "the frozen full-vocabulary BPB threshold"
                ),
            },
            {
                "branch": "curriculum_reallocation_alone",
                "reason": (
                    "LayerCake BPB improved only from 2.2697 to 2.1512 in "
                    "the second 5M-unit segment and remained above 1.7174"
                ),
            },
        ],
        "measured_limiting_factor": (
            "sampled-vocabulary training is not transferring sufficient "
            "probability mass to the full 50,257-way tied inference head"
        ),
        "next_bounded_experiment": {
            "name": "chunked_exact_softmax_calibration",
            "change": (
                "add infrequent chunked exact next-token normalization while "
                "preserving paired data, exact inference architecture, CPU-only "
                "execution, and physically sparse instruction-cake updates"
            ),
            "prerequisite": (
                "profile wall time, dense-gradient memory, and BPB gain on one "
                "short validation-only branch before any additional 5M run"
            ),
            "nearby_sweeps_authorized": False,
        },
        "gates": {
            "random_initialization": True,
            "pretrained_weights_loaded": False,
            "promoted_accelerator_hours": 0.0,
            "three_seed_time_to_quality": False,
            "sample_efficiency_ratio_le_0p5": False,
            "wall_time_to_quality_ratio_le_0p5": False,
            "phase2_inference_sentinel": "NOT_YET_DUE_NO_QUALITY_CROSSING",
        },
    }
    write(PHASE / "continuation_status.json", summary)
    ledger_rows = [
        {
            "event": "phase3_paired_5m_completed",
            "status": "QUALITY_THRESHOLD_NOT_REACHED",
            "evidence": "results/moonshot/phase3/continuation_status.json",
            "layercake_bpb": lc5["quality"]["bits_per_byte"],
            "transformer_bpb": tf5["quality"]["bits_per_byte"],
            "wall_ratio": summary["valid_paired_seed9824_5m"][
                "ratios_layercake_over_transformer"
            ]["end_to_end_wall_time"],
            "negative_evidence_preserved": True,
        },
        {
            "event": "phase3_adaptive_10m_layercake_completed",
            "status": "QUALITY_THRESHOLD_NOT_REACHED",
            "validation_bpb": lc10["quality"]["bits_per_byte"],
            "measured_limiting_factor": summary["measured_limiting_factor"],
            "next_bounded_experiment": summary["next_bounded_experiment"][
                "name"
            ],
            "negative_evidence_preserved": True,
        },
    ]
    ledger = PHASE / "experiment_ledger.jsonl"
    ledger.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
