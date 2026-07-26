"""Validate the Phase 3 cumulative-unit schedule profile and freeze one run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results" / "moonshot" / "phase3"
CONTROL = (
    PHASE
    / "learning_curves"
    / "layercake_seed9824_units0p5m_stateless_from15p5m.json"
)
CANDIDATE = (
    PHASE
    / "learning_curves"
    / "layercake_seed9824_units0p5m_schedule_from15p5m.json"
)
PARENT = (
    PHASE
    / "learning_curves"
    / "layercake_seed9824_units5m_exact_from10p5m.json"
)
PRIOR_DECISION = PHASE / "rowwise_optimizer_decision.json"
OUTPUT = PHASE / "schedule_profile_decision.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evidence(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha(path),
    }


def _trace(document: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            row["step"],
            row["instruction_batch"],
            row["route"],
            row["candidate_vocabulary_rows"],
            row["exact_softmax"] is not None,
        )
        for row in document["step_records"]
    ]


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"schedule decision is immutable: {OUTPUT}")
    control = _read(CONTROL)
    candidate = _read(CANDIDATE)
    parent = _read(PARENT)
    prior = _read(PRIOR_DECISION)
    control_accounting = control["accounting"]
    candidate_accounting = candidate["accounting"]
    configuration = candidate["configuration"]
    parent_bpb = parent["quality"]["bits_per_byte"]
    checks = {
        "authorized_by_prior_decision": (
            prior["next_bounded_experiment"]["name"]
            == "cumulative_unit_cosine_schedule"
        ),
        "same_parent": (
            control["training_parent"]["model_sha256"]
            == candidate["training_parent"]["model_sha256"]
            == parent["checkpoint"]["model_sha256"]
        ),
        "same_seed": control["seed"] == candidate["seed"] == 9824,
        "same_trace": _trace(control) == _trace(candidate),
        "same_units": (
            control_accounting["model_visible_nonpadding_units"]
            == candidate_accounting["model_visible_nonpadding_units"]
        ),
        "same_raw_bytes": (
            control_accounting["raw_utf8_training_bytes_exposed"]
            == candidate_accounting["raw_utf8_training_bytes_exposed"]
        ),
        "same_evaluation_coverage": (
            control["quality"]["covered_raw_bytes"]
            == candidate["quality"]["covered_raw_bytes"]
            and control["quality"]["evaluated_tokens"]
            == candidate["quality"]["evaluated_tokens"]
        ),
        "schedule_frozen": (
            configuration["cumulative_units_before"] == 15502767
            and configuration["schedule_total_units"] == 30_000_000
            and configuration["schedule_warmup_units"] == 1_000_000
            and configuration["schedule_minimum_ratio"] == 0.1
        ),
        "candidate_beats_control": (
            candidate["quality"]["bits_per_byte"]
            < control["quality"]["bits_per_byte"]
        ),
        "candidate_improves_parent": (
            candidate["quality"]["bits_per_byte"] < parent_bpb
        ),
        "cpu_only": (
            candidate["device"] == "cpu"
            and candidate["accelerator_hours"] == 0.0
            and candidate["cuda_memory_allocated_before_and_after"] == [0, 0]
        ),
        "inference_unchanged": (
            candidate["phase2_inference_architecture_unchanged"] is True
            and candidate["final_inference_auxiliaries"] == []
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError("schedule profile failed: " + ", ".join(failures))
    result = {
        "format": "layercake-phase3-schedule-profile-decision/1",
        "status": "CONTINUOUS_RUN_AUTHORIZED",
        "promotion_eligible": False,
        "checks": checks,
        "prior_decision": _evidence(PRIOR_DECISION),
        "parent": {
            **_evidence(PARENT),
            "validation_bpb": parent_bpb,
            "checkpoint_sha256": parent["checkpoint"]["model_sha256"],
        },
        "control": {
            **_evidence(CONTROL),
            "validation_bpb": control["quality"]["bits_per_byte"],
            "accounting": control_accounting,
        },
        "candidate": {
            **_evidence(CANDIDATE),
            "validation_bpb": candidate["quality"]["bits_per_byte"],
            "checkpoint_sha256": candidate["checkpoint"]["model_sha256"],
            "first_learning_rate": candidate["step_records"][0][
                "learning_rate_schedule"
            ],
            "last_learning_rate": candidate["step_records"][-1][
                "learning_rate_schedule"
            ],
            "accounting": candidate_accounting,
        },
        "paired_deltas": {
            "candidate_minus_control_bpb": (
                candidate["quality"]["bits_per_byte"]
                - control["quality"]["bits_per_byte"]
            ),
            "candidate_minus_parent_bpb": (
                candidate["quality"]["bits_per_byte"] - parent_bpb
            ),
            "candidate_over_control_wall_time": (
                candidate_accounting["end_to_end_wall_time_seconds"]
                / control_accounting["end_to_end_wall_time_seconds"]
            ),
            "candidate_over_control_rss": (
                candidate_accounting["peak_process_resident_memory_bytes"]
                / control_accounting["peak_process_resident_memory_bytes"]
            ),
        },
        "frozen_continuous_run": {
            "system": "layercake_complete",
            "seed": 9824,
            "random_initialization": True,
            "pretrained_weights_loaded": False,
            "resume_parent": None,
            "target_model_visible_nonpadding_units": 30_000_000,
            "immutable_evaluation_interval_units": 5_000_000,
            "warmup_steps_with_uncharged_updates": 0,
            "sequence_units": 1020,
            "negative_vocabulary_rows": 2048,
            "prediction_horizons": [1, 2, 4],
            "instruction_period_steps": 32,
            "exact_softmax_period_steps": 8,
            "exact_softmax_positions": 64,
            "exact_softmax_chunk_rows": 4096,
            "exact_softmax_weight": 1.0,
            "vocabulary_optimizer": "stateless hybrid sparse/dense SGD",
            "schedule_total_units": 30_000_000,
            "schedule_warmup_units": 1_000_000,
            "schedule_minimum_ratio": 0.1,
            "threads": 14,
            "nearby_schedule_or_optimizer_sweeps_authorized": False,
        },
        "decision_basis": (
            "The sole frozen schedule reversed the measured resume regression "
            "at matched trace and units. One from-scratch single-process run "
            "is authorized so optimizer state is continuous and all 5M-unit "
            "quality points are immutable. Passing remains contingent on every "
            "pre-existing Phase 3 gate."
        ),
    }
    result["decision_sha256"] = _canonical_sha(result)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
