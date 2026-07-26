"""Close the bounded exact-softmax scale branch from immutable raw evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results" / "moonshot" / "phase3"
PROFILE_DECISION = PHASE / "exact_softmax_profile_decision.json"
PROFILE = (
    PHASE
    / "learning_curves"
    / "layercake_seed9824_units0p5m_exact_from10m.json"
)
SCALE = (
    PHASE
    / "learning_curves"
    / "layercake_seed9824_units5m_exact_from10p5m.json"
)
PRIOR = (
    PHASE
    / "learning_curves"
    / "layercake_seed9824_units10m_adaptive.json"
)
FIRST = (
    PHASE
    / "learning_curves"
    / "layercake_seed9824_units5m_paired.json"
)
OUTPUT = PHASE / "exact_softmax_scale_decision.json"
THRESHOLD = 1.7174


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


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"branch decision is immutable: {OUTPUT}")
    decision = _read(PROFILE_DECISION)
    profile = _read(PROFILE)
    scale = _read(SCALE)
    prior = _read(PRIOR)
    first = _read(FIRST)
    configuration = scale["configuration"]
    exact_steps = [
        row for row in scale["step_records"]
        if row["exact_softmax"] is not None
    ]
    checks = {
        "profile_authorized_exactly_one_scale": (
            decision["status"] == "SCALE_ONCE_AUTHORIZED"
            and decision["frozen_scale_configuration"][
                "additional_scale_runs_authorized_after_this"
            ] is False
        ),
        "scale_parent_is_profile_checkpoint": (
            scale["training_parent"]["model_sha256"]
            == profile["checkpoint"]["model_sha256"]
            == decision["frozen_scale_configuration"][
                "parent_checkpoint_sha256"
            ]
        ),
        "frozen_exact_configuration_used": (
            configuration["exact_softmax_period_steps"] == 8
            and configuration["exact_softmax_positions"] == 64
            and configuration["exact_softmax_chunk_rows"] == 4096
            and configuration["exact_softmax_weight"] == 1.0
            and configuration["instruction_period_steps"] == 32
        ),
        "scale_budget_reached": (
            scale["accounting"]["model_visible_nonpadding_units"]
            >= 5_000_000
        ),
        "cpu_only": (
            scale["device"] == "cpu"
            and scale["accelerator_hours"] == 0.0
            and scale["cuda_memory_allocated_before_and_after"] == [0, 0]
        ),
        "exact_steps_executed": len(exact_steps) > 0,
        "full_vocabulary_physically_updated": all(
            row["exact_softmax"]["exact_softmax_vocabulary_rows"] == 50257
            and row["sparse_vocabulary_rows_updated"] == 50257
            for row in exact_steps
        ),
        "inference_architecture_unchanged": (
            scale["phase2_inference_architecture_unchanged"] is True
            and scale["final_inference_auxiliaries"] == []
        ),
        "quality_threshold_not_crossed": (
            scale["quality"]["bits_per_byte"] > THRESHOLD
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError("exact scale verification failed: " + ", ".join(failures))
    cumulative_units = sum(
        row["accounting"]["model_visible_nonpadding_units"]
        for row in (first, prior, profile, scale)
    )
    cumulative_raw_bytes = sum(
        row["accounting"]["raw_utf8_training_bytes_exposed"]
        for row in (first, prior, profile, scale)
    )
    cumulative_wall = sum(
        row["accounting"]["end_to_end_wall_time_seconds"]
        for row in (first, prior, profile, scale)
    )
    bpb_gain = (
        profile["quality"]["bits_per_byte"]
        - scale["quality"]["bits_per_byte"]
    )
    result = {
        "format": "layercake-phase3-exact-softmax-scale-decision/1",
        "status": "CLOSED_INSUFFICIENT",
        "promotion_eligible": False,
        "phase3_status": "OPEN_CONTINUATION_REQUIRED",
        "checks": checks,
        "profile_decision": _evidence(PROFILE_DECISION),
        "scale_run": {
            **_evidence(SCALE),
            "checkpoint_sha256": scale["checkpoint"]["model_sha256"],
            "validation_bpb": scale["quality"]["bits_per_byte"],
            "quality_gap_to_locked_threshold": (
                scale["quality"]["bits_per_byte"] - THRESHOLD
            ),
            "bpb_gain_during_scale_segment": bpb_gain,
            "exact_steps": len(exact_steps),
            "accounting": scale["accounting"],
        },
        "same_random_init_lineage_cumulative": {
            "model_visible_nonpadding_units": cumulative_units,
            "raw_utf8_training_bytes_exposed": cumulative_raw_bytes,
            "end_to_end_wall_time_seconds": cumulative_wall,
            "end_to_end_wall_time_hours": cumulative_wall / 3600.0,
            "validation_bpb": scale["quality"]["bits_per_byte"],
        },
        "branch_decision": {
            "exact_softmax_calibration_alone": "CLOSED_INSUFFICIENT",
            "additional_exact_period_weight_position_or_chunk_sweeps": False,
            "reason": (
                "The sole authorized 5M-unit scale run remained 0.32757 BPB "
                "above the locked threshold. Exact normalization is physically "
                "verified and bounded, but its gain is insufficient as a "
                "standalone repair."
            ),
        },
        "measured_limiting_factor": (
            "The tied 50,257-row vocabulary still uses stateless SGD: sampled "
            "rows retain no adaptive learning history, while exact steps must "
            "remain stateless to avoid dense optimizer moments."
        ),
        "next_bounded_experiment": {
            "name": "rowwise_adaptive_sparse_vocabulary_state",
            "change": (
                "Replace stateless SGD on sampled vocabulary rows with one "
                "scalar second-moment accumulator per row; retain stateless "
                "dense SGD on the already frozen exact-softmax steps."
            ),
            "profile": (
                "one paired 0.5M-unit validation branch from the same parent, "
                "same trace, exact schedule, CPU, and evaluation coverage"
            ),
            "nearby_optimizer_or_learning_rate_sweeps_authorized": False,
            "inference_architecture_change": False,
        },
        "negative_evidence_preserved": True,
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
