"""Close the bounded row-wise vocabulary optimizer branch."""

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
    / "layercake_seed9824_units0p5m_rowwise_from15p5m.json"
)
PARENT_DECISION = PHASE / "exact_softmax_scale_decision.json"
GRADIENT_PROFILE = PHASE / "profiles" / "vocabulary_gradient_profile.json"
OUTPUT = PHASE / "rowwise_optimizer_decision.json"


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


def _evidence(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha(path),
    }


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"optimizer decision is immutable: {OUTPUT}")
    control = _read(CONTROL)
    candidate = _read(CANDIDATE)
    parent_decision = _read(PARENT_DECISION)
    gradient = _read(GRADIENT_PROFILE)
    control_accounting = control["accounting"]
    candidate_accounting = candidate["accounting"]
    measured_rate = gradient["sampled_step"][
        "suggested_rowwise_adagrad_learning_rate"
    ]
    checks = {
        "authorized_by_prior_decision": (
            parent_decision["next_bounded_experiment"]["name"]
            == "rowwise_adaptive_sparse_vocabulary_state"
        ),
        "same_parent": (
            control["training_parent"]["model_sha256"]
            == candidate["training_parent"]["model_sha256"]
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
        "measured_learning_rate_used": (
            abs(
                candidate["configuration"][
                    "rowwise_adagrad_learning_rate"
                ]
                - measured_rate
            )
            < 1.0e-18
        ),
        "rowwise_state_physically_executed": (
            candidate_accounting["vocabulary_optimizer_state_rows"] == 50257
            and candidate_accounting[
                "vocabulary_optimizer_logical_state_bytes"
            ] == 603084
        ),
        "candidate_worse_than_control": (
            candidate["quality"]["bits_per_byte"]
            > control["quality"]["bits_per_byte"]
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
        raise RuntimeError("row-wise decision failed: " + ", ".join(failures))
    result = {
        "format": "layercake-phase3-rowwise-optimizer-decision/1",
        "status": "CLOSED_NEGATIVE",
        "promotion_eligible": False,
        "checks": checks,
        "gradient_profile": _evidence(GRADIENT_PROFILE),
        "control": {
            **_evidence(CONTROL),
            "validation_bpb": control["quality"]["bits_per_byte"],
            "checkpoint_sha256": control["checkpoint"]["model_sha256"],
            "accounting": control_accounting,
        },
        "candidate": {
            **_evidence(CANDIDATE),
            "validation_bpb": candidate["quality"]["bits_per_byte"],
            "checkpoint_sha256": candidate["checkpoint"]["model_sha256"],
            "accounting": candidate_accounting,
        },
        "paired_deltas": {
            "candidate_minus_control_bpb": (
                candidate["quality"]["bits_per_byte"]
                - control["quality"]["bits_per_byte"]
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
        "branch_decision": {
            "rowwise_adaptive_sparse_vocabulary_state": "CLOSED_NEGATIVE",
            "nearby_optimizer_or_learning_rate_sweeps": False,
            "reason": (
                "The scale-matched adaptive row state increased validation "
                "BPB by 0.03537 at identical trace, units, bytes, and "
                "evaluation coverage."
            ),
        },
        "new_measured_limiting_factor": (
            "Both resumed forks regressed from the 15.5M parent BPB. The "
            "diagnostic checkpoint format resets dense AdamW state and "
            "restarts the full 3e-4 learning rate on every continuation, "
            "creating a measured late-training stability discontinuity."
        ),
        "next_bounded_experiment": {
            "name": "cumulative_unit_cosine_schedule",
            "change": (
                "Use one cumulative-unit learning-rate schedule derived from "
                "a frozen 30M-unit feasibility horizon; keep architecture, "
                "data trace, exact-softmax schedule, and stateless vocabulary "
                "optimizer unchanged."
            ),
            "profile": (
                "one 0.5M-unit candidate from the same 15.5M parent compared "
                "with the already completed stateless constant-rate control"
            ),
            "learning_rate_sweeps_authorized": False,
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
