"""Validate the bounded Phase 3 exact-softmax profile and freeze its decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results" / "moonshot" / "phase3"
SAMPLED = (
    PHASE
    / "learning_curves"
    / "layercake_seed9824_units0p5m_sampled_from10m.json"
)
EXACT = (
    PHASE
    / "learning_curves"
    / "layercake_seed9824_units0p5m_exact_from10m.json"
)
TRANSFORMER = (
    PHASE
    / "learning_curves"
    / "transformer_seed9824_units5m_paired.json"
)
OUTPUT = PHASE / "exact_softmax_profile_decision.json"


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
        raise RuntimeError(f"decision evidence is immutable: {OUTPUT}")
    sampled = _read(SAMPLED)
    exact = _read(EXACT)
    transformer = _read(TRANSFORMER)
    sampled_accounting = sampled["accounting"]
    exact_accounting = exact["accounting"]
    exact_steps = [
        row for row in exact["step_records"]
        if row["exact_softmax"] is not None
    ]
    checks = {
        "same_parent_checkpoint": (
            sampled["training_parent"]["model_sha256"]
            == exact["training_parent"]["model_sha256"]
        ),
        "same_seed": sampled["seed"] == exact["seed"] == 9824,
        "same_trace": _trace(sampled) == _trace(exact),
        "same_model_visible_units": (
            sampled_accounting["model_visible_nonpadding_units"]
            == exact_accounting["model_visible_nonpadding_units"]
        ),
        "same_raw_utf8_bytes": (
            sampled_accounting["raw_utf8_training_bytes_exposed"]
            == exact_accounting["raw_utf8_training_bytes_exposed"]
        ),
        "same_evaluation_coverage": (
            sampled["quality"]["covered_raw_bytes"]
            == exact["quality"]["covered_raw_bytes"]
            and sampled["quality"]["evaluated_tokens"]
            == exact["quality"]["evaluated_tokens"]
        ),
        "exact_configuration_frozen": (
            exact["configuration"]["exact_softmax_period_steps"] == 8
            and exact["configuration"]["exact_softmax_positions"] == 64
            and exact["configuration"]["exact_softmax_chunk_rows"] == 4096
            and exact["configuration"]["exact_softmax_weight"] == 1.0
        ),
        "exact_steps_executed": len(exact_steps) > 0,
        "all_exact_steps_cover_full_vocabulary": all(
            row["exact_softmax"]["exact_softmax_vocabulary_rows"] == 50257
            and row["sparse_vocabulary_rows_updated"] == 50257
            and row["vocabulary_gradient_kind"]
            == "dense_exact_full_vocabulary"
            for row in exact_steps
        ),
        "sampled_steps_remain_sparse": all(
            row["vocabulary_gradient_kind"] == "row_sparse_sampled"
            for row in exact["step_records"]
            if row["exact_softmax"] is None
        ),
        "bpb_gain_positive": (
            exact["quality"]["bits_per_byte"]
            < sampled["quality"]["bits_per_byte"]
        ),
        "rss_below_measured_transformer": (
            exact_accounting["peak_process_resident_memory_bytes"]
            < transformer["accounting"]["peak_process_resident_memory_bytes"]
        ),
        "cpu_only": (
            exact["device"] == "cpu"
            and exact["accelerator_hours"] == 0.0
            and exact["cuda_memory_allocated_before_and_after"] == [0, 0]
        ),
        "inference_architecture_unchanged": (
            exact["phase2_inference_architecture_unchanged"] is True
            and exact["final_inference_auxiliaries"] == []
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError("exact-softmax profile failed: " + ", ".join(failures))
    ratios = {
        field: (
            exact_accounting[field] / sampled_accounting[field]
        )
        for field in (
            "end_to_end_wall_time_seconds",
            "measured_step_wall_time_seconds",
            "estimated_executed_cpu_multiply_accumulates",
            "peak_process_resident_memory_bytes",
        )
    }
    decision = {
        "format": "layercake-phase3-exact-softmax-profile-decision/1",
        "status": "SCALE_ONCE_AUTHORIZED",
        "promotion_eligible": False,
        "role": (
            "bounded validation-only profile required by the prior "
            "continuation decision"
        ),
        "checks": checks,
        "paired_profile": {
            "sampled": {
                **_evidence(SAMPLED),
                "validation_bpb": sampled["quality"]["bits_per_byte"],
                "checkpoint_sha256": sampled["checkpoint"]["model_sha256"],
            },
            "exact": {
                **_evidence(EXACT),
                "validation_bpb": exact["quality"]["bits_per_byte"],
                "checkpoint_sha256": exact["checkpoint"]["model_sha256"],
                "exact_steps": len(exact_steps),
            },
            "exact_minus_sampled_bpb": (
                exact["quality"]["bits_per_byte"]
                - sampled["quality"]["bits_per_byte"]
            ),
            "ratios_exact_over_sampled": ratios,
            "model_visible_units_each": exact_accounting[
                "model_visible_nonpadding_units"
            ],
            "raw_utf8_bytes_each": exact_accounting[
                "raw_utf8_training_bytes_exposed"
            ],
        },
        "measured_transformer_memory_reference": {
            **_evidence(TRANSFORMER),
            "peak_process_resident_memory_bytes": transformer["accounting"][
                "peak_process_resident_memory_bytes"
            ],
        },
        "frozen_scale_configuration": {
            "parent_checkpoint": exact["checkpoint"]["path"],
            "parent_checkpoint_sha256": exact["checkpoint"]["model_sha256"],
            "additional_model_visible_nonpadding_units": 5_000_000,
            "instruction_period_steps": 32,
            "exact_softmax_period_steps": 8,
            "exact_softmax_positions": 64,
            "exact_softmax_chunk_rows": 4096,
            "exact_softmax_weight": 1.0,
            "nearby_sweeps_authorized": False,
            "additional_scale_runs_authorized_after_this": False,
        },
        "decision_basis": (
            "At identical trace, units, bytes, and evaluation coverage, "
            "infrequent exact normalization produced a positive full-"
            "vocabulary BPB gain while remaining CPU-only and below the "
            "measured transformer training-memory peak. One scale run is "
            "authorized to determine whether the gain compounds. This profile "
            "does not satisfy or modify any Phase 3 promotion gate."
        ),
    }
    decision["decision_sha256"] = _canonical_sha(decision)
    OUTPUT.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
