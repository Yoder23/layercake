"""Typed evidence verifier for the governance retirement of Phase 3.

This verifier deliberately proves no training-efficiency claim.  It checks that
the former experiment protocol and completed negative evidence remain intact,
that the sealed Phase 2 parent is the handoff lineage, and that every retirement
gate is derived from raw evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


class Phase3RetirementEvidenceError(RuntimeError):
    """Raised when Phase 3 retirement evidence is missing or misleading."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase3RetirementEvidenceError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase3RetirementEvidenceError(f"{path} must contain a JSON object")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise Phase3RetirementEvidenceError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _require_file(root: Path, relative: str, expected: str | None = None) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise Phase3RetirementEvidenceError(f"path escapes repository: {relative}") from error
    if not path.is_file():
        raise Phase3RetirementEvidenceError(f"required evidence is missing: {relative}")
    if expected is not None and _sha(path) != expected:
        raise Phase3RetirementEvidenceError(f"evidence hash mismatch: {relative}")
    return path


def _git(root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise Phase3RetirementEvidenceError(
            f"git {' '.join(arguments)} failed: {detail}"
        )
    return process.stdout.strip()


def validate_phase3_retirement_bundle(root: Path, phase_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    phase_dir = phase_dir.resolve()
    decision = _read(phase_dir / "retirement_decision.json")
    payload = _read(phase_dir / "certificate_payload.json")
    raw = _read(phase_dir / "raw_runs" / "retirement.json")
    tests = _read(phase_dir / "retirement_test_results.json")
    contract = _read(root / "moonshot" / "phase3_retirement_lock.json")
    stop = _read(phase_dir / "bounded_scratch_control_stop.json")

    if decision.get("format") != "layercake-phase3-retirement-decision/1":
        raise Phase3RetirementEvidenceError("retirement decision format is invalid")
    if decision.get("status") != "RETIRED_BY_GOVERNANCE":
        raise Phase3RetirementEvidenceError("retirement was not explicitly authorized")
    if decision.get("scientific_training_efficiency_passed") is not False:
        raise Phase3RetirementEvidenceError(
            "retirement must explicitly deny a scientific training-efficiency pass"
        )
    if payload.get("format") != "layercake-phase3-retirement-payload/1":
        raise Phase3RetirementEvidenceError("retirement payload format is invalid")
    if payload.get("disposition") != "RETIRED_BY_GOVERNANCE":
        raise Phase3RetirementEvidenceError("retirement payload disposition is invalid")
    if payload.get("scientific_training_efficiency_passed") is not False:
        raise Phase3RetirementEvidenceError("payload fabricates a scientific pass")
    if raw.get("format") != "layercake-phase3-retirement-raw/1":
        raise Phase3RetirementEvidenceError("retirement raw evidence format is invalid")
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise Phase3RetirementEvidenceError("retirement requires exactly one raw decision record")

    required_gates = contract.get("retirement_gates")
    if not isinstance(required_gates, dict) or set(required_gates) != {
        "retirement_authorized",
        "historical_evidence_preserved",
        "no_scientific_training_pass_claimed",
        "phase2_parent_sealed",
        "future_abi_recertification_required",
    }:
        raise Phase3RetirementEvidenceError("retirement gate contract is incomplete")
    if any(records[0].get(name) != 1.0 for name in required_gates):
        raise Phase3RetirementEvidenceError("a retirement raw gate is not satisfied")

    legacy = contract["legacy_contract"]
    _require_file(root, legacy["path"], legacy["sha256"])
    if stop.get("format") != "layercake-phase3-bounded-scratch-control-stop/1":
        raise Phase3RetirementEvidenceError("stopped-control evidence format is invalid")
    if stop.get("status") != "STOPPED_BY_GOVERNANCE":
        raise Phase3RetirementEvidenceError("bounded scratch control was not closed")
    if stop.get("uncheckpointed_tail", {}).get("promotion_credit") is not False:
        raise Phase3RetirementEvidenceError("uncheckpointed training tail received credit")

    milestone = stop["preserved_completed_milestone"]
    _require_file(root, milestone["evidence"], milestone["file_sha256"])
    checkpoint = (root / milestone["checkpoint_directory"]).resolve()
    if not checkpoint.is_dir():
        raise Phase3RetirementEvidenceError("completed milestone checkpoint is missing")
    expected_checkpoint_hashes = {
        "model.safetensors": milestone["model_sha256"],
        "dense_optimizer_state.pt": milestone["dense_optimizer_state_sha256"],
    }
    for name, expected in expected_checkpoint_hashes.items():
        _require_file(root, (checkpoint / name).relative_to(root).as_posix(), expected)

    campaign = _read(root / "moonshot" / "campaign.yaml")
    if campaign.get("phases", {}).get("phase2_cpu_quality_speed") != "SEALED":
        raise Phase3RetirementEvidenceError("Phase 2 parent is not sealed")
    phase2 = campaign.get("phase_records", {}).get("phase2", {})
    if phase2.get("completion_tag") != contract["sealed_parent"]["completion_tag"]:
        raise Phase3RetirementEvidenceError("Phase 2 completion tag lineage changed")
    if _git(root, "cat-file", "-t", contract["sealed_parent"]["completion_tag"]) != "tag":
        raise Phase3RetirementEvidenceError("Phase 2 completion tag is not annotated")
    _require_file(
        root,
        contract["sealed_parent"]["final_core_manifest"],
        contract["sealed_parent"]["final_core_manifest_sha256"],
    )

    if tests.get("format") != "layercake-phase3-retirement-tests/1":
        raise Phase3RetirementEvidenceError("retirement test evidence format is invalid")
    if (
        tests.get("status") != "PASS"
        or tests.get("tests", 0) <= 0
        or tests.get("failures") != 0
        or tests.get("errors") != 0
    ):
        raise Phase3RetirementEvidenceError("retirement regression evidence is not green")
    _require_file(root, tests["junit_path"], tests["junit_sha256"])

    forbidden = {
        "time_to_matched_quality_ratio",
        "sample_efficiency_ratio",
        "phase2_cpu_speed_retention",
    }
    claim_ids = {
        claim.get("gate_id")
        for claim in payload.get("claims", [])
        if isinstance(claim, dict)
    }
    if claim_ids & forbidden:
        raise Phase3RetirementEvidenceError(
            "retirement payload contains a forbidden scientific training claim"
        )

    return {
        "disposition": "RETIRED_BY_GOVERNANCE",
        "scientific_training_efficiency_passed": False,
        "legacy_contract_sha256": legacy["sha256"],
        "completed_milestone_units": milestone[
            "observed_model_visible_nonpadding_units"
        ],
        "completed_milestone_validation_bpb": milestone[
            "validation_bits_per_byte"
        ],
        "phase2_parent_tag": contract["sealed_parent"]["completion_tag"],
        "all_retirement_gates": "PASS",
        "regression_tests": tests["tests"],
    }
