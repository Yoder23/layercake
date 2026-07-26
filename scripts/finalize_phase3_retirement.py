"""Derive an honest Phase 3 governance-retirement evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "results" / "moonshot" / "phase3"
CONTRACT = ROOT / "moonshot" / "phase3_retirement_lock.json"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _junit(path: Path) -> dict[str, int | float]:
    document = ET.parse(path)
    root = document.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("JUnit document contains no test suite")
    totals: dict[str, int | float] = {
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "duration_seconds": 0.0,
    }
    for suite in suites:
        for name in ("tests", "failures", "errors", "skipped"):
            totals[name] = int(totals[name]) + int(suite.attrib.get(name, 0))
        totals["duration_seconds"] = float(totals["duration_seconds"]) + float(
            suite.attrib.get("time", 0.0)
        )
    return totals


def finalize(junit: Path) -> dict:
    contract = _read(CONTRACT)
    stop = _read(PHASE / "bounded_scratch_control_stop.json")
    milestone = _read(
        PHASE
        / "learning_curves"
        / "milestones"
        / "layercake_seed9824_continuous30m_schedule_units5000000.json"
    )
    if stop["preserved_completed_milestone"]["file_sha256"] != _sha(
        PHASE
        / "learning_curves"
        / "milestones"
        / "layercake_seed9824_continuous30m_schedule_units5000000.json"
    ):
        raise ValueError("stopped-run record does not hash the completed milestone")
    if milestone["quality"]["bits_per_byte"] <= 1.7174:
        raise ValueError("retirement evidence unexpectedly crossed the old quality gate")

    junit = junit.resolve()
    junit.relative_to(ROOT)
    totals = _junit(junit)
    test_result = {
        "format": "layercake-phase3-retirement-tests/1",
        "status": (
            "PASS"
            if totals["tests"] > 0
            and totals["failures"] == 0
            and totals["errors"] == 0
            else "FAIL"
        ),
        **totals,
        "passed": (
            int(totals["tests"])
            - int(totals["failures"])
            - int(totals["errors"])
            - int(totals["skipped"])
        ),
        "junit_path": junit.relative_to(ROOT).as_posix(),
        "junit_sha256": _sha(junit),
        "command": "pytest -q --junitxml=results/moonshot/phase3/pytest-retirement.xml",
    }
    _write(PHASE / "retirement_test_results.json", test_result)
    if test_result["status"] != "PASS":
        raise ValueError("regression suite did not pass")

    record = {
        "record_id": "phase3-governance-retirement",
        **contract["retirement_gates"],
    }
    raw = {
        "format": "layercake-phase3-retirement-raw/1",
        "records": [record],
        "scientific_training_efficiency_metrics": {
            "old_quality_threshold_reached": False,
            "old_three_seed_time_to_quality_gate_reached": False,
            "old_sample_efficiency_gate_reached": False,
            "completed_milestone_validation_bits_per_byte": milestone["quality"][
                "bits_per_byte"
            ],
            "old_validation_bits_per_byte_max": 1.7174,
        },
    }
    raw_path = PHASE / "raw_runs" / "retirement.json"
    _write(raw_path, raw)
    raw_sha = _sha(raw_path)

    claims = []
    for gate_id, value in contract["retirement_gates"].items():
        claims.append(
            {
                "gate_id": gate_id,
                "kind": "governance_retirement",
                "promoted": True,
                "raw_artifact": raw_path.relative_to(ROOT).as_posix(),
                "raw_sha256": raw_sha,
                "derivation": {
                    "operation": "mean",
                    "field": gate_id,
                    "where": {"record_id": "phase3-governance-retirement"},
                },
                "value": value,
            }
        )
    payload = {
        "format": "layercake-phase3-retirement-payload/1",
        "disposition": "RETIRED_BY_GOVERNANCE",
        "scientific_training_efficiency_passed": False,
        "claims": claims,
        "headline_claims": [],
    }
    _write(PHASE / "certificate_payload.json", payload)

    decision = {
        "format": "layercake-phase3-retirement-decision/1",
        "status": "RETIRED_BY_GOVERNANCE",
        "scientific_training_efficiency_passed": False,
        "reason": contract["authorization"]["decision"],
        "contract": {
            "path": CONTRACT.relative_to(ROOT).as_posix(),
            "sha256": _sha(CONTRACT),
        },
        "legacy_contract": contract["legacy_contract"],
        "phase2_parent": contract["sealed_parent"],
        "completed_control": {
            "milestone_path": stop["preserved_completed_milestone"]["evidence"],
            "milestone_sha256": stop["preserved_completed_milestone"]["file_sha256"],
            "validation_bits_per_byte": milestone["quality"]["bits_per_byte"],
            "old_quality_gate_passed": False,
        },
        "stopped_control": {
            "path": "results/moonshot/phase3/bounded_scratch_control_stop.json",
            "sha256": _sha(PHASE / "bounded_scratch_control_stop.json"),
            "uncheckpointed_tail_promotion_credit": False,
        },
        "raw_retirement_evidence": {
            "path": raw_path.relative_to(ROOT).as_posix(),
            "sha256": raw_sha,
        },
        "test_evidence": {
            "path": "results/moonshot/phase3/retirement_test_results.json",
            "sha256": _sha(PHASE / "retirement_test_results.json"),
        },
        "future_abi_policy": contract["future_abi_policy"],
    }
    decision["evidence_sha256"] = _canonical_sha(decision)
    _write(PHASE / "retirement_decision.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--junit",
        type=Path,
        default=PHASE / "pytest-retirement.xml",
    )
    arguments = parser.parse_args()
    print(json.dumps(finalize(arguments.junit), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
