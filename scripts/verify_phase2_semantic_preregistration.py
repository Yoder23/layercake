"""Read-only verification of the Phase 2 semantic-encoder preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main() -> int:
    boundary_path = "moonshot/phase2_core_cake_boundary.json"
    preregistration_path = (
        "moonshot/phase2_semantic_encoder_preregistration.json"
    )
    campaign = _read("moonshot/campaign.yaml")
    task = _read("results/moonshot/phase2_recertification/task_state.json")
    preregistration = _read(preregistration_path)
    amendments = {
        row["amendment_id"]: row
        for row in campaign["forward_contract_amendments"]
    }
    checks = {
        "phase2_open": campaign["phases"]["phase2_cpu_quality_speed"] == "OPEN",
        "phase3_locked": campaign["phases"]["phase3_training_speed"] == "LOCKED",
        "boundary_locked": _read(boundary_path)["status"] == "LOCKED",
        "boundary_hash": (
            amendments["phase2-core-cake-boundary/1"]["contract_sha256"]
            == _sha(boundary_path)
        ),
        "preregistration_status": (
            preregistration["status"] == "PREREGISTERED"
        ),
        "preregistration_hash": (
            amendments[
                "phase2-semantic-encoder-preregistration/1"
            ]["contract_sha256"] == _sha(preregistration_path)
        ),
        "closed_branch_evidence_hash": (
            preregistration["falsified_prior_mechanism"]["evidence_sha256"]
            == _sha(
                preregistration["falsified_prior_mechanism"]["evidence"]
            )
        ),
        "single_active_candidate": (
            task["active_candidate"]
            == preregistration["hypothesis"]["id"]
        ),
        "phase3_task_lock": task["phase3_status"] == "LOCKED",
        "no_nearby_sweeps": (
            preregistration["training_contract"][
                "nearby_architecture_sweeps_allowed"
            ] is False
        ),
        "same_lineage_final_evidence": (
            "one integrated checkpoint"
            in preregistration["promotion_rule"].lower()
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "format": "layercake-phase2-semantic-preregistration-verifier/1",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "boundary_sha256": _sha(boundary_path),
        "preregistration_sha256": _sha(preregistration_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
