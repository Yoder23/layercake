"""Read-only verification of the active Phase 2 contextual-token hypothesis."""

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
        "moonshot/phase2_contextual_token_memory_preregistration.json"
    )
    campaign = _read("moonshot/campaign.yaml")
    task = _read("results/moonshot/phase2_recertification/task_state.json")
    preregistration = _read(preregistration_path)
    decision_path = preregistration["prior_branch_decision"]
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
                "phase2-contextual-token-memory-preregistration/1"
            ]["contract_sha256"] == _sha(preregistration_path)
        ),
        "prior_branch_decision_hash": (
            task["latest_batch_sha256"] == _sha(decision_path)
        ),
        "prior_branch_closed": (
            _read(decision_path)["decision"] == "CLOSED_NEGATIVE"
        ),
        "single_active_candidate": (
            task["active_candidate"] == preregistration["hypothesis"]["id"]
        ),
        "phase3_task_lock": task["phase3_status"] == "LOCKED",
        "no_nearby_sweeps": (
            preregistration["training_contract"][
                "nearby_architecture_sweeps_allowed"
            ] is False
        ),
        "direct_memory_not_dense_slots": all((
            preregistration["hypothesis"][
                "learned_semantic_slot_compression"
            ] is False,
            preregistration["hypothesis"][
                "dense_slot_to_vocabulary_state"
            ] is False,
        )),
        "same_lineage_final_evidence": (
            "one integrated checkpoint"
            in preregistration["promotion_rule"].lower()
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "format": (
            "layercake-phase2-contextual-token-preregistration-verifier/1"
        ),
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "boundary_sha256": _sha(boundary_path),
        "prior_branch_decision_sha256": _sha(decision_path),
        "preregistration_sha256": _sha(preregistration_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
