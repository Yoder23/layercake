"""Read-only verification of the active factorized-control hypothesis."""

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
    prereg_path = "moonshot/phase2_factorized_control_preregistration.json"
    campaign = _read("moonshot/campaign.yaml")
    task = _read("results/moonshot/phase2_recertification/task_state.json")
    prereg = _read(prereg_path)
    prior_path = prereg["prior_contextual_branch_decision"]
    amendments = {
        row["amendment_id"]: row
        for row in campaign["forward_contract_amendments"]
    }
    checks = {
        "phase2_open": campaign["phases"]["phase2_cpu_quality_speed"] == "OPEN",
        "phase3_locked": campaign["phases"]["phase3_training_speed"] == "LOCKED",
        "preregistration_status": prereg["status"] == "PREREGISTERED",
        "preregistration_hash": (
            amendments["phase2-factorized-control-preregistration/1"][
                "contract_sha256"
            ] == _sha(prereg_path)
        ),
        "prior_contextual_branch_closed": (
            _read(prior_path)["decision"] == "CLOSED_NEGATIVE"
        ),
        "prior_contextual_branch_hash": (
            task["latest_batch_sha256"] == _sha(prior_path)
        ),
        "prior_shared_branch_hash": (
            prereg["prior_shared_tokenizer_decision_sha256"]
            == _sha(prereg["prior_shared_tokenizer_decision"])
        ),
        "single_active_candidate": (
            task["active_candidate"] == prereg["hypothesis"]["id"]
        ),
        "factor_records_fixed": (
            prereg["hypothesis"]["control_records"]
            == [
                "global prompt record",
                "neural task record",
                "neural topic-span record",
            ]
        ),
        "multi_horizon_locked": (
            prereg["training_contract"][
                "multi_horizon_recovery_schedule"
            ] == [8, 32, 64]
        ),
        "no_hard_constraints": (
            prereg["hypothesis"][
                "inference_templates_or_hard_constraints"
            ] is False
        ),
        "no_nearby_sweeps": (
            prereg["training_contract"][
                "nearby_architecture_sweeps_allowed"
            ] is False
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "format": (
            "layercake-phase2-factorized-control-preregistration-verifier/1"
        ),
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "prior_contextual_branch_sha256": _sha(prior_path),
        "preregistration_sha256": _sha(prereg_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
