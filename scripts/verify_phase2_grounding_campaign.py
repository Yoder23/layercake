"""Read-only independent verification of the closed Phase 2 grounding campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/moonshot/phase2_grounding_campaign"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    decision_path = OUT / "branch_decision.json"
    verifier_path = OUT / "adversarial_verifier.json"
    decision = _read(decision_path)
    verifier = _read(verifier_path)
    unsigned_decision = dict(decision)
    decision_sha = unsigned_decision.pop("decision_sha256")
    unsigned_verifier = dict(verifier)
    verifier_sha = unsigned_verifier.pop("verifier_sha256")
    checks = {
        "decision_payload_hash": _canonical_sha(unsigned_decision) == decision_sha,
        "verifier_payload_hash": _canonical_sha(unsigned_verifier) == verifier_sha,
        "verifier_binds_decision_file": (
            verifier["decision_file_sha256"] == _sha(decision_path)
        ),
        "verifier_binds_decision_payload": (
            verifier["decision_payload_sha256"] == decision_sha
        ),
        "embedded_checks_pass": all(verifier["checks"].values()),
        "no_embedded_failures": verifier["failures"] == [],
        "exactly_two_hypotheses": len(decision["hypothesis_rows"]) == 2,
        "every_hypothesis_failed": all(
            row["development_pass"] is False
            for row in decision["hypothesis_rows"]
        ),
        "no_winner": decision["winner"] is None,
        "no_selected_architecture": decision["selected_architecture"] is None,
        "no_three_seed_promotion": (
            decision["three_seed_promotion_authorized"] is False
        ),
        "phase2_not_sealed": decision["phase2_r3_tag_created"] is False,
        "phase3_locked": (
            decision["phase3_status"] == "LOCKED"
            and decision["phase3_execution_authorized"] is False
        ),
        "no_test_access": decision["test_accessed"] is False,
        "phase4_contract_hash": (
            _sha(ROOT / decision["phase4_cpu_cake_training_contract"]["path"])
            == decision["phase4_cpu_cake_training_contract"]["sha256"]
        ),
    }
    for index, row in enumerate(decision["hypothesis_rows"]):
        checks[f"hypothesis_{index}_checkpoint_hash"] = (
            _sha(ROOT / row["checkpoint"] / "model.safetensors")
            == row["checkpoint_sha256"]
        )
        checks[f"hypothesis_{index}_screen_hash"] = (
            _sha(ROOT / row["screen"]["path"]) == row["screen"]["sha256"]
        )
        checks[f"hypothesis_{index}_audit_hash"] = (
            _sha(ROOT / row["semantic_audit"]["path"])
            == row["semantic_audit"]["sha256"]
        )
    control = decision["speed_and_distributional_quality_control"]
    for label in ("screen", "semantic_audit", "benchmark_128", "benchmark_1024"):
        checks[f"control_{label}_hash"] = (
            _sha(ROOT / control[label]["path"]) == control[label]["sha256"]
        )
    for label, evidence in decision["diagnosis"]["evidence"].items():
        checks[f"diagnosis_{label}_hash"] = (
            _sha(ROOT / evidence["path"]) == evidence["sha256"]
        )
    task = _read(
        ROOT / "results/moonshot/phase2_recertification/task_state.json"
    )
    campaign = _read(ROOT / "moonshot/campaign.yaml")
    checks.update({
        "task_phase2_open": (
            task["phase2_status"] == "OPEN_RESEARCH_DIRECTION_REQUIRED"
        ),
        "task_phase3_locked": task["phase3_status"] == "LOCKED",
        "campaign_phase2_open": (
            campaign["phases"]["phase2_cpu_quality_speed"] == "OPEN"
        ),
        "campaign_phase3_locked": (
            campaign["phases"]["phase3_training_speed"] == "LOCKED"
        ),
        "campaign_binds_phase4_contract": (
            campaign["forward_contract_amendments"][0]["contract_sha256"]
            == decision["phase4_cpu_cake_training_contract"]["sha256"]
        ),
    })
    failures = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "format": "layercake-phase2-grounding-independent-verification/1",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "decision_sha256": decision_sha,
        "verifier_sha256": verifier_sha,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
