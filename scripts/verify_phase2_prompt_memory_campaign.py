"""Read-only adversarial verification for the closed prompt-memory campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/moonshot/phase2_prompt_memory"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    decision_path = OUT / "branch_decision.json"
    verifier_path = OUT / "adversarial_verifier.json"
    decision = _read(decision_path)
    verifier = _read(verifier_path)
    unsigned_decision = dict(decision)
    claimed_decision_sha = unsigned_decision.pop("decision_sha256")
    unsigned_verifier = dict(verifier)
    claimed_verifier_sha = unsigned_verifier.pop("verifier_sha256")
    checks = {
        "decision_payload_hash": _canonical(unsigned_decision)
        == claimed_decision_sha,
        "verifier_payload_hash": _canonical(unsigned_verifier)
        == claimed_verifier_sha,
        "verifier_binds_decision_file": verifier["decision_file_sha256"]
        == _sha(decision_path),
        "verifier_binds_decision_payload": verifier[
            "decision_payload_sha256"
        ] == claimed_decision_sha,
        "embedded_checks_pass": all(verifier["checks"].values()),
        "no_embedded_failures": verifier["failures"] == [],
        "no_winner": decision["winner"] is None,
        "no_three_seed_promotion": decision[
            "three_seed_promotion_authorized"
        ] is False,
        "phase2_not_sealed": decision["phase2_r3_tag_created"] is False,
        "phase3_locked": (
            decision["phase3_status"] == "LOCKED"
            and decision["phase3_execution_authorized"] is False
        ),
        "every_candidate_failed": all(
            row["continuation_pass"] is False
            for row in decision["pareto_rows"]
        ),
        "three_predeclared_candidates": len(decision["pareto_rows"]) == 3,
        "no_test_access": decision["test_accessed"] is False,
    }
    for index, row in enumerate(decision["pareto_rows"]):
        checkpoint = ROOT / row["checkpoint"] / "model.safetensors"
        screen = ROOT / row["screen"]["path"]
        audit = ROOT / row["semantic_audit"]["path"]
        checks[f"candidate_{index}_checkpoint_hash"] = (
            _sha(checkpoint) == row["checkpoint_sha256"]
        )
        checks[f"candidate_{index}_screen_hash"] = (
            _sha(screen) == row["screen"]["sha256"]
        )
        checks[f"candidate_{index}_audit_hash"] = (
            _sha(audit) == row["semantic_audit"]["sha256"]
        )
    for label, evidence in decision["evidence"].items():
        checks[f"{label}_hash"] = (
            _sha(ROOT / evidence["path"]) == evidence["sha256"]
        )
    failures = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "format": "layercake-phase2-prompt-memory-independent-verification/1",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "decision_sha256": claimed_decision_sha,
        "verifier_sha256": claimed_verifier_sha,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
