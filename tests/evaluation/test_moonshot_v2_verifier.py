from pathlib import Path

from layercake.evaluation.moonshot_verifier import (
    ALLOWED,
    REQUIRED_GATES,
    verify_moonshot_v2,
)


def test_v2_verifier_fails_closed_when_evidence_is_missing(tmp_path):
    root = Path(__file__).resolve().parents[2]
    certificate = verify_moonshot_v2(root, tmp_path)
    assert certificate["moonshot_proven"] is False
    assert certificate["overall_status"] == "FAIL"
    assert set(certificate["gates"]) == set(REQUIRED_GATES)
    assert all(gate["status"] in ALLOWED for gate in certificate["gates"].values())
    assert certificate["gates"]["repository_regression"]["status"] == "INVALID_EVIDENCE"
