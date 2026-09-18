from __future__ import annotations

from pathlib import Path

import pytest

import layercake.evaluation.phase8_evidence as phase8


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PRESENT = (ROOT / phase8.PAYLOAD).is_file()


def _evidence_matches_current_lineage() -> bool:
    if not EVIDENCE_PRESENT:
        return False
    try:
        phase8._validate_framework(ROOT)
    except phase8.Phase8EvidenceError:
        return False
    return True


EVIDENCE_READY = _evidence_matches_current_lineage()


@pytest.mark.skipif(
    not EVIDENCE_PRESENT or EVIDENCE_READY,
    reason="no stale Phase 8 evidence is present on this checkout",
)
def test_phase8_stale_evidence_fails_closed_on_development_head():
    with pytest.raises(phase8.Phase8EvidenceError, match="sealed product components changed"):
        phase8.derive_phase8_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 8 evidence is produced only after framework freeze",
)
def test_phase8_metrics_recompute_from_bound_raw_evidence():
    result = phase8.derive_phase8_metrics(ROOT)
    assert result["status"] == "PROVEN"
    assert result["metrics"] == {
        "prior_required_gate_retention": 1.0,
        "clean_room_reproduction": 1.0,
        "adversarial_falsification_findings_resolved": 1.0,
    }
    assert result["details"]["adversarial"]["attacks"] >= 24
    assert result["details"]["adversarial"]["categories"] >= 24


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 8 evidence is produced only after framework freeze",
)
def test_phase8_package_lineage_mutation_is_rejected(monkeypatch):
    changed = dict(phase8.PACKAGES)
    path, _ = changed["python"]
    changed["python"] = (path, "0" * 64)
    monkeypatch.setattr(phase8, "PACKAGES", changed)
    with pytest.raises(
        phase8.Phase8EvidenceError,
        match="package/ABI manifest is stale",
    ):
        phase8.derive_phase8_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 8 evidence is produced only after framework freeze",
)
def test_phase8_transformer_digest_mutation_is_rejected(monkeypatch):
    monkeypatch.setattr(phase8, "QWEN_DIGEST", "0" * 64)
    with pytest.raises(
        phase8.Phase8EvidenceError,
        match="fresh performance protocol identity is invalid",
    ):
        phase8.derive_phase8_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 8 evidence is produced only after framework freeze",
)
def test_phase8_contract_mutation_is_rejected(monkeypatch):
    monkeypatch.setattr(phase8, "CONTRACT_SHA256", "0" * 64)
    with pytest.raises(
        phase8.Phase8EvidenceError,
        match="Phase 8 preregistration changed",
    ):
        phase8.derive_phase8_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 8 evidence is produced only after framework freeze",
)
def test_phase8_claim_boundaries_remain_explicit():
    result = phase8.derive_phase8_metrics(ROOT)
    details = result["details"]
    assert details["training"] == {
        "status": "RETIRED_BY_GOVERNANCE",
        "scientific_training_efficiency_passed": False,
        "product_gate": False,
    }
    assert details["mobile"]["physical_mobile_performance_claimed"] is False
    assert all(
        value is False
        for value in details["claim_boundaries"].values()
    )
