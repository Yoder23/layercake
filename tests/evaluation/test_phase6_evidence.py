from __future__ import annotations

from pathlib import Path

import pytest

import layercake.evaluation.phase6_evidence as phase6


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PRESENT = (ROOT / phase6.PAYLOAD).is_file()


def _evidence_matches_current_lineage() -> bool:
    if not EVIDENCE_PRESENT:
        return False
    try:
        phase6._validate_framework(ROOT)
    except phase6.Phase6EvidenceError:
        return False
    return True


EVIDENCE_READY = _evidence_matches_current_lineage()


@pytest.mark.skipif(
    not EVIDENCE_PRESENT or EVIDENCE_READY,
    reason="no stale Phase 6 evidence is present on this checkout",
)
def test_phase6_stale_evidence_fails_closed_on_development_head():
    with pytest.raises(phase6.Phase6EvidenceError, match="dependent components changed"):
        phase6.derive_phase6_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 6 evidence is produced only after the framework freeze",
)
def test_phase6_metrics_recompute_from_raw_evidence():
    result = phase6.derive_phase6_metrics(ROOT)
    assert result["metrics"]["top1_accuracy"] >= 0.95
    assert result["metrics"]["topk_recall"] >= 0.98
    assert result["metrics"]["false_specialist_activation"] <= 0.02
    assert result["metrics"]["inactive_cake_forward_calls"] == 0.0


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 6 evidence is produced only after the framework freeze",
)
def test_phase6_package_lineage_mutation_is_rejected(monkeypatch):
    changed = dict(phase6.PACKAGE_HASHES)
    changed["python"] = "0" * 64
    monkeypatch.setattr(phase6, "PACKAGE_HASHES", changed)
    with pytest.raises(phase6.Phase6EvidenceError, match="sealed python package changed"):
        phase6.derive_phase6_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 6 evidence is produced only after the framework freeze",
)
def test_phase6_router_profile_mutation_is_rejected(monkeypatch):
    monkeypatch.setattr(phase6, "PROFILES_SHA256", "0" * 64)
    with pytest.raises(
        phase6.Phase6EvidenceError, match="routing profiles changed"
    ):
        phase6.derive_phase6_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 6 evidence is produced only after the framework freeze",
)
def test_phase6_claim_boundaries_remain_explicit():
    result = phase6.derive_phase6_metrics(ROOT)
    assert result["claim_boundaries"]["real_promoted_neural_capabilities"] == 3
    assert result["claim_boundaries"]["latent_neural_fusion_claimed"] is False
    assert result["claim_boundaries"]["hundred_domain_quality_claimed"] is False
