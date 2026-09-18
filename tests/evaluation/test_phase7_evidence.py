from __future__ import annotations

from pathlib import Path

import pytest

import layercake.evaluation.phase7_evidence as phase7


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PRESENT = (ROOT / phase7.PAYLOAD).is_file()


def _evidence_matches_current_lineage() -> bool:
    if not EVIDENCE_PRESENT:
        return False
    try:
        phase7._validate_framework(ROOT)
    except phase7.Phase7EvidenceError:
        return False
    return True


EVIDENCE_READY = _evidence_matches_current_lineage()


@pytest.mark.skipif(
    not EVIDENCE_PRESENT or EVIDENCE_READY,
    reason="no stale Phase 7 evidence is present on this checkout",
)
def test_phase7_stale_evidence_fails_closed_on_development_head():
    with pytest.raises(phase7.Phase7EvidenceError, match="dependent components changed"):
        phase7.derive_phase7_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 7 evidence is produced only after the framework freeze",
)
def test_phase7_metrics_recompute_from_bound_raw_evidence():
    result = phase7.derive_phase7_metrics(ROOT)
    metrics = result["metrics"]
    assert metrics["cpu_cpu_throughput_ratio"] >= 5.0
    assert metrics["cpu_cpu_median_latency_ratio"] <= 0.25
    assert metrics["gpu_gpu_throughput_ratio"] > 1.0
    assert metrics["cpu_gpu_throughput_ratio"] >= 1.0
    assert metrics["cpu_gpu_median_latency_ratio"] <= 1.0
    assert metrics["general_quality_noninferior"] == 1.0
    assert metrics["mixed_domain_quality_superior"] == 1.0
    assert metrics["promoted_domain_success_retention"] == 1.0


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 7 evidence is produced only after the framework freeze",
)
def test_phase7_package_lineage_mutation_is_rejected(monkeypatch):
    changed = dict(phase7.PACKAGE_HASHES)
    changed["python"] = "0" * 64
    monkeypatch.setattr(phase7, "PACKAGE_HASHES", changed)
    with pytest.raises(
        phase7.Phase7EvidenceError,
        match="sealed python package changed",
    ):
        phase7.derive_phase7_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 7 evidence is produced only after the framework freeze",
)
def test_phase7_transformer_gpu_digest_mutation_is_rejected(monkeypatch):
    monkeypatch.setattr(phase7, "QWEN_DIGEST", "0" * 64)
    with pytest.raises(
        phase7.Phase7EvidenceError,
        match="GPU performance evidence identity is invalid",
    ):
        phase7.derive_phase7_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 7 evidence is produced only after the framework freeze",
)
def test_phase7_claim_boundaries_remain_explicit():
    result = phase7.derive_phase7_metrics(ROOT)
    boundaries = result["claim_boundaries"]
    assert boundaries["physical_mobile_hardware_claimed"] is False
    assert boundaries["gpu_training_dominance_claimed"] is False
    assert boundaries["latent_neural_fusion_claimed"] is False
