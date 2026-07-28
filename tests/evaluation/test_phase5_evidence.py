from __future__ import annotations

from pathlib import Path

import pytest

import layercake.evaluation.phase5_evidence as phase5


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_READY = (ROOT / phase5.PAYLOAD).is_file()


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 5 evidence is produced after the framework freeze",
)
def test_phase5_metrics_recompute_from_raw_evidence():
    result = phase5.derive_phase5_metrics(ROOT)
    assert result["metrics"]["real_neural_domain_count"] == 3.0
    assert (
        result["metrics"][
            "minimum_domain_functional_error_reduction"
        ]
        >= 5.0
    )
    assert (
        result["metrics"][
            "generic_authoring_requires_source_edits"
        ]
        == 0.0
    )
    assert result["metrics"]["inactive_cake_effect"] == 0.0


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 5 evidence is produced after the framework freeze",
)
def test_phase5_python_parent_hash_mutation_is_rejected(monkeypatch):
    monkeypatch.setattr(phase5, "PYTHON_PACKAGE_SHA256", "0" * 64)
    with pytest.raises(
        phase5.Phase5EvidenceError,
        match="sealed Python package changed",
    ):
        phase5.derive_phase5_metrics(ROOT)


@pytest.mark.skipif(
    not EVIDENCE_READY,
    reason="Phase 5 evidence is produced after the framework freeze",
)
def test_phase5_cannot_borrow_routing_or_fusion_credit():
    summary = phase5.validate_phase5_bundle(
        ROOT, ROOT / "results/moonshot/phase5"
    )
    assert summary["routing_claimed"] is False
    assert summary["fusion_or_composition_claimed"] is False
