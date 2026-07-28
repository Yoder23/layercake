import json
from pathlib import Path

import pytest

import layercake.evaluation.phase4_evidence as phase4


ROOT = Path(__file__).resolve().parents[2]


def test_phase4_raw_metrics_recompute_from_bound_evidence():
    result = phase4.derive_phase4_metrics(ROOT)
    assert result["seeds"]["unique_seeds"] == 3
    assert [
        row["functional_successes"]
        for row in result["seeds"]["seeds"]
    ] == [63, 64, 64]
    assert len(result["package"]["source_success_task_ids"]) == 128
    assert [
        row["device"] for row in result["package"]["receivers"]
    ] == ["cpu", "cpu", "cuda:0"]
    assert result["metrics"]["functional_error_ratio"] == 0.0
    assert (
        result["metrics"]["core_plus_cake_cpu_transformer_ratio"]
        >= 2.0
    )
    assert result["benchmark"]["median_paired_throughput_ratio"] >= 2.0
    assert result["benchmark"]["bootstrap_lower"] >= 2.0


def test_phase4_direct_interface_does_not_claim_semantic_fusion():
    abi = json.loads(
        (
            ROOT
            / "moonshot/phase4_canonical_direct_decoder_abi_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert abi["version"] == phase4.DIRECT_ABI_VERSION
    assert "same-shape semantic residual fusion" in abi[
        "claim_boundary"
    ]["does_not_prove"]
    assert "multi-domain composition" in abi["claim_boundary"][
        "does_not_prove"
    ]


def test_phase4_package_hash_mutation_is_rejected(monkeypatch):
    monkeypatch.setattr(phase4, "PACKAGE_SHA256", "0" * 64)
    with pytest.raises(
        phase4.Phase4EvidenceError, match="archive changed"
    ):
        phase4.derive_phase4_metrics(ROOT)
