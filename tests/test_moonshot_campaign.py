from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from layercake.moonshot_campaign import (
    PHASE_KEYS,
    CampaignVerificationError,
    changed_components,
    recompute_derivation,
    sha256_file,
    validate_artifact_manifest,
    validate_baselines,
    validate_campaign_state,
    validate_matched_quality,
    validate_required_gates,
    validate_seed_evidence,
    validate_semantic_portability,
    verify_derived_claim,
)


def _claim_contract() -> dict:
    return {
        "allowed_phase_statuses": ["LOCKED", "OPEN", "PASS"],
        "phase_requirements": {
            "1": {
                "required_baselines": [
                    "bpe_reference",
                    "optimized_cpu_transformer",
                    "optimized_gpu_transformer",
                    "byte_transformer",
                    "fastest_existing_layercake",
                    "highest_quality_existing_layercake",
                ]
            }
        },
    }


def _lineage() -> dict:
    return {
        "source_commit": "a" * 40,
        "architecture_id": None,
        "architecture_hash": None,
        "abi_hash": None,
        "data_hashes": {},
        "core_checkpoint_hashes": {},
        "transformer_checkpoint_hashes": {},
        "cake_package_hashes": {},
        "router_hash": None,
        "runtime_hashes": {},
    }


def _campaign(current: int = 0) -> dict:
    phases = {
        key: "PASS" if index < current else "OPEN" if index == current else "LOCKED"
        for index, key in enumerate(PHASE_KEYS)
    }
    return {
        "campaign": "layercake-moonshot",
        "campaign_version": 1,
        "current_phase": current,
        "phases": phases,
        "lineage": _lineage(),
    }


def test_campaign_state_accepts_only_one_ordered_open_phase() -> None:
    validate_campaign_state(_campaign(0), _claim_contract())
    validate_campaign_state(_campaign(5), _claim_contract())

    invalid = _campaign(1)
    invalid["phases"]["phase4_portable_domain"] = "OPEN"
    with pytest.raises(CampaignVerificationError, match="future phase"):
        validate_campaign_state(invalid, _claim_contract())


def test_campaign_state_rejects_manual_pass_without_advancing_current_phase() -> None:
    invalid = _campaign(0)
    invalid["phases"]["phase0_governance"] = "PASS"
    with pytest.raises(CampaignVerificationError, match="current_phase"):
        validate_campaign_state(invalid, _claim_contract())


def test_raw_derivations_recompute_mean_quantile_and_ratio() -> None:
    raw = {
        "records": [
            {"system": "layercake", "latency": 1.0},
            {"system": "layercake", "latency": 3.0},
            {"system": "transformer", "latency": 8.0},
            {"system": "transformer", "latency": 12.0},
        ]
    }
    assert recompute_derivation(
        raw, {"operation": "mean", "field": "latency", "where": {"system": "layercake"}}
    ) == 2.0
    assert recompute_derivation(
        raw,
        {
            "operation": "nearest_rank_quantile",
            "quantile": 0.95,
            "field": "latency",
            "where": {"system": "transformer"},
        },
    ) == 12.0
    assert recompute_derivation(
        raw,
        {
            "operation": "ratio",
            "numerator": {"field": "latency", "where": {"system": "transformer"}},
            "denominator": {"field": "latency", "where": {"system": "layercake"}},
        },
    ) == 5.0


def test_headline_claim_must_match_hashed_raw_derivation(tmp_path: Path) -> None:
    path = tmp_path / "results/moonshot/phase2/raw_runs/run.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"records": [{"throughput": 10.0}, {"throughput": 20.0}]}))
    claim = {
        "raw_artifact": "results/moonshot/phase2/raw_runs/run.json",
        "raw_sha256": sha256_file(path),
        "derivation": {"operation": "mean", "field": "throughput"},
        "value": 15.0,
    }
    assert verify_derived_claim(tmp_path, 2, claim) == 15.0

    hard_coded = dict(claim)
    hard_coded.pop("derivation")
    with pytest.raises(CampaignVerificationError, match="hard-coded"):
        verify_derived_claim(tmp_path, 2, hard_coded)

    wrong = dict(claim, value=99.0)
    with pytest.raises(CampaignVerificationError, match="does not recompute"):
        verify_derived_claim(tmp_path, 2, wrong)


def test_missing_and_omitted_failed_seeds_are_rejected() -> None:
    with pytest.raises(CampaignVerificationError, match="missing seeds"):
        validate_seed_evidence([{"records": [{"seed": 1}, {"seed": 2}]}], 3)
    with pytest.raises(CampaignVerificationError, match="not preserved"):
        validate_seed_evidence(
            [{"records": [{"seed": 1}, {"seed": 2}, {"seed": 3}], "failed_seeds": [4]}],
            3,
        )
    assert validate_seed_evidence(
        [{"records": [{"seed": 1}, {"seed": 2}, {"seed": 3}, {"seed": 4}], "failed_seeds": [4]}],
        3,
    ) == [1, 2, 3, 4]


def _baseline(identifier: str, *, eager: bool = False) -> dict:
    return {
        "id": identifier,
        "runtime": {
            "name": "llama.cpp" if "transformer" in identifier else "pytorch-control",
            "version": "test-version",
            "execution": "eager_python" if eager else "native",
            "deployment_grade": True,
            "kv_cache": True,
        },
    }


def test_phase1_baseline_inventory_rejects_missing_or_eager_optimized_baseline() -> None:
    required = _claim_contract()["phase_requirements"]["1"]["required_baselines"]
    certificate = {"baselines": [_baseline(identifier) for identifier in required]}
    validate_baselines(certificate, _claim_contract())

    missing = {"baselines": certificate["baselines"][:-1]}
    with pytest.raises(CampaignVerificationError, match="missing"):
        validate_baselines(missing, _claim_contract())

    eager = {"baselines": list(certificate["baselines"])}
    eager["baselines"][1] = _baseline("optimized_cpu_transformer", eager=True)
    with pytest.raises(CampaignVerificationError, match="eager Python"):
        validate_baselines(eager, _claim_contract())


def test_speed_claims_fail_closed_without_every_matched_quality_dimension() -> None:
    benchmark = {
        "matched_quality_dimensions": [
            "heldout_bpb",
            "functional_task_quality",
            "instruction_following",
            "invalid_output_rate",
            "repetition",
            "coherence",
            "domain_success",
        ]
    }
    certificate = {
        "claims": [{"kind": "throughput", "promoted": True}],
        "quality_match": {
            name: True for name in benchmark["matched_quality_dimensions"]
        }
        | {
            "layercake_checkpoint_sha256": "a" * 64,
            "transformer_checkpoint_sha256": "b" * 64,
        },
    }
    validate_matched_quality(certificate, benchmark)
    certificate["quality_match"]["coherence"] = False
    with pytest.raises(CampaignVerificationError, match="quality-unmatched"):
        validate_matched_quality(certificate, benchmark)


def test_semantic_portability_requires_nonempty_source_successes_and_no_calibration() -> None:
    certificate = {
        "claims": [{"kind": "semantic_portability", "promoted": True}],
        "semantic_portability": {
            "source_success_task_ids": [],
            "receivers": [],
        },
    }
    with pytest.raises(CampaignVerificationError, match="source task successes"):
        validate_semantic_portability(certificate)

    certificate["semantic_portability"] = {
        "source_success_task_ids": ["python-1", "python-2"],
        "receivers": [
            {
                "id": "host-b",
                "success_task_ids": ["python-1", "python-2"],
                "receiver_training_examples": 0,
                "calibration_performed": False,
            }
        ],
    }
    validate_semantic_portability(certificate)
    certificate["semantic_portability"]["receivers"][0]["calibration_performed"] = True
    with pytest.raises(CampaignVerificationError, match="calibration"):
        validate_semantic_portability(certificate)


def test_artifact_hash_manifest_detects_stale_certificate_inputs(tmp_path: Path) -> None:
    artifact = tmp_path / "raw.json"
    artifact.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    validate_artifact_manifest(tmp_path, {"raw.json": digest})
    artifact.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(CampaignVerificationError, match="stale or modified"):
        validate_artifact_manifest(tmp_path, {"raw.json": digest})


def test_component_change_detection_is_fail_closed() -> None:
    assert changed_components({"architecture": "a", "data": "b"}, {"architecture": "a", "data": "b"}) == []
    assert changed_components(
        {"architecture": "a", "data": "b"},
        {"architecture": "changed", "data": "b", "runtime": "new"},
    ) == ["architecture", "runtime"]


def test_required_gate_threshold_is_loaded_from_contract_and_recomputed(tmp_path: Path) -> None:
    raw_path = tmp_path / "results/moonshot/phase2/raw_runs/gate.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps({"records": [{"ratio": 2.1}]}), encoding="utf-8")
    claim = {
        "gate_id": "cpu_throughput_ratio",
        "raw_artifact": "results/moonshot/phase2/raw_runs/gate.json",
        "raw_sha256": sha256_file(raw_path),
        "derivation": {"operation": "mean", "field": "ratio"},
        "value": 2.1,
    }
    contract = {
        "phase_requirements": {
            "2": {
                "required_gates": [
                    {"id": "cpu_throughput_ratio", "operator": "ge", "threshold": 2.0}
                ]
            }
        }
    }
    assert validate_required_gates(tmp_path, 2, {"claims": [claim]}, contract) == {
        "cpu_throughput_ratio": 2.1
    }
    claim["value"] = 1.9
    raw_path.write_text(json.dumps({"records": [{"ratio": 1.9}]}), encoding="utf-8")
    claim["raw_sha256"] = sha256_file(raw_path)
    with pytest.raises(CampaignVerificationError, match="gate cpu_throughput_ratio failed"):
        validate_required_gates(tmp_path, 2, {"claims": [claim]}, contract)
