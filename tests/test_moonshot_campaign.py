from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

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
    verify_sealed,
)


def _claim_contract() -> dict:
    return {
        "allowed_phase_statuses": ["LOCKED", "OPEN", "CANDIDATE", "PASS", "SEALED"],
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
        "model_source_commit": "a" * 40,
        "governance_commit": "b" * 40,
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
        key: "SEALED" if index < current else "OPEN" if index == current else "LOCKED"
        for index, key in enumerate(PHASE_KEYS)
    }
    return {
        "campaign": "layercake-moonshot",
        "campaign_version": 1,
        "current_phase": current,
        "phases": phases,
        "lineage": _lineage(),
    }


def test_campaign_state_accepts_only_one_ordered_active_phase() -> None:
    validate_campaign_state(_campaign(0), _claim_contract())
    validate_campaign_state(_campaign(5), _claim_contract())

    invalid = _campaign(1)
    invalid["phases"]["phase4_portable_domain"] = "OPEN"
    with pytest.raises(CampaignVerificationError, match="future phase"):
        validate_campaign_state(invalid, _claim_contract())


def test_campaign_state_rejects_manual_pass_without_advancing_current_phase() -> None:
    invalid = _campaign(0)
    invalid["phases"]["phase0_governance"] = "SEALED"
    with pytest.raises(CampaignVerificationError, match="current_phase|LOCKED"):
        validate_campaign_state(invalid, _claim_contract())


@pytest.mark.parametrize("status", ["OPEN", "CANDIDATE", "PASS"])
def test_active_phase_lifecycle_states_do_not_unlock_the_future(status: str) -> None:
    campaign = _campaign(1)
    campaign["phases"]["phase1_benchmark_truth"] = status
    validate_campaign_state(campaign, _claim_contract())
    campaign["phases"]["phase2_cpu_quality_speed"] = "OPEN"
    with pytest.raises(CampaignVerificationError, match="future phase"):
        validate_campaign_state(campaign, _claim_contract())


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
            "runtime_manifest": {"path": "runtime.json", "sha256": "a" * 64},
            "deployment_evidence": {"path": "trace.json", "sha256": "b" * 64},
            "kv_cache_evidence": {"path": "trace.json", "sha256": "b" * 64},
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


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, check=True
    )
    return result.stdout


def _sealed_fixture(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "sealed-repository"
    repository.mkdir()
    source_root = Path(__file__).resolve().parents[1]
    (repository / "moonshot").mkdir()
    for name in (
        "claim_contract.yaml", "invalidation_matrix.yaml", "benchmark_contract.yaml",
        "data_contract.yaml", "security_contract.yaml",
    ):
        shutil.copyfile(source_root / "moonshot" / name, repository / "moonshot" / name)
    phase_dir = repository / "results/moonshot/phase0"
    phase_dir.mkdir(parents=True)
    evidence = phase_dir / "evidence.json"
    evidence.write_text('{"measured": 1}\n', encoding="utf-8")
    campaign = _campaign(1)
    campaign["phase_records"] = {"phase0": {}}
    (repository / "moonshot/campaign.yaml").write_text(
        json.dumps(campaign, indent=2), encoding="utf-8"
    )
    _git(repository, "init")
    _git(repository, "config", "user.email", "campaign-test@example.invalid")
    _git(repository, "config", "user.name", "Campaign Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "release evidence")
    release = _git(repository, "rev-parse", "HEAD")
    digest = hashlib.sha256(
        _git_bytes(repository, "show", f"{release}:results/moonshot/phase0/evidence.json")
    ).hexdigest()
    tag = "layercake-moonshot-phase0"
    seal = {
        "format": "layercake-moonshot-seal/1",
        "campaign_version": 1,
        "phase": 0,
        "status": "SEALED",
        "release_commit": release,
        "completion_tag": tag,
        "sealed_artifact_hashes": {"results/moonshot/phase0/evidence.json": digest},
        "required_tag_kind": "annotated",
        "required_worktree_state": "clean",
    }
    (phase_dir / "seal.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")
    campaign["phase_records"]["phase0"] = {
        "phase_release_commit": release,
        "phase_tag": tag,
        "seal": "results/moonshot/phase0/seal.json",
    }
    (repository / "moonshot/campaign.yaml").write_text(
        json.dumps(campaign, indent=2), encoding="utf-8"
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "seal metadata")
    return repository, tag


def test_sealed_verifier_rejects_lightweight_tag_and_accepts_annotated_tag(tmp_path: Path) -> None:
    repository, tag = _sealed_fixture(tmp_path)
    _git(repository, "tag", tag)
    with pytest.raises(CampaignVerificationError, match="not annotated"):
        verify_sealed(repository, 0)
    _git(repository, "tag", "-d", tag)
    _git(repository, "tag", "-a", tag, "-m", "verified seal")
    assert verify_sealed(repository, 0)["passed"] is True


def test_sealed_verifier_rejects_dirty_or_post_tag_state(tmp_path: Path) -> None:
    repository, tag = _sealed_fixture(tmp_path)
    _git(repository, "tag", "-a", tag, "-m", "verified seal")
    evidence = repository / "results/moonshot/phase0/evidence.json"
    evidence.write_text('{"measured": 2}\n', encoding="utf-8")
    with pytest.raises(CampaignVerificationError, match="clean worktree"):
        verify_sealed(repository, 0)
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "post seal mutation")
    with pytest.raises(CampaignVerificationError, match="modified after sealing"):
        verify_sealed(repository, 0)
