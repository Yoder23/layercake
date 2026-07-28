import json
from pathlib import Path

import pytest

from layercake.evaluation.phase3_retirement_evidence import (
    Phase3RetirementEvidenceError,
    validate_phase3_retirement_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_phase3_retirement_cannot_be_described_as_training_efficiency_pass():
    contract = _read("moonshot/phase3_retirement_lock.json")
    assert contract["status"] == "RETIRED_BY_GOVERNANCE_PENDING_SEAL"
    assert contract["authorization"][
        "phase3_scientific_training_efficiency_pass_claimed"
    ] is False
    assert contract["authorization"][
        "additional_major_random_init_english_optimization_authorized"
    ] is False
    assert contract["handoff_to_phase4"]["training_speed_is_phase4_gate"] is False
    assert contract["handoff_to_phase4"][
        "all_non_training_speed_moonshot_gates_remain"
    ] is True


def test_phase3_claim_contract_contains_only_retirement_gates():
    claim = _read("moonshot/claim_contract.yaml")
    phase3 = claim["phase_requirements"]["3"]
    assert phase3["name"] == "retired_training_efficiency_controls"
    assert {gate["id"] for gate in phase3["required_gates"]} == {
        "retirement_authorized",
        "historical_evidence_preserved",
        "no_scientific_training_pass_claimed",
        "phase2_parent_sealed",
        "future_abi_recertification_required",
    }
    assert "foundation_time_to_quality_faster" not in claim["moonshot_claims"]
    assert "domain_time_to_functional_quality_faster" not in claim["moonshot_claims"]


def test_phase4_keeps_product_gates_and_reports_training_cost_without_speed_gate():
    contract = _read("moonshot/phase4_cpu_cake_training_lock.json")
    assert contract["status"] == "OPEN"
    assert contract["current_acquisition_path"]["external_abi_required"] is False
    assert contract["reported_not_promoted"]["training_speedup"] is True
    assert contract["promotion_gates"]["functional_error_reduction_over_frozen_core_min"] == 5.0
    assert contract["promotion_gates"][
        "core_plus_cake_optimized_cpu_transformer_ratio_min"
    ] == 2.0
    claim = _read("moonshot/claim_contract.yaml")
    phase4_ids = {
        gate["id"] for gate in claim["phase_requirements"]["4"]["required_gates"]
    }
    assert "cake_adaptation_time_ratio" not in phase4_ids
    assert "cake_peak_training_memory_ratio" not in phase4_ids


def test_retirement_verifier_fails_closed_when_bundle_is_missing(tmp_path):
    with pytest.raises(Phase3RetirementEvidenceError, match="cannot read"):
        validate_phase3_retirement_bundle(ROOT, tmp_path)
