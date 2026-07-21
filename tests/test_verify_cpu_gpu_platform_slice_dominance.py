from __future__ import annotations

import argparse
import copy

from scripts.verify_cpu_gpu_platform_slice_dominance import verify


def _args():
    return argparse.Namespace(
        min_param_ratio=5.0,
        min_training_speed_ratio=1.0,
        min_training_cost_ratio=5.0,
        min_training_byte_ratio=1.0,
        min_source_cpu_generation_speed_ratio=5.0,
        min_source_gpu_generation_speed_ratio=1.0,
        min_generation_cost_ratio=5.0,
        min_gpu_generation_cost_ratio=1.0,
        min_domain_cpu_generation_speed_ratio=5.0,
        min_domain_gpu_generation_speed_ratio=1.0,
        min_quality_ratio=1.0,
        min_relevance_ratio=1.0,
        min_portable_memory_match=1.0,
    )


def _source():
    return {
        "status": "PASS",
        "gates": {
            "bpb_non_inferior": True,
            "no_more_training_bytes": True,
        },
        "ratios": {
            "parameter_ratio_transformer_over_layercake": 9.0,
            "bpb_ratio_layercake_over_transformer": 0.5,
            "training_speed_ratio": 2.0,
            "cpu_generation_speed_ratio": 6.0,
            "gpu_generation_speed_ratio": 1.2,
            "cpu_quality_ratio": 1.1,
            "gpu_quality_ratio": 1.1,
        },
        "metrics": {
            "layercake": {
                "params": 1_000_000,
                "train_seconds": 10.0,
                "train_bytes": 1_000_000,
            },
            "transformer": {
                "params": 9_000_000,
                "train_seconds": 20.0,
                "train_bytes": 2_000_000,
            },
        },
    }


def _transfer():
    return {
        "status": "PASS",
        "gates": {
            "receiver_inherits_cpu_generation_win": True,
            "receiver_inherits_gpu_generation_win": True,
            "receiver_inherits_training_win": True,
            "receiver_inherits_quality_win": True,
        },
        "metrics": {
            "transfer_ppl_ratio": 1.0,
            "transfer_max_logit_diff": 0.0,
            "transfer_max_abi_diff": 0.0,
            "transfer_generation_exact": True,
        },
    }


def _instruction():
    return {
        "status": "PASS",
        "gates": {
            "cpu_layercake_exact_relevance_full": True,
            "cpu_layercake_paraphrase_relevance_full": True,
            "gpu_layercake_exact_relevance_full": True,
            "gpu_layercake_paraphrase_relevance_full": True,
        },
        "ratios": {
            "cpu_generation_speed_ratio": 50.0,
            "gpu_generation_speed_ratio": 5.0,
            "cpu_quality_ratio": 1.1,
            "gpu_quality_ratio": 1.1,
            "cpu_relevance_ratio": 1.5,
            "gpu_relevance_ratio": 1.5,
        },
    }


def _portable():
    return {
        "status": "PASS",
        "gates": {"child_certificate_gate": True},
        "ratios": {
            "cpu_generation_speed_ratio": 50.0,
            "gpu_generation_speed_ratio": 5.0,
            "cpu_quality_ratio": 1.1,
            "gpu_quality_ratio": 1.1,
            "cpu_relevance_ratio": 2.0,
            "gpu_relevance_ratio": 2.0,
        },
        "metrics": {
            "layercake_cpu_generation": {"portable_memory_match_rate": 1.0},
            "layercake_gpu_generation": {"portable_memory_match_rate": 1.0},
        },
    }


def _conflict():
    row = _portable()
    row["gates"].update(
        {
            "cpu_samples_no_forbidden": True,
            "gpu_samples_no_forbidden": True,
        }
    )
    return row


def _abstain():
    return {
        "status": "PASS",
        "gates": {"child_certificate_gate": True},
        "ratios": {
            "cpu_generation_speed_ratio": 50.0,
            "gpu_generation_speed_ratio": 5.0,
            "cpu_quality_ratio": 1.1,
            "gpu_quality_ratio": 1.1,
        },
        "metrics": {
            "layercake_cpu_generation": {
                "abstention_required_count": 2,
                "samples_abstentions_pass": True,
                "portable_memory_match_rate_effective": 1.0,
            },
            "layercake_gpu_generation": {
                "abstention_required_count": 2,
                "samples_abstentions_pass": True,
                "portable_memory_match_rate_effective": 1.0,
            },
        },
    }


def _verify(**overrides):
    rows = {
        "source_certificate": _source(),
        "transfer_certificate": _transfer(),
        "instruction_generalization_certificate": _instruction(),
        "portable_mixed_certificate": _portable(),
        "conflicting_isolation_certificate": _conflict(),
        "ood_abstention_certificate": _abstain(),
    }
    rows.update(overrides)
    return verify(args=_args(), **rows)


def test_platform_slice_passes_when_every_child_gate_and_ratio_passes():
    result = _verify()
    assert result["status"] == "PASS"


def test_platform_slice_fails_source_gpu_regression():
    source = _source()
    source["ratios"]["gpu_generation_speed_ratio"] = 0.99
    result = _verify(source_certificate=source)
    assert result["status"] == "FAIL"
    assert result["gates"]["source_gpu_generation_noninferior"] is False


def test_platform_slice_fails_hidden_false_child_gate():
    mixed = _portable()
    mixed["gates"]["child_certificate_gate"] = False
    result = _verify(portable_mixed_certificate=mixed)
    assert result["status"] == "FAIL"
    assert result["gates"]["portable_mixed_all_gates_pass"] is False


def test_platform_slice_fails_when_ood_prompts_are_missing():
    abstain = copy.deepcopy(_abstain())
    abstain["metrics"]["layercake_cpu_generation"]["abstention_required_count"] = 0
    abstain["metrics"]["layercake_gpu_generation"]["abstention_required_count"] = 0
    result = _verify(ood_abstention_certificate=abstain)
    assert result["status"] == "FAIL"
    assert result["gates"]["ood_required_prompts_present"] is False


def test_platform_slice_fails_training_cost_proxy_regression():
    source = _source()
    source["metrics"]["transformer"]["train_seconds"] = 1.0
    result = _verify(source_certificate=source)
    assert result["status"] == "FAIL"
    assert result["gates"]["source_train_cost_proxy_met"] is False


def test_platform_slice_fails_training_byte_efficiency_regression():
    source = _source()
    source["metrics"]["transformer"]["train_bytes"] = 500_000
    result = _verify(source_certificate=source)
    assert result["status"] == "FAIL"
    assert result["gates"]["source_train_byte_efficiency_met"] is False
