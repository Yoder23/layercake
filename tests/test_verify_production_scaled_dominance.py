from __future__ import annotations

import argparse

from scripts.verify_production_scaled_dominance import _scaled_cost_gates


def _args():
    return argparse.Namespace(
        min_training_cost_ratio=5.0,
        min_training_byte_ratio=1.0,
        min_generation_cost_ratio=5.0,
        min_gpu_generation_cost_ratio=1.0,
    )


def _result():
    return {
        "status": "PASS",
        "gates": {"base_gate": True},
        "metrics": {
            "layercake": {
                "params": 2_000_000,
                "train_seconds": 20.0,
                "train_bytes": 10_000_000,
            },
            "transformer": {
                "params": 10_000_000,
                "train_seconds": 100.0,
                "train_bytes": 50_000_000,
            },
        },
        "ratios": {
            "parameter_ratio_transformer_over_layercake": 5.0,
            "cpu_generation_speed_ratio": 5.0,
            "gpu_generation_speed_ratio": 1.2,
        },
    }


def test_scaled_dominance_cost_gates_pass_when_cost_and_bytes_win():
    result = _scaled_cost_gates(_result(), _args())
    assert result["status"] == "PASS"
    assert result["gates"]["training_cost_proxy_met"] is True
    assert result["gates"]["training_byte_efficiency_met"] is True
    assert result["ratios"]["training_cost_proxy_ratio"] == 25.0
    assert result["ratios"]["cpu_generation_cost_proxy_ratio"] == 25.0


def test_scaled_dominance_cost_gates_fail_training_cost_regression():
    row = _result()
    row["metrics"]["transformer"]["train_seconds"] = 5.0
    result = _scaled_cost_gates(row, _args())
    assert result["status"] == "FAIL"
    assert result["gates"]["training_cost_proxy_met"] is False


def test_scaled_dominance_cost_gates_fail_gpu_cost_regression():
    row = _result()
    row["ratios"]["gpu_generation_speed_ratio"] = 0.1
    result = _scaled_cost_gates(row, _args())
    assert result["status"] == "FAIL"
    assert result["gates"]["gpu_generation_cost_proxy_met"] is False
