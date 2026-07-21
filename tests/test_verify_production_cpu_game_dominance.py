from __future__ import annotations

import argparse

from scripts.verify_production_cpu_game_dominance import verify


def _args(**kwargs):
    defaults = {
        "max_same_size_param_ratio": 1.10,
        "min_training_speed_ratio": 1.0,
        "min_generation_speed_ratio": 5.0,
        "min_quality_ratio": 1.0,
        "max_first_token_p95_ms": None,
        "max_response_p95_ms": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _training(params: int, bpb: float, seconds: float, train_bytes: int = 1_000_000):
    return {
        "latest": {
            "trainable_params": params,
            "bpb": bpb,
            "elapsed_seconds": seconds,
            "train_bytes": train_bytes,
        }
    }


def _generation(bps: float, quality: float, device: str = "cpu"):
    return {
        "device": device,
        "metrics": {
            "generation_bytes_per_second": bps,
            "quality_score": quality,
        },
    }


def test_production_cpu_game_gate_passes_for_same_size_5x_cpu_no_quality_loss():
    result = verify(
        layercake_training=_training(1_000_000, 1.90, 80.0),
        transformer_training=_training(1_050_000, 2.00, 100.0),
        layercake_generation=_generation(5_000.0, 0.82),
        transformer_generation=_generation(1_000.0, 0.82),
        args=_args(),
    )

    assert result["status"] == "PASS"
    assert result["gates"]["generation_speed_5x_met"] is True


def test_production_cpu_game_gate_fails_below_5x_generation():
    result = verify(
        layercake_training=_training(1_000_000, 1.90, 80.0),
        transformer_training=_training(1_050_000, 2.00, 100.0),
        layercake_generation=_generation(4_900.0, 0.82),
        transformer_generation=_generation(1_000.0, 0.82),
        args=_args(),
    )

    assert result["status"] == "FAIL"
    assert result["gates"]["generation_speed_5x_met"] is False


def test_production_cpu_game_gate_fails_when_comparator_is_not_same_size():
    result = verify(
        layercake_training=_training(1_000_000, 1.90, 80.0),
        transformer_training=_training(50_000_000, 2.00, 100.0),
        layercake_generation=_generation(5_000.0, 0.82),
        transformer_generation=_generation(1_000.0, 0.82),
        args=_args(),
    )

    assert result["status"] == "FAIL"
    assert result["gates"]["same_size_comparator"] is False
