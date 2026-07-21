from __future__ import annotations

import argparse

from scripts.verify_production_1m_vs_5m_dominance import verify


def _args():
    return argparse.Namespace(
        min_param_ratio=5.0,
        min_training_speed_ratio=1.0,
        min_cpu_generation_speed_ratio=5.0,
        min_gpu_generation_speed_ratio=1.0,
        min_quality_ratio=1.0,
    )


def _training(params: int, bpb: float, seconds: float, train_bytes: int = 1_000_000):
    return {"latest": {"trainable_params": params, "bpb": bpb, "elapsed_seconds": seconds, "train_bytes": train_bytes}}


def _training_with_eval(params: int, train_bpb: float, eval_bpb: float, seconds: float, train_bytes: int = 1_000_000):
    return {
        "latest": {
            "trainable_params": params,
            "bpb": train_bpb,
            "eval_bpb": eval_bpb,
            "elapsed_seconds": seconds,
            "train_bytes": train_bytes,
        }
    }


def _generation(
    device: str,
    bps: float,
    quality: float,
    text: str = "take cover wait behind stone then move safely when the archer reloads",
):
    return {
        "device": device,
        "metrics": {"generation_bytes_per_second": bps, "quality_score": quality},
        "samples": [
            {
                "text": text,
                "printable_ratio": 1.0,
                "alpha_space_ratio": 1.0,
                "max_repeat_8gram": 1.0,
                "distinct_word_ratio": 1.0,
                "one_char_word_ratio": 0.0,
                "unique_word_count": 11.0,
                "unique_alpha_char_count": 18.0,
            }
        ],
    }


def test_1m_vs_5m_gate_passes_with_cpu_5x_gpu_quality_and_samples():
    result = verify(
        layercake_training=_training(1_000_000, 1.9, 80.0),
        transformer_training=_training(5_200_000, 2.0, 100.0),
        layercake_cpu_generation=_generation("cpu", 5_000.0, 0.9),
        transformer_cpu_generation=_generation("cpu", 1_000.0, 0.9),
        layercake_gpu_generation=_generation("cuda", 2_000.0, 0.9),
        transformer_gpu_generation=_generation("cuda", 1_500.0, 0.9),
        args=_args(),
    )
    assert result["status"] == "PASS"


def test_1m_vs_5m_gate_prefers_heldout_eval_bpb_over_train_bpb():
    result = verify(
        layercake_training=_training_with_eval(1_000_000, 1.0, 2.1, 80.0),
        transformer_training=_training_with_eval(5_200_000, 2.0, 2.0, 100.0),
        layercake_cpu_generation=_generation("cpu", 5_000.0, 0.9),
        transformer_cpu_generation=_generation("cpu", 1_000.0, 0.9),
        layercake_gpu_generation=_generation("cuda", 2_000.0, 0.9),
        transformer_gpu_generation=_generation("cuda", 1_500.0, 0.9),
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["bpb_non_inferior"] is False


def test_1m_vs_5m_gate_fails_degenerate_samples_even_when_quality_score_is_high():
    degenerate = _generation("cpu", 5_000.0, 0.99, text="        ")
    degenerate["samples"][0].update(
        {
            "distinct_word_ratio": 0.0,
            "one_char_word_ratio": 0.0,
            "unique_word_count": 0.0,
            "unique_alpha_char_count": 0.0,
        }
    )
    result = verify(
        layercake_training=_training(1_000_000, 1.9, 80.0),
        transformer_training=_training(5_200_000, 2.0, 100.0),
        layercake_cpu_generation=degenerate,
        transformer_cpu_generation=_generation("cpu", 1_000.0, 0.9),
        layercake_gpu_generation={**degenerate, "device": "cuda"},
        transformer_gpu_generation=_generation("cuda", 1_500.0, 0.9),
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_samples_nonempty"] is False


def test_1m_vs_5m_gate_fails_static_letter_loop_samples():
    static_loop = _generation(
        "cpu",
        5_000.0,
        0.99,
        text="SESESESESSSESESES EEEESEEESEEEEESEE E EESEEES SESESEEESESESEE",
    )
    static_loop["samples"][0].update(
        {
            "max_repeat_8gram": 2.0,
            "distinct_word_ratio": 0.33,
            "one_char_word_ratio": 0.16,
            "unique_word_count": 3.0,
            "unique_alpha_char_count": 1.0,
        }
    )
    result = verify(
        layercake_training=_training(1_000_000, 1.9, 80.0),
        transformer_training=_training(5_200_000, 2.0, 100.0),
        layercake_cpu_generation=static_loop,
        transformer_cpu_generation=_generation("cpu", 1_000.0, 0.9),
        layercake_gpu_generation={**static_loop, "device": "cuda"},
        transformer_gpu_generation=_generation("cuda", 1_500.0, 0.9),
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_samples_lexically_diverse"] is False


def test_1m_vs_5m_gate_fails_when_transformer_is_not_5x_larger():
    result = verify(
        layercake_training=_training(1_000_000, 1.9, 80.0),
        transformer_training=_training(4_900_000, 2.0, 100.0),
        layercake_cpu_generation=_generation("cpu", 5_000.0, 0.9),
        transformer_cpu_generation=_generation("cpu", 1_000.0, 0.9),
        layercake_gpu_generation=_generation("cuda", 2_000.0, 0.9),
        transformer_gpu_generation=_generation("cuda", 1_500.0, 0.9),
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["transformer_at_least_5x_params"] is False
