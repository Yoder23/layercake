from __future__ import annotations

import argparse

from scripts.verify_instruction_generalization_dominance import verify


def _args():
    return argparse.Namespace(
        min_cpu_generation_speed_ratio=5.0,
        min_gpu_generation_speed_ratio=1.0,
        min_quality_ratio=1.0,
        min_relevance_ratio=1.0,
        min_layercake_category_relevance=1.0,
        min_layercake_paraphrase_alias_rate=1.0,
    )


def _sample(
    *,
    category: str = "paraphrase",
    relevant: bool = True,
    runtime_path: str = "semantic_instruction_alias",
    text: str = " Create space first, identify the faster threat, then rotate safely.",
):
    return {
        "prompt": "Question: Two enemies show up at once. What is the safest opening move? Answer:",
        "category": category,
        "text": text,
        "runtime_path": runtime_path,
        "printable_ratio": 1.0,
        "alpha_space_ratio": 0.95,
        "max_repeat_8gram": 1.0,
        "distinct_word_ratio": 0.9,
        "one_char_word_ratio": 0.0,
        "unique_word_count": 10.0,
        "unique_alpha_char_count": 17.0,
        "relevance_pass": relevant,
    }


def _generation(
    *,
    device: str,
    bps: float,
    quality: float = 0.95,
    relevance_rate: float = 1.0,
    exact_relevance: float = 1.0,
    paraphrase_relevance: float = 1.0,
    paraphrase_alias: float = 1.0,
    relevant: bool = True,
):
    return {
        "device": device,
        "metrics": {
            "generation_bytes_per_second": bps,
            "quality_score": quality,
            "relevance_rate": relevance_rate,
            "alias_match_rate": paraphrase_alias,
            "category_metrics": {
                "exact": {"count": 1, "relevance_rate": exact_relevance, "alias_match_rate": 1.0},
                "paraphrase": {
                    "count": 1,
                    "relevance_rate": paraphrase_relevance,
                    "alias_match_rate": paraphrase_alias,
                },
            },
        },
        "samples": [
            _sample(category="exact", relevant=relevant),
            _sample(category="paraphrase", relevant=relevant),
        ],
    }


def test_instruction_generalization_gate_passes_with_5x_cpu_gpu_and_paraphrases():
    result = verify(
        layercake_cpu_generation=_generation(device="cpu", bps=5000.0),
        transformer_cpu_generation=_generation(
            device="cpu",
            bps=1000.0,
            paraphrase_alias=0.0,
        ),
        layercake_gpu_generation=_generation(device="cuda", bps=1500.0),
        transformer_gpu_generation=_generation(
            device="cuda",
            bps=1400.0,
            paraphrase_alias=0.0,
        ),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "PASS"},
        args=_args(),
    )
    assert result["status"] == "PASS"


def test_instruction_generalization_gate_fails_when_paraphrase_relevance_misses():
    result = verify(
        layercake_cpu_generation=_generation(
            device="cpu",
            bps=5000.0,
            paraphrase_relevance=0.0,
            relevant=False,
        ),
        transformer_cpu_generation=_generation(device="cpu", bps=1000.0, paraphrase_alias=0.0),
        layercake_gpu_generation=_generation(
            device="cuda",
            bps=1500.0,
            paraphrase_relevance=0.0,
            relevant=False,
        ),
        transformer_gpu_generation=_generation(device="cuda", bps=1400.0, paraphrase_alias=0.0),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "PASS"},
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_layercake_paraphrase_relevance_full"] is False
    assert result["gates"]["cpu_samples_relevant"] is False


def test_instruction_generalization_gate_fails_when_cpu_speed_is_below_5x():
    result = verify(
        layercake_cpu_generation=_generation(device="cpu", bps=4000.0),
        transformer_cpu_generation=_generation(device="cpu", bps=1000.0, paraphrase_alias=0.0),
        layercake_gpu_generation=_generation(device="cuda", bps=1500.0),
        transformer_gpu_generation=_generation(device="cuda", bps=1400.0, paraphrase_alias=0.0),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "PASS"},
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_generation_5x_met"] is False


def test_instruction_generalization_gate_requires_transfer_certificate_pass():
    result = verify(
        layercake_cpu_generation=_generation(device="cpu", bps=5000.0),
        transformer_cpu_generation=_generation(device="cpu", bps=1000.0, paraphrase_alias=0.0),
        layercake_gpu_generation=_generation(device="cuda", bps=1500.0),
        transformer_gpu_generation=_generation(device="cuda", bps=1400.0, paraphrase_alias=0.0),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "FAIL"},
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["transfer_certificate_pass"] is False
