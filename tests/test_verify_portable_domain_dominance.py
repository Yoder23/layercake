from __future__ import annotations

import argparse

from scripts.verify_portable_domain_dominance import verify


def _args():
    return argparse.Namespace(
        required_categories="app,website,game",
        min_domain_setup_speed_ratio=1.0,
        min_cpu_generation_speed_ratio=5.0,
        min_gpu_generation_speed_ratio=1.0,
        min_quality_ratio=1.0,
        min_relevance_ratio=1.0,
        min_layercake_memory_match=1.0,
        min_layercake_abstention_rate=0.0,
        min_layercake_category_relevance=1.0,
        min_layercake_category_memory_match=1.0,
    )


def _sample(category: str, *, relevant: bool = True):
    return {
        "prompt": f"Question: {category} prompt? Answer:",
        "category": category,
        "text": " The local cache keeps writing available offline and sync resumes later.",
        "runtime_path": "portable_corpus_memory",
        "printable_ratio": 1.0,
        "alpha_space_ratio": 0.95,
        "max_repeat_8gram": 1.0,
        "distinct_word_ratio": 0.9,
        "one_char_word_ratio": 0.0,
        "unique_word_count": 10.0,
        "unique_alpha_char_count": 18.0,
        "forbidden_keyword_hits": 0,
        "expect_abstain": False,
        "abstention_pass": True,
        "relevance_pass": relevant,
    }


def _generation(
    *,
    device: str,
    bps: float,
    quality: float = 0.95,
    relevance_rate: float = 1.0,
    memory_rate: float = 1.0,
    relevant: bool = True,
    forbidden_hits: int = 0,
    abstention_rate: float = 0.0,
    include_abstention: bool = False,
    abstention_pass: bool = True,
):
    samples = [_sample(category, relevant=relevant) for category in ("app", "website", "game")]
    if samples:
        samples[0]["forbidden_keyword_hits"] = forbidden_hits
    if include_abstention:
        samples.append(
            {
                "prompt": "Question: Unknown fact? Answer:",
                "category": "unknown",
                "text": " I do not have that information in the attached domain layer.",
                "runtime_path": "portable_corpus_abstain" if abstention_pass else "neural_layercake",
                "printable_ratio": 1.0,
                "alpha_space_ratio": 0.95,
                "max_repeat_8gram": 1.0,
                "distinct_word_ratio": 0.9,
                "one_char_word_ratio": 0.0,
                "unique_word_count": 10.0,
                "unique_alpha_char_count": 18.0,
                "forbidden_keyword_hits": 0,
                "expect_abstain": True,
                "abstention_pass": abstention_pass,
                "relevance_pass": abstention_pass,
            }
        )
    return {
        "device": device,
        "metrics": {
            "generation_bytes_per_second": bps,
            "quality_score": quality,
            "relevance_rate": relevance_rate,
            "portable_memory_match_rate": memory_rate,
            "abstention_rate": abstention_rate,
            "domain_setup_seconds": 0.01,
            "category_metrics": {
                category: {
                    "count": 1,
                    "relevance_rate": relevance_rate,
                    "portable_memory_match_rate": memory_rate,
                }
                for category in ("app", "website", "game")
            },
        },
        "samples": samples,
    }


def _training(seconds: float = 10.0):
    return {"latest": {"elapsed_seconds": seconds, "trainable_params": 5_000_000, "train_bytes": 1_000_000}}


def test_portable_domain_gate_passes_with_cpu_5x_relevance_and_transfer():
    result = verify(
        layercake_cpu_generation=_generation(device="cpu", bps=5000.0),
        transformer_cpu_generation=_generation(device="cpu", bps=1000.0, memory_rate=0.0),
        layercake_gpu_generation=_generation(device="cuda", bps=1500.0),
        transformer_gpu_generation=_generation(device="cuda", bps=1400.0, memory_rate=0.0),
        transformer_training=_training(),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "PASS"},
        args=_args(),
    )
    assert result["status"] == "PASS"


def test_portable_domain_gate_fails_when_cpu_generation_is_not_5x():
    result = verify(
        layercake_cpu_generation=_generation(device="cpu", bps=4000.0),
        transformer_cpu_generation=_generation(device="cpu", bps=1000.0, memory_rate=0.0),
        layercake_gpu_generation=_generation(device="cuda", bps=1500.0),
        transformer_gpu_generation=_generation(device="cuda", bps=1400.0, memory_rate=0.0),
        transformer_training=_training(),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "PASS"},
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_generation_5x_met"] is False


def test_portable_domain_gate_fails_when_category_relevance_misses():
    result = verify(
        layercake_cpu_generation=_generation(
            device="cpu",
            bps=5000.0,
            relevance_rate=0.0,
            relevant=False,
        ),
        transformer_cpu_generation=_generation(device="cpu", bps=1000.0, memory_rate=0.0),
        layercake_gpu_generation=_generation(
            device="cuda",
            bps=1500.0,
            relevance_rate=0.0,
            relevant=False,
        ),
        transformer_gpu_generation=_generation(device="cuda", bps=1400.0, memory_rate=0.0),
        transformer_training=_training(),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "PASS"},
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_app_relevance_full"] is False
    assert result["gates"]["cpu_samples_relevant"] is False


def test_portable_domain_gate_fails_on_forbidden_cross_domain_fact():
    result = verify(
        layercake_cpu_generation=_generation(
            device="cpu",
            bps=5000.0,
            forbidden_hits=1,
        ),
        transformer_cpu_generation=_generation(device="cpu", bps=1000.0, memory_rate=0.0),
        layercake_gpu_generation=_generation(
            device="cuda",
            bps=1500.0,
            forbidden_hits=1,
        ),
        transformer_gpu_generation=_generation(device="cuda", bps=1400.0, memory_rate=0.0),
        transformer_training=_training(),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "PASS"},
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_samples_no_forbidden"] is False


def test_portable_domain_gate_passes_required_abstention():
    args = _args()
    args.min_layercake_abstention_rate = 0.25
    layercake_cpu = _generation(
        device="cpu",
        bps=5000.0,
        abstention_rate=0.25,
        include_abstention=True,
        abstention_pass=True,
    )
    layercake_cpu["metrics"]["portable_memory_match_rate"] = 0.75
    layercake_gpu = _generation(
        device="cuda",
        bps=1500.0,
        abstention_rate=0.25,
        include_abstention=True,
        abstention_pass=True,
    )
    layercake_gpu["metrics"]["portable_memory_match_rate"] = 0.75
    result = verify(
        layercake_cpu_generation=layercake_cpu,
        transformer_cpu_generation=_generation(device="cpu", bps=1000.0, memory_rate=0.0),
        layercake_gpu_generation=layercake_gpu,
        transformer_gpu_generation=_generation(device="cuda", bps=1400.0, memory_rate=0.0),
        transformer_training=_training(),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "PASS"},
        args=args,
    )
    assert result["status"] == "PASS"
    assert result["metrics"]["layercake_cpu_generation"][
        "portable_memory_match_rate_effective"
    ] == 1.0


def test_portable_domain_gate_fails_missing_required_abstention():
    args = _args()
    args.min_layercake_abstention_rate = 0.25
    result = verify(
        layercake_cpu_generation=_generation(
            device="cpu",
            bps=5000.0,
            abstention_rate=0.0,
            include_abstention=True,
            abstention_pass=False,
        ),
        transformer_cpu_generation=_generation(device="cpu", bps=1000.0, memory_rate=0.0),
        layercake_gpu_generation=_generation(
            device="cuda",
            bps=1500.0,
            abstention_rate=0.0,
            include_abstention=True,
            abstention_pass=False,
        ),
        transformer_gpu_generation=_generation(device="cuda", bps=1400.0, memory_rate=0.0),
        transformer_training=_training(),
        source_certificate={"status": "PASS"},
        transfer_certificate={"status": "PASS"},
        args=args,
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_abstention_rate_met"] is False
    assert result["gates"]["cpu_samples_abstentions_pass"] is False
