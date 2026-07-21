import argparse

from scripts.verify_companion_runtime_dominance import verify


def _generation(device, bps, quality, relevance):
    return {
        "device": device,
        "metrics": {
            "generation_bytes_per_second": bps,
            "quality_score": quality,
            "relevance_rate": relevance,
            "category_metrics": {
                "game_tactics": {"relevance_rate": relevance},
                "game_recovery": {"relevance_rate": relevance},
                "companion_style": {"relevance_rate": relevance},
            },
        },
        "samples": [
            {
                "text": "Breathe, guard, create space, and recover tempo.",
                "printable_ratio": 1.0,
                "alpha_space_ratio": 0.95,
                "max_repeat_8gram": 1.0,
                "relevance_pass": relevance >= 1.0,
                "trimmed": False,
            }
        ],
    }


def test_verify_companion_runtime_dominance_passes():
    args = argparse.Namespace(
        min_training_step=10000,
        min_cpu_generation_speed_ratio=5.0,
        min_gpu_generation_speed_ratio=1.0,
        min_quality_ratio=1.0,
        min_relevance_ratio=1.0,
        min_layercake_relevance=1.0,
        required_categories="game_tactics,game_recovery,companion_style",
    )
    result = verify(
        layercake_cpu=_generation("cpu", 1000.0, 0.95, 1.0),
        transformer_cpu=_generation("cpu", 100.0, 0.90, 0.5),
        layercake_gpu=_generation("cuda", 1000.0, 0.95, 1.0),
        transformer_gpu=_generation("cuda", 900.0, 0.90, 0.5),
        training_metrics={
            "status": "COMPLETE",
            "latest": {"step": 10000, "eval_bpb": 3.5},
        },
        args=args,
    )
    assert result["status"] == "PASS"
    assert result["ratios"]["cpu_generation_speed_ratio"] == 10.0
