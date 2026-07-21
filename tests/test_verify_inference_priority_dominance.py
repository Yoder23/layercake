import argparse

from scripts.verify_inference_priority_dominance import verify


def _args():
    return argparse.Namespace(
        param_tolerance=0.05,
        min_eval_bytes=1_000_000.0,
        max_bpb_ratio=1.0,
        min_cpu_speed_ratio=5.0,
    )


def _training(params, bpb, seconds=100.0):
    return {
        "parameters": params,
        "eval_bpb": bpb,
        "eval_bytes": 1_000_000,
        "elapsed_seconds": seconds,
        "estimated_total_training_bytes": 10_000_000,
    }


def _generation(layercake_bps, bpe_bps):
    return {
        "layercake": {"bytes_per_second": layercake_bps},
        "bpe": {"bytes_per_second": bpe_bps},
    }


def _quality(passes=True):
    return {
        "quality_gates": {
            "layercake_printable": passes,
            "layercake_distinct_trigram_at_least_bpe": passes,
            "layercake_max_repeat_8gram_no_worse_than_bpe": passes,
        }
    }


def test_inference_priority_gate_passes_cpu_deployment_dominance():
    result = verify(
        layercake_training=_training(15_000_000, 1.9),
        transformer_training=_training(15_100_000, 2.0),
        cpu_generation=_generation(500.0, 50.0),
        cpu_quality=_quality(True),
        gpu_generation=_generation(200.0, 100.0),
        args=_args(),
    )
    assert result["status"] == "PASS"
    assert result["gates"]["heldout_bpb_noninferior"] is True
    assert result["gates"]["cpu_generation_5x_faster"] is True


def test_inference_priority_gate_rejects_bad_quality_or_bpb():
    result = verify(
        layercake_training=_training(15_000_000, 2.1),
        transformer_training=_training(15_100_000, 2.0),
        cpu_generation=_generation(500.0, 50.0),
        cpu_quality=_quality(False),
        gpu_generation=None,
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["heldout_bpb_noninferior"] is False
    assert result["gates"]["cpu_generation_quality_passes"] is False
