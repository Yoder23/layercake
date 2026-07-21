import argparse

from scripts.verify_breakthrough_equal_size_dominance import verify


def _args():
    return argparse.Namespace(
        param_tolerance=0.05,
        min_eval_bytes=1_000_000.0,
        min_quality_bpb_improvement_ratio=5.0,
        min_training_speed_ratio=5.0,
        min_training_cost_ratio=5.0,
        max_train_byte_ratio=1.0,
        min_inference_speed_ratio=5.0,
        min_generation_quality_ratio=5.0,
        min_task_score_ratio=5.0,
        min_relevance_ratio=1.0,
        min_layercake_relevance=1.0,
    )


def _training(params, bpb, seconds, train_bytes=10_000_000, eval_bytes=1_000_000):
    return {
        "latest": {
            "trainable_params": params,
            "eval_bpb": bpb,
            "elapsed_total_seconds": seconds,
            "train_bytes": train_bytes,
            "eval_bytes": eval_bytes,
        }
    }


def _generation(bps, quality=0.1, task=0.1, relevance=1.0):
    return {
        "metrics": {
            "generation_bytes_per_second": bps,
            "quality_score": quality,
            "task_score": task,
            "relevance_rate": relevance,
        },
        "samples": [{"text": "sample"}],
    }


def test_breakthrough_gate_passes_only_with_5x_equal_size_evidence():
    result = verify(
        layercake_training=_training(15_000_000, 0.4, 20.0, train_bytes=9_000_000),
        transformer_training=_training(15_100_000, 2.1, 120.0),
        layercake_cpu_generation=_generation(5000.0, quality=0.75, task=0.6),
        transformer_cpu_generation=_generation(900.0, quality=0.1, task=0.1),
        layercake_gpu_generation=_generation(6000.0, quality=0.8, task=0.7),
        transformer_gpu_generation=_generation(1000.0, quality=0.1, task=0.1),
        args=_args(),
    )
    assert result["status"] == "PASS"
    assert result["gates"]["heldout_bpb_5x_better"] is True
    assert result["gates"]["training_5x_faster"] is True
    assert result["gates"]["cpu_inference_5x_faster"] is True


def test_breakthrough_gate_rejects_current_incremental_dominance_shape():
    result = verify(
        layercake_training=_training(14_220_866, 6.589, 105.1, train_bytes=8_192_000),
        transformer_training=_training(10_696_704, 2.677, 89.7, train_bytes=20_747_305),
        layercake_cpu_generation=_generation(896.7, quality=0.926, task=0.0, relevance=0.0),
        transformer_cpu_generation=_generation(142.7, quality=0.873, task=0.0, relevance=0.0),
        layercake_gpu_generation=_generation(596.1, quality=0.926, task=0.0, relevance=0.0),
        transformer_gpu_generation=_generation(577.6, quality=0.873, task=0.0, relevance=0.0),
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["equal_size_parameter_window"] is False
    assert result["gates"]["heldout_bpb_5x_better"] is False
    assert result["gates"]["training_5x_faster"] is False
    assert result["gates"]["gpu_inference_5x_faster"] is False


def test_breakthrough_gate_rejects_missing_generation_evidence():
    result = verify(
        layercake_training=_training(15_000_000, 0.4, 20.0, train_bytes=9_000_000),
        transformer_training=_training(15_100_000, 2.1, 120.0),
        layercake_cpu_generation=None,
        transformer_cpu_generation=None,
        layercake_gpu_generation=None,
        transformer_gpu_generation=None,
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["cpu_generation_evidence_present"] is False
    assert result["gates"]["gpu_generation_evidence_present"] is False


def test_positive_quality_over_zero_baseline_counts_as_breakthrough_ratio():
    result = verify(
        layercake_training=_training(15_000_000, 0.4, 20.0, train_bytes=9_000_000),
        transformer_training=_training(15_100_000, 2.1, 120.0),
        layercake_cpu_generation=_generation(5000.0, quality=1.0, task=1.0),
        transformer_cpu_generation=_generation(900.0, quality=0.0, task=1.0),
        layercake_gpu_generation=_generation(6000.0, quality=1.0, task=1.0),
        transformer_gpu_generation=_generation(1000.0, quality=0.0, task=1.0),
        args=_args(),
    )
    assert result["gates"]["cpu_generation_or_task_quality_5x"] is True
    assert result["gates"]["gpu_generation_or_task_quality_5x"] is True
    assert result["ratios"]["cpu_generation_quality_ratio_layercake_over_transformer"] >= 5.0


def test_breakthrough_certificate_reports_required_pass_targets():
    result = verify(
        layercake_training=_training(10_000_000, 1.0, 50.0, train_bytes=9_000_000),
        transformer_training=_training(20_000_000, 2.0, 100.0, train_bytes=10_000_000),
        layercake_cpu_generation=_generation(5000.0, quality=1.0, task=1.0),
        transformer_cpu_generation=_generation(900.0, quality=0.0, task=1.0),
        layercake_gpu_generation=_generation(6000.0, quality=1.0, task=1.0),
        transformer_gpu_generation=_generation(1000.0, quality=0.0, task=1.0),
        args=_args(),
    )
    assert result["required_for_pass"]["max_layercake_bpb_for_5x"] == 0.4
    assert result["required_for_pass"]["max_layercake_train_seconds_for_5x_speed"] == 20.0
    assert result["required_for_pass"]["max_layercake_train_seconds_for_5x_cost_proxy"] == 40.0
    assert result["shortfall"]["layercake_bpb_over_5x_target"] == 2.5
