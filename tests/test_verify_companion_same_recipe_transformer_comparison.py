import argparse

from scripts.verify_companion_same_recipe_transformer_comparison import verify


def _lc_training(step=11000):
    return {
        "status": "COMPLETE",
        "latest": {
            "elapsed_seconds": 10.0,
            "counted_pretrain_seconds": 1.0,
            "step": step,
            "train_bytes": 800.0,
            "trainable_params": 100,
            "eval_bpb": 2.0,
        },
    }


def _tx_training(step=11000):
    return {
        "status": "COMPLETE",
        "latest": {
            "elapsed_total_seconds": 30.0,
            "elapsed_seconds": 25.0,
            "tokenizer_seconds": 5.0,
            "step": step,
            "train_bytes": 900.0,
            "trainable_params": 120,
            "eval_bpb": 2.5,
        },
    }


def _generation(device, bps, quality, relevance):
    return {
        "device": device,
        "metrics": {
            "generation_bytes_per_second": bps,
            "quality_score": quality,
            "relevance_rate": relevance,
        },
        "samples": [
            {
                "text": "Retreat, guard, create space, and recover tempo.",
                "relevance_pass": relevance >= 1.0,
                "printable_ratio": 1.0,
                "max_repeat_8gram": 1.0,
            }
        ],
    }


def test_same_recipe_companion_comparison_passes():
    args = argparse.Namespace(
        min_param_ratio=1.0,
        min_layercake_step=11000,
        min_transformer_step=11000,
        min_training_speed_ratio=1.0,
        max_eval_bpb_ratio=1.0,
        max_training_byte_exposure_ratio=1.0,
        min_cpu_speed_ratio=1.0,
        min_gpu_speed_ratio=1.0,
        min_quality_ratio=1.0,
        min_layercake_relevance=1.0,
    )
    result = verify(
        layercake_training_rows=[_lc_training()],
        transformer_training=_tx_training(),
        layercake_cpu_generation=_generation("cpu", 1000.0, 0.95, 1.0),
        transformer_cpu_generation=_generation("cpu", 100.0, 0.8, 0.0),
        layercake_gpu_generation=_generation("cuda", 1000.0, 0.95, 1.0),
        transformer_gpu_generation=_generation("cuda", 500.0, 0.8, 0.0),
        args=args,
    )
    assert result["status"] == "PASS"
    assert result["ratios"]["training_wall_clock_speed_ratio"] > 1.0


def test_same_recipe_companion_comparison_fails_worse_neural_bpb():
    args = argparse.Namespace(
        min_param_ratio=1.0,
        min_layercake_step=11000,
        min_transformer_step=11000,
        min_training_speed_ratio=1.0,
        max_eval_bpb_ratio=1.0,
        max_training_byte_exposure_ratio=1.0,
        min_cpu_speed_ratio=1.0,
        min_gpu_speed_ratio=1.0,
        min_quality_ratio=1.0,
        min_layercake_relevance=1.0,
    )
    result = verify(
        layercake_training_rows=[_lc_training()],
        transformer_training={**_tx_training(), "latest": {**_tx_training()["latest"], "eval_bpb": 1.5}},
        layercake_cpu_generation=_generation("cpu", 1000.0, 0.95, 1.0),
        transformer_cpu_generation=_generation("cpu", 100.0, 0.8, 0.0),
        layercake_gpu_generation=_generation("cuda", 1000.0, 0.95, 1.0),
        transformer_gpu_generation=_generation("cuda", 500.0, 0.8, 0.0),
        args=args,
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["layercake_eval_bpb_noninferior"] is False


def test_same_recipe_companion_comparison_fails_more_training_bytes():
    args = argparse.Namespace(
        min_param_ratio=1.0,
        min_layercake_step=11000,
        min_transformer_step=11000,
        min_training_speed_ratio=1.0,
        max_eval_bpb_ratio=1.0,
        max_training_byte_exposure_ratio=1.0,
        min_cpu_speed_ratio=1.0,
        min_gpu_speed_ratio=1.0,
        min_quality_ratio=1.0,
        min_layercake_relevance=1.0,
    )
    lc = _lc_training()
    lc["latest"]["train_bytes"] = 1000.0
    result = verify(
        layercake_training_rows=[lc],
        transformer_training=_tx_training(),
        layercake_cpu_generation=_generation("cpu", 1000.0, 0.95, 1.0),
        transformer_cpu_generation=_generation("cpu", 100.0, 0.8, 0.0),
        layercake_gpu_generation=_generation("cuda", 1000.0, 0.95, 1.0),
        transformer_gpu_generation=_generation("cuda", 500.0, 0.8, 0.0),
        args=args,
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["layercake_no_more_training_bytes"] is False
