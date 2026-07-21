import argparse

from scripts.verify_true_dominance_claim import verify


def _args():
    return argparse.Namespace(
        min_param_ratio=1.0,
        min_eval_bytes=100_000.0,
        max_eval_bpb_ratio=1.0,
        min_training_speed_ratio=1.0,
        min_training_cost_ratio=1.0,
        max_train_byte_ratio=1.0,
        min_cpu_generation_speed_ratio=1.0,
        min_gpu_generation_speed_ratio=1.0,
        min_quality_ratio=1.0,
        min_relevance_ratio=1.0,
        min_layercake_relevance=1.0,
    )


def _lc_config():
    return {
        "model": {"domain_cache_override": False, "domain_cache_order": 0},
        "training": {
            "data_roots": ["data/train"],
            "eval_data_roots": ["data/eval"],
            "initialize_domain_cache_from_corpus": False,
        },
    }


def _tx_config():
    return {
        "model": {},
        "training": {
            "data_roots": ["data/train_tx"],
            "eval_data_roots": ["data/eval_tx"],
            "eval_bytes": 100_000,
        },
    }


def _training(params, bpb, seconds, train_bytes=1_000_000, eval_bytes=100_000):
    return {
        "status": "COMPLETE",
        "latest": {
            "trainable_params": params,
            "eval_bpb": bpb,
            "elapsed_seconds": seconds,
            "train_bytes": train_bytes,
            "eval_bytes": eval_bytes,
        },
    }


def _generation(device, bps, quality=0.9, relevance=1.0, runtime_path="layercake"):
    return {
        "device": device,
        "metrics": {
            "generation_bytes_per_second": bps,
            "quality_score": quality,
            "relevance_rate": relevance,
        },
        "samples": [
            {
                "runtime_path": runtime_path,
                "text": "Retreat, guard, create space, and recover tempo.",
                "printable_ratio": 1.0,
                "max_repeat_8gram": 1.0,
            }
        ],
    }


def _transfer():
    return {
        "status": "PASS",
        "gates": {
            "transfer_max_logit_diff_exact": True,
            "transfer_ppl_ratio_exact": True,
            "transfer_generation_exact": True,
        },
    }


def test_true_dominance_gate_passes_clean_artifacts():
    result = verify(
        layercake_config=_lc_config(),
        transformer_config=_tx_config(),
        layercake_training=_training(1_000_000, 1.9, 10.0, train_bytes=900_000),
        transformer_training=_training(2_000_000, 2.0, 20.0),
        layercake_cpu_generation=_generation("cpu", 2000.0),
        transformer_cpu_generation=_generation("cpu", 1000.0, runtime_path="bpe"),
        layercake_gpu_generation=_generation("cuda", 1500.0),
        transformer_gpu_generation=_generation("cuda", 1000.0, runtime_path="bpe"),
        transfer_certificate=_transfer(),
        args=_args(),
    )
    assert result["status"] == "PASS"


def test_true_dominance_gate_rejects_cache_overlap_small_eval_and_alias_runtime():
    lc_config = _lc_config()
    lc_config["model"]["domain_cache_override"] = True
    lc_config["model"]["domain_cache_order"] = 16
    lc_config["training"]["eval_data_roots"] = ["data/train"]
    result = verify(
        layercake_config=lc_config,
        transformer_config=_tx_config(),
        layercake_training=_training(1_000_000, 1.9, 10.0, eval_bytes=5360),
        transformer_training=_training(2_000_000, 2.0, 20.0),
        layercake_cpu_generation=_generation(
            "cpu", 2000.0, runtime_path="semantic_instruction_alias"
        ),
        transformer_cpu_generation=_generation("cpu", 1000.0, runtime_path="bpe"),
        layercake_gpu_generation=_generation(
            "cuda", 1500.0, runtime_path="semantic_instruction_alias"
        ),
        transformer_gpu_generation=_generation("cuda", 1000.0, runtime_path="bpe"),
        transfer_certificate=_transfer(),
        args=_args(),
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["layercake_train_eval_split_disjoint"] is False
    assert result["gates"]["layercake_domain_cache_disabled"] is False
    assert result["gates"]["heldout_eval_bytes_met"] is False
    assert result["gates"]["cpu_generation_raw_neural_only"] is False
