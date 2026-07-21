from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(row: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = row
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def _roots(config: dict[str, Any], key: str) -> set[str]:
    values = _get(config, f"training.{key}", default=[]) or []
    return {str(Path(item).as_posix()).lower().rstrip("/") for item in values}


def _split_disjoint(config: dict[str, Any]) -> bool:
    train = _roots(config, "data_roots")
    eval_roots = _roots(config, "eval_data_roots")
    return bool(train) and bool(eval_roots) and train.isdisjoint(eval_roots)


def _domain_cache_disabled(config: dict[str, Any]) -> bool:
    model = config.get("model", {})
    training = config.get("training", {})
    return (
        not bool(model.get("domain_cache_override", False))
        and int(model.get("domain_cache_order", 0)) <= 0
        and not bool(training.get("initialize_domain_cache_from_corpus", False))
    )


def _training_summary(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    latest = metrics.get("latest", metrics)
    configured_eval_bytes = float(_get(config, "training.eval_bytes", default=0.0) or 0.0)
    return {
        "params": float(
            latest.get("trainable_params", metrics.get("trainable_params", 0.0))
        ),
        "eval_bpb": float(
            latest.get("eval_bpb", latest.get("bpb", metrics.get("eval_bpb", 0.0)))
        ),
        "train_seconds": float(
            latest.get(
                "elapsed_total_seconds",
                latest.get("elapsed_seconds", metrics.get("train_seconds", 0.0)),
            )
        ),
        "train_bytes": float(latest.get("train_bytes", metrics.get("train_bytes", 0.0))),
        "eval_bytes": float(
            latest.get(
                "eval_bytes",
                metrics.get("eval_bytes", configured_eval_bytes),
            )
        ),
    }


def _raw_generation_summary(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    samples = row.get("samples", [])
    runtimes = {str(sample.get("runtime_path", row.get("runtime", ""))) for sample in samples}
    return {
        "device": str(row.get("device", "")).lower(),
        "bytes_per_second": float(metrics.get("generation_bytes_per_second", 0.0)),
        "quality_score": float(metrics.get("quality_score", 0.0)),
        "relevance_rate": float(metrics.get("relevance_rate", 0.0)),
        "sample_count": len(samples),
        "raw_neural_only": all(
            runtime in {
                "",
                "layercake",
                "neural_layercake",
                "neural_layercake_patch_prediction",
                "bpe",
                "neural_bpe_transformer",
            }
            for runtime in runtimes
        ),
        "samples_clean": bool(samples)
        and all(bool(str(sample.get("text", "")).strip()) for sample in samples)
        and all(float(sample.get("printable_ratio", 0.0)) >= 0.95 for sample in samples)
        and all(float(sample.get("max_repeat_8gram", 999.0)) <= 4.0 for sample in samples),
    }


def verify(
    *,
    layercake_config: dict[str, Any],
    transformer_config: dict[str, Any],
    layercake_training: dict[str, Any],
    transformer_training: dict[str, Any],
    layercake_cpu_generation: dict[str, Any],
    transformer_cpu_generation: dict[str, Any],
    layercake_gpu_generation: dict[str, Any],
    transformer_gpu_generation: dict[str, Any],
    transfer_certificate: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lc_train = _training_summary(layercake_training, layercake_config)
    tx_train = _training_summary(transformer_training, transformer_config)
    lc_cpu = _raw_generation_summary(layercake_cpu_generation)
    tx_cpu = _raw_generation_summary(transformer_cpu_generation)
    lc_gpu = _raw_generation_summary(layercake_gpu_generation)
    tx_gpu = _raw_generation_summary(transformer_gpu_generation)
    ratios = {
        "parameter_ratio_transformer_over_layercake": _ratio(
            tx_train["params"], lc_train["params"]
        ),
        "eval_bpb_ratio_layercake_over_transformer": _ratio(
            lc_train["eval_bpb"], tx_train["eval_bpb"]
        ),
        "training_speed_ratio": _ratio(
            tx_train["train_seconds"], lc_train["train_seconds"]
        ),
        "training_cost_proxy_ratio": _ratio(
            tx_train["params"] * tx_train["train_seconds"],
            lc_train["params"] * lc_train["train_seconds"],
        ),
        "training_byte_ratio_transformer_over_layercake": _ratio(
            tx_train["train_bytes"], lc_train["train_bytes"]
        ),
        "cpu_generation_speed_ratio": _ratio(
            lc_cpu["bytes_per_second"], tx_cpu["bytes_per_second"]
        ),
        "gpu_generation_speed_ratio": _ratio(
            lc_gpu["bytes_per_second"], tx_gpu["bytes_per_second"]
        ),
        "cpu_quality_ratio": _ratio(lc_cpu["quality_score"], tx_cpu["quality_score"]),
        "gpu_quality_ratio": _ratio(lc_gpu["quality_score"], tx_gpu["quality_score"]),
        "cpu_relevance_ratio": _ratio(lc_cpu["relevance_rate"], tx_cpu["relevance_rate"]),
        "gpu_relevance_ratio": _ratio(lc_gpu["relevance_rate"], tx_gpu["relevance_rate"]),
    }
    transfer_gates = transfer_certificate.get("gates", {})
    gates = {
        "layercake_train_eval_split_disjoint": _split_disjoint(layercake_config),
        "transformer_train_eval_split_disjoint": _split_disjoint(transformer_config),
        "layercake_domain_cache_disabled": _domain_cache_disabled(layercake_config),
        "transformer_at_least_min_param_ratio": (
            ratios["parameter_ratio_transformer_over_layercake"] >= args.min_param_ratio
        ),
        "heldout_eval_bytes_met": (
            lc_train["eval_bytes"] >= args.min_eval_bytes
            and tx_train["eval_bytes"] >= args.min_eval_bytes
        ),
        "heldout_bpb_noninferior": (
            lc_train["eval_bpb"] > 0.0
            and tx_train["eval_bpb"] > 0.0
            and ratios["eval_bpb_ratio_layercake_over_transformer"]
            <= args.max_eval_bpb_ratio
        ),
        "training_faster": ratios["training_speed_ratio"] >= args.min_training_speed_ratio,
        "training_cost_lower": (
            ratios["training_cost_proxy_ratio"] >= args.min_training_cost_ratio
        ),
        "no_more_training_bytes": (
            lc_train["train_bytes"] <= tx_train["train_bytes"] * args.max_train_byte_ratio
        ),
        "cpu_generation_faster": (
            ratios["cpu_generation_speed_ratio"] >= args.min_cpu_generation_speed_ratio
        ),
        "gpu_generation_noninferior": (
            ratios["gpu_generation_speed_ratio"] >= args.min_gpu_generation_speed_ratio
        ),
        "cpu_quality_noninferior": ratios["cpu_quality_ratio"] >= args.min_quality_ratio,
        "gpu_quality_noninferior": ratios["gpu_quality_ratio"] >= args.min_quality_ratio,
        "cpu_relevance_noninferior": (
            lc_cpu["relevance_rate"] >= args.min_layercake_relevance
            and ratios["cpu_relevance_ratio"] >= args.min_relevance_ratio
        ),
        "gpu_relevance_noninferior": (
            lc_gpu["relevance_rate"] >= args.min_layercake_relevance
            and ratios["gpu_relevance_ratio"] >= args.min_relevance_ratio
        ),
        "cpu_generation_raw_neural_only": lc_cpu["raw_neural_only"],
        "gpu_generation_raw_neural_only": lc_gpu["raw_neural_only"],
        "cpu_samples_clean": lc_cpu["samples_clean"],
        "gpu_samples_clean": lc_gpu["samples_clean"],
        "transfer_certificate_pass": transfer_certificate.get("status") == "PASS",
        "transfer_logits_exact": bool(
            transfer_gates.get("transfer_max_logit_diff_exact")
            or transfer_gates.get("transfer_generation_exact")
        ),
        "transfer_ppl_exact": bool(transfer_gates.get("transfer_ppl_ratio_exact")),
        "transfer_generation_exact": bool(transfer_gates.get("transfer_generation_exact")),
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "status": "PASS" if not blockers else "FAIL",
        "scope": (
            "Strict true-dominance gate: held-out English/domain modeling, raw neural "
            "generation, exact game-domain transfer, lower training cost, and CPU/GPU "
            "generation efficiency. Domain caches, train/eval overlap, alias lookup, "
            "and tiny held-out slices are blockers."
        ),
        "blockers": blockers,
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake_training": lc_train,
            "transformer_training": tx_train,
            "layercake_cpu_generation": lc_cpu,
            "transformer_cpu_generation": tx_cpu,
            "layercake_gpu_generation": lc_gpu,
            "transformer_gpu_generation": tx_gpu,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the strict LayerCake true-dominance claim.")
    parser.add_argument("--layercake-config", required=True, type=Path)
    parser.add_argument("--transformer-config", required=True, type=Path)
    parser.add_argument("--layercake-training", required=True, type=Path)
    parser.add_argument("--transformer-training", required=True, type=Path)
    parser.add_argument("--layercake-cpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-cpu-generation", required=True, type=Path)
    parser.add_argument("--layercake-gpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-gpu-generation", required=True, type=Path)
    parser.add_argument("--transfer-certificate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-param-ratio", type=float, default=1.0)
    parser.add_argument("--min-eval-bytes", type=float, default=100_000.0)
    parser.add_argument("--max-eval-bpb-ratio", type=float, default=1.0)
    parser.add_argument("--min-training-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-training-cost-ratio", type=float, default=1.0)
    parser.add_argument("--max-train-byte-ratio", type=float, default=1.0)
    parser.add_argument("--min-cpu-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-gpu-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    parser.add_argument("--min-relevance-ratio", type=float, default=1.0)
    parser.add_argument("--min-layercake-relevance", type=float, default=1.0)
    args = parser.parse_args()
    result = verify(
        layercake_config=_read(args.layercake_config),
        transformer_config=_read(args.transformer_config),
        layercake_training=_read(args.layercake_training),
        transformer_training=_read(args.transformer_training),
        layercake_cpu_generation=_read(args.layercake_cpu_generation),
        transformer_cpu_generation=_read(args.transformer_cpu_generation),
        layercake_gpu_generation=_read(args.layercake_gpu_generation),
        transformer_gpu_generation=_read(args.transformer_gpu_generation),
        transfer_certificate=_read(args.transfer_certificate),
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
