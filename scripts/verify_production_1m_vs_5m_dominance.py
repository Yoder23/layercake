from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(row: dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        cur: Any = row
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def _ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def _training_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": int(_get(row, "latest.trainable_params", "trainable_params", "params")),
        "bpb": float(_get(row, "latest.eval_bpb", "eval_bpb", "latest.bpb", "bpb")),
        "train_seconds": float(_get(row, "latest.elapsed_seconds", "elapsed_seconds", "train_seconds")),
        "train_bytes": int(_get(row, "latest.train_bytes", "train_bytes", default=0)),
    }


def _generation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples", [])
    nonempty = all(bool(str(sample.get("text", "")).strip()) for sample in samples)
    printable = all(float(sample.get("printable_ratio", 0.0)) >= 0.95 for sample in samples)
    alpha = all(float(sample.get("alpha_space_ratio", 0.0)) >= 0.75 for sample in samples)
    no_repeat = all(float(sample.get("max_repeat_8gram", 999.0)) <= 4.0 for sample in samples)
    lexical = all(
        float(sample.get("unique_word_count", 0.0)) >= 8.0
        and float(sample.get("distinct_word_ratio", 0.0)) >= 0.35
        and float(sample.get("one_char_word_ratio", 1.0)) <= 0.35
        and float(sample.get("unique_alpha_char_count", 0.0)) >= 10.0
        for sample in samples
    )
    return {
        "device": str(row.get("device", "unknown")).lower(),
        "generation_bytes_per_second": float(_get(row, "metrics.generation_bytes_per_second")),
        "quality_score": float(_get(row, "metrics.quality_score", default=0.0)),
        "samples_nonempty": bool(samples) and nonempty,
        "samples_printable": printable,
        "samples_alpha": alpha,
        "samples_no_repeat_8": no_repeat,
        "samples_lexically_diverse": bool(samples) and lexical,
    }


def verify(
    *,
    layercake_training: dict[str, Any],
    transformer_training: dict[str, Any],
    layercake_cpu_generation: dict[str, Any],
    transformer_cpu_generation: dict[str, Any],
    layercake_gpu_generation: dict[str, Any],
    transformer_gpu_generation: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lc = _training_metrics(layercake_training)
    tx = _training_metrics(transformer_training)
    lc_cpu = _generation_metrics(layercake_cpu_generation)
    tx_cpu = _generation_metrics(transformer_cpu_generation)
    lc_gpu = _generation_metrics(layercake_gpu_generation)
    tx_gpu = _generation_metrics(transformer_gpu_generation)
    ratios = {
        "parameter_ratio_transformer_over_layercake": _ratio(tx["params"], lc["params"]),
        "bpb_ratio_layercake_over_transformer": _ratio(lc["bpb"], tx["bpb"]),
        "training_speed_ratio": _ratio(tx["train_seconds"], lc["train_seconds"]),
        "cpu_generation_speed_ratio": _ratio(lc_cpu["generation_bytes_per_second"], tx_cpu["generation_bytes_per_second"]),
        "gpu_generation_speed_ratio": _ratio(lc_gpu["generation_bytes_per_second"], tx_gpu["generation_bytes_per_second"]),
        "cpu_quality_ratio": _ratio(lc_cpu["quality_score"], tx_cpu["quality_score"]),
        "gpu_quality_ratio": _ratio(lc_gpu["quality_score"], tx_gpu["quality_score"]),
    }
    sample_gates = {
        "cpu_samples_nonempty": lc_cpu["samples_nonempty"],
        "cpu_samples_printable": lc_cpu["samples_printable"],
        "cpu_samples_alpha": lc_cpu["samples_alpha"],
        "cpu_samples_no_repeat_8": lc_cpu["samples_no_repeat_8"],
        "cpu_samples_lexically_diverse": lc_cpu["samples_lexically_diverse"],
        "gpu_samples_nonempty": lc_gpu["samples_nonempty"],
        "gpu_samples_printable": lc_gpu["samples_printable"],
        "gpu_samples_alpha": lc_gpu["samples_alpha"],
        "gpu_samples_no_repeat_8": lc_gpu["samples_no_repeat_8"],
        "gpu_samples_lexically_diverse": lc_gpu["samples_lexically_diverse"],
    }
    gates = {
        "transformer_at_least_5x_params": ratios["parameter_ratio_transformer_over_layercake"] >= args.min_param_ratio,
        "bpb_non_inferior": lc["bpb"] <= tx["bpb"],
        "training_speed_met": ratios["training_speed_ratio"] >= args.min_training_speed_ratio,
        "no_more_training_bytes": lc["train_bytes"] <= tx["train_bytes"] if lc["train_bytes"] and tx["train_bytes"] else True,
        "cpu_devices": lc_cpu["device"] == "cpu" and tx_cpu["device"] == "cpu",
        "gpu_devices": lc_gpu["device"] == "cuda" and tx_gpu["device"] == "cuda",
        "cpu_generation_5x_met": ratios["cpu_generation_speed_ratio"] >= args.min_cpu_generation_speed_ratio,
        "gpu_generation_noninferior": ratios["gpu_generation_speed_ratio"] >= args.min_gpu_generation_speed_ratio,
        "cpu_quality_noninferior": ratios["cpu_quality_ratio"] >= args.min_quality_ratio,
        "gpu_quality_noninferior": ratios["gpu_quality_ratio"] >= args.min_quality_ratio,
        **sample_gates,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "Production 1M LayerCake vs 5M tokenizer-transformer dominance certificate",
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake": lc,
            "transformer": tx,
            "layercake_cpu_generation": lc_cpu,
            "transformer_cpu_generation": tx_cpu,
            "layercake_gpu_generation": lc_gpu,
            "transformer_gpu_generation": tx_gpu,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify 1M LayerCake vs 5M BPE production dominance")
    parser.add_argument("--layercake-training", required=True, type=Path)
    parser.add_argument("--transformer-training", required=True, type=Path)
    parser.add_argument("--layercake-cpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-cpu-generation", required=True, type=Path)
    parser.add_argument("--layercake-gpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-gpu-generation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-param-ratio", type=float, default=5.0)
    parser.add_argument("--min-training-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-cpu-generation-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-gpu-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    args = parser.parse_args()
    result = verify(
        layercake_training=_read(args.layercake_training),
        transformer_training=_read(args.transformer_training),
        layercake_cpu_generation=_read(args.layercake_cpu_generation),
        transformer_cpu_generation=_read(args.transformer_cpu_generation),
        layercake_gpu_generation=_read(args.layercake_gpu_generation),
        transformer_gpu_generation=_read(args.transformer_gpu_generation),
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
