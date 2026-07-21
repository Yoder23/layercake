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


def _sample_gates(row: dict[str, Any]) -> dict[str, bool]:
    samples = row.get("samples", [])
    return {
        "samples_present": bool(samples),
        "samples_nonempty": all(bool(str(sample.get("text", "")).strip()) for sample in samples),
        "samples_printable": all(float(sample.get("printable_ratio", 0.0)) >= 0.95 for sample in samples),
        "samples_alpha": all(float(sample.get("alpha_space_ratio", 0.0)) >= 0.75 for sample in samples),
        "samples_no_repeat_8": all(float(sample.get("max_repeat_8gram", 999.0)) <= 4.0 for sample in samples),
        "samples_relevant": all(bool(sample.get("relevance_pass", False)) for sample in samples),
        "samples_not_trimmed": all(not bool(sample.get("trimmed", False)) for sample in samples),
    }


def _category_relevance(row: dict[str, Any]) -> dict[str, float]:
    metrics = _get(row, "metrics.category_metrics", {}) or {}
    return {
        str(category): float((values or {}).get("relevance_rate", 0.0))
        for category, values in metrics.items()
    }


def verify(
    *,
    layercake_cpu: dict[str, Any],
    transformer_cpu: dict[str, Any],
    layercake_gpu: dict[str, Any],
    transformer_gpu: dict[str, Any],
    training_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lc_cpu_bps = float(_get(layercake_cpu, "metrics.generation_bytes_per_second", 0.0))
    tx_cpu_bps = float(_get(transformer_cpu, "metrics.generation_bytes_per_second", 0.0))
    lc_gpu_bps = float(_get(layercake_gpu, "metrics.generation_bytes_per_second", 0.0))
    tx_gpu_bps = float(_get(transformer_gpu, "metrics.generation_bytes_per_second", 0.0))
    lc_cpu_quality = float(_get(layercake_cpu, "metrics.quality_score", 0.0))
    tx_cpu_quality = float(_get(transformer_cpu, "metrics.quality_score", 0.0))
    lc_gpu_quality = float(_get(layercake_gpu, "metrics.quality_score", 0.0))
    tx_gpu_quality = float(_get(transformer_gpu, "metrics.quality_score", 0.0))
    lc_cpu_rel = float(_get(layercake_cpu, "metrics.relevance_rate", 0.0))
    tx_cpu_rel = float(_get(transformer_cpu, "metrics.relevance_rate", 0.0))
    lc_gpu_rel = float(_get(layercake_gpu, "metrics.relevance_rate", 0.0))
    tx_gpu_rel = float(_get(transformer_gpu, "metrics.relevance_rate", 0.0))
    latest = training_metrics.get("latest", {})
    lc_cpu_sample_gates = _sample_gates(layercake_cpu)
    lc_gpu_sample_gates = _sample_gates(layercake_gpu)
    category_rates = {
        "cpu": _category_relevance(layercake_cpu),
        "gpu": _category_relevance(layercake_gpu),
    }
    required_categories = [item.strip() for item in args.required_categories.split(",") if item.strip()]
    category_gates = {
        f"cpu_category_{category}_full_relevance": category_rates["cpu"].get(category, 0.0) >= args.min_layercake_relevance
        for category in required_categories
    }
    category_gates.update(
        {
            f"gpu_category_{category}_full_relevance": category_rates["gpu"].get(category, 0.0) >= args.min_layercake_relevance
            for category in required_categories
        }
    )
    ratios = {
        "cpu_generation_speed_ratio": _ratio(lc_cpu_bps, tx_cpu_bps),
        "gpu_generation_speed_ratio": _ratio(lc_gpu_bps, tx_gpu_bps),
        "cpu_quality_ratio": _ratio(lc_cpu_quality, tx_cpu_quality),
        "gpu_quality_ratio": _ratio(lc_gpu_quality, tx_gpu_quality),
        "cpu_relevance_ratio": _ratio(lc_cpu_rel, tx_cpu_rel),
        "gpu_relevance_ratio": _ratio(lc_gpu_rel, tx_gpu_rel),
    }
    gates = {
        "training_complete": training_metrics.get("status") == "COMPLETE",
        "training_reached_min_step": int(latest.get("step", 0)) >= args.min_training_step,
        "training_has_eval_bpb": float(latest.get("eval_bpb", 0.0)) > 0.0,
        "layercake_cpu_device": str(layercake_cpu.get("device", "")).lower() == "cpu",
        "layercake_gpu_device": str(layercake_gpu.get("device", "")).lower() == "cuda",
        "transformer_cpu_device": str(transformer_cpu.get("device", "")).lower() == "cpu",
        "transformer_gpu_device": str(transformer_gpu.get("device", "")).lower() == "cuda",
        "cpu_generation_speed": ratios["cpu_generation_speed_ratio"] >= args.min_cpu_generation_speed_ratio,
        "gpu_generation_speed": ratios["gpu_generation_speed_ratio"] >= args.min_gpu_generation_speed_ratio,
        "cpu_quality": ratios["cpu_quality_ratio"] >= args.min_quality_ratio,
        "gpu_quality": ratios["gpu_quality_ratio"] >= args.min_quality_ratio,
        "cpu_relevance": lc_cpu_rel >= args.min_layercake_relevance and ratios["cpu_relevance_ratio"] >= args.min_relevance_ratio,
        "gpu_relevance": lc_gpu_rel >= args.min_layercake_relevance and ratios["gpu_relevance_ratio"] >= args.min_relevance_ratio,
        **{f"cpu_{key}": value for key, value in lc_cpu_sample_gates.items()},
        **{f"gpu_{key}": value for key, value in lc_gpu_sample_gates.items()},
        **category_gates,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": (
            "Production companion runtime gate. This compares the 15M LayerCake "
            "checkpoint plus its bounded companion/domain runtime against the saved "
            "10M BPE transformer on the same companion prompt suite. It is not a "
            "universal open-domain language-model claim."
        ),
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "training": {
                "step": int(latest.get("step", 0)),
                "trainable_params": int(latest.get("trainable_params", 0)),
                "train_bytes": int(latest.get("train_bytes", 0)),
                "elapsed_seconds": float(latest.get("elapsed_seconds", 0.0)),
                "eval_bpb": float(latest.get("eval_bpb", 0.0)),
                "steps_per_second": float(latest.get("steps_per_second", 0.0)),
            },
            "layercake_cpu": layercake_cpu.get("metrics", {}),
            "transformer_cpu": transformer_cpu.get("metrics", {}),
            "layercake_gpu": layercake_gpu.get("metrics", {}),
            "transformer_gpu": transformer_gpu.get("metrics", {}),
            "layercake_category_relevance": category_rates,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the production companion runtime gate.")
    parser.add_argument("--layercake-cpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-cpu-generation", required=True, type=Path)
    parser.add_argument("--layercake-gpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-gpu-generation", required=True, type=Path)
    parser.add_argument("--training-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-training-step", type=int, default=10000)
    parser.add_argument("--min-cpu-generation-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-gpu-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    parser.add_argument("--min-relevance-ratio", type=float, default=1.0)
    parser.add_argument("--min-layercake-relevance", type=float, default=1.0)
    parser.add_argument(
        "--required-categories",
        default="game_tactics,game_recovery,companion_style",
    )
    args = parser.parse_args()
    result = verify(
        layercake_cpu=_read(args.layercake_cpu_generation),
        transformer_cpu=_read(args.transformer_cpu_generation),
        layercake_gpu=_read(args.layercake_gpu_generation),
        transformer_gpu=_read(args.transformer_gpu_generation),
        training_metrics=_read(args.training_metrics),
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
