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


def _layercake_training_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one LayerCake training metrics file is required")
    latest_rows = [row.get("latest", {}) for row in rows]
    final_latest = latest_rows[-1]
    elapsed = 0.0
    pretrain = 0.0
    for latest in latest_rows:
        elapsed += float(latest.get("elapsed_seconds", 0.0))
        pretrain += float(latest.get("counted_pretrain_seconds", 0.0))
    return {
        "status_complete": all(row.get("status") == "COMPLETE" for row in rows),
        "phase_count": len(rows),
        "elapsed_seconds": elapsed,
        "counted_pretrain_seconds": pretrain,
        "elapsed_total_seconds": elapsed + pretrain,
        "step": int(final_latest.get("step", 0)),
        "train_bytes": float(final_latest.get("train_bytes", 0.0)),
        "trainable_params": int(final_latest.get("trainable_params", 0)),
        "eval_bpb": float(final_latest.get("eval_bpb", 0.0)),
    }


def _bpe_training_summary(row: dict[str, Any]) -> dict[str, Any]:
    latest = row.get("latest", {})
    return {
        "status_complete": row.get("status") == "COMPLETE",
        "elapsed_seconds": float(latest.get("elapsed_seconds", 0.0)),
        "tokenizer_seconds": float(latest.get("tokenizer_seconds", 0.0)),
        "elapsed_total_seconds": float(
            latest.get(
                "elapsed_total_seconds",
                float(latest.get("elapsed_seconds", 0.0))
                + float(latest.get("tokenizer_seconds", 0.0)),
            )
        ),
        "step": int(latest.get("step", 0)),
        "train_bytes": float(latest.get("train_bytes", 0.0)),
        "trainable_params": int(latest.get("trainable_params", 0)),
        "eval_bpb": float(latest.get("eval_bpb", 0.0)),
    }


def _generation_summary(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples", [])
    return {
        "device": str(row.get("device", "")).lower(),
        "generation_bytes_per_second": float(
            _get(row, "metrics.generation_bytes_per_second", 0.0)
        ),
        "quality_score": float(_get(row, "metrics.quality_score", 0.0)),
        "relevance_rate": float(_get(row, "metrics.relevance_rate", 0.0)),
        "sample_count": len(samples),
        "samples_nonempty": bool(samples)
        and all(bool(str(sample.get("text", "")).strip()) for sample in samples),
        "samples_relevant": bool(samples)
        and all(bool(sample.get("relevance_pass", False)) for sample in samples),
        "samples_printable": bool(samples)
        and all(float(sample.get("printable_ratio", 0.0)) >= 0.95 for sample in samples),
        "samples_no_repeat_8": bool(samples)
        and all(float(sample.get("max_repeat_8gram", 999.0)) <= 4.0 for sample in samples),
    }


def verify(
    *,
    layercake_training_rows: list[dict[str, Any]],
    transformer_training: dict[str, Any],
    layercake_cpu_generation: dict[str, Any],
    transformer_cpu_generation: dict[str, Any],
    layercake_gpu_generation: dict[str, Any],
    transformer_gpu_generation: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lc_train = _layercake_training_summary(layercake_training_rows)
    tx_train = _bpe_training_summary(transformer_training)
    lc_cpu = _generation_summary(layercake_cpu_generation)
    tx_cpu = _generation_summary(transformer_cpu_generation)
    lc_gpu = _generation_summary(layercake_gpu_generation)
    tx_gpu = _generation_summary(transformer_gpu_generation)
    ratios = {
        "transformer_param_ratio": _ratio(
            tx_train["trainable_params"],
            lc_train["trainable_params"],
        ),
        "eval_bpb_ratio_layercake_over_transformer": _ratio(
            lc_train["eval_bpb"],
            tx_train["eval_bpb"],
        ),
        "training_wall_clock_speed_ratio": _ratio(
            tx_train["elapsed_total_seconds"],
            lc_train["elapsed_total_seconds"],
        ),
        "training_byte_exposure_ratio_layercake_over_transformer": _ratio(
            lc_train["train_bytes"],
            tx_train["train_bytes"],
        ),
        "cpu_generation_speed_ratio": _ratio(
            lc_cpu["generation_bytes_per_second"],
            tx_cpu["generation_bytes_per_second"],
        ),
        "gpu_generation_speed_ratio": _ratio(
            lc_gpu["generation_bytes_per_second"],
            tx_gpu["generation_bytes_per_second"],
        ),
        "cpu_quality_ratio": _ratio(lc_cpu["quality_score"], tx_cpu["quality_score"]),
        "gpu_quality_ratio": _ratio(lc_gpu["quality_score"], tx_gpu["quality_score"]),
        "cpu_relevance_ratio": _ratio(lc_cpu["relevance_rate"], tx_cpu["relevance_rate"]),
        "gpu_relevance_ratio": _ratio(lc_gpu["relevance_rate"], tx_gpu["relevance_rate"]),
    }
    gates = {
        "layercake_training_complete": lc_train["status_complete"],
        "transformer_training_complete": tx_train["status_complete"],
        "transformer_at_least_same_size": ratios["transformer_param_ratio"] >= args.min_param_ratio,
        "layercake_reached_training_step": lc_train["step"] >= args.min_layercake_step,
        "transformer_reached_training_step": tx_train["step"] >= args.min_transformer_step,
        "layercake_eval_bpb_noninferior": (
            lc_train["eval_bpb"] > 0.0
            and tx_train["eval_bpb"] > 0.0
            and ratios["eval_bpb_ratio_layercake_over_transformer"]
            <= args.max_eval_bpb_ratio
        ),
        "layercake_no_more_training_bytes": (
            ratios["training_byte_exposure_ratio_layercake_over_transformer"]
            <= args.max_training_byte_exposure_ratio
        ),
        "layercake_faster_training_wall_clock": (
            ratios["training_wall_clock_speed_ratio"] >= args.min_training_speed_ratio
        ),
        "cpu_devices": lc_cpu["device"] == "cpu" and tx_cpu["device"] == "cpu",
        "gpu_devices": lc_gpu["device"] == "cuda" and tx_gpu["device"] == "cuda",
        "cpu_generation_faster": ratios["cpu_generation_speed_ratio"] >= args.min_cpu_speed_ratio,
        "gpu_generation_faster": ratios["gpu_generation_speed_ratio"] >= args.min_gpu_speed_ratio,
        "cpu_quality_better": ratios["cpu_quality_ratio"] >= args.min_quality_ratio,
        "gpu_quality_better": ratios["gpu_quality_ratio"] >= args.min_quality_ratio,
        "cpu_relevance_full": lc_cpu["relevance_rate"] >= args.min_layercake_relevance,
        "gpu_relevance_full": lc_gpu["relevance_rate"] >= args.min_layercake_relevance,
        "cpu_relevance_better": lc_cpu["relevance_rate"] > tx_cpu["relevance_rate"],
        "gpu_relevance_better": lc_gpu["relevance_rate"] > tx_gpu["relevance_rate"],
        "cpu_layercake_samples_clean": (
            lc_cpu["samples_nonempty"]
            and lc_cpu["samples_relevant"]
            and lc_cpu["samples_printable"]
            and lc_cpu["samples_no_repeat_8"]
        ),
        "gpu_layercake_samples_clean": (
            lc_gpu["samples_nonempty"]
            and lc_gpu["samples_relevant"]
            and lc_gpu["samples_printable"]
            and lc_gpu["samples_no_repeat_8"]
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": (
            "Same-recipe digital-companion comparison: 15M LayerCake companion "
            "checkpoint plus bounded domain runtime vs a freshly trained 16M BPE "
            "transformer on the same RedPajama + Ember Road + companion corpus recipe. "
            "This is a scoped game-companion result, not a universal open-domain claim."
        ),
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
    parser = argparse.ArgumentParser(
        description="Verify same-recipe LayerCake companion vs BPE transformer comparison."
    )
    parser.add_argument("--layercake-training", required=True, nargs="+", type=Path)
    parser.add_argument("--transformer-training", required=True, type=Path)
    parser.add_argument("--layercake-cpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-cpu-generation", required=True, type=Path)
    parser.add_argument("--layercake-gpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-gpu-generation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-param-ratio", type=float, default=1.0)
    parser.add_argument("--min-layercake-step", type=int, default=11000)
    parser.add_argument("--min-transformer-step", type=int, default=11000)
    parser.add_argument("--min-training-speed-ratio", type=float, default=1.0)
    parser.add_argument("--max-eval-bpb-ratio", type=float, default=1.0)
    parser.add_argument("--max-training-byte-exposure-ratio", type=float, default=1.0)
    parser.add_argument("--min-cpu-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-gpu-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    parser.add_argument("--min-layercake-relevance", type=float, default=1.0)
    args = parser.parse_args()
    result = verify(
        layercake_training_rows=[_read(path) for path in args.layercake_training],
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
