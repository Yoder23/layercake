from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def _metric(row: dict[str, Any], name: str) -> float:
    return float(row.get("metrics", {}).get(name, 0.0))


def _examples(layercake: dict[str, Any], bpe: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = []
    for lc_sample, bpe_sample in zip(layercake.get("samples", []), bpe.get("samples", [])):
        rows.append(
            {
                "prompt": lc_sample.get("prompt"),
                "category": lc_sample.get("category"),
                "layercake_text": lc_sample.get("text"),
                "layercake_relevance": lc_sample.get("relevance_pass"),
                "layercake_runtime_path": lc_sample.get("runtime_path"),
                "transformer_text": bpe_sample.get("text"),
                "transformer_relevance": bpe_sample.get("relevance_pass"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake-cpu", type=Path, required=True)
    parser.add_argument("--layercake-gpu", type=Path, required=True)
    parser.add_argument("--bpe-cpu", type=Path, required=True)
    parser.add_argument("--bpe-gpu", type=Path, required=True)
    parser.add_argument("--bpe-training", type=Path, required=True)
    parser.add_argument("--layercake-training", type=Path, required=True)
    parser.add_argument("--portable-training", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=12)
    args = parser.parse_args()

    lc_cpu = _read(args.layercake_cpu)
    lc_gpu = _read(args.layercake_gpu)
    bpe_cpu = _read(args.bpe_cpu)
    bpe_gpu = _read(args.bpe_gpu)
    bpe_training = _read(args.bpe_training)
    layercake_training = _read(args.layercake_training)
    portable_training = _read(args.portable_training)
    transfer = _read(args.transfer)

    ratios = {
        "cpu_speed": _ratio(_metric(lc_cpu, "generation_bytes_per_second"), _metric(bpe_cpu, "generation_bytes_per_second")),
        "gpu_speed": _ratio(_metric(lc_gpu, "generation_bytes_per_second"), _metric(bpe_gpu, "generation_bytes_per_second")),
        "cpu_relevance": _ratio(_metric(lc_cpu, "relevance_rate"), _metric(bpe_cpu, "relevance_rate")),
        "gpu_relevance": _ratio(_metric(lc_gpu, "relevance_rate"), _metric(bpe_gpu, "relevance_rate")),
        "cpu_quality": _ratio(_metric(lc_cpu, "quality_score"), _metric(bpe_cpu, "quality_score")),
        "gpu_quality": _ratio(_metric(lc_gpu, "quality_score"), _metric(bpe_gpu, "quality_score")),
    }
    gates = {
        "bpe_trained": bpe_training.get("status") == "COMPLETE",
        "layercake_core_trained": layercake_training.get("status") == "COMPLETE",
        "portable_game_layer_trained": portable_training.get("status") == "TRAINED",
        "layercake_cpu_relevance_full": _metric(lc_cpu, "relevance_rate") == 1.0,
        "layercake_gpu_relevance_full": _metric(lc_gpu, "relevance_rate") == 1.0,
        "layercake_cpu_faster_than_transformer": ratios["cpu_speed"] >= 5.0,
        "layercake_gpu_faster_than_transformer": ratios["gpu_speed"] >= 5.0,
        "layercake_cpu_quality_noninferior": _metric(lc_cpu, "quality_score") >= _metric(bpe_cpu, "quality_score"),
        "layercake_gpu_quality_noninferior": _metric(lc_gpu, "quality_score") >= _metric(bpe_gpu, "quality_score"),
        "lossless_game_layer_transfer": transfer.get("status") == "PASS",
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mode": "game_companion_domain_layer_runtime",
        "claim_scope": (
            "Game companion domain-layer runtime over a trained LayerCake core, compared "
            "with a freshly trained BPE transformer on English plus Ember Road game data. "
            "This is not a fair-neural no-layer LLM claim."
        ),
        "artifacts": {
            "layercake_cpu": str(args.layercake_cpu),
            "layercake_gpu": str(args.layercake_gpu),
            "bpe_cpu": str(args.bpe_cpu),
            "bpe_gpu": str(args.bpe_gpu),
            "transfer": str(args.transfer),
            "examples": str(args.examples),
        },
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake_cpu": lc_cpu.get("metrics", {}),
            "layercake_gpu": lc_gpu.get("metrics", {}),
            "bpe_cpu": bpe_cpu.get("metrics", {}),
            "bpe_gpu": bpe_gpu.get("metrics", {}),
            "portable_training": portable_training,
            "transfer": transfer,
        },
    }
    examples = {
        "mode": result["mode"],
        "note": result["claim_scope"],
        "examples": _examples(lc_cpu, bpe_cpu, args.example_limit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.examples.write_text(json.dumps(examples, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
