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


def _metrics(training: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": int(_get(training, "latest.trainable_params", "trainable_params", "params")),
        "bpb": float(_get(training, "latest.bpb", "latest.eval_bpb", "eval_bpb", "general_bpb", "bpb")),
        "train_seconds": float(
            _get(training, "latest.elapsed_seconds", "elapsed_seconds", "train.elapsed_seconds", "train_seconds")
        ),
        "train_bytes": int(_get(training, "latest.train_bytes", "train_bytes", "data.train_bytes", default=0)),
        "device": str(_get(generation, "device", "metrics.device", default="unknown")).lower(),
        "generation_bytes_per_second": float(
            _get(
                generation,
                "metrics.generation_bytes_per_second",
                "generation_bytes_per_second",
                "mean_bytes_per_second",
            )
        ),
        "quality_score": float(
            _get(generation, "metrics.quality_score", "quality_score", "qa_quality_mean", default=1.0)
        ),
        "first_token_p95_ms": _get(generation, "metrics.first_token_p95_ms", "first_token_p95_ms"),
        "response_p95_ms": _get(generation, "metrics.response_p95_ms", "response_p95_ms"),
    }


def verify(
    *,
    layercake_training: dict[str, Any],
    transformer_training: dict[str, Any],
    layercake_generation: dict[str, Any],
    transformer_generation: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lc = _metrics(layercake_training, layercake_generation)
    tx = _metrics(transformer_training, transformer_generation)
    param_ratio = _ratio(max(lc["params"], tx["params"]), min(lc["params"], tx["params"]))
    ratios = {
        "same_size_parameter_ratio": param_ratio,
        "bpb_ratio_layercake_over_transformer": _ratio(lc["bpb"], tx["bpb"]),
        "training_speed_ratio": _ratio(tx["train_seconds"], lc["train_seconds"]),
        "generation_speed_ratio": _ratio(lc["generation_bytes_per_second"], tx["generation_bytes_per_second"]),
        "quality_ratio": _ratio(lc["quality_score"], tx["quality_score"]),
    }
    gates = {
        "layercake_generation_is_cpu": lc["device"] == "cpu",
        "transformer_generation_is_cpu": tx["device"] == "cpu",
        "same_size_comparator": param_ratio <= args.max_same_size_param_ratio,
        "no_more_training_bytes": lc["train_bytes"] <= tx["train_bytes"] if lc["train_bytes"] and tx["train_bytes"] else True,
        "bpb_non_inferior": lc["bpb"] <= tx["bpb"],
        "training_speed_met": ratios["training_speed_ratio"] >= args.min_training_speed_ratio,
        "generation_speed_5x_met": ratios["generation_speed_ratio"] >= args.min_generation_speed_ratio,
        "quality_non_inferior": ratios["quality_ratio"] >= args.min_quality_ratio,
    }
    if args.max_first_token_p95_ms is not None:
        gates["first_token_p95_met"] = lc["first_token_p95_ms"] is not None and float(lc["first_token_p95_ms"]) <= args.max_first_token_p95_ms
    if args.max_response_p95_ms is not None:
        gates["response_p95_met"] = lc["response_p95_ms"] is not None and float(lc["response_p95_ms"]) <= args.max_response_p95_ms
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "Production CPU/game same-size LayerCake-vs-tokenizer-transformer dominance gate",
        "claim_boundary": (
            "Passing this gate is required before claiming 5x CPU same-size deployment speed "
            "with no quality loss. Short-run asymmetric probes do not satisfy this gate."
        ),
        "thresholds": {
            "max_same_size_param_ratio": args.max_same_size_param_ratio,
            "min_training_speed_ratio": args.min_training_speed_ratio,
            "min_generation_speed_ratio": args.min_generation_speed_ratio,
            "min_quality_ratio": args.min_quality_ratio,
            "max_first_token_p95_ms": args.max_first_token_p95_ms,
            "max_response_p95_ms": args.max_response_p95_ms,
        },
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake": lc,
            "transformer": tx,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify production CPU/game same-size LayerCake dominance")
    parser.add_argument("--layercake-training", required=True, type=Path)
    parser.add_argument("--transformer-training", required=True, type=Path)
    parser.add_argument("--layercake-generation", required=True, type=Path)
    parser.add_argument("--transformer-generation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-same-size-param-ratio", type=float, default=1.10)
    parser.add_argument("--min-training-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-generation-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    parser.add_argument("--max-first-token-p95-ms", type=float, default=None)
    parser.add_argument("--max-response-p95-ms", type=float, default=None)
    args = parser.parse_args()
    result = verify(
        layercake_training=_read(args.layercake_training),
        transformer_training=_read(args.transformer_training),
        layercake_generation=_read(args.layercake_generation),
        transformer_generation=_read(args.transformer_generation),
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
