from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def verify(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    lc = row["layercake"]
    bpe = row["baseline"]
    cost = row["cost_proxy_param_seconds"]
    lc_params = int(lc["params"])
    bpe_params = int(bpe["params"])
    lc_bpb = float(lc["general_bpb"])
    bpe_bpb = float(bpe["general_bpb"])
    lc_train = float(lc["train"]["elapsed_seconds"])
    bpe_raw_train = float(bpe["train"]["elapsed_seconds"])
    bpe_total_train = float(bpe["train"].get("elapsed_total_seconds", bpe_raw_train))
    lc_gen = float(lc["generation"]["mean_bytes_per_second"])
    bpe_gen = float(bpe["generation"]["mean_bytes_per_second"])
    lc_quality = float(lc["qa_quality_mean"])
    bpe_quality = float(bpe["qa_quality_mean"])
    ratios = {
        "parameter_ratio_transformer_over_layercake": _ratio(bpe_params, lc_params),
        "bpb_ratio_layercake_over_transformer": _ratio(lc_bpb, bpe_bpb),
        "raw_training_speed_ratio": _ratio(bpe_raw_train, lc_train),
        "total_training_speed_ratio": _ratio(bpe_total_train, lc_train),
        "cost_proxy_ratio": _ratio(float(cost["baseline"]), float(cost["layercake"])),
        "generation_speed_ratio": _ratio(lc_gen, bpe_gen),
        "quality_ratio": _ratio(lc_quality, bpe_quality),
    }
    gates = {
        "benchmark_status_pass": row.get("status") == "PASS",
        "layercake_within_param_cap": lc_params <= args.max_layercake_params,
        "transformer_meets_param_floor": bpe_params >= args.min_transformer_params,
        "parameter_ratio_met": ratios["parameter_ratio_transformer_over_layercake"] >= args.min_param_ratio,
        "bpb_lower": lc_bpb < bpe_bpb,
        "raw_training_speed_met": ratios["raw_training_speed_ratio"] >= args.min_raw_train_ratio,
        "total_training_speed_met": ratios["total_training_speed_ratio"] >= args.min_total_train_ratio,
        "cost_proxy_ratio_met": ratios["cost_proxy_ratio"] >= args.min_cost_ratio,
        "generation_speed_met": ratios["generation_speed_ratio"] >= args.min_generation_speed_ratio,
        "quality_ratio_met": ratios["quality_ratio"] >= args.min_quality_ratio,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "tier": row.get("tier"),
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake_params": lc_params,
            "transformer_params": bpe_params,
            "layercake_bpb": lc_bpb,
            "transformer_bpb": bpe_bpb,
            "layercake_train_seconds": lc_train,
            "transformer_raw_train_seconds": bpe_raw_train,
            "transformer_total_train_seconds": bpe_total_train,
            "layercake_generation_bytes_per_second": lc_gen,
            "transformer_generation_bytes_per_second": bpe_gen,
            "layercake_quality": lc_quality,
            "transformer_quality": bpe_quality,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify asymmetric LayerCake-vs-larger-transformer ladder artifacts")
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-layercake-params", type=float, default=1_000_000)
    parser.add_argument("--min-transformer-params", type=float, default=50_000_000)
    parser.add_argument("--min-param-ratio", type=float, default=50.0)
    parser.add_argument("--min-raw-train-ratio", type=float, default=1.0)
    parser.add_argument("--min-total-train-ratio", type=float, default=1.0)
    parser.add_argument("--min-cost-ratio", type=float, default=5.0)
    parser.add_argument("--min-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    args = parser.parse_args()
    row = json.loads(args.artifact.read_text(encoding="utf-8"))
    result = {
        "scope": "Asymmetric LayerCake-vs-larger-tokenizer-transformer certificate",
        "artifact": str(args.artifact),
        "thresholds": {
            "max_layercake_params": args.max_layercake_params,
            "min_transformer_params": args.min_transformer_params,
            "min_param_ratio": args.min_param_ratio,
            "min_raw_train_ratio": args.min_raw_train_ratio,
            "min_total_train_ratio": args.min_total_train_ratio,
            "min_cost_ratio": args.min_cost_ratio,
            "min_generation_speed_ratio": args.min_generation_speed_ratio,
            "min_quality_ratio": args.min_quality_ratio,
        },
        **verify(row, args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
