from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_SCALES = ("1m", "2m", "5m", "10m")


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _qa_metrics(row: dict[str, Any]) -> dict[str, float]:
    samples = row.get("qa_samples", [])
    lc_quality: list[float] = []
    bpe_quality: list[float] = []
    lc_repeat: list[float] = []
    bpe_repeat: list[float] = []
    lc_keywords: list[float] = []
    bpe_keywords: list[float] = []
    lc_alpha: list[float] = []
    bpe_alpha: list[float] = []
    for sample in samples:
        lc = sample.get("layercake", {})
        bpe = sample.get("baseline", {})
        lc_quality.append(float(lc.get("quality_score", 0.0)))
        bpe_quality.append(float(bpe.get("quality_score", 0.0)))
        lc_repeat.append(float(lc.get("max_token_repeat", 1e9)))
        bpe_repeat.append(float(bpe.get("max_token_repeat", 1e9)))
        lc_keywords.append(float(lc.get("keyword_score", 0.0)))
        bpe_keywords.append(float(bpe.get("keyword_score", 0.0)))
        lc_alpha.append(float(lc.get("alpha_ratio", 0.0)))
        bpe_alpha.append(float(bpe.get("alpha_ratio", 0.0)))
    return {
        "layercake_quality_mean": _mean(lc_quality),
        "baseline_quality_mean": _mean(bpe_quality),
        "layercake_repeat_mean": _mean(lc_repeat),
        "baseline_repeat_mean": _mean(bpe_repeat),
        "layercake_keyword_mean": _mean(lc_keywords),
        "baseline_keyword_mean": _mean(bpe_keywords),
        "layercake_alpha_mean": _mean(lc_alpha),
        "baseline_alpha_mean": _mean(bpe_alpha),
    }


def _scale_gate(row: dict[str, Any]) -> dict[str, Any]:
    lc = row.get("layercake", {})
    bpe = row.get("baseline", {})
    lc_train = lc.get("train", {})
    bpe_train = bpe.get("train", {})
    cost = row.get("cost_proxy_param_seconds", {})
    qa = _qa_metrics(row)

    lc_raw_seconds = float(lc_train.get("elapsed_seconds", 1e18))
    bpe_raw_seconds = float(bpe_train.get("elapsed_seconds", 0.0))
    bpe_total_seconds = float(
        bpe_train.get("elapsed_total_seconds", bpe_raw_seconds)
    )
    lc_bpb = float(lc.get("general_bpb", 1e18))
    bpe_bpb = float(bpe.get("general_bpb", 0.0))
    lc_params = int(lc.get("params", 10**18))
    bpe_params = int(bpe.get("params", 0))
    lc_cost = float(cost.get("layercake", 1e30))
    bpe_cost = float(cost.get("baseline", 0.0))

    gates = {
        "artifact_scale_status_pass": row.get("status") == "PASS",
        "params_no_larger": lc_params <= bpe_params,
        "bpb_strictly_lower": lc_bpb < bpe_bpb,
        "raw_training_faster_excluding_tokenizer": lc_raw_seconds < bpe_raw_seconds,
        "training_faster_including_tokenizer": lc_raw_seconds < bpe_total_seconds,
        "cost_proxy_lower": lc_cost < bpe_cost,
        "generation_quality_noninferior": qa["layercake_quality_mean"]
        >= qa["baseline_quality_mean"],
        "generation_repetition_no_worse": qa["layercake_repeat_mean"]
        <= qa["baseline_repeat_mean"],
        "generation_keyword_no_worse": qa["layercake_keyword_mean"]
        >= qa["baseline_keyword_mean"],
        "generation_alpha_no_worse": qa["layercake_alpha_mean"]
        >= qa["baseline_alpha_mean"],
    }
    return {
        "scale": row.get("scale"),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "failed": _failed(gates),
        "metrics": {
            "layercake_params": lc_params,
            "baseline_params": bpe_params,
            "layercake_bpb": lc_bpb,
            "baseline_bpb": bpe_bpb,
            "layercake_raw_train_seconds": lc_raw_seconds,
            "baseline_raw_train_seconds": bpe_raw_seconds,
            "baseline_tokenizer_prep_seconds": float(
                bpe_train.get("prep_seconds", 0.0)
            ),
            "baseline_total_train_seconds": bpe_total_seconds,
            "layercake_cost_proxy_param_seconds": lc_cost,
            "baseline_cost_proxy_param_seconds": bpe_cost,
            **qa,
        },
    }


def verify(
    artifact: Path,
    output: Path,
    min_train_bytes: int,
    min_eval_bytes: int,
    min_steps: int,
) -> dict[str, Any]:
    data = _load(artifact)
    rows = {row.get("scale"): row for row in data.get("scales", [])}
    scale_results = [
        _scale_gate(rows[scale]) if scale in rows else {
            "scale": scale,
            "status": "FAIL",
            "gates": {"scale_artifact_exists": False},
            "failed": ["scale_artifact_exists"],
            "metrics": {},
        }
        for scale in REQUIRED_SCALES
    ]
    evidence_gates = {
        "artifact_status_pass": data.get("status") == "PASS",
        "has_required_scales": all(scale in rows for scale in REQUIRED_SCALES),
        "train_bytes_at_least_minimum": int(data.get("train_bytes", 0))
        >= min_train_bytes,
        "eval_bytes_at_least_minimum": int(data.get("eval_bytes", 0))
        >= min_eval_bytes,
        "steps_at_least_minimum": int(data.get("steps", 0)) >= min_steps,
    }
    gates = {
        **evidence_gates,
        "all_scales_strict_pass": all(
            row["status"] == "PASS" for row in scale_results
        ),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": (
            "Strict 1M-10M LayerCake-vs-tokenizer-transformer dominance. "
            "PASS requires every scale to independently win params, BPB, raw "
            "training speed, tokenizer-inclusive training speed, cost proxy, "
            "and generation-quality/repetition gates."
        ),
        "artifact": str(artifact.relative_to(ROOT) if artifact.is_relative_to(ROOT) else artifact),
        "gates": gates,
        "failed": _failed(gates),
        "minimums": {
            "train_bytes": min_train_bytes,
            "eval_bytes": min_eval_bytes,
            "steps": min_steps,
        },
        "observed": {
            "train_bytes": data.get("train_bytes"),
            "eval_bytes": data.get("eval_bytes"),
            "steps": data.get("steps"),
            "device": data.get("device"),
        },
        "scales": scale_results,
        "claim_boundary": (
            "FAIL means LayerCake cannot be marketed as making tokenized "
            "transformers obsolete at 1M-10M on this evidence."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default="results/micro_scale_curriculum_frontier_v2.json",
    )
    parser.add_argument(
        "--output",
        default="results/micro_1m10m_strict_dominance_certificate.json",
    )
    parser.add_argument("--min-train-bytes", type=int, default=1_000_000)
    parser.add_argument("--min-eval-bytes", type=int, default=100_000)
    parser.add_argument("--min-steps", type=int, default=500)
    args = parser.parse_args()

    artifact = Path(args.artifact)
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    result = verify(
        artifact=artifact,
        output=output,
        min_train_bytes=args.min_train_bytes,
        min_eval_bytes=args.min_eval_bytes,
        min_steps=args.min_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
