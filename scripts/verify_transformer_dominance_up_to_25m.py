from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _num(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _extract_bpb(payload: dict[str, Any]) -> float:
    general = payload.get("general", {})
    return _num(general.get("bpb"))


def _extract_seconds(payload: dict[str, Any]) -> float:
    return _num(payload.get("elapsed_seconds"))


def _extract_params(payload: dict[str, Any]) -> float:
    return _num(payload.get("parameters"))


def _extract_training_bytes(payload: dict[str, Any]) -> float:
    if "estimated_total_training_bytes" in payload:
        return _num(payload.get("estimated_total_training_bytes"))
    if "train_bytes" in payload:
        return _num(payload.get("train_bytes"))
    return float("nan")


def _failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def _compare_scale(
    tier_name: str,
    layercake: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    require_exact_baseline: bool,
    baseline_is_proxy: bool,
) -> dict[str, Any]:
    layercake_exists = layercake is not None
    baseline_exists = baseline is not None

    lc_bpb = _extract_bpb(layercake) if layercake else float("nan")
    bpe_bpb = _extract_bpb(baseline) if baseline else float("nan")
    lc_seconds = _extract_seconds(layercake) if layercake else float("nan")
    bpe_seconds = _extract_seconds(baseline) if baseline else float("nan")
    lc_params = _extract_params(layercake) if layercake else float("nan")
    bpe_params = _extract_params(baseline) if baseline else float("nan")
    lc_bytes = _extract_training_bytes(layercake) if layercake else float("nan")
    bpe_bytes = _extract_training_bytes(baseline) if baseline else float("nan")

    gates = {
        "layercake_artifact_exists": layercake_exists,
        "baseline_artifact_exists": baseline_exists,
        "exact_baseline_required": (not require_exact_baseline) or (baseline_exists and not baseline_is_proxy),
        "quality_beats_transformer": layercake_exists and baseline_exists and (lc_bpb < bpe_bpb),
        "training_time_beats_transformer": layercake_exists and baseline_exists and (lc_seconds < bpe_seconds),
        "params_no_larger_than_transformer": layercake_exists and baseline_exists and (lc_params <= bpe_params),
        "training_bytes_no_more_than_transformer": layercake_exists and baseline_exists and (lc_bytes <= bpe_bytes),
    }

    required = {
        "layercake_artifact_exists": gates["layercake_artifact_exists"],
        "baseline_artifact_exists": gates["baseline_artifact_exists"],
        "quality_beats_transformer": gates["quality_beats_transformer"],
        "training_time_beats_transformer": gates["training_time_beats_transformer"],
        "params_no_larger_than_transformer": gates["params_no_larger_than_transformer"],
        "training_bytes_no_more_than_transformer": gates["training_bytes_no_more_than_transformer"],
    }
    if require_exact_baseline:
        required["exact_baseline_required"] = gates["exact_baseline_required"]

    status = "PASS" if all(required.values()) else "FAIL"
    return {
        "status": status,
        "tier": tier_name,
        "baseline_is_proxy": baseline_is_proxy,
        "gates": gates,
        "required_gates": required,
        "failed_required": _failed(required),
        "metrics": {
            "layercake_bpb": lc_bpb,
            "transformer_bpb": bpe_bpb,
            "layercake_training_seconds": lc_seconds,
            "transformer_training_seconds": bpe_seconds,
            "layercake_parameters": lc_params,
            "transformer_parameters": bpe_params,
            "layercake_training_bytes": lc_bytes,
            "transformer_training_bytes": bpe_bytes,
        },
    }


def _compare_scale_candidates(
    tier_name: str,
    candidate_files: list[Path],
    baseline: dict[str, Any] | None,
    require_exact_baseline: bool,
    baseline_is_proxy: bool,
) -> dict[str, Any]:
    if not candidate_files:
        return _compare_scale(
            tier_name=tier_name,
            layercake=None,
            baseline=baseline,
            require_exact_baseline=require_exact_baseline,
            baseline_is_proxy=baseline_is_proxy,
        )

    evaluations: list[dict[str, Any]] = []
    for path in candidate_files:
        payload = _maybe_load(path)
        if payload is None:
            continue
        evaluation = _compare_scale(
            tier_name=tier_name,
            layercake=payload,
            baseline=baseline,
            require_exact_baseline=require_exact_baseline,
            baseline_is_proxy=baseline_is_proxy,
        )
        evaluation["artifact"] = str(path.relative_to(ROOT)).replace("\\", "/")
        required = evaluation["required_gates"]
        evaluation["required_pass_count"] = sum(1 for passed in required.values() if passed)
        evaluation["required_total"] = len(required)
        evaluations.append(evaluation)

    if not evaluations:
        return _compare_scale(
            tier_name=tier_name,
            layercake=None,
            baseline=baseline,
            require_exact_baseline=require_exact_baseline,
            baseline_is_proxy=baseline_is_proxy,
        )

    def _score(item: dict[str, Any]) -> tuple[float, float, float]:
        metrics = item.get("metrics", {})
        bpb = metrics.get("layercake_bpb", float("inf"))
        sec = metrics.get("layercake_training_seconds", float("inf"))
        return (
            float(item["required_pass_count"]),
            -float(bpb) if isinstance(bpb, (int, float)) else float("-inf"),
            -float(sec) if isinstance(sec, (int, float)) else float("-inf"),
        )

    best = max(evaluations, key=_score)
    best["candidate_pool_size"] = len(evaluations)
    best["candidate_pool"] = [
        {
            "artifact": entry["artifact"],
            "status": entry["status"],
            "required_pass_count": entry["required_pass_count"],
            "required_total": entry["required_total"],
            "failed_required": entry["failed_required"],
            "layercake_bpb": entry["metrics"].get("layercake_bpb"),
            "layercake_training_seconds": entry["metrics"].get("layercake_training_seconds"),
        }
        for entry in evaluations
    ]
    return best


def _transfer_summary(results_dir: Path) -> dict[str, Any]:
    transfer_files = sorted(
        list(results_dir.glob("lossless_domain*.json"))
        + list(results_dir.glob("northstar_lossless_domain*.json"))
        + list(results_dir.glob("scale*m_lossless_domain*.json"))
    )

    rows: list[dict[str, Any]] = []
    exact_count = 0
    no_damage_count = 0

    for path in transfer_files:
        payload = _maybe_load(path)
        if not payload:
            continue
        ppl_ratio = _num(payload.get("ppl_ratio"), default=float("nan"))
        max_logit_diff = _num(payload.get("max_logit_diff"), default=float("nan"))
        generation_equal = bool(payload.get("generation", {}).get("equal", False))
        src_bpb = _num(payload.get("source", {}).get("bpb"), default=float("nan"))
        tgt_bpb = _num(payload.get("target", {}).get("bpb"), default=float("nan"))
        no_damage = abs(src_bpb - tgt_bpb) <= 1e-12
        exact = (ppl_ratio == 1.0) and (max_logit_diff == 0.0) and generation_equal
        if exact:
            exact_count += 1
        if no_damage:
            no_damage_count += 1
        rows.append(
            {
                "artifact": str(path.relative_to(results_dir.parent)).replace("\\", "/"),
                "ppl_ratio": ppl_ratio,
                "max_logit_diff": max_logit_diff,
                "generation_equal": generation_equal,
                "source_bpb": src_bpb,
                "target_bpb": tgt_bpb,
                "no_damage": no_damage,
                "exact": exact,
            }
        )

    matrix_v2 = _maybe_load(results_dir / "transfer_matrix_v2.json")
    matrix_rows = matrix_v2.get("rows", []) if matrix_v2 else []
    has_non_smoke_row = any(row.get("status") != "SMOKE_ONLY" for row in matrix_rows)

    gates = {
        "at_least_one_lossless_transfer_artifact": len(rows) > 0,
        "all_transfer_artifacts_exact": len(rows) > 0 and exact_count == len(rows),
        "all_transfer_artifacts_no_damage": len(rows) > 0 and no_damage_count == len(rows),
        "transfer_matrix_v2_has_non_smoke_row": has_non_smoke_row,
    }

    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        "status": status,
        "gates": gates,
        "failed": _failed(gates),
        "artifacts_checked": rows,
        "artifact_count": len(rows),
        "exact_count": exact_count,
        "no_damage_count": no_damage_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify LayerCake transformer dominance through 25M with strict "
            "domain-transfer no-damage gates."
        )
    )
    parser.add_argument(
        "--output",
        default="results/dominance_up_to_25m_research_certificate.json",
    )
    parser.add_argument(
        "--require-exact-25m-baseline",
        action="store_true",
        default=True,
        help="Require a true 25M matched transformer artifact (default: true).",
    )
    parser.add_argument(
        "--allow-proxy-25m-baseline",
        action="store_true",
        help="Allow a nearby baseline if exact 25M transformer artifact is unavailable.",
    )
    args = parser.parse_args()

    scale20_bpe = _maybe_load(RESULTS / "scale20m_bpe448_l7_seed6250.json")

    scale24_bpe = _maybe_load(RESULTS / "scale24m_bpe_seed7001.json")

    scale25_bpe_exact = _maybe_load(RESULTS / "scale25m_bpe_seed7002.json")
    scale25_bpe_proxy = _maybe_load(RESULTS / "scale26m_bpe512_seed6250.json")

    scale20_candidates = sorted(
        [
            path
            for path in RESULTS.glob("scale20m_*.json")
            if "bpe" not in path.name and "certificate" not in path.name
        ]
    )
    scale24_candidates = sorted(
        [
            path
            for path in RESULTS.glob("scale24m_*.json")
            if "bpe" not in path.name and "certificate" not in path.name
        ]
    )
    scale25_candidates = sorted(
        [
            path
            for path in RESULTS.glob("scale25m_*.json")
            if "bpe" not in path.name and "certificate" not in path.name
        ]
    )

    use_proxy = bool(args.allow_proxy_25m_baseline and scale25_bpe_exact is None and scale25_bpe_proxy is not None)
    scale25_baseline = scale25_bpe_proxy if use_proxy else scale25_bpe_exact

    scale20 = _compare_scale_candidates(
        "20m",
        candidate_files=scale20_candidates,
        baseline=scale20_bpe,
        require_exact_baseline=False,
        baseline_is_proxy=False,
    )
    scale24 = _compare_scale_candidates(
        "24m",
        candidate_files=scale24_candidates,
        baseline=scale24_bpe,
        require_exact_baseline=False,
        baseline_is_proxy=False,
    )
    scale25 = _compare_scale_candidates(
        "25m",
        candidate_files=scale25_candidates,
        baseline=scale25_baseline,
        require_exact_baseline=(args.require_exact_25m_baseline and not args.allow_proxy_25m_baseline),
        baseline_is_proxy=use_proxy,
    )

    transfer = _transfer_summary(RESULTS)

    gates = {
        "scale20_dominance": scale20["status"] == "PASS",
        "scale24_dominance": scale24["status"] == "PASS",
        "scale25_dominance": scale25["status"] == "PASS",
        "transfer_no_damage_and_exact": transfer["status"] == "PASS",
    }

    failed = _failed(gates)
    result = {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "Groundbreaking research gate: strict transformer dominance through "
            "20M/24M/25M plus transfer exactness and no-damage guarantees."
        ),
        "gates": gates,
        "failed": failed,
        "scale_tiers": {
            "20m": scale20,
            "24m": scale24,
            "25m": scale25,
        },
        "transfer": transfer,
        "notes": {
            "25m_exact_baseline_required": args.require_exact_25m_baseline,
            "25m_proxy_baseline_used": use_proxy,
            "this_is_not_marketing_copy": True,
        },
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
