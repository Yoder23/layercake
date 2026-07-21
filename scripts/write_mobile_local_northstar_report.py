from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BREAKTHROUGH = RESULTS / "breakthrough_equal"


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _get(row: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    cur: Any = row
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def _failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def build_report(
    *,
    fair_neural: dict[str, Any] | None,
    game_domain: dict[str, Any] | None,
    cpu_resources: dict[str, Any] | None,
    exact_cpu_resources: dict[str, Any] | None = None,
    mobile_int8: dict[str, Any] | None,
    platform: dict[str, Any] | None,
) -> dict[str, Any]:
    artifact_bytes = float(_get(cpu_resources, "layercake.artifact_bytes", 0.0))
    peak_rss = float(_get(cpu_resources, "layercake.peak_rss_bytes", 0.0))
    generation_ratio = float(_get(cpu_resources, "metrics.generation_speed_ratio", 0.0))
    prefill_ratio = float(_get(cpu_resources, "metrics.prefill_speed_ratio", 0.0))
    int8_payload_bytes = float(_get(mobile_int8, "payload_bytes", 0.0))
    int8_generation_bps = float(
        _get(mobile_int8, "greedy_generation.bytes_per_second", 0.0)
    )

    best_candidate = _get(fair_neural, "best_current_candidate")
    best_fair = _get(fair_neural, f"candidates.{best_candidate}", {}) if best_candidate else {}
    fair_cpu_heldout_exact = float(_get(best_fair, "cpu_heldout.layercake_exact", 0.0))
    fair_gpu_heldout_exact = float(_get(best_fair, "gpu_heldout.layercake_exact", 0.0))
    fair_cpu_heldout_speed = float(
        _get(best_fair, "cpu_heldout.speed_ratio_layercake_over_transformer", 0.0)
    )
    fair_gpu_heldout_speed = float(
        _get(best_fair, "gpu_heldout.speed_ratio_layercake_over_transformer", 0.0)
    )

    game_cpu_speed = _ratio(
        float(_get(game_domain, "metrics.layercake_cpu.generation_bytes_per_second", 0.0)),
        float(_get(game_domain, "metrics.bpe_cpu.generation_bytes_per_second", 0.0)),
    )
    game_gpu_speed = _ratio(
        float(_get(game_domain, "metrics.layercake_gpu.generation_bytes_per_second", 0.0)),
        float(_get(game_domain, "metrics.bpe_gpu.generation_bytes_per_second", 0.0)),
    )

    gates = {
        "required_artifacts_present": all(
            item is not None
            for item in [fair_neural, game_domain, cpu_resources, mobile_int8, platform]
        ),
        "small_deployment_artifact_under_64mb": 0 < artifact_bytes <= 64 * 1024 * 1024,
        "desktop_peak_rss_under_512mb": 0 < peak_rss <= 512 * 1024 * 1024,
        "int8_domain_payload_under_1mb": 0 < int8_payload_bytes <= 1 * 1024 * 1024,
        "int8_cpu_generation_nontrivial": int8_generation_bps >= 512.0,
        "desktop_cpu_generation_faster_than_bpe": generation_ratio > 1.0,
        "desktop_cpu_prefill_faster_than_bpe": prefill_ratio > 1.0,
        "domain_runtime_cpu_5x": game_cpu_speed >= 5.0,
        "domain_runtime_gpu_5x": game_gpu_speed >= 5.0,
        "domain_runtime_quality_full": bool(
            _get(game_domain, "gates.layercake_cpu_relevance_full", False)
            and _get(game_domain, "gates.layercake_gpu_relevance_full", False)
            and _get(game_domain, "gates.lossless_game_layer_transfer", False)
        ),
        "fair_neural_cpu_5x_speed": fair_cpu_heldout_speed >= 5.0,
        "fair_neural_gpu_5x_speed": fair_gpu_heldout_speed >= 5.0,
        "fair_neural_high_exact_quality": (
            fair_cpu_heldout_exact >= 0.95 and fair_gpu_heldout_exact >= 0.95
        ),
        "platform_matrix_pass": _get(platform, "status") == "PASS",
        "real_phone_hardware_evidence": False,
    }
    failed = _failed(gates)
    return {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "Mobile-local north-star gate for LayerCake. Desktop CPU and CUDA "
            "measurements are useful proxies, but this report refuses a phone "
            "dominance claim without real phone hardware evidence. Domain-layer "
            "runtime wins and fair-neural LLM wins are tracked separately."
        ),
        "claim_boundaries": {
        "valid_today": [
                "LayerCake has small desktop deployment artifacts in the measured 15M-class proxy.",
                "LayerCake has strong domain-layer runtime results on the game companion benchmark.",
            "LayerCake has fair-neural schema/action candidates with 5x CPU/GPU speed but insufficient exact heldout quality.",
            ],
            "not_yet_valid": [
                "Universal dominance over any local phone LLM.",
                "Fair-neural high-quality LLM dominance on heldout transduction.",
                "Android/iOS thermal, battery, NPU, or sustained latency claims.",
            ],
        },
        "gates": gates,
        "failed": failed,
        "metrics": {
            "desktop_layercake_artifact_bytes": artifact_bytes,
            "desktop_layercake_peak_rss_bytes": peak_rss,
            "desktop_cpu_generation_speed_ratio_over_bpe": generation_ratio,
            "desktop_cpu_prefill_speed_ratio_over_bpe": prefill_ratio,
            "desktop_exact_prefill_speed_ratio_over_bpe": float(
                _get(exact_cpu_resources, "metrics.prefill_speed_ratio", prefill_ratio)
            ),
            "desktop_fast_prefill_active": bool(
                _get(cpu_resources, "layercake_deployment_mode.fast_prefill_active", False)
            ),
            "int8_domain_payload_bytes": int8_payload_bytes,
            "int8_domain_generation_bytes_per_second": int8_generation_bps,
            "game_domain_cpu_speed_ratio_over_bpe": game_cpu_speed,
            "game_domain_gpu_speed_ratio_over_bpe": game_gpu_speed,
            "best_fair_neural_candidate": best_candidate,
            "fair_neural_cpu_heldout_exact": fair_cpu_heldout_exact,
            "fair_neural_gpu_heldout_exact": fair_gpu_heldout_exact,
            "fair_neural_cpu_heldout_speed_ratio": fair_cpu_heldout_speed,
            "fair_neural_gpu_heldout_speed_ratio": fair_gpu_heldout_speed,
        },
        "next_architecture_targets": [
            (
                "Keep fast aligned prefill as an explicit deployment mode and validate "
                "quality impact; exact prefill remains below BPE on the desktop proxy."
                if prefill_ratio > 1.0
                else "Reduce prefill latency by caching or compressing global-core prefill; current desktop prefill ratio is below 1.0."
            ),
            "Continue argmax/pointer-copy work until fair-neural heldout exact quality reaches at least 0.95.",
            "Add real Android or iOS benchmark harness before claiming phone dominance.",
            "Keep domain-layer runtime as product path, but do not use it as proof of fair-neural LLM dominance.",
        ],
        "artifacts": {
            "fair_neural": "results/breakthrough_equal/schema_action_fair_neural_candidate_report.json",
            "game_domain": "results/breakthrough_equal/game_companion_scale_report.json",
            "cpu_resources": "results/cpu_deployment_resources_certificate.json",
            "cpu_resources_fast_prefill": "results/cpu_deployment_resources_fast_prefill_certificate.json",
            "mobile_int8_proxy": "results/portable_domain_mobile_cpu_int8.json",
            "platform_matrix": "results/platform_benchmark_dominance_certificate.json",
        },
    }


def main() -> int:
    fast_cpu_resources = _read(RESULTS / "cpu_deployment_resources_fast_prefill_certificate.json")
    exact_cpu_resources = _read(RESULTS / "cpu_deployment_resources_certificate.json")
    report = build_report(
        fair_neural=_read(BREAKTHROUGH / "schema_action_fair_neural_candidate_report.json"),
        game_domain=_read(BREAKTHROUGH / "game_companion_scale_report.json"),
        cpu_resources=fast_cpu_resources or exact_cpu_resources,
        exact_cpu_resources=exact_cpu_resources,
        mobile_int8=_read(RESULTS / "portable_domain_mobile_cpu_int8.json"),
        platform=_read(RESULTS / "platform_benchmark_dominance_certificate.json"),
    )
    out = RESULTS / "mobile_local_northstar_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
