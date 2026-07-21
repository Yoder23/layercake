from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> int:
    northstar = _load("northstar_mobile_certificate.json")
    transition = _load("scale15m_transition_frontier_certificate.json")
    receiver = _load("receiver_frontier_certificate.json")
    cpu_resources = _load("cpu_deployment_resources_certificate.json")
    mobile_domain = _load("mobile_domain_win_certificate.json")
    transfer = _load("transfer_matrix_v2.json")

    missing = [
        name
        for name, value in {
            "northstar_mobile_certificate.json": northstar,
            "scale15m_transition_frontier_certificate.json": transition,
            "receiver_frontier_certificate.json": receiver,
            "cpu_deployment_resources_certificate.json": cpu_resources,
            "mobile_domain_win_certificate.json": mobile_domain,
            "transfer_matrix_v2.json": transfer,
        }.items()
        if value is None
    ]
    if missing:
        result = {
            "status": "FAIL",
            "scope": "Platform matrix dominance gate across CPU/GPU/mobile and end-to-end benchmark dimensions.",
            "missing_artifacts": missing,
            "gates": {"required_artifacts_present": False},
            "failed": ["required_artifacts_present"],
        }
        out = RESULTS / "platform_benchmark_dominance_certificate.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    ns_gates = northstar.get("required_gates", {})
    tr_gates = transition.get("required_gates", {})
    rc_gates = receiver.get("required_gates", {})
    cr_gates = cpu_resources.get("required_gates", {})
    md_gates = mobile_domain.get("required_gates", {})
    md_targets = mobile_domain.get("research_targets", {})

    transfer_all_exact = bool(transfer.get("summary", {}).get("all_exact", False))

    gpu_ratio_ns = _as_float(northstar.get("metrics", {}).get("gpu_generation_speed_ratio"))
    gpu_lc_bps = _as_float(mobile_domain.get("metrics", {}).get("layercake_gpu_bytes_per_second"))
    gpu_tf_bps = _as_float(mobile_domain.get("metrics", {}).get("transformer_adapter_gpu_bytes_per_second"))

    cpu_prefill_ratio = _as_float(cpu_resources.get("metrics", {}).get("prefill_speed_ratio"))
    cpu_gen_ratio = _as_float(cpu_resources.get("metrics", {}).get("generation_speed_ratio"))

    gates = {
        "cpu_quality_dominance": bool(
            ns_gates.get("core_quality_seed6250")
            and ns_gates.get("core_quality_seed6263")
            and tr_gates.get("source_at_least_0_5pct_better_general_bpb")
        ),
        "cpu_training_efficiency_dominance": bool(
            ns_gates.get("mean_training_time")
            and ns_gates.get("lower_domain_training_time")
            and tr_gates.get("source_at_least_1pct_faster_training")
        ),
        "cpu_generation_dominance": bool(
            ns_gates.get("faster_cpu_generation_seed6250")
            and ns_gates.get("faster_cpu_generation_seed6263")
            and ns_gates.get("faster_cpu_generation_norepeat8")
            and cr_gates.get("layercake_generation_faster_than_bpe")
            and rc_gates.get("receiver_faster_cpu_generation")
        ),
        "cpu_prefill_latency_dominance": bool(
            ns_gates.get("faster_batch1_prefill")
            and cr_gates.get("layercake_prefill_faster_than_bpe")
        ),
        "memory_artifact_dominance": bool(
            ns_gates.get("smaller_model")
            and ns_gates.get("smaller_domain_payload")
            and cr_gates.get("layercake_parameter_memory_lower_than_bpe")
            and cr_gates.get("layercake_artifact_smaller_than_bpe")
            and cr_gates.get("layercake_peak_rss_no_more_than_bpe")
        ),
        "transfer_exactness_dominance": bool(
            ns_gates.get("lossless_migration_seed6250")
            and ns_gates.get("lossless_migration_seed6263")
            and tr_gates.get("transfer_generation_exact")
            and tr_gates.get("transfer_max_logit_diff_exact")
            and tr_gates.get("transfer_ppl_ratio_exact")
            and transfer_all_exact
        ),
        "mobile_proxy_dominance": bool(
            md_gates.get("better_domain_bpb")
            and md_gates.get("lower_training_wall_time")
            and md_gates.get("smaller_deployment_artifact")
            and md_gates.get("faster_single_thread_cpu")
            and md_gates.get("exact_cross_host_transfer")
        ),
        "gpu_generation_dominance": bool(
            (gpu_ratio_ns is not None and gpu_ratio_ns > 1.0)
            and (gpu_lc_bps is not None and gpu_tf_bps is not None and gpu_lc_bps > gpu_tf_bps)
        ),
        "gpu_prefill_latency_dominance": bool(md_targets.get("gpu_prefill_win", False)),
        "real_mobile_hardware_evidence": bool(md_targets.get("real_mobile_hardware", False)),
    }

    failed = _failed(gates)

    gaps: dict[str, Any] = {
        "cpu_prefill_speed_ratio_target_minus_current": None,
        "gpu_generation_ratio_target_minus_current": None,
        "gpu_prefill_evidence_gap": None,
        "real_mobile_hardware_gap": None,
    }
    if cpu_prefill_ratio is not None:
        gaps["cpu_prefill_speed_ratio_target_minus_current"] = 1.0 - cpu_prefill_ratio
    if gpu_ratio_ns is not None:
        gaps["gpu_generation_ratio_target_minus_current"] = 1.0 - gpu_ratio_ns
    gaps["gpu_prefill_evidence_gap"] = 0 if md_targets.get("gpu_prefill_win", False) else 1
    gaps["real_mobile_hardware_gap"] = 0 if md_targets.get("real_mobile_hardware", False) else 1

    result = {
        "status": "PASS" if not failed else "FAIL",
        "scope": "Platform matrix dominance gate across CPU/GPU/mobile and end-to-end benchmark dimensions.",
        "gates": gates,
        "failed": failed,
        "metrics": {
            "cpu_generation_speed_ratio": cpu_gen_ratio,
            "cpu_prefill_speed_ratio": cpu_prefill_ratio,
            "gpu_generation_speed_ratio_northstar": gpu_ratio_ns,
            "gpu_layercake_bytes_per_second": gpu_lc_bps,
            "gpu_transformer_bytes_per_second": gpu_tf_bps,
            "transfer_artifact_count": transfer.get("summary", {}).get("artifact_count", 0),
            "transfer_all_exact": transfer_all_exact,
        },
        "gaps": gaps,
        "artifacts": {
            "northstar": "results/northstar_mobile_certificate.json",
            "transition_15m": "results/scale15m_transition_frontier_certificate.json",
            "receiver": "results/receiver_frontier_certificate.json",
            "cpu_resources": "results/cpu_deployment_resources_certificate.json",
            "mobile_domain": "results/mobile_domain_win_certificate.json",
            "transfer": "results/transfer_matrix_v2.json",
        },
    }

    out = RESULTS / "platform_benchmark_dominance_certificate.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
