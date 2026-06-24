from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, value in gates.items() if not value]


def main() -> int:
    transition = load("results/scale15m_transition_frontier_certificate.json")
    cpu_quality = load("results/scale15m_transition_generation_quality_cpu1_norepeat4.json")
    gpu_quality = load("results/scale15m_transition_generation_quality_gpu_norepeat4.json")
    game_ready = load("results/game_ready_mobile_llm_certificate.json")
    bpe_prefill = load("results/scale15m_bpe_matched_inference_batch1.json")
    northstar_prefill = load("results/scale15m_northstar_arpatch2_inference_batch1.json")
    domain = load("results/mobile_domain_win_certificate.json")

    transition_metrics = transition["metrics"]
    patch_prefill = next(
        row for row in northstar_prefill["rows"] if row["path"] == "patch_base"
    )
    layercake_prefill_ms = patch_prefill["seconds"] * 1000 / northstar_prefill["iterations"]
    cpu_gates = {
        "cpu_generation_faster_than_bpe": (
            cpu_quality["layercake"]["bytes_per_second"]
            > cpu_quality["bpe"]["bytes_per_second"]
        ),
        "cpu_generation_quality_gates_pass": all(cpu_quality["quality_gates"].values()),
        "cpu_generation_at_least_2x_bpe": (
            cpu_quality["layercake"]["bytes_per_second"]
            >= 2.0 * cpu_quality["bpe"]["bytes_per_second"]
        ),
    }
    gpu_gates = {
        "gpu_generation_faster_than_bpe": (
            gpu_quality["layercake"]["bytes_per_second"]
            > gpu_quality["bpe"]["bytes_per_second"]
        ),
        "gpu_generation_quality_gates_pass": all(gpu_quality["quality_gates"].values()),
    }
    latency_gates = {
        "batch1_prefill_faster_than_bpe": layercake_prefill_ms < bpe_prefill["median_ms"],
    }
    training_gates = {
        "lower_bpb_than_bpe": transition_metrics["layercake_general_bpb"] < transition_metrics["bpe_general_bpb"],
        "faster_training_than_bpe": transition_metrics["layercake_training_seconds"] < transition_metrics["bpe_training_seconds"],
        "fewer_training_bytes_than_bpe": transition_metrics["layercake_training_bytes"] <= transition_metrics["bpe_training_bytes"],
        "smaller_core_than_bpe": transition_metrics["layercake_parameters"] < transition_metrics["bpe_parameters"],
    }
    domain_gates = {
        "domain_quality_beats_adapter": domain["required_gates"]["better_domain_bpb"],
        "domain_training_faster_than_adapter": domain["required_gates"]["lower_training_wall_time"],
        "domain_payload_smaller_than_adapter": domain["required_gates"]["smaller_deployment_artifact"],
        "domain_cpu_faster_than_adapter": domain["required_gates"]["faster_single_thread_cpu"],
        "domain_transfer_exact": domain["required_gates"]["exact_cross_host_transfer"],
    }
    promoted = {
        "training_quality_cost": {
            "status": "PASS" if not failed(training_gates) else "FAIL",
            "required_gates": training_gates,
            "failed": failed(training_gates),
        },
        "cpu_generation": {
            "status": "PASS" if not failed(cpu_gates) else "FAIL",
            "required_gates": cpu_gates,
            "failed": failed(cpu_gates),
        },
        "latency": {
            "status": "PASS" if not failed(latency_gates) else "FAIL",
            "required_gates": latency_gates,
            "failed": failed(latency_gates),
        },
        "domain_layers": {
            "status": "PASS" if not failed(domain_gates) else "FAIL",
            "required_gates": domain_gates,
            "failed": failed(domain_gates),
        },
        "game_ready_cpu_mobile_proxy": {
            "status": game_ready["status"],
            "required_gates": game_ready["required_gates"],
            "failed": game_ready["failed_required"],
        },
    }
    open_or_failing = {
        "gpu_generation": {
            "status": "OPEN" if failed(gpu_gates) else "PASS",
            "required_gates": gpu_gates,
            "failed": failed(gpu_gates),
            "reason": (
                "GPU generation quality gates pass, but LayerCake generation is slower than BPE "
                "on the retained CUDA benchmark. This blocks any across-the-board GPU claim."
                if failed(gpu_gates)
                else ""
            ),
        }
    }
    promoted_failures = {
        name: group["failed"]
        for name, group in promoted.items()
        if group["status"] != "PASS" or group["failed"]
    }
    result = {
        "status": "PASS" if not promoted_failures else "FAIL",
        "claim_boundary": (
            "PASS means promoted CPU/mobile-proxy, training, latency, and domain gates pass. "
            "OPEN GPU generation prevents any across-the-board CPU+GPU dominance claim."
        ),
        "promoted_scorecard": promoted,
        "open_or_failing_scorecard": open_or_failing,
        "failed_promoted_groups": promoted_failures,
        "metrics": {
            "cpu_layercake_bytes_per_second": cpu_quality["layercake"]["bytes_per_second"],
            "cpu_bpe_bytes_per_second": cpu_quality["bpe"]["bytes_per_second"],
            "gpu_layercake_bytes_per_second": gpu_quality["layercake"]["bytes_per_second"],
            "gpu_bpe_bytes_per_second": gpu_quality["bpe"]["bytes_per_second"],
            "layercake_prefill_ms": layercake_prefill_ms,
            "bpe_prefill_ms": bpe_prefill["median_ms"],
            "layercake_generation_sample_cpu": cpu_quality["layercake"]["utf8"],
            "bpe_generation_sample_cpu": cpu_quality["bpe"]["utf8"],
        },
    }
    output = RESULTS / "cross_backend_quality_scorecard.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
