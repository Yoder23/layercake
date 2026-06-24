from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def main() -> int:
    transition = load("results/scale15m_transition_frontier_certificate.json")
    mobile_domain = load("results/mobile_domain_win_certificate.json")
    receiver = load("results/receiver_frontier_certificate.json")
    matrix = load("results/transformer_dominance_matrix.json")
    resources = load("results/cpu_deployment_resources_certificate.json")

    metrics = transition["metrics"]
    domain_metrics = mobile_domain["metrics"]
    receiver_metrics = receiver["metrics"]
    gates = {
        "core_frontier_promoted": transition["status"] == "PASS",
        "dominance_matrix_promoted_tiers_pass": matrix["status"] == "PASS",
        "cpu_generation_at_least_2x_bpe": metrics["cpu_generation_speed_ratio"] >= 2.0,
        "core_smaller_than_bpe": metrics["layercake_parameters"] < metrics["bpe_parameters"],
        "core_better_bpb_than_bpe": metrics["layercake_general_bpb"] < metrics["bpe_general_bpb"],
        "core_faster_training_than_bpe": (
            metrics["layercake_training_seconds"] < metrics["bpe_training_seconds"]
        ),
        "core_uses_fewer_training_bytes": (
            metrics["layercake_training_bytes"] <= metrics["bpe_training_bytes"]
        ),
        "english_generation_printable": metrics["generation_printable_rate"] >= 0.95,
        "english_generation_alpha_space": metrics["generation_alpha_space_rate"] >= 0.85,
        "english_generation_diverse": metrics["generation_distinct_trigram_rate"] >= 0.80,
        "english_generation_no_8gram_loop": not metrics["generation_has_repeated_8gram"],
        "domain_payload_smaller_than_adapter": (
            domain_metrics["layercake_artifact_bytes"]
            < domain_metrics["transformer_adapter_artifact_bytes"]
        ),
        "domain_training_faster_than_adapter": (
            domain_metrics["layercake_training_seconds"]
            < domain_metrics["transformer_adapter_training_seconds"]
        ),
        "domain_cpu_at_least_2x_adapter": (
            domain_metrics["layercake_cpu_bytes_per_second"]
            >= 2.0 * domain_metrics["transformer_adapter_cpu_bytes_per_second"]
        ),
        "domain_quality_beats_adapter": (
            domain_metrics["layercake_domain_bpb"]
            < domain_metrics["transformer_adapter_domain_bpb"]
        ),
        "lossless_transfer_exact": (
            metrics["transfer_ppl_ratio"] == 1.0
            and metrics["transfer_max_logit_diff"] == 0.0
            and metrics["transfer_generation_equal"]
        ),
        "receiver_after_transfer_beats_transformer": (
            receiver["status"] == "PASS"
            and receiver["required_gates"]["receiver_smaller"]
            and receiver["required_gates"]["receiver_better_general_bpb"]
            and receiver["required_gates"]["receiver_faster_training"]
            and receiver["required_gates"]["receiver_faster_cpu_generation"]
        ),
        "cpu_deployment_peak_memory_measured_and_no_worse_than_bpe": (
            resources["required_gates"]["measured_in_isolated_processes"]
            and resources["required_gates"]["layercake_peak_rss_no_more_than_bpe"]
        ),
        "cpu_deployment_artifact_smaller_than_bpe": (
            resources["required_gates"]["layercake_artifact_smaller_than_bpe"]
        ),
        "cpu_deployment_generation_faster_than_bpe": (
            resources["required_gates"]["layercake_generation_faster_than_bpe"]
        ),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": (
            "Game-ready CPU/mobile proxy gate. This certifies a CPU-first desktop/mobile-proxy "
            "deployment thesis with installable domain payloads. It is not a real Android/iOS, "
            "battery, thermal, NPU, or task-level game dialogue benchmark."
        ),
        "required_gates": gates,
        "failed_required": _failed(gates),
        "metrics": {
            "core_parameters": metrics["layercake_parameters"],
            "bpe_parameters": metrics["bpe_parameters"],
            "core_general_bpb": metrics["layercake_general_bpb"],
            "bpe_general_bpb": metrics["bpe_general_bpb"],
            "core_training_seconds": metrics["layercake_training_seconds"],
            "bpe_training_seconds": metrics["bpe_training_seconds"],
            "cpu_generation_speed_ratio": metrics["cpu_generation_speed_ratio"],
            "english_generation_sample": metrics["layercake_generation_utf8"],
            "domain_payload_bytes": domain_metrics["layercake_artifact_bytes"],
            "adapter_payload_bytes": domain_metrics["transformer_adapter_artifact_bytes"],
            "domain_training_seconds": domain_metrics["layercake_training_seconds"],
            "adapter_training_seconds": domain_metrics["transformer_adapter_training_seconds"],
            "domain_cpu_bytes_per_second": domain_metrics["layercake_cpu_bytes_per_second"],
            "adapter_cpu_bytes_per_second": domain_metrics["transformer_adapter_cpu_bytes_per_second"],
            "domain_bpb": domain_metrics["layercake_domain_bpb"],
            "adapter_domain_bpb": domain_metrics["transformer_adapter_domain_bpb"],
            "transfer_ppl_ratio": metrics["transfer_ppl_ratio"],
            "transfer_max_logit_diff": metrics["transfer_max_logit_diff"],
            "receiver_cpu_generation_speed_ratio": (
                receiver_metrics["receiver_cpu_generation_speed_ratio"]
            ),
            "cpu_deployment_peak_rss_ratio": resources["metrics"]["peak_rss_ratio"],
            "cpu_deployment_artifact_ratio": resources["metrics"]["artifact_ratio"],
            "cpu_deployment_parameter_memory_ratio": (
                resources["metrics"]["parameter_memory_ratio"]
            ),
            "cpu_deployment_generation_speed_ratio": (
                resources["metrics"]["generation_speed_ratio"]
            ),
        },
        "open_requirements_for_real_game_shipping": {
            "real_mobile_device_latency": False,
            "memory_peak_measurement": (
                resources["required_gates"]["measured_in_isolated_processes"]
                and resources["required_gates"]["layercake_peak_rss_no_more_than_bpe"]
            ),
            "battery_or_thermal_measurement": False,
            "game_dialogue_domain_dataset": False,
            "task_level_game_qa_or_npc_eval": False,
            "native_int8_runtime": False,
        },
        "open_cpu_deployment_resource_items": {
            "isolated_prefill_microbench_faster_than_bpe": resources[
                "required_gates"
            ]["layercake_prefill_faster_than_bpe"],
        },
    }
    output = RESULTS / "game_ready_mobile_llm_certificate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
