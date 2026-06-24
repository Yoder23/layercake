from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> int:
    transition = load("results/scale15m_transition_frontier_certificate.json")
    matrix = load("results/transformer_dominance_matrix.json")
    scorecard = load("results/cross_backend_quality_scorecard.json")
    game_ready = load("results/game_ready_mobile_llm_certificate.json")
    many_domain = load("results/many_domain_game_layers_certificate.json")
    game_workflow = load("results/game_domain_training_workflow_certificate.json")
    cross_domain = load("results/cross_domain_smoke_frontier_certificate.json")
    cross_domain_adapter = load(
        "results/cross_domain_adapter_frontier_certificate.json"
    )
    resources = load("results/cpu_deployment_resources_certificate.json")
    receiver = load("results/receiver_frontier_certificate.json")

    promoted_gates = {
        "base_frontier_source_core": transition["status"] == "PASS",
        "transformer_dominance_matrix_promoted_tiers": matrix["status"] == "PASS",
        "cross_backend_promoted_scorecard": scorecard["status"] == "PASS",
        "game_ready_cpu_mobile_proxy": game_ready["status"] == "PASS",
        "many_domain_install_migration_isolation": many_domain["status"] == "PASS",
        "game_domain_training_deployment_workflow": game_workflow["status"] == "PASS",
        "cross_domain_smoke_transfer_workflow": cross_domain["status"] == "PASS",
        "cross_domain_adapter_frontier": cross_domain_adapter["status"] == "PASS",
        "receiver_after_transfer": receiver["status"] == "PASS",
    }
    open_items = {
        "gpu_generation_speed": (
            scorecard["open_or_failing_scorecard"]["gpu_generation"]["status"]
            == "OPEN"
        ),
        "full_corpus_20m_training_time": (
            matrix["unpromoted_tiers"]["full_corpus_20m_source"]["status"]
            == "OPEN"
        ),
        "real_mobile_device_latency": (
            not game_ready["open_requirements_for_real_game_shipping"][
                "real_mobile_device_latency"
            ]
        ),
        "memory_peak_measurement": (
            not (
                game_ready["open_requirements_for_real_game_shipping"][
                    "memory_peak_measurement"
                ]
                and resources["required_gates"][
                    "layercake_peak_rss_no_more_than_bpe"
                ]
            )
        ),
        "isolated_cpu_prefill_microbench": (
            not resources["required_gates"]["layercake_prefill_faster_than_bpe"]
        ),
        "battery_or_thermal_measurement": (
            not game_ready["open_requirements_for_real_game_shipping"][
                "battery_or_thermal_measurement"
            ]
        ),
        "native_int8_runtime": (
            not game_ready["open_requirements_for_real_game_shipping"][
                "native_int8_runtime"
            ]
        ),
        "trained_game_dialogue_payload": (
            not many_domain["open_requirements_for_real_game_domains"][
                "trained_game_dialogue_payload"
            ]
        ),
        "trained_game_lore_payload": (
            not many_domain["open_requirements_for_real_game_domains"][
                "trained_game_lore_payload"
            ]
        ),
        "trained_game_quest_state_payload": (
            not many_domain["open_requirements_for_real_game_domains"][
                "trained_game_quest_state_payload"
            ]
        ),
        "task_level_npc_eval": (
            not many_domain["open_requirements_for_real_game_domains"][
                "task_level_npc_eval"
            ]
        ),
        "domain_routing_policy_eval": (
            not many_domain["open_requirements_for_real_game_domains"][
                "domain_routing_policy_eval"
            ]
        ),
    }
    failed_promoted = [
        name for name, passed in promoted_gates.items() if not passed
    ]
    result = {
        "status": "PASS" if not failed_promoted else "FAIL",
        "northstar_claim_status": "OPEN" if any(open_items.values()) else "PASS",
        "claim_boundary": (
            "PASS status means the currently promoted frontier evidence is internally "
            "consistent. northstar_claim_status remains OPEN until GPU generation speed, "
            "20M training-time dominance, real mobile/device latency, battery/thermal, native int8, "
            "and trained game-domain/task gates pass."
        ),
        "promoted_gates": promoted_gates,
        "failed_promoted_gates": failed_promoted,
        "open_northstar_items": {
            name: is_open for name, is_open in open_items.items() if is_open
        },
        "certificates": {
            "base_frontier": "results/scale15m_transition_frontier_certificate.json",
            "dominance_matrix": "results/transformer_dominance_matrix.json",
            "cross_backend_scorecard": "results/cross_backend_quality_scorecard.json",
            "game_ready_mobile_proxy": "results/game_ready_mobile_llm_certificate.json",
            "many_domain_game_layers": "results/many_domain_game_layers_certificate.json",
            "game_domain_training_workflow": "results/game_domain_training_workflow_certificate.json",
            "cross_domain_smoke": "results/cross_domain_smoke_frontier_certificate.json",
            "cross_domain_adapter_frontier": "results/cross_domain_adapter_frontier_certificate.json",
            "cpu_deployment_resources": "results/cpu_deployment_resources_certificate.json",
            "receiver_frontier": "results/receiver_frontier_certificate.json",
        },
        "key_metrics": {
            "base_bpb": transition["metrics"]["layercake_general_bpb"],
            "bpe_bpb": transition["metrics"]["bpe_general_bpb"],
            "cpu_generation_speed_ratio": (
                game_ready["metrics"]["cpu_generation_speed_ratio"]
            ),
            "gpu_layercake_bytes_per_second": (
                scorecard["metrics"]["gpu_layercake_bytes_per_second"]
            ),
            "gpu_bpe_bytes_per_second": (
                scorecard["metrics"]["gpu_bpe_bytes_per_second"]
            ),
            "many_domain_count": many_domain["metrics"]["installed_domains"],
            "many_domain_max_interference": (
                many_domain["metrics"]["max_cross_domain_interference"]
            ),
            "game_workflow_domain_bpb": game_workflow["metrics"]["domain_bpb"],
            "game_workflow_top1_byte_accuracy": (
                game_workflow["metrics"]["domain_top1_byte_accuracy"]
            ),
            "cross_domain_mean_bpb": cross_domain["metrics"]["mean_bpb"],
            "cross_domain_min_top1_byte_accuracy": (
                cross_domain["metrics"]["min_top1_byte_accuracy"]
            ),
            "cross_domain_adapter_max_bpb_delta": (
                cross_domain_adapter["metrics"]["max_domain_bpb_delta"]
            ),
            "cross_domain_adapter_min_training_speed_ratio": (
                cross_domain_adapter["metrics"][
                    "min_training_speed_ratio_adapter_over_layercake"
                ]
            ),
            "receiver_cpu_generation_speed_ratio": (
                receiver["metrics"]["receiver_cpu_generation_speed_ratio"]
            ),
            "cpu_deployment_peak_rss_ratio": resources["metrics"]["peak_rss_ratio"],
            "cpu_deployment_artifact_ratio": resources["metrics"]["artifact_ratio"],
        },
    }
    output = RESULTS / "frontier_model_northstar_certificate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
