import json
from pathlib import Path


def test_many_domain_game_layers_certificate_schema():
    data = json.loads(
        Path("results/many_domain_game_layers_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "PASS"
    assert data["domain_ids"] == [
        "game_dialogue",
        "game_lore",
        "game_quest_state",
    ]
    gates = data["required_gates"]
    for gate in [
        "three_domains_installed",
        "domain_specs_are_distinct",
        "payload_function_reused",
        "all_domains_transfer_exact_logits",
        "all_domains_generation_exact_after_transfer",
        "installing_other_domains_does_not_change_selected_domain",
    ]:
        assert gates[gate] is True
    assert data["metrics"]["max_cross_domain_interference"] == 0.0
    assert data["open_requirements_for_real_game_domains"]["task_level_npc_eval"] is False


def test_frontier_model_northstar_certificate_schema():
    data = json.loads(
        Path("results/frontier_model_northstar_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "PASS"
    assert data["northstar_claim_status"] == "OPEN"
    for gate in [
        "base_frontier_source_core",
        "transformer_dominance_matrix_promoted_tiers",
        "cross_backend_promoted_scorecard",
        "game_ready_cpu_mobile_proxy",
        "many_domain_install_migration_isolation",
        "game_domain_training_deployment_workflow",
        "cross_domain_smoke_transfer_workflow",
        "cross_domain_adapter_frontier",
        "receiver_after_transfer",
    ]:
        assert data["promoted_gates"][gate] is True
    open_items = data["open_northstar_items"]
    assert open_items["gpu_generation_speed"] is True
    assert "memory_peak_measurement" not in open_items
    assert open_items["isolated_cpu_prefill_microbench"] is True
    assert open_items["trained_game_dialogue_payload"] is True
    assert open_items["task_level_npc_eval"] is True
