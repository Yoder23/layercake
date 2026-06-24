import json
from pathlib import Path


def test_game_ready_mobile_llm_certificate_schema():
    data = json.loads(
        Path("results/game_ready_mobile_llm_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "PASS"
    gates = data["required_gates"]
    for gate in [
        "core_frontier_promoted",
        "cpu_generation_at_least_2x_bpe",
        "core_smaller_than_bpe",
        "core_better_bpb_than_bpe",
        "core_faster_training_than_bpe",
        "english_generation_printable",
        "english_generation_diverse",
        "domain_payload_smaller_than_adapter",
        "domain_training_faster_than_adapter",
        "domain_cpu_at_least_2x_adapter",
        "domain_quality_beats_adapter",
        "lossless_transfer_exact",
        "receiver_after_transfer_beats_transformer",
        "cpu_deployment_peak_memory_measured_and_no_worse_than_bpe",
        "cpu_deployment_artifact_smaller_than_bpe",
        "cpu_deployment_generation_faster_than_bpe",
    ]:
        assert gates[gate] is True
    open_items = data["open_requirements_for_real_game_shipping"]
    assert open_items["real_mobile_device_latency"] is False
    assert open_items["memory_peak_measurement"] is True
    assert open_items["game_dialogue_domain_dataset"] is False
    assert open_items["task_level_game_qa_or_npc_eval"] is False
    assert (
        data["open_cpu_deployment_resource_items"][
            "isolated_prefill_microbench_faster_than_bpe"
        ]
        is False
    )
