import json
from pathlib import Path


def test_game_domain_training_workflow_certificate_schema():
    data = json.loads(
        Path("results/game_domain_training_workflow_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "PASS"
    gates = data["required_gates"]
    for gate in [
        "custom_game_domain_file_was_used",
        "training_completed",
        "int8_quantized_payload_created",
        "source_receiver_transfer_exact",
        "transfer_ppl_ratio_exact",
        "transfer_max_logit_diff_exact",
        "transfer_generation_exact",
        "smoke_domain_bpb_under_3",
        "smoke_domain_top1_accuracy_over_50pct",
    ]:
        assert gates[gate] is True
    assert data["metrics"]["transfer_ppl_ratio"] == 1.0
    assert data["metrics"]["transfer_max_logit_diff"] == 0.0
    assert data["open_requirements_for_production_game_domain"]["user_game_corpus"] is False
