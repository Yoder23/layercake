import json
from pathlib import Path


def test_cross_domain_adapter_frontier_certificate_schema():
    data = json.loads(
        Path("results/cross_domain_adapter_frontier_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "PASS"
    assert data["required_gates"]["four_domains_tested"] is True
    assert data["required_gates"]["all_domains_beat_adapter"] is True
    for domain in [
        "game_dialogue",
        "game_lore",
        "game_quest_state",
        "technical_text",
    ]:
        gates = data["domains"][domain]["required_gates"]
        assert gates["layercake_lower_domain_bpb"] is True
        assert gates["layercake_faster_domain_training"] is True
        assert gates["layercake_smaller_payload"] is True
        assert gates["layercake_transfer_exact"] is True
    assert data["metrics"]["max_domain_bpb_delta"] < 0.0
    assert data["metrics"]["max_payload_ratio"] < 1.0
    assert data["metrics"]["min_training_speed_ratio_adapter_over_layercake"] > 1.0
