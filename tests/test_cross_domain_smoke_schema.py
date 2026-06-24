import json
from pathlib import Path


def test_cross_domain_smoke_frontier_certificate_schema():
    data = json.loads(
        Path("results/cross_domain_smoke_frontier_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "PASS"
    assert set(data["domains"]) == {
        "game_dialogue",
        "game_lore",
        "game_quest_state",
        "technical_text",
    }
    assert data["required_gates"]["four_domains_tested"] is True
    assert data["required_gates"]["all_domains_pass"] is True
    assert data["required_gates"]["all_transfers_exact"] is True
    assert data["required_gates"]["all_quality_smoke_gates_pass"] is True
    assert data["metrics"]["max_transfer_logit_diff"] == 0.0
    assert data["metrics"]["min_top1_byte_accuracy"] > 0.50
