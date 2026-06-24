import json
from pathlib import Path


def test_cross_backend_quality_scorecard_schema():
    data = json.loads(
        Path("results/cross_backend_quality_scorecard.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "PASS"
    promoted = data["promoted_scorecard"]
    for group in [
        "training_quality_cost",
        "cpu_generation",
        "latency",
        "domain_layers",
        "game_ready_cpu_mobile_proxy",
    ]:
        assert promoted[group]["status"] == "PASS"
        assert promoted[group]["failed"] == []
    gpu = data["open_or_failing_scorecard"]["gpu_generation"]
    assert gpu["status"] == "OPEN"
    assert "gpu_generation_faster_than_bpe" in gpu["failed"]
    assert gpu["required_gates"]["gpu_generation_quality_gates_pass"] is True
    assert data["metrics"]["cpu_layercake_bytes_per_second"] > data["metrics"]["cpu_bpe_bytes_per_second"]
    assert data["metrics"]["gpu_layercake_bytes_per_second"] < data["metrics"]["gpu_bpe_bytes_per_second"]
