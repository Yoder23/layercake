import json
from pathlib import Path


def test_cpu_deployment_resources_certificate_schema():
    data = json.loads(
        Path("results/cpu_deployment_resources_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["status"] == "OPEN"
    gates = data["required_gates"]
    for gate in [
        "measured_in_isolated_processes",
        "layercake_parameter_memory_lower_than_bpe",
        "layercake_artifact_smaller_than_bpe",
        "layercake_peak_rss_no_more_than_bpe",
        "layercake_generation_faster_than_bpe",
    ]:
        assert gates[gate] is True
    assert gates["layercake_prefill_faster_than_bpe"] is False
    metrics = data["metrics"]
    assert metrics["peak_rss_ratio"] <= 1.0
    assert metrics["artifact_ratio"] <= 1.0
    assert metrics["generation_speed_ratio"] > 1.0
