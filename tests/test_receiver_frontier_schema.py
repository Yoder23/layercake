import json
from pathlib import Path


def test_receiver_frontier_schema():
    data = json.loads(Path("results/receiver_frontier_certificate.json").read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert "receiver_faster_cpu_generation" in data["required_gates"]
