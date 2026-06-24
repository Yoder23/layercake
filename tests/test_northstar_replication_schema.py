import json
from pathlib import Path


def test_northstar_certificate_schema():
    data = json.loads(Path("results/northstar_mobile_certificate.json").read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert "gpu_generation_speed_ratio" in data["metrics"]
