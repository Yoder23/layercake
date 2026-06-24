import json
from pathlib import Path


def test_mobile_cpu_schema_if_present():
    path = Path("results/dominance/mobile_cpu_proxy.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "source_cpu_generation_speed_ratio" in data
    assert data["real_mobile_device"] is False
