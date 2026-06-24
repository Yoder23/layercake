import json
from pathlib import Path


def test_domain_adaptation_schema_if_present():
    path = Path("results/dominance/domain_adaptation_dominance.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "migrated_domain_bpb" in data
    assert "transformer_adapter_domain_bpb" in data
