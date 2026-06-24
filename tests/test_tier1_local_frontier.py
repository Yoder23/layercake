import json

from scripts.verify_tier1_local_frontier import PROBES


def test_tier1_local_frontier_probe_schema_constant():
    assert "276k" in PROBES
    assert PROBES["276k"].endswith("tier1_local_276k_probe.json")
