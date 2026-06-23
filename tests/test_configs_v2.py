import json
from pathlib import Path


def test_multisize_config_family_has_shared_abi_contract():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs").glob("layercake_*.json"))
    assert len(paths) == 5
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert {config["abi_version"] for config in configs} == {"lc-abi/2"}
    assert {config["d_abi"] for config in configs} == {512}
    for config in configs:
        assert config["input_mode"] in {"tokenized", "byte", "byte_patch"}
        assert config["domain_brick_default"]["type"]
        assert config["intended_device_class"]
        assert config["expected_quantization"]
