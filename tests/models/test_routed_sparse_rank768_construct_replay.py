from pathlib import Path

from layercake.evaluation.routed_sparse_rank768_progressive_construct_replay import _stable


def test_stable_replay_excludes_random_package_identifiers(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    raw = {
        "status": "PASS",
        "protocol": {"path": "original.json", "sha256": "a" * 64},
        "checks": {"one": True},
        "target_parameters": 1,
        "package": {"archive_hash": "random", "state_dict_hash": "random"},
    }
    first = _stable(raw, protocol)
    raw["package"] = {"archive_hash": "different", "state_dict_hash": "different"}
    second = _stable(raw, protocol)
    assert first == second
    assert "package" not in first
