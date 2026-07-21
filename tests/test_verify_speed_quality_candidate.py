from __future__ import annotations

import json

from scripts.verify_speed_quality_candidate import build_certificate


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_speed_quality_candidate_passes_when_all_gates_clear(tmp_path):
    layercake = _write_json(
        tmp_path / "lc.json",
        {"status": "TRAINED", "parameters": 100, "general": {"bpb": 0.9}},
    )
    transformer = _write_json(
        tmp_path / "tx.json",
        {"parameters": 100, "general": {"bpb": 1.0}},
    )
    cpu = _write_json(
        tmp_path / "cpu.json",
        {"speed_ratio": 6.0, "layercake": {"hex": "aa"}, "bpe": {"hex": "bb"}},
    )
    gpu = _write_json(
        tmp_path / "gpu.json",
        {"speed_ratio": 5.5, "layercake": {"hex": "aa"}, "bpe": {"hex": "bb"}},
    )

    cert = build_certificate(
        layercake_train=layercake,
        transformer_train=transformer,
        cpu_generation=cpu,
        gpu_generation=gpu,
        min_speed_ratio=5.0,
        max_bpb_ratio=1.0,
    )

    assert cert["status"] == "PASS"
    assert cert["ratios"]["heldout_bpb_ratio_layercake_over_transformer"] == 0.9


def test_speed_quality_candidate_fails_closed_on_missing_quality(tmp_path):
    layercake = _write_json(
        tmp_path / "lc.json",
        {"status": "TRAINED", "parameters": 100, "general": {"bpb": 1.1}},
    )
    transformer = _write_json(
        tmp_path / "tx.json",
        {"parameters": 100, "general": {"bpb": 1.0}},
    )
    cpu = _write_json(
        tmp_path / "cpu.json",
        {"speed_ratio": 6.0, "layercake": {"hex": "aa"}, "bpe": {"hex": "bb"}},
    )
    gpu = _write_json(
        tmp_path / "gpu.json",
        {"speed_ratio": 5.5, "layercake": {"hex": "aa"}, "bpe": {"hex": "bb"}},
    )

    cert = build_certificate(
        layercake_train=layercake,
        transformer_train=transformer,
        cpu_generation=cpu,
        gpu_generation=gpu,
        min_speed_ratio=5.0,
        max_bpb_ratio=1.0,
    )

    assert cert["status"] == "FAIL"
    assert cert["gates"]["lm_bpb_noninferior"] is False
