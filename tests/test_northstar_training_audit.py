from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from scripts.benchmark_northstar_training_speed import (
    Workload,
    _build_layercake_next_byte_only,
    _bytes_per_token,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "breakthrough_equal"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_training_workload_matches_raw_byte_volume() -> None:
    bytes_per_token = _bytes_per_token()
    workload = Workload(
        raw_sequence_bytes=256,
        batch_size=16,
        transformer_tokens=round(256 / bytes_per_token),
        transformer_bytes_per_token=bytes_per_token,
    )
    ratio = (
        workload.layercake_bytes_per_step
        / workload.transformer_bytes_per_step
    )
    assert 0.99 <= ratio <= 1.01


def test_favorable_layercake_mode_removes_dormant_optimizer_state() -> None:
    model = _build_layercake_next_byte_only(torch.device("meta"))
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert total == 14_804_448
    assert trainable == 14_622_464
    assert 0.95 <= trainable / 14_950_848 <= 1.05


def test_published_training_ratios_recompute_from_raw_repeats() -> None:
    for filename in (
        "northstar_v22_training_speed_recipe.json",
        "northstar_v22_training_speed_favorable_lower_bound.json",
    ):
        document = _load(filename)
        for device_name in ("cpu", "cuda"):
            device = document["devices"][device_name]
            layercake = device["repeat_details"]["layercake"]
            transformer = device["repeat_details"]["transformer"]
            ratios = [
                layercake[index]["logical_bytes_per_second"]
                / transformer[index]["logical_bytes_per_second"]
                for index in range(3)
            ]
            recorded = device["ratios"]
            assert all(
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                for actual, expected in zip(
                    ratios,
                    recorded[
                        "training_throughput_layercake_over_transformer_per_repeat"
                    ],
                )
            )
            assert math.isclose(
                statistics.median(ratios),
                recorded[
                    "median_training_throughput_layercake_over_transformer"
                ],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            assert min(ratios) < 5.0


def test_training_audit_is_valid_but_northstar_is_open() -> None:
    audit = _load("northstar_v22_training_audit.json")
    assert audit["measurement_status"] == "PASS"
    assert audit["failed_measurement_integrity"] == []
    assert audit["training_northstar_status"] == "OPEN"
    assert audit["failed_training_northstar"]
    assert not any(audit["training_northstar_gates"].values())


def test_training_audit_verifier_rebuilds_certificate(tmp_path: Path) -> None:
    output = tmp_path / "training_audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_northstar_training_audit.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rebuilt = json.loads(output.read_text(encoding="utf-8"))
    assert rebuilt["measurement_status"] == "PASS"
    assert rebuilt["training_northstar_status"] == "OPEN"


def test_public_docs_do_not_claim_training_dominance() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    release = (ROOT / "NORTHSTAR_V22_RELEASE.md").read_text(encoding="utf-8")
    claims = (ROOT / "CLAIMS.md").read_text(encoding="utf-8")
    training = (ROOT / "TRAINING_NORTHSTAR.md").read_text(encoding="utf-8")
    assert "Full-core training speed is a separate, currently open gate" in readme
    assert "does not claim faster full-core training" in release
    assert "Faster full-core training" in claims and "OPEN" in claims
    assert "must not claim 5x training dominance" in training
