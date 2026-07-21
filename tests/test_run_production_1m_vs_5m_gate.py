from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_production_1m_vs_5m_gate import build_commands


def test_1m_vs_5m_runner_builds_cpu_gpu_and_verifier_commands(tmp_path: Path):
    lc_config = tmp_path / "lc.json"
    tx_config = tmp_path / "tx.json"
    lc_config.write_text(json.dumps({"training": {"out_dir": "runs/lc", "metrics_path": "metrics.json"}}), encoding="utf-8")
    tx_config.write_text(json.dumps({"training": {"out_dir": "runs/tx", "metrics_path": "metrics.json"}}), encoding="utf-8")
    commands = build_commands(
        argparse.Namespace(
            layercake_config=lc_config,
            transformer_config=tx_config,
            output_dir=Path("results/test_1m5m"),
            cpu_threads=1,
            max_new_bytes=128,
            no_repeat_ngram=8,
        )
    )

    assert "train_byte_core_from_config.py" in commands["train_layercake"][1]
    assert "train_bpe_transformer_from_config.py" in commands["train_transformer"][1]
    assert "cuda" in commands["bench_layercake_cuda"]
    assert "cpu" in commands["bench_transformer_cpu"]
    assert "verify_production_1m_vs_5m_dominance.py" in commands["verify"][1]
