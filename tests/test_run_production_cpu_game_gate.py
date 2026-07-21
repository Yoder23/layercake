from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_production_cpu_game_gate import build_commands


def test_build_commands_uses_real_checkpoint_generation_and_strict_verifier(tmp_path: Path):
    lc_config = tmp_path / "lc.json"
    tx_config = tmp_path / "tx.json"
    lc_config.write_text(
        json.dumps({"training": {"out_dir": "runs_experiment/lc_prod", "metrics_path": "metrics.json"}}),
        encoding="utf-8",
    )
    tx_config.write_text(
        json.dumps({"training": {"out_dir": "runs_experiment/tx_prod", "metrics_path": "training_metrics.json"}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        layercake_config=lc_config,
        transformer_config=tx_config,
        output_dir=Path("results/production_cpu_game_test"),
        cpu_threads=1,
        max_new_bytes=128,
        no_repeat_ngram=8,
        max_same_size_param_ratio=1.10,
        min_training_speed_ratio=1.0,
        min_generation_speed_ratio=5.0,
        min_quality_ratio=1.0,
    )

    commands = build_commands(args)

    assert "train_byte_core_from_config.py" in commands["train_layercake"][1]
    assert "train_bpe_transformer_from_config.py" in commands["train_transformer"][1]
    assert "--device" in commands["bench_layercake_cpu"]
    assert "cpu" in commands["bench_layercake_cpu"]
    assert "benchmark_moonshot_generation.py" in commands["bench_transformer_cpu"][1]
    assert "verify_production_cpu_game_dominance.py" in commands["verify"][1]
    assert "--min-generation-speed-ratio" in commands["verify"]
    assert "5.0" in commands["verify"]
