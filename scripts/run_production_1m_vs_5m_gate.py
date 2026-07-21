from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _read_config(path: Path) -> dict:
    resolved = path if path.is_absolute() else (ROOT / path).resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def _out_dir(config: dict) -> Path:
    path = Path(config["training"]["out_dir"])
    return path if path.is_absolute() else (ROOT / path).resolve()


def _metrics_path(config: dict) -> Path:
    return _out_dir(config) / config["training"].get("metrics_path", "training_metrics.json")


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    lc_cfg = _read_config(args.layercake_config)
    tx_cfg = _read_config(args.transformer_config)
    lc_out = _out_dir(lc_cfg)
    tx_out = _out_dir(tx_cfg)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    commands = {
        "train_layercake": [
            sys.executable,
            str(SCRIPTS / "train_byte_core_from_config.py"),
            "--config",
            str(args.layercake_config),
        ],
        "train_transformer": [
            sys.executable,
            str(SCRIPTS / "train_bpe_transformer_from_config.py"),
            "--config",
            str(args.transformer_config),
        ],
    }
    for device in ("cpu", "cuda"):
        commands[f"bench_layercake_{device}"] = [
            sys.executable,
            str(SCRIPTS / "benchmark_moonshot_generation.py"),
            "--checkpoint",
            str(lc_out / "latest.pt"),
            "--model-kind",
            "layercake",
            "--device",
            device,
            "--cpu-threads",
            str(args.cpu_threads),
            "--max-new-bytes",
            str(args.max_new_bytes),
            "--no-repeat-ngram",
            str(args.no_repeat_ngram),
            "--output",
            str(output_dir / f"layercake_{device}_generation.json"),
        ]
        commands[f"bench_transformer_{device}"] = [
            sys.executable,
            str(SCRIPTS / "benchmark_moonshot_generation.py"),
            "--checkpoint",
            str(tx_out / "latest.pt"),
            "--model-kind",
            "bpe",
            "--device",
            device,
            "--cpu-threads",
            str(args.cpu_threads),
            "--max-new-bytes",
            str(args.max_new_bytes),
            "--no-repeat-ngram",
            str(args.no_repeat_ngram),
            "--output",
            str(output_dir / f"transformer_{device}_generation.json"),
        ]
    commands["verify"] = [
        sys.executable,
        str(SCRIPTS / "verify_production_1m_vs_5m_dominance.py"),
        "--layercake-training",
        str(_metrics_path(lc_cfg)),
        "--transformer-training",
        str(_metrics_path(tx_cfg)),
        "--layercake-cpu-generation",
        str(output_dir / "layercake_cpu_generation.json"),
        "--transformer-cpu-generation",
        str(output_dir / "transformer_cpu_generation.json"),
        "--layercake-gpu-generation",
        str(output_dir / "layercake_cuda_generation.json"),
        "--transformer-gpu-generation",
        str(output_dir / "transformer_cuda_generation.json"),
        "--output",
        str(output_dir / "production_1m_vs_5m_dominance_certificate.json"),
    ]
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full 1M LayerCake vs 5M BPE production gate")
    parser.add_argument("--layercake-config", type=Path, default=Path("configs/production_cpu_game_1m_abipatchcell_layercake.json"))
    parser.add_argument("--transformer-config", type=Path, default=Path("configs/production_cpu_game_5m_bpe.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/production_cpu_game/1m_vs_5m_abipatchcell"))
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-new-bytes", type=int, default=128)
    parser.add_argument("--no-repeat-ngram", type=int, default=8)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = build_commands(args)
    if not args.skip_training:
        _run(commands["train_layercake"], dry_run=args.dry_run)
        _run(commands["train_transformer"], dry_run=args.dry_run)
    if not args.skip_generation:
        _run(commands["bench_layercake_cpu"], dry_run=args.dry_run)
        _run(commands["bench_transformer_cpu"], dry_run=args.dry_run)
        _run(commands["bench_layercake_cuda"], dry_run=args.dry_run)
        _run(commands["bench_transformer_cuda"], dry_run=args.dry_run)
    _run(commands["verify"], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
