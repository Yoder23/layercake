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


def _out_dir(config: dict, fallback: str) -> Path:
    configured = config.get("training", {}).get("out_dir", fallback)
    path = Path(configured)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _metrics_path(config: dict, out_dir: Path) -> Path:
    return out_dir / config.get("training", {}).get("metrics_path", "training_metrics.json")


def _run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    lc_cfg = _read_config(args.layercake_config)
    tx_cfg = _read_config(args.transformer_config)
    lc_out = _out_dir(lc_cfg, "runs_experiment/production_layercake")
    tx_out = _out_dir(tx_cfg, "runs_experiment/production_transformer")
    output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    lc_generation = output_dir / "layercake_cpu_generation.json"
    tx_generation = output_dir / "transformer_cpu_generation.json"
    certificate = output_dir / "production_cpu_game_dominance_certificate.json"
    return {
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
        "bench_layercake_cpu": [
            sys.executable,
            str(SCRIPTS / "benchmark_moonshot_generation.py"),
            "--checkpoint",
            str(lc_out / "latest.pt"),
            "--model-kind",
            "layercake",
            "--device",
            "cpu",
            "--cpu-threads",
            str(args.cpu_threads),
            "--max-new-bytes",
            str(args.max_new_bytes),
            "--no-repeat-ngram",
            str(args.no_repeat_ngram),
            "--output",
            str(lc_generation),
        ],
        "bench_transformer_cpu": [
            sys.executable,
            str(SCRIPTS / "benchmark_moonshot_generation.py"),
            "--checkpoint",
            str(tx_out / "latest.pt"),
            "--model-kind",
            "bpe",
            "--device",
            "cpu",
            "--cpu-threads",
            str(args.cpu_threads),
            "--max-new-bytes",
            str(args.max_new_bytes),
            "--no-repeat-ngram",
            str(args.no_repeat_ngram),
            "--output",
            str(tx_generation),
        ],
        "verify": [
            sys.executable,
            str(SCRIPTS / "verify_production_cpu_game_dominance.py"),
            "--layercake-training",
            str(_metrics_path(lc_cfg, lc_out)),
            "--transformer-training",
            str(_metrics_path(tx_cfg, tx_out)),
            "--layercake-generation",
            str(lc_generation),
            "--transformer-generation",
            str(tx_generation),
            "--output",
            str(certificate),
            "--max-same-size-param-ratio",
            str(args.max_same_size_param_ratio),
            "--min-training-speed-ratio",
            str(args.min_training_speed_ratio),
            "--min-generation-speed-ratio",
            str(args.min_generation_speed_ratio),
            "--min-quality-ratio",
            str(args.min_quality_ratio),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full same-size CPU/game production dominance gate on real checkpoints."
    )
    parser.add_argument("--layercake-config", required=True, type=Path)
    parser.add_argument("--transformer-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/production_cpu_game"))
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-new-bytes", type=int, default=128)
    parser.add_argument("--no-repeat-ngram", type=int, default=8)
    parser.add_argument("--max-same-size-param-ratio", type=float, default=1.10)
    parser.add_argument("--min-training-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-generation-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
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
    _run(commands["verify"], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
