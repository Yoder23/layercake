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
    lc_out = _out_dir(lc_cfg, "runs_experiment/breakthrough_equal_layercake")
    tx_out = _out_dir(tx_cfg, "runs_experiment/breakthrough_equal_bpe")
    output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / args.output_dir).resolve()
    lc_cpu_generation = output_dir / "layercake_cpu_generation.json"
    tx_cpu_generation = output_dir / "transformer_cpu_generation.json"
    lc_gpu_generation = output_dir / "layercake_gpu_generation.json"
    tx_gpu_generation = output_dir / "transformer_gpu_generation.json"
    certificate = output_dir / "breakthrough_equal_size_certificate.json"
    common_generation = [
        "--max-new-bytes",
        str(args.max_new_bytes),
        "--no-repeat-ngram",
        str(args.no_repeat_ngram),
    ]
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
            *common_generation,
            "--output",
            str(lc_cpu_generation),
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
            *common_generation,
            "--output",
            str(tx_cpu_generation),
        ],
        "bench_layercake_gpu": [
            sys.executable,
            str(SCRIPTS / "benchmark_moonshot_generation.py"),
            "--checkpoint",
            str(lc_out / "latest.pt"),
            "--model-kind",
            "layercake",
            "--device",
            "cuda",
            *common_generation,
            "--output",
            str(lc_gpu_generation),
        ],
        "bench_transformer_gpu": [
            sys.executable,
            str(SCRIPTS / "benchmark_moonshot_generation.py"),
            "--checkpoint",
            str(tx_out / "latest.pt"),
            "--model-kind",
            "bpe",
            "--device",
            "cuda",
            *common_generation,
            "--output",
            str(tx_gpu_generation),
        ],
        "verify": [
            sys.executable,
            str(SCRIPTS / "verify_breakthrough_equal_size_dominance.py"),
            "--layercake-training",
            str(_metrics_path(lc_cfg, lc_out)),
            "--transformer-training",
            str(_metrics_path(tx_cfg, tx_out)),
            "--layercake-cpu-generation",
            str(lc_cpu_generation),
            "--transformer-cpu-generation",
            str(tx_cpu_generation),
            "--layercake-gpu-generation",
            str(lc_gpu_generation),
            "--transformer-gpu-generation",
            str(tx_gpu_generation),
            "--output",
            str(certificate),
            "--param-tolerance",
            str(args.param_tolerance),
            "--min-eval-bytes",
            str(args.min_eval_bytes),
            "--min-quality-bpb-improvement-ratio",
            str(args.min_quality_bpb_improvement_ratio),
            "--min-training-speed-ratio",
            str(args.min_training_speed_ratio),
            "--min-training-cost-ratio",
            str(args.min_training_cost_ratio),
            "--max-train-byte-ratio",
            str(args.max_train_byte_ratio),
            "--min-inference-speed-ratio",
            str(args.min_inference_speed_ratio),
            "--min-generation-quality-ratio",
            str(args.min_generation_quality_ratio),
            "--min-task-score-ratio",
            str(args.min_task_score_ratio),
            "--min-relevance-ratio",
            str(args.min_relevance_ratio),
            "--min-layercake-relevance",
            str(args.min_layercake_relevance),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the hard equal-size breakthrough gate on real checkpoints."
    )
    parser.add_argument("--layercake-config", required=True, type=Path)
    parser.add_argument("--transformer-config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/breakthrough_equal"))
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-new-bytes", type=int, default=256)
    parser.add_argument("--no-repeat-ngram", type=int, default=8)
    parser.add_argument("--param-tolerance", type=float, default=0.05)
    parser.add_argument("--min-eval-bytes", type=float, default=1_000_000.0)
    parser.add_argument("--min-quality-bpb-improvement-ratio", type=float, default=5.0)
    parser.add_argument("--min-training-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-training-cost-ratio", type=float, default=5.0)
    parser.add_argument("--max-train-byte-ratio", type=float, default=1.0)
    parser.add_argument("--min-inference-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-generation-quality-ratio", type=float, default=5.0)
    parser.add_argument("--min-task-score-ratio", type=float, default=5.0)
    parser.add_argument("--min-relevance-ratio", type=float, default=1.0)
    parser.add_argument("--min-layercake-relevance", type=float, default=1.0)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-gpu-generation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = build_commands(args)
    if not args.skip_training:
        _run(commands["train_layercake"], dry_run=args.dry_run)
        _run(commands["train_transformer"], dry_run=args.dry_run)
    if not args.skip_generation:
        _run(commands["bench_layercake_cpu"], dry_run=args.dry_run)
        _run(commands["bench_transformer_cpu"], dry_run=args.dry_run)
        if not args.skip_gpu_generation:
            _run(commands["bench_layercake_gpu"], dry_run=args.dry_run)
            _run(commands["bench_transformer_gpu"], dry_run=args.dry_run)
    _run(commands["verify"], dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
