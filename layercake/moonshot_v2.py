"""One-command V2 research, demonstration, and fail-closed verification paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _dump(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _hosts() -> list[Path]:
    return [ROOT / f"artifacts/cores/english-core-{suffix}" for suffix in "abc"]


def _demo() -> dict:
    from layercake.evaluation.orchestration_v2 import run_orchestration_demo
    return run_orchestration_demo(
        host_dirs=_hosts(),
        cake_path=ROOT / "artifacts/cakes/python.cake",
        public_key_path=ROOT / "artifacts/cakes/python.public.pem",
        router_path=ROOT / "artifacts/router/semantic-router.safetensors",
        domain_test_path=ROOT / "data/moonshot/v2/python/python_test.bin",
        registry_root=ROOT / "artifacts/demo/registries",
        output_path=ROOT / "results/moonshot/v2/orchestration_evidence.json",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m layercake.moonshot_v2")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("search")
    train_core = sub.add_parser("train-core")
    train_core.add_argument("--config", default="configs/moonshot/dev/core_english_development.yaml")
    train_core.add_argument("--output", default="artifacts/cores/english-core-a")
    sub.add_parser("train-hosts")
    sub.add_parser("train-cakes")
    sub.add_parser("benchmark")
    sub.add_parser("demo")
    verify = sub.add_parser("verify")
    verify.add_argument("--evidence", default="results/moonshot/v2")
    args = parser.parse_args(argv)

    if args.command == "audit":
        from layercake.evaluation.development import build_development_evidence
        development = build_development_evidence(
            ROOT, ROOT / "results/moonshot/v2/development_evidence.json"
        )
        _dump({
            "baseline": json.loads((ROOT / "results/moonshot/v2/baseline_audit.json").read_text()),
            "development": development,
        })
        return 0
    if args.command == "search":
        from layercake.training.architecture_search import run_architecture_search
        result = run_architecture_search(
            ROOT / "configs/moonshot/search/architecture_search.yaml",
            ROOT / "results/moonshot/v2/architecture_search.json",
        )
        _dump(result)
        return 0 if result["status"] == "PASS" else 1
    if args.command == "train-core":
        from layercake.training.foundation import train_english_core
        _dump(train_english_core(ROOT / args.config, ROOT / args.output))
        return 0
    if args.command == "train-hosts":
        from layercake.training.foundation import train_english_core
        configs = (
            "core_english_development.yaml", "core_english_receiver_b.yaml",
            "core_english_receiver_c.yaml",
        )
        rows = [
            train_english_core(ROOT / "configs/moonshot/dev" / config, output)
            for config, output in zip(configs, _hosts())
        ]
        _dump({"hosts": rows})
        return 0 if all(row["status"] == "PASS" for row in rows) else 1
    if args.command == "train-cakes":
        from layercake.training.cake import train_portable_fusion_cake
        result = train_portable_fusion_cake(
            _hosts()[0], ROOT / "configs/moonshot/dev/cake_python.yaml",
            ROOT / "artifacts/cakes/python.cake",
        )
        _dump(result)
        return 0 if result["status"] == "PASS" else 1
    if args.command == "benchmark":
        from layercake.evaluation.cpu_vs_gpu import benchmark_cpu_vs_gpu
        from layercake.evaluation.incremental import benchmark_incremental_generation
        from layercake.evaluation.routing_v2 import train_and_benchmark_router
        from layercake.evaluation.true_portability import verify_true_cross_host_portability
        rows = {
            "incremental": benchmark_incremental_generation(
                _hosts()[0], ROOT / "data/moonshot/v2/wikitext103/test.bin",
                ROOT / "results/moonshot/v2/incremental_benchmark.json", repeats=3,
            ),
            "routing": train_and_benchmark_router(
                ROOT / "results/moonshot/v2/routing_evidence.json",
                model_path=ROOT / "artifacts/router/semantic-router.safetensors",
            ),
            "portability": verify_true_cross_host_portability(
                cake_path=ROOT / "artifacts/cakes/python.cake",
                public_key_path=ROOT / "artifacts/cakes/python.public.pem",
                host_dirs=_hosts(),
                domain_test_path=ROOT / "data/moonshot/v2/python/python_test.bin",
                output_path=ROOT / "results/moonshot/v2/portability_evidence.json",
            ),
        }
        if torch.cuda.is_available():
            rows["cpu_vs_gpu"] = benchmark_cpu_vs_gpu(
                ROOT / "configs/eval/matched_quality_mixed_workload.yaml",
                core_dir=_hosts()[0], cake_path=ROOT / "artifacts/cakes/python.cake",
                public_key_path=ROOT / "artifacts/cakes/python.public.pem",
                transformer_dir=ROOT / "artifacts/baselines/transformer-mixed",
                router_path=ROOT / "artifacts/router/semantic-router.safetensors",
                output_path=ROOT / "results/moonshot/v2/cpu_vs_gpu_evidence.json",
            )
        else:
            rows["cpu_vs_gpu"] = {"status": "NOT_RUN_NO_HARDWARE"}
        rows["orchestration"] = _demo()
        _dump(rows)
        return 0 if all(row.get("status") == "PASS" for row in rows.values()) else 1
    if args.command == "demo":
        demonstration = _demo()
        from layercake.evaluation.moonshot_verifier import verify_moonshot_v2
        certificate = verify_moonshot_v2(ROOT, ROOT / "results/moonshot/v2")
        _dump({
            "demonstration": demonstration,
            "cpu_vs_gpu": json.loads((ROOT / "results/moonshot/v2/cpu_vs_gpu_evidence.json").read_text()),
            "certificate": certificate,
        })
        return 0 if certificate["moonshot_proven"] else 1
    if args.command == "verify":
        from layercake.evaluation.moonshot_verifier import verify_moonshot_v2
        certificate = verify_moonshot_v2(ROOT, ROOT / args.evidence)
        _dump(certificate)
        return 0 if certificate["moonshot_proven"] else 1
    raise AssertionError("unhandled V2 command")


if __name__ == "__main__":
    raise SystemExit(main())
