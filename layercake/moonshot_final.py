"""Final moonshot command surface over real artifacts and fail-closed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results/moonshot/final"
V2 = ROOT / "results/moonshot/v2"


def _dump(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hosts() -> list[Path]:
    return [ROOT / f"artifacts/cores/english-core-{suffix}" for suffix in "abc"]


def _portability() -> dict:
    from layercake.evaluation.true_portability import verify_true_cross_host_portability

    return verify_true_cross_host_portability(
        cake_path=ROOT / "artifacts/cakes/python.cake",
        public_key_path=ROOT / "artifacts/cakes/python.public.pem",
        host_dirs=_hosts(),
        domain_test_path=ROOT / "data/moonshot/v2/python/python_test.bin",
        output_path=V2 / "portability_evidence.json",
    )


def _orchestration() -> dict:
    from layercake.evaluation.orchestration_v2 import run_orchestration_demo

    return run_orchestration_demo(
        host_dirs=_hosts(),
        cake_path=ROOT / "artifacts/cakes/python.cake",
        public_key_path=ROOT / "artifacts/cakes/python.public.pem",
        router_path=ROOT / "artifacts/router/semantic-router.safetensors",
        domain_test_path=ROOT / "data/moonshot/v2/python/python_test.bin",
        registry_root=ROOT / "artifacts/demo/registries",
        output_path=V2 / "orchestration_evidence.json",
    )


def _direct_benchmark() -> dict:
    from layercake.evaluation.cpu_vs_gpu import benchmark_cpu_vs_gpu

    return benchmark_cpu_vs_gpu(
        ROOT / "configs/eval/matched_quality_mixed_workload.yaml",
        core_dir=_hosts()[0],
        cake_path=ROOT / "artifacts/cakes/python.cake",
        public_key_path=ROOT / "artifacts/cakes/python.public.pem",
        transformer_dir=ROOT / "artifacts/baselines/transformer-mixed",
        router_path=ROOT / "artifacts/router/semantic-router.safetensors",
        output_path=V2 / "cpu_vs_gpu_evidence.json",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m layercake.moonshot_final")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "audit", "search", "train-core", "train-hosts", "train-domains",
        "portability", "routing", "benchmark-cpu", "benchmark-gpu",
        "benchmark-cpu-vs-gpu", "demo", "verify",
    ):
        sub.add_parser(command)
    args = parser.parse_args(argv)

    if args.command == "audit":
        result = {
            "baseline": _read(FINAL / "baseline_audit.json"),
            "current_certificate": (
                _read(FINAL / "release_certificate.json")
                if (FINAL / "release_certificate.json").is_file() else None
            ),
        }
        _dump(result)
        return 0
    if args.command == "search":
        names = (
            "foundation_search", "foundation_quality_search", "foundation_objective_search",
            "foundation_attention_search", "foundation_variable_patch_search",
            "foundation_adaptive_routing_search", "foundation_adaptive_depth_search",
            "foundation_adaptive_context_search",
        )
        ledgers = {name: _read(FINAL / f"{name}.json") for name in names}
        result = {
            "status": "PASS" if all(row.get("runs") for row in ledgers.values()) else "FAIL",
            "selection_split_only": all(not row.get("final_test_accessed", True) for row in ledgers.values()),
            "ledgers": {
                name: {
                    "status": row.get("status"),
                    "selected_candidate": row.get("selected_candidate"),
                    "runs": len(row.get("runs", [])),
                    "failed_runs": len(row.get("failed_runs", [])),
                }
                for name, row in ledgers.items()
            },
        }
        _dump(result)
        return 0 if result["selection_split_only"] else 1
    if args.command == "train-core":
        from layercake.training.patch_campaign import run_variable_patch_campaign

        result = run_variable_patch_campaign(
            ROOT / "configs/moonshot/final/foundation_adaptive_medium_pilot.json",
            FINAL / "foundation_adaptive_medium_pilot.json",
            artifact_root=ROOT / "artifacts/final/adaptive-medium-pilot",
        )
        _dump(result)
        return 0 if result.get("status") == "PASS" else 1
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
        result = {"status": "PASS" if all(row.get("status") == "PASS" for row in rows) else "FAIL", "hosts": rows}
        _dump(result)
        return 0 if result["status"] == "PASS" else 1
    if args.command == "train-domains":
        from layercake.training.cake import train_portable_fusion_cake

        python = train_portable_fusion_cake(
            _hosts()[0],
            ROOT / "configs/moonshot/dev/cake_python.yaml",
            ROOT / "artifacts/cakes/python.cake",
        )
        functional = python.get("evaluation", {}).get("syntax_tasks", {}).get("five_x_error_gate") == "PASS"
        result = {
            "status": "PASS" if functional else "FAIL",
            "python": python,
            "second_domain": {"status": "OPEN"},
            "third_domain": {"status": "OPEN"},
        }
        _dump(result)
        return 0 if result["status"] == "PASS" else 1
    if args.command == "portability":
        result = _portability()
        _dump(result)
        return 0 if result.get("status") == "PASS" else 1
    if args.command == "routing":
        from layercake.evaluation.catalog_scaling import benchmark_catalog_scaling
        from layercake.evaluation.routing_v2 import train_and_benchmark_router

        router = train_and_benchmark_router(
            V2 / "routing_evidence.json",
            model_path=ROOT / "artifacts/router/semantic-router.safetensors",
        )
        catalog = benchmark_catalog_scaling(
            ROOT / "artifacts/router/semantic-router.safetensors",
            ROOT / "artifacts/cakes/python.cake",
            FINAL / "catalog_scaling.json",
        )
        result = {"router": router, "catalog": catalog}
        _dump(result)
        return 0 if router.get("status") == "PASS" and catalog.get("status") == "PASS" else 1
    if args.command == "benchmark-cpu":
        from layercake.evaluation.incremental import benchmark_incremental_generation

        result = benchmark_incremental_generation(
            _hosts()[0], ROOT / "data/moonshot/v2/wikitext103/test.bin",
            V2 / "incremental_benchmark.json", repeats=3,
        )
        _dump(result)
        return 0 if result.get("status") == "PASS" else 1
    if args.command == "benchmark-gpu":
        result = _read(V2 / "cpu_vs_gpu_evidence.json")
        summary = {
            "status": "INVALID_EVIDENCE",
            "reason": "GPU runtime ran in the realistic suite, but no matched-quality LayerCake GPU comparison exists.",
            "transformer_gpu": result.get("systems", {}).get("transformer_gpu"),
            "quality": result.get("quality"),
        }
        _dump(summary)
        return 1
    if args.command == "benchmark-cpu-vs-gpu":
        result = _direct_benchmark()
        _dump(result)
        return 0 if result.get("status") == "PASS" else 1
    if args.command == "demo":
        result = {"orchestration": _orchestration(), "portability": _portability()}
        _dump(result)
        return 0 if all(row.get("status") == "PASS" for row in result.values()) else 1
    if args.command == "verify":
        from layercake.evaluation.moonshot_final_verifier import verify_moonshot_final

        result = verify_moonshot_final(ROOT, FINAL / "release_certificate.json")
        _dump(result)
        return 0 if result["moonshot_proven"] else 1
    raise AssertionError("unhandled final moonshot command")


if __name__ == "__main__":
    raise SystemExit(main())
