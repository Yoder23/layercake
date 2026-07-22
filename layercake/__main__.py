from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


def _run(argv: list[str]) -> int:
    from layercake.cake.registry import CakeRegistry
    from layercake.cake.cli import _trust_store
    from layercake.routing.orchestrator import LocalLayerCakeOrchestrator
    from layercake.routing.policies import CakePermissionPolicy, RoutingPolicy
    from layercake.training.foundation import load_core_checkpoint

    parser = argparse.ArgumentParser(prog="layercake run")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cake", action="append")
    selection.add_argument("--auto-route", action="store_true")
    parser.add_argument("--registry")
    parser.add_argument("--core", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--trust-store")
    parser.add_argument("--router", default="artifacts/router/semantic-router.safetensors")
    parser.add_argument("--max-new-bytes", type=int, default=64)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("prompt")
    args = parser.parse_args(argv)
    registry = CakeRegistry(args.registry)
    policy = RoutingPolicy(permissions=CakePermissionPolicy(
        allowed_permissions=frozenset({"local-inference"}), allow_unsigned_local=True
    ))
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    semantic_router = None
    if args.auto_route and args.router and Path(args.router).is_file():
        from safetensors.torch import load_file
        from layercake.routing.learned_router import CompactSemanticRouter
        semantic_router = CompactSemanticRouter()
        semantic_router.load_state_dict(load_file(args.router), strict=True)
        semantic_router.eval()
    orchestrator = LocalLayerCakeOrchestrator(
        registry, policy=policy, trust_store=_trust_store(args.trust_store), device=device,
        semantic_router=semantic_router,
    )
    core, core_metadata = load_core_checkpoint(args.core, device=device)

    def core_handler(prompt: str) -> str:
        state = core.prefill(
            prompt, route=int(core_metadata["route"]), capture_generated=True
        )
        _, state = core.decode_many(state, args.max_new_bytes)
        return bytes(state.generated_bytes[0].cpu().tolist()).decode("utf-8", errors="replace")

    @torch.inference_mode()
    def cake_handler(prompt: str, modules: list[torch.nn.Module], _route) -> str:
        fusion = [module for module in modules if module.__class__.__name__ == "PortableFusionCake"]
        if fusion:
            if len(fusion) != 1 or len(modules) != 1:
                raise ValueError("portable_fusion currently requires a single selected cake")
            module = fusion[0].to(device)
            state = core.prefill(
                prompt, route=int(core_metadata["route"]), fusion_cake=module,
                capture_generated=True,
            )
            _, state = core.decode_many(state, args.max_new_bytes, fusion_cake=module)
            return bytes(state.generated_bytes[0].cpu().tolist()).decode("utf-8", errors="replace")
        ids = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long, device=device)[None]
        for _ in range(args.max_new_bytes):
            logits = [module(ids)[:, -1] for module in modules]
            ids = torch.cat([ids, torch.stack(logits).mean(dim=0).argmax(-1, keepdim=True)], dim=1)
        return bytes(ids[0, -args.max_new_bytes:].cpu().tolist()).decode("utf-8", errors="replace")

    result = orchestrator.execute(
        args.prompt,
        core_handler=core_handler,
        cake_handler=cake_handler,
        forced=tuple(args.cake or ()) or None,
    )
    rendered = str(result.output) + "\n"
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write(rendered.encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    else:
        print(rendered, end="")
    if args.trace:
        print(json.dumps(result.metrics(), indent=2, sort_keys=True, default=str), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m layercake {core|moonshot|moonshot_v2|cake|benchmark|run} ...", file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "cake":
        from layercake.cake.cli import main as cake_main
        return cake_main(rest)
    if command == "core":
        parser = argparse.ArgumentParser(prog="layercake core")
        sub = parser.add_subparsers(dest="core_command", required=True)
        train = sub.add_parser("train")
        train.add_argument("--config", required=True)
        train.add_argument("--output", required=True)
        args = parser.parse_args(rest)
        from layercake.training.foundation import train_english_core
        print(json.dumps(train_english_core(args.config, args.output), indent=2, sort_keys=True))
        return 0
    if command == "moonshot":
        from layercake.moonshot import main as moonshot_main
        return moonshot_main(rest)
    if command == "moonshot_v2":
        from layercake.moonshot_v2 import main as moonshot_v2_main
        return moonshot_v2_main(rest)
    if command == "benchmark":
        parser = argparse.ArgumentParser(prog="layercake benchmark")
        sub = parser.add_subparsers(dest="benchmark_command", required=True)
        direct = sub.add_parser("cpu-vs-gpu")
        direct.add_argument("--layercake-core", required=True)
        direct.add_argument("--cake", required=True)
        direct.add_argument("--public-key")
        direct.add_argument("--transformer", required=True)
        direct.add_argument("--router", default="artifacts/router/semantic-router.safetensors")
        direct.add_argument("--suite", required=True)
        direct.add_argument("--output", default="results/moonshot/v2/cpu_vs_gpu_evidence.json")
        args = parser.parse_args(rest)
        if args.benchmark_command == "cpu-vs-gpu":
            from layercake.evaluation.cpu_vs_gpu import benchmark_cpu_vs_gpu
            public_key = args.public_key or str(Path(args.cake).with_suffix(".public.pem"))
            result = benchmark_cpu_vs_gpu(
                args.suite, core_dir=args.layercake_core, cake_path=args.cake,
                public_key_path=public_key, transformer_dir=args.transformer,
                router_path=args.router, output_path=args.output,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "PASS" else 1
    if command == "run":
        return _run(rest)
    print(f"unknown layercake command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
