from __future__ import annotations

import argparse
import json
import sys

import torch


def _run(argv: list[str]) -> int:
    from layercake.cake.registry import CakeRegistry
    from layercake.models.foundation import FoundationConfig, LayerCakeFoundation
    from layercake.routing.orchestrator import LocalLayerCakeOrchestrator
    from layercake.routing.policies import CakePermissionPolicy, RoutingPolicy

    parser = argparse.ArgumentParser(prog="layercake run")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cake", action="append")
    selection.add_argument("--auto-route", action="store_true")
    parser.add_argument("--registry")
    parser.add_argument("--max-new-bytes", type=int, default=64)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("prompt")
    args = parser.parse_args(argv)
    registry = CakeRegistry(args.registry)
    policy = RoutingPolicy(
        permissions=CakePermissionPolicy(allow_unsigned_local=True),
    )
    orchestrator = LocalLayerCakeOrchestrator(registry, policy=policy)
    torch.manual_seed(20260721)
    core = LayerCakeFoundation(
        FoundationConfig(d_byte=16, d_model=48, recurrent_layers=1, routed_experts=4, expert_expansion=2, abi_width=16)
    ).eval()

    def core_handler(prompt: str) -> str:
        generated = core.generate(prompt, args.max_new_bytes)
        prompt_length = len(prompt.encode("utf-8"))
        return bytes(generated[0, prompt_length:].tolist()).decode("utf-8", errors="replace")

    @torch.inference_mode()
    def cake_handler(prompt: str, modules: list[torch.nn.Module], _route) -> str:
        ids = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long)[None]
        prompt_length = ids.shape[1]
        for _ in range(args.max_new_bytes):
            logits = [module(ids)[:, -1] for module in modules]
            mean_logits = torch.stack(logits).mean(dim=0)
            ids = torch.cat([ids, mean_logits.argmax(-1, keepdim=True)], dim=1)
        return bytes(ids[0, prompt_length:].tolist()).decode("utf-8", errors="replace")

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
        print("usage: python -m layercake {moonshot|cake|run} ...", file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "cake":
        from layercake.cake.cli import main as cake_main
        return cake_main(rest)
    if command == "moonshot":
        from layercake.moonshot import main as moonshot_main
        return moonshot_main(rest)
    if command == "run":
        return _run(rest)
    print(f"unknown layercake command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
