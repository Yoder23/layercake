from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

import _common
from layercake.portable_domain import (
    LayerCakeRuntime,
    artifact_payload_bytes,
    load_portable_artifact,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU/mobile-proxy benchmark for exact portable domains"
    )
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--generation-bytes", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    artifact = torch.load(args.decoder, map_location="cpu", weights_only=True)
    spec, decoder = load_portable_artifact(artifact, device)
    x = torch.randint(0, 256, (args.batch, args.context), device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            decoder(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_ms = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            decoder(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_ms.append((time.perf_counter() - started) * 1000)

        runtime = LayerCakeRuntime()
        runtime.install_portable_domain(artifact, device)
        started = time.perf_counter()
        generated = runtime.generate(
            b"def layercake_domain(",
            max_new_bytes=args.generation_bytes,
            domain_id=spec.domain_id,
            context_bytes=args.context,
        )
        generation_seconds = time.perf_counter() - started

    median_ms = statistics.median(forward_ms)
    bytes_per_second = args.batch * args.context / (median_ms / 1000)
    result = {
        "status": "PASS",
        "device": str(device),
        "threads": args.threads,
        "domain_id": spec.domain_id,
        "parameters": decoder.parameter_count(),
        "quantization": spec.quantization,
        "payload_bytes": artifact_payload_bytes(artifact),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
        "forward": {
            "batch": args.batch,
            "context_bytes": args.context,
            "median_ms": median_ms,
            "p95_ms": percentile(forward_ms, 0.95),
            "bytes_per_second": bytes_per_second,
        },
        "greedy_generation": {
            "new_bytes": args.generation_bytes,
            "seconds": generation_seconds,
            "bytes_per_second": args.generation_bytes / generation_seconds,
            "output_hex": bytes(generated[0].cpu().tolist()).hex(),
            "output_utf8": bytes(generated[0].cpu().tolist()).decode(
                "utf-8", errors="replace"
            ),
        },
        "scope": (
            "Desktop hardware benchmark; this is not an Android, iOS, NPU, "
            "battery, or thermal benchmark."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
