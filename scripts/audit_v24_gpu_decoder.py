"""Audit exact GPU CountCake decoding across independent prompt offsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from layercake.count_cake_cpu import CountCakeCPUDecoder  # noqa: E402
from layercake.count_cake_triton import CountCakeGPUDecoder  # noqa: E402


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _summary(values: list[float]) -> dict:
    return {
        "values": values,
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--offsets",
        default="0,131072,262144,393216,524288,655360,786432,917504",
    )
    parser.add_argument("--prompt-bytes", type=int, default=1024)
    parser.add_argument("--patches", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA decoder audit requested but CUDA is unavailable")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    bundle_path = Path(args.bundle)
    prompt_path = Path(args.prompt)
    corpus = prompt_path.read_bytes()
    offsets = [int(value) for value in args.offsets.split(",") if value.strip()]
    if not offsets:
        raise ValueError("at least one offset is required")
    for offset in offsets:
        if offset < 0 or offset + args.prompt_bytes > len(corpus):
            raise ValueError(f"prompt offset {offset} is outside the corpus")

    cpu_model, cpu_manifest = load_count_cake_bundle(bundle_path, device="cpu")
    gpu_model, gpu_manifest = load_count_cake_bundle(bundle_path, device="cuda")
    if cpu_manifest != gpu_manifest:
        raise RuntimeError("CPU and GPU bundle manifests differ")
    cpu_decoder = CountCakeCPUDecoder(cpu_model)
    gpu_decoder = CountCakeGPUDecoder(gpu_model)
    generated_bytes = args.patches * gpu_model.patch_size

    # Compile and allocate outside every measured region.
    warm_prompt = torch.tensor(
        list(corpus[offsets[0] : offsets[0] + args.prompt_bytes]),
        device="cuda",
        dtype=torch.long,
    ).reshape(1, -1)
    warm_state = gpu_model.begin_cached_generation(warm_prompt)
    gpu_decoder.prepare(warm_state, generated_bytes=generated_bytes)
    gpu_decoder.generate_cached(warm_state, patches=args.patches)
    torch.cuda.synchronize()

    rows = []
    for offset in offsets:
        prompt = corpus[offset : offset + args.prompt_bytes]
        cpu_rows = torch.tensor(list(prompt), dtype=torch.long).reshape(1, -1)
        cpu_state = cpu_model.begin_cached_generation(cpu_rows)
        cpu_decoder.clear_cache()
        cpu_started = time.perf_counter()
        reference = cpu_decoder.generate_cached(
            cpu_state, patches=args.patches
        ).reshape(-1)
        cpu_seconds = time.perf_counter() - cpu_started

        gpu_seconds = []
        gpu_wall_seconds = []
        certified_fractions = []
        launch_counts = []
        accelerated = None
        for _ in range(args.repeats):
            gpu_rows = torch.tensor(
                list(prompt), device="cuda", dtype=torch.long
            ).reshape(1, -1)
            gpu_state = gpu_model.begin_cached_generation(gpu_rows)
            gpu_decoder.prepare(gpu_state, generated_bytes=generated_bytes)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_started = time.perf_counter()
            start.record()
            candidate = gpu_decoder.generate_cached(
                gpu_state, patches=args.patches
            ).reshape(-1)
            end.record()
            torch.cuda.synchronize()
            wall = time.perf_counter() - wall_started
            device_seconds = start.elapsed_time(end) / 1000.0
            gpu_seconds.append(device_seconds)
            gpu_wall_seconds.append(wall)
            certified_fractions.append(
                float(gpu_state["gpu_certified_bytes"]) / generated_bytes
            )
            launch_counts.append(int(gpu_state["gpu_certificate_launches"]))
            if accelerated is None:
                accelerated = candidate.cpu()
            elif not torch.equal(accelerated, candidate.cpu()):
                raise RuntimeError(f"non-deterministic GPU output at offset {offset}")

        if accelerated is None:
            raise RuntimeError("GPU audit produced no output")
        differences = torch.nonzero(reference != accelerated).flatten()
        equal = not bool(differences.numel())
        rows.append(
            {
                "offset": offset,
                "prompt_sha256": _sha256_bytes(prompt),
                "generated_bytes": generated_bytes,
                "output_sha256": _sha256_bytes(bytes(reference.tolist())),
                "cpu_reference_seconds": cpu_seconds,
                "cpu_reference_bytes_per_second": generated_bytes / cpu_seconds,
                "gpu_device_seconds": _summary(gpu_seconds),
                "gpu_wall_seconds": _summary(gpu_wall_seconds),
                "gpu_device_bytes_per_second": _summary(
                    [generated_bytes / value for value in gpu_seconds]
                ),
                "certified_fraction": _summary(certified_fractions),
                "certificate_launches": _summary(launch_counts),
                "exact_cpu_gpu_equal": equal,
                "first_difference": (
                    None if equal else int(differences[0].item())
                ),
            }
        )

    all_throughputs = [
        value
        for row in rows
        for value in row["gpu_device_bytes_per_second"]["values"]
    ]
    report = {
        "format": "layercake-v24-gpu-decoder-audit/1",
        "status": (
            "PASS" if all(row["exact_cpu_gpu_equal"] for row in rows) else "FAIL"
        ),
        "scope": {
            "prompt_offsets": offsets,
            "prompt_bytes": args.prompt_bytes,
            "generated_bytes_per_prompt": generated_bytes,
            "repeats": args.repeats,
            "timing": "steady-state decode; prompt/index preparation excluded",
        },
        "artifacts": {
            "bundle": {
                "path": str(bundle_path),
                "bytes": bundle_path.stat().st_size,
                "sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            },
            "prompt_corpus": {
                "path": str(prompt_path),
                "bytes": len(corpus),
                "sha256": _sha256_bytes(corpus),
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "aggregate_gpu_device_bytes_per_second": _summary(all_throughputs),
        "prompts": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
