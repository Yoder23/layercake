from __future__ import annotations

import argparse
import random
import statistics
import time
import torch

from _common import emit
from artifact_utils import build_brick, build_models


@torch.no_grad()
def measure(fn, iterations, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    return elapsed, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--brick")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.core, map_location="cpu", weights_only=True)
    byte, patch = build_models(artifact, device)
    byte.eval()
    patch.eval()
    brick = None
    if args.brick:
        brick_artifact = torch.load(args.brick, map_location="cpu", weights_only=True)
        brick = build_brick(brick_artifact["brick_config"], device)
        brick.load_state_dict(brick_artifact["brick"])
        brick.eval()
    batch_size = args.batch or min(artifact["args"].get("batch", 24), 24)
    seq = artifact["args"].get("seq", 128)
    x = torch.randint(0, 256, (batch_size, seq), device=device)
    paths = [
        ("byte_base", lambda: byte(x)),
        ("byte_brick", lambda: byte(x, brick=brick) if brick else byte(x)),
        ("patch_base", lambda: patch(x)),
        ("patch_brick", lambda: patch(x, brick=brick) if brick else patch(x)),
    ]
    for _, fn in paths:
        for _ in range(30):
            fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples = {name: [] for name, _ in paths}
    peaks = {name: [] for name, _ in paths}
    rng = random.Random(42)
    for _ in range(args.rounds):
        order = paths.copy()
        rng.shuffle(order)
        for name, fn in order:
            elapsed, peak = measure(fn, args.iterations, device)
            samples[name].append(elapsed)
            if peak is not None:
                peaks[name].append(peak)
    rows = []
    for name, _ in paths:
        elapsed = statistics.median(samples[name])
        peak = max(peaks[name]) if peaks[name] else None
        units = batch_size * seq * args.iterations
        rows.append({
            "path": name, "seconds": elapsed, "bytes": units,
            "bytes_per_second": units / elapsed, "peak_memory_bytes": peak,
            "round_seconds": samples[name],
        })
    emit(
        {
            "device": str(device), "iterations": args.iterations, "rounds": args.rounds,
            "batch": batch_size, "sequence_bytes": seq,
            "byte_parameters": sum(p.numel() for p in byte.parameters()),
            "patch_parameters": sum(p.numel() for p in patch.parameters()),
            "global_sequence_reduction": patch.patch_size,
            "rows": rows,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
