"""Check exactness and throughput of the recurrent bounded-memory GPU decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from layercake.count_cake_triton import (  # noqa: E402
    CountCakeGPUDecoder,
    fused_recurrent_cached_byte,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--patches", type=int, default=1)
    args = parser.parse_args()
    model, _ = load_count_cake_bundle(args.bundle, device="cuda")
    prompt = Path(args.prompt).read_bytes()[:1024]
    rows = torch.tensor(list(prompt), device="cuda", dtype=torch.long).reshape(1, -1)
    reference_state = model.begin_cached_generation(rows)
    torch.cuda.synchronize()
    reference_started = time.perf_counter()
    reference = model.generate_cached(reference_state, patches=args.patches)
    torch.cuda.synchronize()
    reference_seconds = time.perf_counter() - reference_started

    decoder = CountCakeGPUDecoder(model)
    state = model.begin_cached_generation(rows)
    decoder.prepare(state, generated_bytes=args.patches * model.patch_size)
    torch.cuda.synchronize()
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    accelerated_started = time.perf_counter()
    gpu_start.record()
    accelerated = decoder.generate_cached(state, patches=args.patches)
    gpu_end.record()
    torch.cuda.synchronize()
    accelerated_seconds = time.perf_counter() - accelerated_started
    accelerated_gpu_seconds = gpu_start.elapsed_time(gpu_end) / 1000.0

    oracle_state = model.begin_cached_generation(rows)
    decoder.prepare(
        oracle_state, generated_bytes=args.patches * model.patch_size
    )
    oracle_history = oracle_state["gpu_history"]
    oracle_contexts = oracle_state["gpu_cache_context_keys"]
    oracle_stats = oracle_state["gpu_cache_stats"]
    oracle_map_keys = oracle_state["gpu_cache_map_keys"]
    oracle_map_occupied = oracle_state["gpu_cache_map_occupied"]
    oracle_map_counts = oracle_state["gpu_cache_map_counts"]
    oracle_map_totals = oracle_state["gpu_cache_map_totals"]
    oracle_recent_keys0 = oracle_state["gpu_recent_map_keys0"]
    oracle_recent_keys1 = oracle_state["gpu_recent_map_keys1"]
    oracle_recent_keys2 = oracle_state["gpu_recent_map_keys2"]
    oracle_recent_occupied = oracle_state["gpu_recent_map_occupied"]
    oracle_recent_latest = oracle_state["gpu_recent_map_latest"]
    oracle_position = int(oracle_state["gpu_position"])
    oracle_start = torch.cuda.Event(enable_timing=True)
    oracle_end = torch.cuda.Event(enable_timing=True)
    oracle_start.record()
    for patch_index in range(args.patches):
        expected_patch = reference[
            :, patch_index * model.patch_size : (patch_index + 1) * model.patch_size
        ]
        context = oracle_state["recurrent_state"].squeeze(0)
        composed = context + model.from_abi(model.to_abi(context))
        initial = model.local_projection(composed).unsqueeze(0)
        teacher = torch.cat(
            [
                model.local_bos.reshape(1, 1, -1),
                model.byte_embedding(expected_patch[:, :-1]),
            ],
            dim=1,
        )
        hidden, _ = model.local_core(teacher, initial)
        local = model.local_norm(hidden[0] + model.local_positions.weight)
        high_log = torch.log_softmax(model.high_head(local), dim=-1)
        high_values = torch.arange(16, device="cuda")
        low_hidden = model.low_norm(
            local.unsqueeze(1)
            * (1.0 + model.high_scale(high_values).unsqueeze(0))
            + model.high_embedding(high_values).unsqueeze(0)
        )
        low_log = torch.log_softmax(model.low_head(low_hidden), dim=-1)
        neural = (high_log.unsqueeze(-1) + low_log).flatten(-2).exp()
        gates = torch.sigmoid(model.mixture_gate(local)).squeeze(-1)
        patch_start = oracle_position
        for offset in range(model.patch_size):
            fused_recurrent_cached_byte(
                model,
                oracle_history,
                oracle_contexts,
                oracle_stats,
                oracle_map_keys,
                oracle_map_occupied,
                oracle_map_counts,
                oracle_map_totals,
                oracle_recent_keys0,
                oracle_recent_keys1,
                oracle_recent_keys2,
                oracle_recent_occupied,
                oracle_recent_latest,
                oracle_position,
                neural[offset].clone(),
                gates[offset].clone(),
            )
            oracle_position += 1
        oracle_patch = oracle_history[patch_start:oracle_position].reshape(1, -1)
        feature = torch.tanh(
            model.patch_projection(model.byte_embedding(oracle_patch).flatten(-2))
        ).unsqueeze(1)
        _, oracle_state["recurrent_state"] = model.patch_core(
            feature, oracle_state["recurrent_state"]
        )
    oracle_end.record()
    torch.cuda.synchronize()
    oracle_seconds = oracle_start.elapsed_time(oracle_end) / 1000.0
    oracle_output = oracle_history[-reference.numel() :].reshape_as(reference)
    equal = bool(torch.equal(reference, accelerated))
    differing = torch.nonzero(reference != accelerated).flatten()
    report = {
        "equal": equal,
        "first_difference": None if not differing.numel() else int(differing[0]),
        "bytes": int(reference.numel()),
        "reference_seconds": reference_seconds,
        "accelerated_seconds_including_first_compile": accelerated_seconds,
        "accelerated_gpu_seconds": accelerated_gpu_seconds,
        "oracle_teacher_lower_bound_seconds": oracle_seconds,
        "oracle_teacher_equal": bool(torch.equal(reference, oracle_output)),
        "reference": reference[0].cpu().tolist(),
        "accelerated": accelerated[0].cpu().tolist(),
    }
    print(json.dumps(report, indent=2))
    if not equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
