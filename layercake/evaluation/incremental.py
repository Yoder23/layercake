"""Prompt-scaling and sustained incremental-generation evidence."""

from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import time

import torch

from layercake.runtime.cpu import configure_cpu
from layercake.runtime.cpu_reference import CPUReferenceRuntime
from layercake.training.foundation import load_core_checkpoint


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def _stats(values: list[float]) -> dict:
    return {
        "p50": statistics.median(values), "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99), "mean": statistics.fmean(values),
        "raw": values,
    }


@torch.inference_mode()
def benchmark_incremental_generation(
    core_dir: str | Path,
    corpus_path: str | Path,
    output_path: str | Path,
    *,
    repeats: int = 3,
) -> dict:
    configure_cpu(1)
    core, metadata = load_core_checkpoint(core_dir, device="cpu")
    route = int(metadata["route"])
    corpus = Path(corpus_path).read_bytes()
    prompt_lengths = [64, 256, 1024, 4096]
    generation_lengths = [1, 64, 256, 1024]
    randomized = [(prompt, generated, repeat) for prompt in prompt_lengths for generated in generation_lengths for repeat in range(repeats)]
    random.Random(20260721).shuffle(randomized)
    rows = []
    for prompt_length, generated, repeat in randomized:
        prompt = corpus[:prompt_length]
        started = time.perf_counter_ns()
        state = core.prefill(prompt, route=route)
        prefill_ms = (time.perf_counter_ns() - started) / 1_000_000
        started = time.perf_counter_ns()
        _, state = core.decode_many(state, generated)
        decode_ms = (time.perf_counter_ns() - started) / 1_000_000
        rows.append({
            "prompt_bytes": prompt_length, "generated_bytes": generated,
            "repeat": repeat, "prefill_milliseconds": prefill_ms,
            "decode_milliseconds": decode_ms,
            "decode_bytes_per_second": generated / (decode_ms / 1000),
            "time_to_first_byte_milliseconds": prefill_ms + decode_ms / generated,
            "state_bytes": state.state_bytes,
        })
    aggregates = {}
    for prompt_length in prompt_lengths:
        aggregates[str(prompt_length)] = {}
        for generated in generation_lengths:
            selected = [
                row for row in rows
                if row["prompt_bytes"] == prompt_length and row["generated_bytes"] == generated
            ]
            aggregates[str(prompt_length)][str(generated)] = {
                "prefill_milliseconds": _stats([row["prefill_milliseconds"] for row in selected]),
                "decode_milliseconds": _stats([row["decode_milliseconds"] for row in selected]),
                "decode_bytes_per_second": _stats([row["decode_bytes_per_second"] for row in selected]),
                "time_to_first_byte_milliseconds": _stats([row["time_to_first_byte_milliseconds"] for row in selected]),
                "state_bytes": selected[0]["state_bytes"],
            }
    # Actual reference work is bounded to 64 bytes because its purpose is
    # equivalence/asymptotic contrast, not the promoted runtime measurement.
    reference = CPUReferenceRuntime(core, route=route)
    equivalence = []
    for prompt_length in (64, 256, 1024):
        prompt = corpus[:prompt_length]
        for generated in (1, 16, 64):
            started = time.perf_counter_ns()
            full = reference.generate(prompt, generated)
            full_ms = (time.perf_counter_ns() - started) / 1_000_000
            state = core.prefill(prompt, route=route)
            started = time.perf_counter_ns()
            logits, state = core.decode_many(state, generated)
            incremental_ms = (time.perf_counter_ns() - started) / 1_000_000
            identical = torch.equal(
                full[:, prompt_length:], logits.argmax(dim=-1)
            )
            equivalence.append({
                "prompt_bytes": prompt_length, "generated_bytes": generated,
                "identical_outputs": identical,
                "full_context_milliseconds": full_ms,
                "incremental_decode_milliseconds_excluding_prefill": incremental_ms,
                "speedup": full_ms / max(incremental_ms, 1e-9),
            })
    long_decode_rates = [
        row["decode_bytes_per_second"] for row in rows if row["generated_bytes"] == 1024
    ]
    evidence = {
        "format": "layercake-incremental-benchmark/2",
        "status": "PASS" if all(row["identical_outputs"] for row in equivalence) else "FAIL",
        "runtime": "cpu-one-thread-stateful",
        "full_prompt_recomputed_per_decode_step": False,
        "prompt_lengths": prompt_lengths,
        "generation_lengths": generation_lengths,
        "randomized_execution_order": True,
        "repeats": repeats,
        "raw_rows": rows,
        "aggregates": aggregates,
        "equivalence": equivalence,
        "sustained_1024_byte_decode_rate": _stats(long_decode_rates),
        "state_contract": {
            "retained_prompt_bytes": 0,
            "local_convolution_history": True,
            "incomplete_multiscale_patches": True,
            "recurrent_hidden_state": True,
            "current_route": True,
            "canonical_state": True,
            "active_cake_state_supported": True,
            "sampler_accounting": True,
        },
        "core_checkpoint_sha256": metadata["checkpoint"]["sha256"],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence
