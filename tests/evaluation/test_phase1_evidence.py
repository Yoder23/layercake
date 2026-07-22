from __future__ import annotations

import copy
import hashlib
import itertools

import pytest

from layercake.evaluation.phase1_evidence import (
    Phase1EvidenceError,
    validate_baseline_optimization,
    validate_benchmark_matrix,
    validate_raw_timing_samples,
)


def _row(
    index: int,
    *,
    system: str = "transformer",
    runtime: str = "ollama-cpu",
    device: str = "cpu_one_thread",
    cache: str = "cold",
    mode: str = "deterministic",
    prompt_bucket: str = "short",
    target: int = 64,
    trial: int = 1,
) -> dict:
    started = 1_000_000_000 + index * 10_000_000
    first = started + 1_000_000
    completed = started + 2_000_000
    threads = 1 if device != "cpu_all_core" else 14
    output = bytes([index % 251 + 1]) * target
    return {
        "run_id": f"run-{index}",
        "system_id": system,
        "runtime_id": runtime,
        "model_id": f"{system}-model",
        "model_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "configuration_sha256": "c" * 64,
        "precision": "q4_k_m",
        "seed": 20260722 + trial,
        "trial": trial,
        "device": {"kind": device, "hardware_id": "cpu-0"},
        "threads": {"requested": threads, "observed_limit": threads},
        "prompt": {
            "id": f"prompt-{prompt_bucket}", "sha256": "d" * 64,
            "bytes": 32, "bucket": prompt_bucket,
        },
        "output": {
            "target_bytes": target, "generated_bytes": target,
            "generated_tokens": target, "generated_characters": target,
            "sha256": hashlib.sha256(output).hexdigest(),
        },
        "generation": {"mode": mode, "temperature": 0.0 if mode == "deterministic" else 0.8, "seed": 7},
        "cache_state": {"kind": cache, "procedure": f"measured {cache} procedure"},
        "order": {"randomization_seed": 99, "index": index + 1, "permutation_sha256": "e" * 64},
        "timing": {
            "clock": "perf_counter_ns", "request_started_ns": started,
            "first_output_ns": first, "target_completed_ns": completed,
            "time_to_first_output_seconds": 0.001, "total_latency_seconds": 0.002,
            "phase_timings": {
                "model_load_seconds": 0.0, "prompt_preprocessing_seconds": 0.0001,
                "prefill_seconds": 0.0005, "decode_seconds": 0.0014,
                "measurement": "direct test clock",
            },
        },
        "memory": {
            "method": "process RSS", "resident_bytes": 1000, "peak_resident_bytes": 1200,
            "accelerator_allocation": {"status": "NOT_APPLICABLE_CPU", "method": "none", "peak_bytes": 0},
        },
        "execution": {"command_id": "command-1", "exit_code": 0},
        "status": "PASS",
    }


def _hardware() -> dict:
    return {
        "format": "layercake-phase1-hardware/1",
        "capture": {"command": "capture", "stdout_sha256": "f" * 64},
        "cpu": {"model": "test cpu", "physical_cores": 14, "logical_cores": 20, "instruction_sets": ["AVX2"]},
        "memory": {"total_physical_bytes": 1024},
        "gpus": [],
    }


def _matrix_rows() -> tuple[dict, list[dict]]:
    matrix = {
        "format": "layercake-phase1-benchmark-matrix/1",
        "minimum_trials_per_cell": 2,
        "axes": {
            "cache_states": ["cold", "warm"],
            "generation_modes": ["deterministic", "sampled"],
            "prompt_buckets": ["short", "medium", "long"],
            "output_target_bytes": [64, 256, 1024],
        },
        "systems": [
            {"id": "transformer", "role": "optimized_transformer_baseline", "required_devices": ["cpu_one_thread", "cpu_all_core"]},
            {"id": "layercake", "role": "fastest_layercake_baseline", "required_devices": ["cpu_one_thread", "cpu_all_core"]},
        ],
    }
    rows = []
    for index, values in enumerate(itertools.product(
        ["transformer", "layercake"], ["cpu_one_thread", "cpu_all_core"],
        ["cold", "warm"], ["deterministic", "sampled"],
        ["short", "medium", "long"], [64, 256, 1024], [1, 2],
    )):
        system, device, cache, mode, prompt, target, trial = values
        rows.append(_row(
            index, system=system, runtime="ollama-cpu" if system == "transformer" else "layercake-pytorch",
            device=device, cache=cache, mode=mode, prompt_bucket=prompt, target=target, trial=trial,
        ))
    return matrix, rows


def test_raw_timing_is_directly_recomputed_and_failure_rows_are_preserved() -> None:
    row = _row(0)
    validate_raw_timing_samples({"format": "layercake-phase1-raw-timings/1", "records": [row]})
    failure = copy.deepcopy(row)
    failure["run_id"] = "failed-run"
    failure["status"] = "FAIL"
    failure["execution"]["exit_code"] = 17
    validate_raw_timing_samples({"format": "layercake-phase1-raw-timings/1", "records": [failure]})
    tampered = copy.deepcopy(row)
    tampered["timing"]["total_latency_seconds"] = 99.0
    with pytest.raises(Phase1EvidenceError, match="not derived"):
        validate_raw_timing_samples({"format": "layercake-phase1-raw-timings/1", "records": [tampered]})


def test_raw_timing_rejects_bare_cold_boolean() -> None:
    row = _row(0)
    row["cold"] = True
    with pytest.raises(Phase1EvidenceError, match="bare cold"):
        validate_raw_timing_samples({"format": "layercake-phase1-raw-timings/1", "records": [row]})


def test_complete_matrix_requires_every_repeated_randomized_cell() -> None:
    matrix, rows = _matrix_rows()
    result = validate_benchmark_matrix(matrix, rows, _hardware())
    assert result["transformer"]["required_cells"] == 72
    with pytest.raises(Phase1EvidenceError, match="incomplete"):
        validate_benchmark_matrix(matrix, rows[:-2], _hardware())


def test_optimized_runtime_requires_real_cache_and_thread_traces() -> None:
    _, rows = _matrix_rows()
    transformer_rows = [row for row in rows if row["system_id"] == "transformer"]
    runtime = {
        "format": "layercake-phase1-runtime-manifest/1", "id": "ollama-cpu",
        "executable": {"path": "ollama.exe", "sha256": "1" * 64},
        "version": {"command": "ollama --version", "stdout": "ollama 1", "stdout_sha256": "2" * 64},
        "backend": "llama.cpp", "target_device": "cpu", "precision_contract": "q4_k_m",
        "optimization_evidence": {
            "kv_cache": {"mechanism": "llama.cpp KV cache", "raw_trace_run_ids": [transformer_rows[0]["run_id"]]},
            "kernels": {"implementation": "ggml AVX2", "instruction_sets": ["AVX2"]},
            "threading": {"raw_trace_run_ids": [row["run_id"] for row in transformer_rows[:2]]},
            "batch_one": {"batch_size": 1},
            "device_probe": {"path": "probe.json", "sha256": "3" * 64, "observed_target": "cpu"},
        },
    }
    validate_baseline_optimization(runtime, rows, runtime_id="ollama-cpu")
    runtime["optimization_evidence"]["kv_cache"]["raw_trace_run_ids"] = ["invented"]
    with pytest.raises(Phase1EvidenceError, match="do not exist"):
        validate_baseline_optimization(runtime, rows, runtime_id="ollama-cpu")
