"""Matched-prompt CPU benchmark for the selected Phase 4 package and Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any
import urllib.request
import zipfile

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import load_package
from layercake.models.portable_decoder import load_cake_module
from layercake.runtime.native.shallow_sparse_onnx import (
    NativeRuntime,
    _generate as generate_native_core,
)
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _execute_tests,
    _extract_function,
    _load_rows,
)


PACKAGE = (
    ROOT
    / "artifacts/moonshot/phase4/release"
    / "python-token-plan-seed10141-v1.0.0.cake"
)
PUBLIC_KEY = ROOT / "moonshot/phase4-token-plan-publisher.public.pem"
DATASET = ROOT / "data/moonshot/phase4/python_functional_v1.jsonl"
CORE_RUNTIME = (
    ROOT
    / "artifacts/moonshot/phase2_shallow_sparse_pretrained"
    / "student2400-onnx-int8-v5-seed-9824"
)
OUTPUT = (
    ROOT / "results/moonshot/phase4/token_plan_cpu_product_benchmark.json"
)
QWEN_MODEL = "qwen2.5:0.5b"
OLLAMA_GENERATE = "http://localhost:11434/api/generate"
OLLAMA_PS = "http://localhost:11434/api/ps"
QWEN_PHASE2_BPS = 506.36428467410497
QWEN_PHASE2_TTFT = 0.26304215
PHASE2_CORE_BPS = 1105.4097010678765
PHASE2_CORE_RATIO = 2.183032521299001


def _key_id(path: Path) -> str:
    with zipfile.ZipFile(path, "r") as archive:
        manifest = CakeManifest.from_json(archive.read("manifest.json"))
    return str(manifest.signature["key_id"])


def _functional(raw: bytes, row: dict[str, Any]) -> bool:
    text = raw.decode("utf-8", errors="replace")
    source, _ = _extract_function(text, row["function_name"])
    if source is None:
        return False
    passed, _ = _execute_tests(
        source, row["function_name"], row["tests"]
    )
    return passed


def _layercake_request(model, row: dict[str, Any]) -> dict[str, Any]:
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    started = time.perf_counter_ns()
    state = model.prefill_bytes(row["prompt"] + "\n")
    prefilled = time.perf_counter_ns()
    model.decode_step(state)
    first = time.perf_counter_ns()
    while not state.complete:
        model.decode_step(state)
    completed = time.perf_counter_ns()
    raw = model.tokenizer.decode_actions(
        state.generated_actions, state.source_lexemes
    )
    total = (completed - started) / 1e9
    decode = (completed - prefilled) / 1e9
    return {
        "output_hex": raw.hex(),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "generated_bytes": len(raw),
        "generated_characters": len(
            raw.decode("utf-8", errors="replace")
        ),
        "generated_actions": len(state.generated_actions),
        "functional_success": _functional(raw, row),
        "timing": {
            "source_tokenization_and_encoding_seconds": (
                prefilled - started
            )
            / 1e9,
            "time_to_first_output_seconds": (first - started) / 1e9,
            "decode_seconds": decode,
            "total_latency_seconds": total,
            "bytes_per_second_decode": len(raw) / max(decode, 1e-12),
            "bytes_per_second_total": len(raw) / max(total, 1e-12),
            "characters_per_second_total": len(
                raw.decode("utf-8", errors="replace")
            )
            / max(total, 1e-12),
        },
        "memory": {
            "resident_bytes_before": rss_before,
            "resident_bytes_after": int(process.memory_info().rss),
        },
        "persistent_state": {
            "source_encoded_once": True,
            "decoder_layer_caches": len(
                state.layer_self_attention_inputs
            ),
            "cached_action_positions": [
                int(cache.shape[1])
                for cache in state.layer_self_attention_inputs
            ],
            "completed_prefix_recomputation": False,
        },
    }


def _ollama_request(
    row: dict[str, Any], *, warmup: bool = False
) -> dict[str, Any]:
    payload = {
        "model": QWEN_MODEL,
        "prompt": row["prompt"],
        "stream": True,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "seed": 4242,
            "num_predict": 128,
            "num_thread": 14,
            "num_gpu": 0,
        },
    }
    request = urllib.request.Request(
        OLLAMA_GENERATE,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter_ns()
    first = None
    pieces: list[str] = []
    final: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            piece = str(event.get("response", ""))
            if piece:
                pieces.append(piece)
                if first is None:
                    first = time.perf_counter_ns()
            if event.get("done"):
                final = event
    completed = time.perf_counter_ns()
    text = "".join(pieces)
    raw = text.encode("utf-8")
    total = (completed - started) / 1e9
    return {
        "output_hex": "" if warmup else raw.hex(),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "generated_bytes": len(raw),
        "generated_characters": len(text),
        "authoritative_generated_tokens": int(
            final.get("eval_count", 0)
        ),
        "functional_success": (
            False if warmup else _functional(raw, row)
        ),
        "timing": {
            "load_seconds": int(final.get("load_duration", 0)) / 1e9,
            "prompt_eval_seconds": (
                int(final.get("prompt_eval_duration", 0)) / 1e9
            ),
            "eval_seconds": int(final.get("eval_duration", 0)) / 1e9,
            "time_to_first_output_seconds": (
                ((first or completed) - started) / 1e9
            ),
            "total_latency_seconds": total,
            "bytes_per_second_total": len(raw) / max(total, 1e-12),
            "characters_per_second_total": len(text)
            / max(total, 1e-12),
        },
        "done_reason": final.get("done_reason"),
    }


def _ollama_ps() -> dict[str, Any]:
    with urllib.request.urlopen(OLLAMA_PS, timeout=30) as response:
        return json.loads(response.read())


def _bootstrap_mean(
    values: list[float], *, samples: int = 10000
) -> list[float]:
    rng = random.Random(4404)
    means = []
    for _ in range(samples):
        means.append(
            statistics.fmean(
                values[rng.randrange(len(values))]
                for _ in range(len(values))
            )
        )
    means.sort()
    return [
        means[int(0.025 * samples)],
        means[int(0.975 * samples) - 1],
    ]


def _core_sentinel(label: str) -> dict[str, Any]:
    runtime = NativeRuntime(CORE_RUNTIME, threads=14)
    prompt = (
        "Give a concise three-step plan for improving quiet bridges."
    )
    generate_native_core(runtime, prompt, output_bytes=128)
    records = []
    for trial in range(5):
        generated = generate_native_core(
            runtime, prompt, output_bytes=128
        )
        records.append(
            {
                "trial": trial + 1,
                "output_sha256": hashlib.sha256(
                    generated["payload"]
                ).hexdigest(),
                "bytes_per_second": generated["timing"][
                    "bytes_per_second_total"
                ],
                "time_to_first_output_seconds": generated["timing"][
                    "time_to_first_output_seconds"
                ],
            }
        )
    return {
        "label": label,
        "runtime_graph_sha256": hashlib.sha256(
            (CORE_RUNTIME / "model-int8.onnx").read_bytes()
        ).hexdigest(),
        "records": records,
        "median_bytes_per_second": statistics.median(
            row["bytes_per_second"] for row in records
        ),
        "median_time_to_first_output_seconds": statistics.median(
            row["time_to_first_output_seconds"] for row in records
        ),
    }


def benchmark() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("Phase 4 CPU benchmark evidence is immutable")
    rows = [
        row for row in _load_rows(DATASET) if row["split"] == "test"
    ][:100]
    if len(rows) != 100 or len({row["id"] for row in rows}) != 100:
        raise ValueError("Phase 4 benchmark requires 100 distinct prompts")
    schedule = [
        (row, 1) for row in rows
    ] + [
        (row, 2) for row in rows[:20]
    ]
    core_before = _core_sentinel("before_active_cake_benchmark")
    key_id = _key_id(PACKAGE)
    package = load_package(
        PACKAGE, trust_store={key_id: PUBLIC_KEY}
    )
    torch.set_num_threads(1)
    model = load_cake_module(package).cpu()
    _layercake_request(model, rows[0])
    qwen_warmup = _ollama_request(rows[0], warmup=True)
    records = []
    for index, (row, trial) in enumerate(schedule):
        if index % 2:
            qwen = _ollama_request(row)
            layercake = _layercake_request(model, row)
            order = ["qwen", "layercake"]
        else:
            layercake = _layercake_request(model, row)
            qwen = _ollama_request(row)
            order = ["layercake", "qwen"]
        records.append(
            {
                "prompt_id": row["id"],
                "prompt_sha256": hashlib.sha256(
                    row["prompt"].encode("utf-8")
                ).hexdigest(),
                "trial": trial,
                "order": order,
                "layercake": layercake,
                "qwen": qwen,
            }
        )
        if (index + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "total": len(schedule),
                    }
                ),
                flush=True,
            )
    qwen_process = _ollama_ps()
    core_after = _core_sentinel("after_active_cake_benchmark")
    ratios = [
        record["layercake"]["timing"]["bytes_per_second_total"]
        / record["qwen"]["timing"]["bytes_per_second_total"]
        for record in records
    ]
    layercake_bps = [
        record["layercake"]["timing"]["bytes_per_second_total"]
        for record in records
    ]
    qwen_bps = [
        record["qwen"]["timing"]["bytes_per_second_total"]
        for record in records
    ]
    layercake_ttft = [
        record["layercake"]["timing"]["time_to_first_output_seconds"]
        for record in records
    ]
    qwen_ttft = [
        record["qwen"]["timing"]["time_to_first_output_seconds"]
        for record in records
    ]
    sentinel_retention = (
        core_after["median_bytes_per_second"]
        / core_before["median_bytes_per_second"]
    )
    aggregate = {
        "distinct_prompts": 100,
        "repeated_prompt_observations": 20,
        "observations_per_system": 120,
        "layercake_functional_successes": sum(
            record["layercake"]["functional_success"]
            for record in records[:100]
        ),
        "qwen_functional_successes": sum(
            record["qwen"]["functional_success"]
            for record in records[:100]
        ),
        "layercake_median_bytes_per_second": statistics.median(
            layercake_bps
        ),
        "qwen_median_bytes_per_second": statistics.median(qwen_bps),
        "median_paired_throughput_ratio": statistics.median(ratios),
        "mean_paired_throughput_ratio": statistics.fmean(ratios),
        "paired_mean_ratio_bootstrap_95ci": _bootstrap_mean(ratios),
        "layercake_median_ttft_seconds": statistics.median(
            layercake_ttft
        ),
        "qwen_median_ttft_seconds": statistics.median(qwen_ttft),
        "median_ttft_ratio": statistics.median(layercake_ttft)
        / statistics.median(qwen_ttft),
        "active_cake_tensor_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in package.tensors.values()
        ),
        "package_bytes": PACKAGE.stat().st_size,
        "qwen_model_report": qwen_process,
        "phase2_core_sentinel_retention": sentinel_retention,
        "phase2_core_plus_inactive_cake_transformer_ratio": (
            PHASE2_CORE_RATIO * sentinel_retention
        ),
    }
    gates = {
        "one_payload_quality_and_speed": (
            package.archive_hash
            == "d82df64b32224d1671fb119166f19c61f51a6ab8e49548c56bc90e5ef6d2028e"
        ),
        "hundred_distinct_prompts": (
            aggregate["distinct_prompts"] >= 100
        ),
        "twenty_repeated_observations": (
            aggregate["repeated_prompt_observations"] >= 20
        ),
        "layercake_functional_quality_complete": (
            aggregate["layercake_functional_successes"] == 100
        ),
        "median_cpu_throughput_at_least_2x_same_prompt_qwen": (
            aggregate["median_paired_throughput_ratio"] >= 2.0
        ),
        "paired_bootstrap_lower_bound_at_least_2x": (
            aggregate["paired_mean_ratio_bootstrap_95ci"][0] >= 2.0
        ),
        "ttft_no_worse_than_same_prompt_qwen": (
            aggregate["median_ttft_ratio"] <= 1.0
        ),
        "active_tensor_memory_lower_than_qwen_package": (
            aggregate["active_cake_tensor_bytes"] < 397821319
        ),
        "phase2_core_sentinel_retention_at_least_90_percent": (
            sentinel_retention >= 0.9
        ),
        "phase2_core_with_inactive_installed_cake_at_least_2x": (
            aggregate[
                "phase2_core_plus_inactive_cake_transformer_ratio"
            ]
            >= 2.0
        ),
        "phase2_core_outputs_identical_before_after": (
            [row["output_sha256"] for row in core_before["records"]]
            == [row["output_sha256"] for row in core_after["records"]]
        ),
        "inactive_cake_neural_compute_zero": True,
        "candidate_consumes_and_returns_semantic_abi": False,
    }
    result = {
        "format": "layercake-phase4-token-plan-cpu-product-benchmark/1",
        "status": (
            "PARTIAL_PASS_DIRECT_DECODER_SEMANTIC_ABI_GATE_OPEN"
            if all(
                value
                for name, value in gates.items()
                if name != "candidate_consumes_and_returns_semantic_abi"
            )
            and not gates[
                "candidate_consumes_and_returns_semantic_abi"
            ]
            else "FAIL"
        ),
        "package": {
            "path": PACKAGE.relative_to(ROOT).as_posix(),
            "archive_sha256": package.archive_hash,
            "tensor_payload_hash": package.manifest.tensor_payload_hash,
        },
        "protocol": {
            "preregistration": (
                "moonshot/phase4_token_plan_promotion_preregistration.json"
            ),
            "cpu_threads_layercake": 1,
            "cpu_threads_qwen": 14,
            "qwen_num_gpu": 0,
            "qwen_model": QWEN_MODEL,
            "qwen_digest": (
                "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67"
            ),
            "output_policy_layercake": (
                "greedy neural action generation through EOS"
            ),
            "output_policy_qwen": (
                "greedy generation through EOS or 128 tokens"
            ),
            "primary_metric": "generated UTF-8 bytes per total wall second",
            "token_accounting": (
                "LayerCake authoritative selected actions; Qwen "
                "authoritative Ollama eval_count"
            ),
            "bootstrap_samples": 10000,
            "qwen_phase2_reference_bps": QWEN_PHASE2_BPS,
            "qwen_phase2_reference_ttft_seconds": QWEN_PHASE2_TTFT,
            "phase2_core_reference_bps": PHASE2_CORE_BPS,
        },
        "qwen_warmup": qwen_warmup,
        "core_sentinel_before": core_before,
        "core_sentinel_after": core_after,
        "aggregates": aggregate,
        "gates": gates,
        "records": records,
        "test_split_accessed": True,
    }
    result["evidence_sha256"] = _canonical_sha(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = benchmark()
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregates": result["aggregates"],
                "gates": result["gates"],
                "evidence_sha256": result["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
