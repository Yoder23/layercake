"""Controlled representation bake-off for the LayerCake moonshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from typing import Any, Sequence

import psutil
import torch
from safetensors.torch import load_file, save_file

from layercake.models.baseline_transformer import BytePairTokenizer
from layercake.models.representation_tokenizer import HybridTokenByteTokenizer
from layercake.phase2_recertification import _output_quality
from layercake.training.data import sha256_file
from layercake.training.phase2_sparse_bpe import load_sparse_bpe_checkpoint


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("results/moonshot/representation_bakeoff")
LEDGER = Path("results/moonshot/phase2_recertification/experiment_ledger.jsonl")
QUALITY_MANIFEST = Path("results/moonshot/phase1/quality_suite_manifest.json")
QWEN_QUALITY = Path("results/moonshot/phase1/functional_quality.json")
QWEN_BPS = 506.2597482558446


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha(document: Any) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_ledger(root: Path, event: dict[str, Any]) -> None:
    path = root / LEDGER
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        )


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_hybrid_tokenizer(
    root: Path,
    base_path: Path,
    corpus_path: Path,
    output_path: Path,
    training_bytes: int,
) -> dict[str, Any]:
    """Bind a byte-fallback contract to the shared tokenizer vocabulary."""

    base_path = _resolve(root, base_path)
    corpus_path = _resolve(root, corpus_path)
    output_path = _resolve(root, output_path)
    if output_path.exists():
        raise RuntimeError(f"hybrid tokenizer artifact is immutable: {output_path}")
    started = time.perf_counter()
    base_document = _read(base_path)
    base = BytePairTokenizer([tuple(pair) for pair in base_document["merges"]])
    tokenizer = HybridTokenByteTokenizer(base)
    corpus = corpus_path.read_bytes()[:training_bytes]
    sample = corpus[: min(len(corpus), 200_000)]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    if decoded != sample:
        raise RuntimeError("hybrid tokenizer failed corpus byte round-trip")
    vectors = [
        b"ordinary English text remains compact.",
        "naïve café — 雪".encode("utf-8"),
        b"def parse_http2_id(value_7): return value_7[0]",
        b"C:\\models\\layercake\\weights.bin",
        b"\\xff malformed " + bytes([0xFF, 0xFE, 0x80]),
        b"line 1\\n  line 2\\t{exact:[format]}",
    ]
    tests = []
    for payload in vectors:
        ids = tokenizer.encode(payload)
        observed = tokenizer.decode(ids)
        tests.append({
            "input_hex": payload.hex(),
            "input_sha256": hashlib.sha256(payload).hexdigest(),
            "token_ids": ids,
            "round_trip_exact": observed == payload,
            "raw_byte_units": sum(token_id < 256 for token_id in ids),
            "total_units": len(ids),
        })
    if not all(row["round_trip_exact"] for row in tests):
        raise RuntimeError("hybrid tokenizer failed a deterministic test vector")
    elapsed = time.perf_counter() - started
    specification = tokenizer.canonical_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write(output_path, specification)
    document = {
        "format": "layercake-representation-tokenizer-manifest/1",
        "status": "PASS",
        "representation": "hybrid_token_byte",
        "tokenizer_path": output_path.relative_to(root).as_posix(),
        "tokenizer_sha256": sha256_file(output_path),
        "base_tokenizer_path": base_path.relative_to(root).as_posix(),
        "base_tokenizer_sha256": sha256_file(base_path),
        "training_corpus_path": corpus_path.relative_to(root).as_posix(),
        "training_corpus_sha256": sha256_file(corpus_path),
        "training_bytes": len(corpus),
        "build_and_verification_seconds": elapsed,
        "vocab_size": tokenizer.vocab_size,
        "merges": len(tokenizer.merges),
        "sample_raw_bytes": len(sample),
        "sample_sha256": hashlib.sha256(sample).hexdigest(),
        "sample_units": len(encoded),
        "sample_bytes_per_unit": len(sample) / len(encoded),
        "sample_round_trip_exact": decoded == sample,
        "test_vectors": tests,
        "contract": specification["hybrid_contract"],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    _write(manifest_path, document)
    _append_ledger(root, {
        "event": "representation_tokenizer_built",
        "representation": "hybrid_token_byte",
        "tokenizer": output_path.relative_to(root).as_posix(),
        "tokenizer_sha256": document["tokenizer_sha256"],
        "manifest": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "training_bytes": len(corpus),
        "build_and_verification_seconds": elapsed,
        "round_trip_exact": True,
    })
    return document


def strip_planner_checkpoint(
    root: Path,
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Remove the prohibited planner buffer without changing neural tensors."""

    source_path = _resolve(root, source_path)
    output_path = _resolve(root, output_path)
    if output_path.exists():
        raise RuntimeError(f"planner-stripped artifact is immutable: {output_path}")
    metadata = _read(source_path / "metadata.json")
    if not bool(metadata.get("english_planner", {}).get("enabled")):
        raise ValueError("source checkpoint does not contain an enabled planner")
    source_checkpoint = source_path / "model.safetensors"
    state = load_file(str(source_checkpoint), device="cpu")
    planner = state.pop("english_planner_spec", None)
    if planner is None:
        raise ValueError("source checkpoint has no planner buffer")
    neural_hashes = {
        name: hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, value in state.items()
    }
    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint = output_path / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in state.items()
        },
        str(checkpoint),
    )
    tokenizer_source = source_path / "tokenizer.json"
    tokenizer_output = output_path / "tokenizer.json"
    shutil.copyfile(tokenizer_source, tokenizer_output)
    architecture = dict(metadata["architecture"])
    architecture["constrained_english_planner"] = False
    architecture["architecture_version"] = (
        "layercake-sparse-bpe-core/4-neural-only"
    )
    converted = dict(metadata)
    converted.update({
        "format": "layercake-representation-planner-stripped-checkpoint/1",
        "status": "PASS",
        "architecture": architecture,
        "checkpoint": {
            "path": checkpoint.relative_to(root).as_posix(),
            "sha256": sha256_file(checkpoint),
        },
        "tokenizer": {
            "path": tokenizer_output.relative_to(root).as_posix(),
            "sha256": sha256_file(tokenizer_output),
        },
        "representation": {
            "class": "shared_tokenizer",
            "external_input": "UTF-8 bytes",
            "external_output": "UTF-8 bytes",
        },
        "english_planner": {
            "enabled": False,
            "stripped": True,
            "source_checkpoint_buffer_sha256": hashlib.sha256(
                planner.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            "runtime_planner_available": False,
        },
        "safety_conversion": {
            "source_checkpoint_path": source_checkpoint.relative_to(root).as_posix(),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "removed_tensors": ["english_planner_spec"],
            "neural_tensor_count": len(state),
            "neural_tensor_hashes": neural_hashes,
            "neural_tensors_changed": False,
            "training_performed": False,
        },
    })
    _write(output_path / "metadata.json", converted)
    loaded_model, _, loaded_metadata = _load_candidate(output_path)
    if loaded_model.planner_sha256() is not None:
        raise RuntimeError("planner-stripped checkpoint still exposes a planner")
    if loaded_metadata["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise RuntimeError("planner-stripped checkpoint hash is stale")
    result = {
        "status": "PASS",
        "output": output_path.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tokenizer_sha256": sha256_file(tokenizer_output),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "neural_tensors_changed": False,
        "planner_available": False,
    }
    _append_ledger(root, {
        "event": "prohibited_planner_stripped",
        **result,
    })
    return result


def enable_prompt_attention_checkpoint(
    root: Path,
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Enable cached prompt attention without changing checkpoint tensors."""

    source_path = _resolve(root, source_path)
    output_path = _resolve(root, output_path)
    if output_path.exists():
        raise RuntimeError(f"prompt-attention artifact is immutable: {output_path}")
    metadata = _read(source_path / "metadata.json")
    architecture = dict(metadata["architecture"])
    if not bool(architecture.get("prompt_conditioning")):
        raise ValueError("prompt attention requires prompt conditioning")
    if bool(architecture.get("constrained_english_planner")):
        raise ValueError("planner must be stripped before prompt attention")
    architecture["prompt_attention_pooling"] = True
    architecture["architecture_version"] = (
        "layercake-sparse-bpe-core/5-cached-prompt-attention"
    )
    output_path.mkdir(parents=True, exist_ok=False)
    checkpoint_source = source_path / "model.safetensors"
    checkpoint_output = output_path / "model.safetensors"
    tokenizer_source = source_path / "tokenizer.json"
    tokenizer_output = output_path / "tokenizer.json"
    shutil.copyfile(checkpoint_source, checkpoint_output)
    shutil.copyfile(tokenizer_source, tokenizer_output)
    converted = dict(metadata)
    converted.update({
        "format": "layercake-representation-prompt-attention-checkpoint/1",
        "architecture": architecture,
        "checkpoint": {
            "path": checkpoint_output.relative_to(root).as_posix(),
            "sha256": sha256_file(checkpoint_output),
        },
        "tokenizer": {
            "path": tokenizer_output.relative_to(root).as_posix(),
            "sha256": sha256_file(tokenizer_output),
        },
        "architecture_conversion": {
            "source_checkpoint_path": checkpoint_source.relative_to(root).as_posix(),
            "source_checkpoint_sha256": sha256_file(checkpoint_source),
            "neural_tensors_changed": False,
            "training_performed": False,
            "decode_graph_changed": False,
            "prefill_change": "cached neural attention pooling over prompt units",
        },
    })
    _write(output_path / "metadata.json", converted)
    loaded_model, _, _ = _load_candidate(output_path)
    if not loaded_model.config.prompt_attention_pooling:
        raise RuntimeError("prompt-attention contract did not load")
    result = {
        "status": "PASS",
        "output": output_path.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_output),
        "tokenizer_sha256": sha256_file(tokenizer_output),
        "neural_tensors_changed": False,
        "decode_graph_changed": False,
    }
    _append_ledger(root, {
        "event": "cached_prompt_attention_enabled",
        **result,
    })
    return result


def _load_prompts(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    path = root / QUALITY_MANIFEST
    document = _read(path)
    prompts = list(document["prompts"])
    if len(prompts) < 100 or len({row["id"] for row in prompts}) < 100:
        raise ValueError("representation screen requires 100 frozen prompts")
    for prompt in prompts:
        payload = prompt["text"].encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != prompt["sha256"]:
            raise ValueError(f"prompt hash mismatch: {prompt['id']}")
    return path, prompts


def _load_candidate(checkpoint: Path):
    model, tokenizer, metadata = load_sparse_bpe_checkpoint(
        checkpoint, device="cpu"
    )
    if bool(metadata.get("english_planner", {}).get("enabled")):
        raise ValueError("planner-enabled checkpoints are prohibited from the bake-off")
    if bool(metadata["architecture"].get("constrained_english_planner")):
        raise ValueError("planner-enabled architecture is prohibited from the bake-off")
    return model.eval(), tokenizer, metadata


def _select_token(
    logits: torch.Tensor,
    generated_ids: Sequence[int],
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    scores = logits.clone()
    if repetition_penalty != 1.0 and generated_ids:
        repeated = sorted(set(generated_ids[-64:]))
        token_ids = torch.tensor(repeated, device=scores.device)
        values = scores[0, token_ids]
        scores[0, token_ids] = torch.where(
            values < 0,
            values * repetition_penalty,
            values / repetition_penalty,
        )
    size = no_repeat_ngram_size
    if size > 1 and len(generated_ids) >= size - 1:
        prefix = tuple(generated_ids[-(size - 1):])
        banned = {
            generated_ids[index + size - 1]
            for index in range(len(generated_ids) - size + 1)
            if tuple(generated_ids[index:index + size - 1]) == prefix
        }
        if banned:
            scores[0, torch.tensor(sorted(banned), device=scores.device)] = (
                -torch.inf
            )
    return scores.argmax(dim=-1)


def _generate(
    model,
    tokenizer,
    prompt_bytes: bytes,
    output_bytes: int,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> dict[str, Any]:
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    started = time.perf_counter_ns()
    prompt_ids = tokenizer.encode(prompt_bytes)
    tokenized = time.perf_counter_ns()
    if not prompt_ids:
        raise ValueError("empty tokenized prompt")
    if len(prompt_ids) >= model.config.max_tokens:
        raise ValueError("tokenized prompt exceeds checkpoint context")
    with torch.inference_mode():
        state = model.prefill(torch.tensor([prompt_ids], dtype=torch.long))
        first_ready = time.perf_counter_ns()
        generated = bytearray()
        generated_ids: list[int] = []
        decode_started = first_ready
        sparse_steps = 0
        while len(generated) < output_bytes:
            if len(prompt_ids) + len(generated_ids) >= model.config.max_tokens:
                raise RuntimeError("checkpoint context ended before byte target")
            selected = _select_token(
                state.next_logits,
                generated_ids,
                repetition_penalty,
                no_repeat_ngram_size,
            )
            token_id = int(selected.item())
            generated.extend(tokenizer.decode([token_id]))
            generated_ids.append(token_id)
            _, state = model.decode_step(state, next_token=selected)
            counts = model.cakes.last_assignment_counts
            if counts is None or int(counts.sum().item()) != 1:
                raise RuntimeError("decode step did not physically dispatch one expert")
            sparse_steps += 1
        completed = time.perf_counter_ns()
    payload = bytes(generated)
    rss_after = int(process.memory_info().rss)
    tokenization_seconds = (tokenized - started) / 1e9
    prefill_seconds = (first_ready - tokenized) / 1e9
    decode_seconds = (completed - decode_started) / 1e9
    total_seconds = (completed - started) / 1e9
    return {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_ids),
        "generated_token_ids": generated_ids,
        "generated_bytes": payload,
        "tokenization_seconds": tokenization_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "total_latency_seconds": total_seconds,
        "time_to_first_output_seconds": (first_ready - started) / 1e9,
        "bytes_per_second_decode": len(payload) / decode_seconds,
        "bytes_per_second_total": len(payload) / total_seconds,
        "resident_memory_bytes_before": rss_before,
        "resident_memory_bytes_after": rss_after,
        "sparse_decode_steps": sparse_steps,
        "maximum_active_experts_per_generated_token": 1,
        "external_path_counters": {
            "planner_calls": 0,
            "retrieval_calls": 0,
            "stored_answer_calls": 0,
            "template_calls": 0,
            "forced_token_calls": 0,
        },
    }


def screen_checkpoint(
    root: Path,
    checkpoint: Path,
    output_path: Path,
    output_bytes: int,
    threads: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    """Run the frozen 100-prompt autonomous neural product screen."""

    checkpoint = _resolve(root, checkpoint)
    output_path = _resolve(root, output_path)
    torch.set_num_threads(threads)
    model, tokenizer, metadata = _load_candidate(checkpoint)
    manifest_path, prompts = _load_prompts(root)
    with torch.inference_mode():
        warm_ids = tokenizer.encode(b"Warm autonomous LayerCake generation.")
        warm = model.prefill(torch.tensor([warm_ids], dtype=torch.long))
        _, _ = model.decode_step(warm)
    records = []
    for prompt in prompts:
        result = _generate(
            model,
            tokenizer,
            prompt["text"].encode("utf-8"),
            output_bytes,
            repetition_penalty,
            no_repeat_ngram_size,
        )
        payload = result.pop("generated_bytes")
        token_ids = result.pop("generated_token_ids")
        records.append({
            "prompt_id": prompt["id"],
            "prompt_sha256": prompt["sha256"],
            "category": prompt["category"],
            "generated_hex": payload.hex(),
            "generated_sha256": hashlib.sha256(payload).hexdigest(),
            "generated_token_ids_sha256": _canonical_sha(token_ids),
            "metrics": _output_quality(payload),
            "timing": {
                key: result[key]
                for key in (
                    "tokenization_seconds",
                    "prefill_seconds",
                    "decode_seconds",
                    "total_latency_seconds",
                    "time_to_first_output_seconds",
                    "bytes_per_second_decode",
                    "bytes_per_second_total",
                    "resident_memory_bytes_before",
                    "resident_memory_bytes_after",
                )
            },
            "execution": {
                "prompt_tokens": result["prompt_tokens"],
                "generated_tokens": result["generated_tokens"],
                "sparse_decode_steps": result["sparse_decode_steps"],
                "maximum_active_experts_per_generated_token": 1,
                "external_path_counters": result["external_path_counters"],
            },
        })
    metric_names = tuple(records[0]["metrics"])
    aggregates = {
        name: statistics.mean(
            float(record["metrics"][name]) for record in records
        )
        for name in metric_names
    }
    for name in (
        "tokenization_seconds",
        "time_to_first_output_seconds",
        "total_latency_seconds",
        "bytes_per_second_decode",
        "bytes_per_second_total",
        "resident_memory_bytes_after",
    ):
        aggregates[f"median_{name}"] = statistics.median(
            float(record["timing"][name]) for record in records
        )
    qwen = _read(root / QWEN_QUALITY)[
        "systems"
    ]["qwen25-05b-cpu"]["aggregates"]
    comparison = {
        "repetition_rate_delta_layercake_minus_qwen": (
            aggregates["repetition_rate"] - qwen["repetition_rate"]
        ),
        "word_diversity_delta_layercake_minus_qwen": (
            aggregates["word_diversity"] - qwen["word_diversity"]
        ),
        "valid_utf8_delta_layercake_minus_qwen": (
            aggregates["valid_utf8"] - qwen["valid_utf8"]
        ),
        "transformer_relative_median_decode_throughput": (
            aggregates["median_bytes_per_second_decode"] / QWEN_BPS
        ),
        "product_surface_noninferiority_pass": (
            aggregates["repetition_rate"] <= qwen["repetition_rate"] + 0.02
            and aggregates["word_diversity"] >= qwen["word_diversity"] - 0.02
            and aggregates["invalid_output"] <= qwen["invalid_output"]
        ),
    }
    representation = (
        "hybrid_token_byte"
        if isinstance(tokenizer, HybridTokenByteTokenizer)
        else "shared_tokenizer"
    )
    command = " ".join([
        sys.executable,
        "-m",
        "layercake.representation_bakeoff",
        "screen",
        "--checkpoint",
        checkpoint.relative_to(root).as_posix(),
        "--output",
        output_path.relative_to(root).as_posix(),
        "--output-bytes",
        str(output_bytes),
        "--threads",
        str(threads),
        "--repetition-penalty",
        str(repetition_penalty),
        "--no-repeat-ngram-size",
        str(no_repeat_ngram_size),
    ])
    document = {
        "format": "layercake-representation-functional-screen/1",
        "status": "PASS",
        "representation": representation,
        "candidate": checkpoint.parent.name + "/" + checkpoint.name,
        "checkpoint_path": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "tokenizer_sha256": metadata["tokenizer"]["sha256"],
        "architecture": metadata["architecture"],
        "parameters": metadata["parameters"],
        "quality": metadata["quality"],
        "training": metadata["training"],
        "prompt_manifest": manifest_path.relative_to(root).as_posix(),
        "prompt_manifest_sha256": sha256_file(manifest_path),
        "distinct_prompts": len(prompts),
        "output_bytes_per_prompt_minimum": output_bytes,
        "decoding": {
            "mode": (
                "greedy"
                if repetition_penalty == 1.0 and no_repeat_ngram_size == 0
                else "deterministic_logit_control"
            ),
            "source": "checkpoint neural token-or-byte logits",
            "external_override": False,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        },
        "threads": threads,
        "test_accessed": False,
        "exact_command": command,
        "records": records,
        "aggregates": aggregates,
        "qwen_product_reference_aggregates": qwen,
        "comparison": comparison,
    }
    document["screen_sha256"] = _canonical_sha(document)
    _write(output_path, document)
    _append_ledger(root, {
        "event": "representation_functional_screen_completed",
        "representation": representation,
        "checkpoint_hash": document["checkpoint_sha256"],
        "screen": output_path.relative_to(root).as_posix(),
        "screen_sha256": document["screen_sha256"],
        "distinct_prompts": len(prompts),
        "output_bytes": output_bytes,
        "aggregates": aggregates,
        "comparison": comparison,
        "test_accessed": False,
    })
    return {
        "status": "PASS",
        "representation": representation,
        "checkpoint_sha256": document["checkpoint_sha256"],
        "screen_sha256": document["screen_sha256"],
        "aggregates": aggregates,
        "comparison": comparison,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def benchmark_checkpoint(
    root: Path,
    checkpoint: Path,
    output_path: Path,
    output_bytes: int,
    repetitions: int,
    threads: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    if repetitions < 20:
        raise ValueError("headline timing requires at least 20 observations")
    checkpoint = _resolve(root, checkpoint)
    output_path = _resolve(root, output_path)
    torch.set_num_threads(threads)
    model, tokenizer, metadata = _load_candidate(checkpoint)
    _, prompts = _load_prompts(root)
    prompt = prompts[0]
    warm = _generate(
        model,
        tokenizer,
        prompt["text"].encode("utf-8"),
        min(16, output_bytes),
        repetition_penalty,
        no_repeat_ngram_size,
    )
    del warm
    records = []
    for index in range(repetitions):
        result = _generate(
            model,
            tokenizer,
            prompt["text"].encode("utf-8"),
            output_bytes,
            repetition_penalty,
            no_repeat_ngram_size,
        )
        payload = result.pop("generated_bytes")
        token_ids = result.pop("generated_token_ids")
        records.append({
            "observation_index": index,
            "generated_sha256": hashlib.sha256(payload).hexdigest(),
            "generated_token_ids_sha256": _canonical_sha(token_ids),
            **result,
        })
    metric_names = (
        "tokenization_seconds",
        "prefill_seconds",
        "decode_seconds",
        "total_latency_seconds",
        "time_to_first_output_seconds",
        "bytes_per_second_decode",
        "bytes_per_second_total",
        "resident_memory_bytes_before",
        "resident_memory_bytes_after",
    )
    summary = {}
    for name in metric_names:
        values = [float(row[name]) for row in records]
        summary[name] = {
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
        }
    summary["observations"] = repetitions
    ratio = summary["bytes_per_second_decode"]["median"] / QWEN_BPS
    representation = (
        "hybrid_token_byte"
        if isinstance(tokenizer, HybridTokenByteTokenizer)
        else "shared_tokenizer"
    )
    document = {
        "format": "layercake-representation-timing/1",
        "status": "PASS",
        "representation": representation,
        "checkpoint_path": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "tokenizer_sha256": metadata["tokenizer"]["sha256"],
        "parameters": metadata["parameters"],
        "output_bytes_minimum": output_bytes,
        "repetitions": repetitions,
        "threads": threads,
        "decoding": {
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        },
        "prompt_id": prompt["id"],
        "prompt_sha256": prompt["sha256"],
        "raw_observations": records,
        "summary": summary,
        "optimized_transformer_median_bytes_per_second": QWEN_BPS,
        "transformer_relative_median_decode_throughput": ratio,
        "cpu_2x_gate_pass": ratio >= 2.0,
        "test_accessed": False,
    }
    document["benchmark_sha256"] = _canonical_sha(document)
    _write(output_path, document)
    _append_ledger(root, {
        "event": "representation_timing_completed",
        "representation": representation,
        "checkpoint_hash": document["checkpoint_sha256"],
        "benchmark": output_path.relative_to(root).as_posix(),
        "benchmark_sha256": document["benchmark_sha256"],
        "output_bytes": output_bytes,
        "observations": repetitions,
        "median_bytes_per_second": summary["bytes_per_second_decode"]["median"],
        "transformer_relative_ratio": ratio,
        "cpu_2x_gate_pass": ratio >= 2.0,
    })
    return {
        "status": "PASS",
        "representation": representation,
        "checkpoint_sha256": document["checkpoint_sha256"],
        "benchmark_sha256": document["benchmark_sha256"],
        "median_bytes_per_second": summary["bytes_per_second_decode"]["median"],
        "transformer_relative_ratio": ratio,
        "cpu_2x_gate_pass": ratio >= 2.0,
    }


def certify_finetune_provenance(
    root: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Correct inherited conversion flags after a real neural fine-tune."""

    checkpoint_path = _resolve(root, checkpoint_path)
    metadata_path = checkpoint_path / "metadata.json"
    metadata = _read(metadata_path)
    parent_record = metadata.get("parent_checkpoint")
    distillation = metadata.get("instruction_distillation")
    if not parent_record or not distillation:
        raise ValueError("checkpoint is not an instruction-fine-tuned artifact")
    checkpoint = checkpoint_path / "model.safetensors"
    parent_checkpoint = _resolve(root, Path(parent_record["path"]))
    current_sha = sha256_file(checkpoint)
    parent_sha = sha256_file(parent_checkpoint)
    if current_sha == parent_sha:
        raise RuntimeError("fine-tuned checkpoint is byte-identical to its parent")
    prior_metadata_sha = sha256_file(metadata_path)
    correction = {
        "format": "layercake-finetune-provenance-correction/1",
        "prior_metadata_sha256": prior_metadata_sha,
        "parent_checkpoint_path": parent_checkpoint.relative_to(root).as_posix(),
        "parent_checkpoint_sha256": parent_sha,
        "checkpoint_sha256": current_sha,
        "training_performed": True,
        "neural_tensors_changed_from_parent": True,
        "instruction_steps": int(distillation["steps"]),
        "instruction_corpus_sha256": distillation["corpus_sha256"],
        "reason": (
            "Inherited no-neural-change flags apply only to the preceding "
            "planner-removal or architecture-contract conversion."
        ),
    }
    metadata["fine_tune_provenance"] = correction
    for conversion_key in ("architecture_conversion", "safety_conversion"):
        if conversion_key in metadata:
            metadata[conversion_key] = {
                **metadata[conversion_key],
                "subsequent_fine_tune_performed": True,
                "neural_tensors_changed_after_conversion": True,
            }
    history = list(metadata.get("metadata_correction_history", []))
    history.append(correction)
    metadata["metadata_correction_history"] = history
    _write(metadata_path, metadata)
    result = {
        "status": "PASS",
        "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
        "checkpoint_sha256": current_sha,
        "parent_checkpoint_sha256": parent_sha,
        "prior_metadata_sha256": prior_metadata_sha,
        "corrected_metadata_sha256": sha256_file(metadata_path),
        "neural_tensors_changed_from_parent": True,
    }
    _append_ledger(root, {
        "event": "finetune_provenance_corrected",
        **result,
    })
    return result


def _evidence_record(root: Path, path: Path) -> dict[str, Any]:
    resolved = _resolve(root, path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _bootstrap_median_interval(
    values: Sequence[float],
    *,
    seed: int = 20260723,
    samples: int = 10_000,
) -> dict[str, Any]:
    generator = random.Random(seed)
    observed = [float(value) for value in values]
    estimates = []
    for _ in range(samples):
        estimates.append(statistics.median(
            observed[generator.randrange(len(observed))]
            for _ in range(len(observed))
        ))
    estimates.sort()
    return {
        "method": "nonparametric bootstrap of the median",
        "seed": seed,
        "resamples": samples,
        "observations": len(observed),
        "estimate": statistics.median(observed),
        "confidence_level": 0.95,
        "lower": estimates[int(0.025 * samples)],
        "upper": estimates[min(samples - 1, int(0.975 * samples))],
    }


def finalize_bakeoff(root: Path) -> dict[str, Any]:
    """Close the bounded bake-off from raw evidence without promoting a loser."""

    evidence_root = root / EVIDENCE
    output_paths = (
        evidence_root / "pareto_frontier.json",
        evidence_root / "representation_decision.json",
        evidence_root / "release_report.md",
    )
    if any(path.exists() for path in output_paths):
        raise RuntimeError("representation decision package is immutable")
    for directory in (
        "dynamic_patch_branch",
        "raw_runs",
        "training_curves",
        "generation_samples",
        "performance",
        "memory",
        "statistical_analysis",
    ):
        (evidence_root / directory).mkdir(parents=True, exist_ok=True)

    byte_frontier_path = Path(
        "results/moonshot/representation_bakeoff/byte_branch/pareto_frontier.json"
    )
    byte_decision_path = Path(
        "results/moonshot/representation_bakeoff/byte_branch/branch_decision.json"
    )
    token_screen_path = Path(
        "results/moonshot/representation_bakeoff/token_branch/"
        "functional_screen_prompt_attention_curated_seed9824_decodecontrol640.json"
    )
    token_semantic_path = Path(
        "results/moonshot/representation_bakeoff/token_branch/"
        "functional_semantic_audit_prompt_attention_curated_seed9824_decodecontrol640.json"
    )
    token_128_path = Path(
        "results/moonshot/representation_bakeoff/performance/"
        "token_prompt_attention_128x20.json"
    )
    token_1024_path = Path(
        "results/moonshot/representation_bakeoff/performance/"
        "token_prompt_attention_1024x20.json"
    )
    token_checkpoint_path = Path(
        "artifacts/moonshot/representation_bakeoff/"
        "shared_tokenizer_prompt_attention_curated/seed-9824"
    )
    hybrid_screen_path = Path(
        "results/moonshot/representation_bakeoff/hybrid_branch/"
        "functional_screen_seed9824_decodecontrol.json"
    )
    hybrid_semantic_path = Path(
        "results/moonshot/representation_bakeoff/hybrid_branch/"
        "functional_semantic_audit_seed9824_decodecontrol.json"
    )
    hybrid_128_path = Path(
        "results/moonshot/representation_bakeoff/performance/"
        "hybrid_token_byte_128x20.json"
    )
    hybrid_1024_path = Path(
        "results/moonshot/representation_bakeoff/performance/"
        "hybrid_token_byte_1024x20.json"
    )
    hybrid_checkpoint_path = Path(
        "artifacts/moonshot/representation_bakeoff/"
        "hybrid_token_byte_10m/seed-9824"
    )
    hybrid_manifest_path = Path(
        "results/moonshot/representation_bakeoff/hybrid_branch/"
        "hybrid_token_byte_2304.manifest.json"
    )

    byte_frontier = _read(root / byte_frontier_path)
    byte_decision = _read(root / byte_decision_path)
    token_screen = _read(root / token_screen_path)
    token_semantic = _read(root / token_semantic_path)
    token_128 = _read(root / token_128_path)
    token_1024 = _read(root / token_1024_path)
    token_metadata = _read(root / token_checkpoint_path / "metadata.json")
    hybrid_screen = _read(root / hybrid_screen_path)
    hybrid_semantic = _read(root / hybrid_semantic_path)
    hybrid_128 = _read(root / hybrid_128_path)
    hybrid_1024 = _read(root / hybrid_1024_path)
    hybrid_metadata = _read(root / hybrid_checkpoint_path / "metadata.json")
    hybrid_manifest = _read(root / hybrid_manifest_path)

    quality_ceiling = float(
        byte_frontier["locked_gates"]["quality_ceiling_validation_bpb"]
    )
    qwen_bps = float(
        byte_frontier["locked_gates"][
            "optimized_transformer_median_bytes_per_second"
        ]
    )
    qwen_row = byte_frontier["rows"][0]
    byte_row = byte_frontier["rows"][1]
    pointer_row = byte_frontier["rows"][2]

    token_row = {
        "branch": "B_SHARED_TOKENIZER",
        "candidate": "shared_tokenizer_prompt_attention_curated/seed-9824",
        "architecture_sha256": _canonical_sha(token_metadata["architecture"]),
        "representation_sha256": _canonical_sha(token_metadata["representation"]),
        "tokenizer_sha256": token_metadata["tokenizer"]["sha256"],
        "checkpoint_sha256": token_metadata["checkpoint"]["sha256"],
        "data_hashes": token_metadata["data"],
        "validation_bpb": token_metadata["quality"]["validation"]["bits_per_byte"],
        "selection_bpb": token_metadata["quality"]["architecture_selection"][
            "bits_per_byte"
        ],
        "quality_ceiling_bpb": quality_ceiling,
        "language_model_quality_pass": (
            token_metadata["quality"]["validation"]["bits_per_byte"]
            <= quality_ceiling
        ),
        "functional_surface_pass": token_screen["comparison"][
            "product_surface_noninferiority_pass"
        ],
        "functional_semantic_pass": token_semantic["comparison"][
            "product_semantic_noninferiority_pass"
        ],
        "core_adherence_pass_rate": token_semantic["systems"]["layercake"][
            "aggregates"
        ]["core_adherence_pass"],
        "topic_token_recall": token_semantic["systems"]["layercake"][
            "aggregates"
        ]["topic_token_recall"],
        "repetition_rate_4gram": token_screen["aggregates"]["repetition_rate"],
        "word_diversity": token_screen["aggregates"]["word_diversity"],
        "valid_utf8_rate": token_screen["aggregates"]["valid_utf8"],
        "median_cpu_bytes_per_second_128": token_128["summary"][
            "bytes_per_second_decode"
        ]["median"],
        "transformer_relative_throughput_128": token_128[
            "transformer_relative_median_decode_throughput"
        ],
        "median_cpu_bytes_per_second_1024": token_1024["summary"][
            "bytes_per_second_decode"
        ]["median"],
        "transformer_relative_throughput_1024": token_1024[
            "transformer_relative_median_decode_throughput"
        ],
        "median_time_to_first_output_seconds": token_screen["aggregates"][
            "median_time_to_first_output_seconds"
        ],
        "median_process_resident_bytes": token_screen["aggregates"][
            "median_resident_memory_bytes_after"
        ],
        "active_parameters": token_metadata["parameters"]["active"],
        "total_parameters": token_metadata["parameters"]["total"],
        "active_fraction": token_metadata["parameters"]["active_fraction"],
        "physical_top1_sparse_execution": (
            token_metadata["routing"]["physically_dispatched"]
            and token_metadata["routing"]["maximum_active_experts_per_token"] == 1
        ),
        "persistent_incremental_state": token_metadata["incremental_state"][
            "implemented"
        ],
        "planner_enabled": bool(
            token_metadata.get("english_planner", {}).get("enabled")
        ),
        "training_wall_seconds": token_metadata["instruction_distillation"][
            "wall_seconds"
        ],
        "raw_bytes_seen_before_instruction_finetune": token_metadata["training"][
            "cumulative_raw_bytes_seen"
        ],
        "rejection_reasons": [
            "functional semantic noninferiority failed",
            "functional surface noninferiority failed",
            "process-resident memory exceeds the deployment transformer",
            "shared-tokenizer training provenance is incomplete for production",
            "canonical portable-cake ABI is not implemented by this candidate",
        ],
    }
    hybrid_row = {
        "branch": "C_HYBRID_TOKEN_BYTE",
        "candidate": "hybrid_token_byte_10m/seed-9824",
        "architecture_sha256": _canonical_sha(hybrid_metadata["architecture"]),
        "representation_sha256": _canonical_sha(hybrid_metadata["representation"]),
        "tokenizer_sha256": hybrid_metadata["tokenizer"]["sha256"],
        "checkpoint_sha256": hybrid_metadata["checkpoint"]["sha256"],
        "data_hashes": hybrid_metadata["data"],
        "validation_bpb": hybrid_metadata["quality"]["validation"]["bits_per_byte"],
        "selection_bpb": hybrid_metadata["quality"]["architecture_selection"][
            "bits_per_byte"
        ],
        "quality_ceiling_bpb": quality_ceiling,
        "language_model_quality_pass": (
            hybrid_metadata["quality"]["validation"]["bits_per_byte"]
            <= quality_ceiling
        ),
        "functional_surface_pass": hybrid_screen["comparison"][
            "product_surface_noninferiority_pass"
        ],
        "functional_semantic_pass": hybrid_semantic["comparison"][
            "product_semantic_noninferiority_pass"
        ],
        "core_adherence_pass_rate": hybrid_semantic["systems"]["layercake"][
            "aggregates"
        ]["core_adherence_pass"],
        "topic_token_recall": hybrid_semantic["systems"]["layercake"][
            "aggregates"
        ]["topic_token_recall"],
        "repetition_rate_4gram": hybrid_screen["aggregates"]["repetition_rate"],
        "word_diversity": hybrid_screen["aggregates"]["word_diversity"],
        "valid_utf8_rate": hybrid_screen["aggregates"]["valid_utf8"],
        "byte_fallback_round_trip_exact": (
            hybrid_manifest["sample_round_trip_exact"]
            and all(
                row["round_trip_exact"]
                for row in hybrid_manifest["test_vectors"]
            )
        ),
        "median_cpu_bytes_per_second_128": hybrid_128["summary"][
            "bytes_per_second_decode"
        ]["median"],
        "transformer_relative_throughput_128": hybrid_128[
            "transformer_relative_median_decode_throughput"
        ],
        "median_cpu_bytes_per_second_1024": hybrid_1024["summary"][
            "bytes_per_second_decode"
        ]["median"],
        "transformer_relative_throughput_1024": hybrid_1024[
            "transformer_relative_median_decode_throughput"
        ],
        "median_time_to_first_output_seconds": hybrid_screen["aggregates"][
            "median_time_to_first_output_seconds"
        ],
        "median_process_resident_bytes": hybrid_screen["aggregates"][
            "median_resident_memory_bytes_after"
        ],
        "active_parameters": hybrid_metadata["parameters"]["active"],
        "total_parameters": hybrid_metadata["parameters"]["total"],
        "active_fraction": hybrid_metadata["parameters"]["active_fraction"],
        "physical_top1_sparse_execution": (
            hybrid_metadata["routing"]["physically_dispatched"]
            and hybrid_metadata["routing"]["maximum_active_experts_per_token"] == 1
        ),
        "persistent_incremental_state": hybrid_metadata["incremental_state"][
            "implemented"
        ],
        "planner_enabled": bool(
            hybrid_metadata.get("english_planner", {}).get("enabled")
        ),
        "training_wall_seconds": hybrid_metadata["training"]["wall_seconds"],
        "raw_bytes_seen": hybrid_metadata["training"][
            "cumulative_raw_bytes_seen"
        ],
        "rejection_reasons": [
            "functional semantic noninferiority failed",
            "functional surface noninferiority failed",
            "1024-byte CPU throughput is below 2x",
            "process-resident memory exceeds the deployment transformer",
            "canonical portable-cake ABI is not implemented by this candidate",
        ],
    }

    token_branch = {
        "format": "layercake-representation-branch-decision/1",
        "status": "CLOSED_NEGATIVE",
        "branch": "B_SHARED_TOKENIZER",
        "candidate": token_row,
        "commands": [
            token_screen["exact_command"],
            (
                f"{sys.executable} -m layercake.representation_bakeoff benchmark "
                f"--checkpoint {token_checkpoint_path.as_posix()} --output "
                f"{token_128_path.as_posix()} --output-bytes 128 "
                "--repetitions 20 --threads 1 --repetition-penalty 1.15 "
                "--no-repeat-ngram-size 4"
            ),
            (
                f"{sys.executable} -m layercake.representation_bakeoff benchmark "
                f"--checkpoint {token_checkpoint_path.as_posix()} --output "
                f"{token_1024_path.as_posix()} --output-bytes 1024 "
                "--repetitions 20 --threads 1 --repetition-penalty 1.15 "
                "--no-repeat-ngram-size 4"
            ),
        ],
        "failed_seeds": [9824],
        "promotion_authorized": False,
        "additional_nearby_variants_authorized": False,
        "test_accessed": False,
        "evidence": [
            _evidence_record(root, token_screen_path),
            _evidence_record(root, token_semantic_path),
            _evidence_record(root, token_128_path),
            _evidence_record(root, token_1024_path),
            _evidence_record(root, token_checkpoint_path / "metadata.json"),
            _evidence_record(root, token_checkpoint_path / "model.safetensors"),
            _evidence_record(root, token_checkpoint_path / "tokenizer.json"),
        ],
    }
    hybrid_branch = {
        "format": "layercake-representation-branch-decision/1",
        "status": "CLOSED_NEGATIVE",
        "branch": "C_HYBRID_TOKEN_BYTE",
        "candidate": hybrid_row,
        "commands": [
            hybrid_screen["exact_command"],
            (
                f"{sys.executable} -m layercake.representation_bakeoff benchmark "
                f"--checkpoint {hybrid_checkpoint_path.as_posix()} --output "
                f"{hybrid_128_path.as_posix()} --output-bytes 128 "
                "--repetitions 20 --threads 1 --repetition-penalty 1.15 "
                "--no-repeat-ngram-size 4"
            ),
            (
                f"{sys.executable} -m layercake.representation_bakeoff benchmark "
                f"--checkpoint {hybrid_checkpoint_path.as_posix()} --output "
                f"{hybrid_1024_path.as_posix()} --output-bytes 1024 "
                "--repetitions 20 --threads 1 --repetition-penalty 1.15 "
                "--no-repeat-ngram-size 4"
            ),
        ],
        "failed_seeds": [9824],
        "promotion_authorized": False,
        "additional_nearby_variants_authorized": False,
        "test_accessed": False,
        "evidence": [
            _evidence_record(root, hybrid_screen_path),
            _evidence_record(root, hybrid_semantic_path),
            _evidence_record(root, hybrid_128_path),
            _evidence_record(root, hybrid_1024_path),
            _evidence_record(root, hybrid_checkpoint_path / "metadata.json"),
            _evidence_record(root, hybrid_checkpoint_path / "model.safetensors"),
            _evidence_record(root, hybrid_checkpoint_path / "tokenizer.json"),
            _evidence_record(root, hybrid_manifest_path),
        ],
    }
    _write(evidence_root / "token_branch" / "branch_decision.json", token_branch)
    _write(evidence_root / "hybrid_branch" / "branch_decision.json", hybrid_branch)

    dynamic_branch = {
        "format": "layercake-representation-branch-decision/1",
        "status": "NOT_AUTHORIZED",
        "branch": "D_DYNAMIC_LATENT_PATCH",
        "reason": (
            "No profiler or quality evidence identified a materially distinct "
            "dynamic-patch mechanism likely to improve both semantic adherence "
            "and long-horizon CPU throughput."
        ),
        "runs": 0,
        "checkpoint": None,
    }
    _write(
        evidence_root / "dynamic_patch_branch" / "not_authorized.json",
        dynamic_branch,
    )

    statistics_document = {
        "format": "layercake-representation-bootstrap-analysis/1",
        "status": "PASS",
        "optimized_transformer_median_bytes_per_second": qwen_bps,
        "intervals": {
            "shared_tokenizer_128": _bootstrap_median_interval([
                row["bytes_per_second_decode"]
                for row in token_128["raw_observations"]
            ]),
            "shared_tokenizer_1024": _bootstrap_median_interval([
                row["bytes_per_second_decode"]
                for row in token_1024["raw_observations"]
            ]),
            "hybrid_token_byte_128": _bootstrap_median_interval([
                row["bytes_per_second_decode"]
                for row in hybrid_128["raw_observations"]
            ]),
            "hybrid_token_byte_1024": _bootstrap_median_interval([
                row["bytes_per_second_decode"]
                for row in hybrid_1024["raw_observations"]
            ]),
        },
        "notes": (
            "Each headline interval uses 20 repeated observations. The "
            "100-prompt functional screens are distinct-prompt evidence and "
            "are not mislabeled as repeated timing."
        ),
    }
    _write(
        evidence_root / "statistical_analysis" / "bootstrap_throughput.json",
        statistics_document,
    )

    memory_document = {
        "format": "layercake-representation-memory-comparison/1",
        "locked_metric": "median process resident bytes after generation",
        "optimized_transformer": {
            "candidate": qwen_row["candidate"],
            "median_bytes": qwen_row["median_process_resident_bytes"],
        },
        "byte_pointer": {
            "candidate": pointer_row["candidate"],
            "median_bytes": pointer_row["median_process_resident_bytes"],
            "lower_than_transformer": False,
        },
        "shared_tokenizer": {
            "candidate": token_row["candidate"],
            "median_bytes": token_row["median_process_resident_bytes"],
            "lower_than_transformer": (
                token_row["median_process_resident_bytes"]
                < qwen_row["median_process_resident_bytes"]
            ),
        },
        "hybrid_token_byte": {
            "candidate": hybrid_row["candidate"],
            "median_bytes": hybrid_row["median_process_resident_bytes"],
            "lower_than_transformer": (
                hybrid_row["median_process_resident_bytes"]
                < qwen_row["median_process_resident_bytes"]
            ),
        },
    }
    _write(evidence_root / "memory" / "comparison.json", memory_document)

    training_document = {
        "format": "layercake-representation-training-curves/1",
        "shared_tokenizer": {
            "checkpoint_sha256": token_row["checkpoint_sha256"],
            "language_model_curve": token_metadata["training"]["curves"],
            "instruction_curve": token_metadata["instruction_distillation"][
                "curves"
            ],
            "instruction_wall_seconds": token_row["training_wall_seconds"],
        },
        "hybrid_token_byte": {
            "checkpoint_sha256": hybrid_row["checkpoint_sha256"],
            "language_model_curve": hybrid_metadata["training"]["curves"],
            "training_wall_seconds": hybrid_row["training_wall_seconds"],
            "tokenizer_build_seconds": hybrid_manifest[
                "build_and_verification_seconds"
            ],
        },
        "promotion_assessment": (
            "No branch is promotable, so fixed-compute three-seed and formal "
            "Phase 3 time-to-quality measurements are not authorized."
        ),
    }
    _write(evidence_root / "training_curves" / "summary.json", training_document)

    generation_manifest = {
        "format": "layercake-representation-generation-sample-manifest/1",
        "records_are_embedded_in": [
            _evidence_record(root, token_screen_path),
            _evidence_record(root, hybrid_screen_path),
            _evidence_record(
                root,
                Path(
                    "results/moonshot/phase2_recertification/"
                    "functional_screen_pointer_gate_step300_greedy128.json"
                ),
            ),
        ],
        "autonomous_generation": True,
        "external_override": False,
        "note": (
            "Generated byte payloads remain in the immutable raw screens to "
            "avoid duplicating or altering negative evidence."
        ),
    }
    _write(
        evidence_root / "generation_samples" / "manifest.json",
        generation_manifest,
    )

    raw_paths = [
        byte_frontier_path,
        byte_decision_path,
        token_screen_path,
        token_semantic_path,
        token_128_path,
        token_1024_path,
        hybrid_screen_path,
        hybrid_semantic_path,
        hybrid_128_path,
        hybrid_1024_path,
        hybrid_manifest_path,
    ]
    raw_manifest = {
        "format": "layercake-representation-raw-evidence-manifest/1",
        "files": [_evidence_record(root, path) for path in raw_paths],
        "historical_evidence_overwritten": False,
    }
    raw_manifest["manifest_sha256"] = _canonical_sha(raw_manifest)
    _write(evidence_root / "raw_runs" / "manifest.json", raw_manifest)

    pareto = {
        "format": "layercake-representation-pareto-frontier/1",
        "status": "COMPLETE_NO_PROMOTION",
        "locked_gates": byte_frontier["locked_gates"],
        "comparators": {
            "same_scale_architecture": {
                "evidence": "results/moonshot/phase1/comparison_certificate.json",
                "quality_and_speed_use_one_transformer": True,
            },
            "matched_quality_product": {
                "candidate": qwen_row,
                "quality_evidence": "results/moonshot/phase1/functional_quality.json",
                "speed_evidence": qwen_row["evidence"],
                "single_deployment_lineage": True,
            },
        },
        "rows": [
            {
                "branch": "PRODUCT_COMPARATOR",
                **qwen_row,
            },
            {
                "branch": "A_BYTE_PATCH",
                **byte_row,
            },
            {
                "branch": "A_BYTE_PATCH_FINAL",
                **pointer_row,
            },
            token_row,
            hybrid_row,
        ],
        "promotion_rule": (
            "A row must pass held-out quality, autonomous functional quality, "
            "2x CPU at short and long horizons, lower process memory, physical "
            "sparsity, incremental state, and portability readiness using one "
            "checkpoint."
        ),
        "promoted_candidate": None,
    }
    pareto["pareto_sha256"] = _canonical_sha(pareto)
    _write(evidence_root / "pareto_frontier.json", pareto)

    diagnosis = {
        "format": "layercake-representation-falsifiable-diagnosis/1",
        "status": "ACTIONABLE_NO_WINNER",
        "observations": {
            "byte_native": (
                "Language-model BPB and CPU speed can pass separately, but "
                "autonomous prompt adherence collapses; the final pointer gate "
                "also loses the 2x speed and memory gates."
            ),
            "shared_tokenizer": (
                "Cached prompt attention improves topic-token recall from 0.12 "
                "to 0.225 and preserves 2x long-horizon speed, but core "
                "adherence remains 0.0 and process RSS remains too high."
            ),
            "hybrid": (
                "Exact byte fallback and language-model BPB pass, but semantic "
                "adherence fails and throughput falls to 1.91x at 1024 bytes."
            ),
        },
        "falsifiable_hypothesis": (
            "The limiting factor is not unit compression or next-unit "
            "language modeling. It is prompt-to-response conditional "
            "generalization: a single pooled prompt vector cannot preserve "
            "task, topic, and requested structure. A materially distinct "
            "multi-slot prompt-state architecture trained on compositionally "
            "disjoint instruction tasks should raise topic recall to at least "
            "0.82 and core adherence to at least 0.55 without adding recurrent "
            "decode work; otherwise this hypothesis is rejected."
        ),
        "predeclared_refutation_gates": {
            "validation_bpb_maximum": quality_ceiling,
            "topic_token_recall_minimum": 0.82,
            "core_adherence_pass_rate_minimum": 0.55,
            "functional_surface_noninferiority_required": True,
            "cpu_ratio_10m_continuation_minimum": 1.5,
            "cpu_ratio_promotion_minimum": 2.0,
            "long_horizon_cpu_ratio_minimum": 2.0,
            "median_process_resident_bytes_maximum_exclusive": qwen_row[
                "median_process_resident_bytes"
            ],
        },
        "explicitly_prohibited_restarts": [
            "nearby byte pointer or auxiliary-loss variants",
            "another uniform or scalar-attention prompt pooling variant",
            "larger-data continuation of a semantically failed checkpoint",
            "planner, template, retrieval, forced continuation, or stored answers",
            "renaming an already failed architecture search",
        ],
    }
    _write(
        evidence_root / "statistical_analysis" / "falsifiable_diagnosis.json",
        diagnosis,
    )

    decision = {
        "format": "layercake-representation-decision/1",
        "status": "NO_PROMOTABLE_REPRESENTATION",
        "winner": None,
        "phase2_status": "OPEN_CONTINUATION_REQUIRED",
        "phase3_status": "LOCKED",
        "phase2_r3_tag_created": False,
        "reason": (
            "No single autonomous checkpoint passes the locked functional "
            "quality, 2x short/long CPU, and lower-memory gates."
        ),
        "branch_status": {
            "A_BYTE_PATCH": byte_decision["status"],
            "B_SHARED_TOKENIZER": token_branch["status"],
            "C_HYBRID_TOKEN_BYTE": hybrid_branch["status"],
            "D_DYNAMIC_LATENT_PATCH": dynamic_branch["status"],
        },
        "three_seed_recertification_authorized": False,
        "phase3_execution_authorized": False,
        "pareto_frontier": _evidence_record(
            root, EVIDENCE / "pareto_frontier.json"
        ),
        "diagnosis": _evidence_record(
            root,
            EVIDENCE / "statistical_analysis" / "falsifiable_diagnosis.json",
        ),
        "exact_next_command": (
            f"{sys.executable} -m layercake.representation_bakeoff "
            "verify-decision --decision "
            "results/moonshot/representation_bakeoff/representation_decision.json"
        ),
        "test_accessed": False,
    }
    decision["decision_sha256"] = _canonical_sha(decision)
    _write(evidence_root / "representation_decision.json", decision)

    report = f"""# LayerCake representation bake-off release report

Status: **CONTINUATION REQUIRED**

No representation is promoted. Phase 2 remains open and Phase 3 remains locked.

| Candidate | Validation BPB | 128-byte CPU ratio | 1024-byte CPU ratio | Core adherence | Topic recall | Process RSS | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5 0.5B CPU | n/a | 1.00x | 1.00x | 0.60 | 0.87 | {qwen_row["median_process_resident_bytes"]} | locked comparator |
| Byte sliding-cosine | {byte_row["validation_bpb"]:.4f} | {byte_row["transformer_relative_throughput"]:.2f}x | n/a | failed | failed | {byte_row["median_process_resident_bytes"]} | reject |
| Byte pointer gate | {pointer_row["validation_bpb"]:.4f} | {pointer_row["transformer_relative_throughput"]:.2f}x | {pointer_row["transformer_relative_throughput_1024_diagnostic"]:.2f}x | 0.00 | {byte_decision["gate_results"]["autonomous_functional_noninferiority"]["topic_token_recall"]:.2f} | {pointer_row["median_process_resident_bytes"]} | reject |
| Shared tokenizer | {token_row["validation_bpb"]:.4f} | {token_row["transformer_relative_throughput_128"]:.2f}x | {token_row["transformer_relative_throughput_1024"]:.2f}x | {token_row["core_adherence_pass_rate"]:.2f} | {token_row["topic_token_recall"]:.3f} | {int(token_row["median_process_resident_bytes"])} | reject |
| Hybrid token-byte | {hybrid_row["validation_bpb"]:.4f} | {hybrid_row["transformer_relative_throughput_128"]:.2f}x | {hybrid_row["transformer_relative_throughput_1024"]:.2f}x | {hybrid_row["core_adherence_pass_rate"]:.2f} | {hybrid_row["topic_token_recall"]:.3f} | {int(hybrid_row["median_process_resident_bytes"])} | reject |

The byte pointer run and every negative checkpoint are preserved. No nearby
byte, gate, pointer, auxiliary-loss, hidden-size, or decoder variant was
launched after the authorized pointer run.

The next campaign must test the falsifiable multi-slot prompt-state diagnosis;
it may not restart any closed representation search under a new name.
"""
    (evidence_root / "release_report.md").write_text(report, encoding="utf-8")

    continuation = """Continue the LayerCake Phase 2 representation campaign from the sealed no-winner bake-off. First run the exact verifier command in task_state.json. Do not rerun the byte pointer, auxiliary-loss, gate, hidden-size, byte-decoder, uniform-pooling, scalar-attention-pooling, or failed hybrid/token checkpoints. Use results/moonshot/representation_bakeoff/statistical_analysis/falsifiable_diagnosis.json as the predeclared hypothesis: implement at most one materially distinct multi-slot cached prompt-state architecture whose recurrent decode graph does not grow, train only the 10M continuation tier, and apply the locked BPB, autonomous 100-prompt semantic, 20-repeat short/long CPU, TTFT, process-memory, incremental-state, physical-sparsity, and no-external-path gates. Continue to 100M and three seeds only if every continuation gate passes. Keep Phase 3 locked until moonshot_campaign verify-phase certifies a single winning checkpoint.
"""
    (evidence_root / "continuation_prompt.txt").write_text(
        continuation, encoding="utf-8"
    )

    task_path = root / (
        "results/moonshot/phase2_recertification/task_state.json"
    )
    task = _read(task_path)
    task.update({
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_stage": "PHASE2_REPRESENTATION_DIAGNOSIS",
        "phase2_status": "OPEN_CONTINUATION_REQUIRED",
        "phase3_status": "LOCKED",
        "active_candidate": None,
        "latest_batch": (
            "results/moonshot/representation_bakeoff/"
            "representation_decision.json"
        ),
        "continuation_command": decision["exact_next_command"],
        "continuation_prompt": (
            "results/moonshot/representation_bakeoff/continuation_prompt.txt"
        ),
        "representation_branches": {
            "byte_patch": "CLOSED_NEGATIVE",
            "shared_tokenizer": "CLOSED_NEGATIVE",
            "hybrid_token_byte": "CLOSED_NEGATIVE",
            "dynamic_latent_patch": "NOT_AUTHORIZED",
        },
        "representation_winner": None,
        "phase2_r3_tag_created": False,
    })
    completed = list(task.get("completed_batches", []))
    completed.extend([
        "hybrid token-byte 10m screen and repeated timing",
        "shared-tokenizer neural-only screens",
        "cached prompt-attention bounded candidate",
        "representation bakeoff closed with no winner",
    ])
    task["completed_batches"] = completed
    task["remaining_gates"] = [
        "falsify or validate materially distinct multi-slot prompt state",
        "one checkpoint functional semantic noninferiority",
        "one checkpoint short and long CPU throughput >=2x",
        "one checkpoint process-resident memory below transformer",
        "canonical representation-agnostic cake ABI",
        "three-seed Phase 2 r3 recertification",
        "Phase 3 matched-quality training-speed proof",
    ]
    _write(task_path, task)
    task["latest_batch_sha256"] = sha256_file(
        root / task["latest_batch"]
    )
    _write(task_path, task)

    _append_ledger(root, {
        "event": "representation_bakeoff_closed_no_winner",
        "decision": (
            "results/moonshot/representation_bakeoff/"
            "representation_decision.json"
        ),
        "decision_sha256": decision["decision_sha256"],
        "pareto_sha256": pareto["pareto_sha256"],
        "branch_status": decision["branch_status"],
        "phase2_status": decision["phase2_status"],
        "phase3_status": decision["phase3_status"],
        "test_accessed": False,
    })
    return {
        "status": "CONTINUATION_REQUIRED",
        "decision_sha256": decision["decision_sha256"],
        "pareto_sha256": pareto["pareto_sha256"],
        "phase2_status": decision["phase2_status"],
        "phase3_status": decision["phase3_status"],
        "winner": None,
    }


def verify_decision(root: Path, decision_path: Path) -> dict[str, Any]:
    """Adversarially verify that a no-winner decision cannot unlock Phase 3."""

    decision_path = _resolve(root, decision_path)
    decision = _read(decision_path)
    failures = []
    decision_payload = dict(decision)
    recorded_decision_payload_sha = decision_payload.pop("decision_sha256", None)
    if _canonical_sha(decision_payload) != recorded_decision_payload_sha:
        failures.append("decision payload hash mismatch")
    if decision.get("winner") is not None:
        failures.append("no-winner decision unexpectedly names a winner")
    if decision.get("phase3_status") != "LOCKED":
        failures.append("Phase 3 is not locked")
    if decision.get("phase2_r3_tag_created"):
        failures.append("Phase 2 r3 tag was claimed without a winner")
    if decision.get("three_seed_recertification_authorized"):
        failures.append("three-seed recertification was authorized")
    if decision.get("phase3_execution_authorized"):
        failures.append("Phase 3 execution was authorized")
    if set(decision.get("branch_status", {}).values()) != {
        "CLOSED_NEGATIVE",
        "NOT_AUTHORIZED",
    }:
        failures.append("branch closure set is incomplete")
    pareto_record = decision["pareto_frontier"]
    pareto_path = root / pareto_record["path"]
    if sha256_file(pareto_path) != pareto_record["sha256"]:
        failures.append("Pareto frontier hash mismatch")
    pareto = _read(pareto_path)
    pareto_payload = dict(pareto)
    recorded_pareto_payload_sha = pareto_payload.pop("pareto_sha256", None)
    if _canonical_sha(pareto_payload) != recorded_pareto_payload_sha:
        failures.append("Pareto payload hash mismatch")
    if pareto.get("promoted_candidate") is not None:
        failures.append("Pareto frontier promotes a candidate")
    for row in pareto["rows"]:
        if row.get("branch") in {
            "A_BYTE_PATCH",
            "A_BYTE_PATCH_FINAL",
            "B_SHARED_TOKENIZER",
            "C_HYBRID_TOKEN_BYTE",
        }:
            quality = bool(row.get("language_model_quality_pass", True))
            semantics = bool(row.get("functional_semantic_pass", False))
            surface = bool(row.get("functional_surface_pass", False))
            speed128 = float(
                row.get(
                    "transformer_relative_throughput_128",
                    row.get("transformer_relative_throughput", 0.0),
                )
            ) >= 2.0
            speed1024 = float(
                row.get(
                    "transformer_relative_throughput_1024",
                    row.get(
                        "transformer_relative_throughput_1024_diagnostic",
                        0.0,
                    ),
                )
            ) >= 2.0
            memory = float(row.get("median_process_resident_bytes", math.inf))
            memory_pass = memory < 214_990_848
            if all((quality, semantics, surface, speed128, speed1024, memory_pass)):
                failures.append(
                    f"row {row['candidate']} appears promotable but was rejected"
                )
    branch_decisions = [
        root / EVIDENCE / "byte_branch" / "branch_decision.json",
        root / EVIDENCE / "token_branch" / "branch_decision.json",
        root / EVIDENCE / "hybrid_branch" / "branch_decision.json",
    ]
    for branch_path in branch_decisions:
        branch = _read(branch_path)
        if branch.get("status") != "CLOSED_NEGATIVE":
            failures.append(f"{branch_path.name} is not closed negative")
        for record in branch.get("evidence", []):
            if isinstance(record, str):
                evidence_path = _resolve(root, Path(record))
                if not evidence_path.is_file():
                    failures.append(f"missing branch evidence: {record}")
                continue
            evidence_path = _resolve(root, Path(record["path"]))
            if (
                not evidence_path.is_file()
                or sha256_file(evidence_path) != record["sha256"]
            ):
                failures.append(
                    f"branch evidence hash mismatch: {record['path']}"
                )
    screen_paths = [
        root / EVIDENCE / "token_branch"
        / "functional_screen_prompt_attention_curated_seed9824_decodecontrol640.json",
        root / EVIDENCE / "hybrid_branch"
        / "functional_screen_seed9824_decodecontrol.json",
    ]
    for screen_path in screen_paths:
        screen = _read(screen_path)
        if screen.get("distinct_prompts") != 100:
            failures.append(f"{screen_path.name} does not contain 100 prompts")
        if len(screen.get("records", [])) != 100:
            failures.append(f"{screen_path.name} raw record count is not 100")
        for record in screen.get("records", []):
            counters = record.get("execution", {}).get(
                "external_path_counters", {}
            )
            if any(int(value) != 0 for value in counters.values()):
                failures.append(
                    f"{screen_path.name} uses a prohibited external path"
                )
                break
            if (
                record.get("execution", {}).get(
                    "maximum_active_experts_per_generated_token"
                )
                != 1
            ):
                failures.append(
                    f"{screen_path.name} lacks physical top-1 dispatch"
                )
                break
    diagnosis_record = decision["diagnosis"]
    diagnosis_path = root / diagnosis_record["path"]
    if sha256_file(diagnosis_path) != diagnosis_record["sha256"]:
        failures.append("falsifiable diagnosis hash mismatch")
    result = {
        "format": "layercake-representation-adversarial-verifier/1",
        "status": "PASS" if not failures else "FAIL",
        "decision_path": decision_path.relative_to(root).as_posix(),
        "decision_file_sha256": sha256_file(decision_path),
        "decision_payload_sha256": decision.get("decision_sha256"),
        "checks": {
            "winner_is_null": decision.get("winner") is None,
            "phase2_remains_open": decision.get("phase2_status", "").startswith(
                "OPEN"
            ),
            "phase3_locked": decision.get("phase3_status") == "LOCKED",
            "phase2_r3_tag_not_claimed": not decision.get(
                "phase2_r3_tag_created"
            ),
            "all_required_branches_terminal": not failures,
            "decision_payload_hash_matches": (
                _canonical_sha(decision_payload)
                == recorded_decision_payload_sha
            ),
            "pareto_payload_hash_matches": (
                _canonical_sha(pareto_payload)
                == recorded_pareto_payload_sha
            ),
            "pareto_hash_matches": (
                sha256_file(pareto_path) == pareto_record["sha256"]
            ),
        },
        "failures": failures,
    }
    result["verifier_sha256"] = _canonical_sha(result)
    output = root / EVIDENCE / "statistical_analysis" / "adversarial_verifier.json"
    _write(output, result)
    if failures:
        raise RuntimeError("; ".join(failures))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m layercake.representation_bakeoff"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("build-hybrid-tokenizer")
    command.add_argument("--base", type=Path, required=True)
    command.add_argument("--corpus", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--training-bytes", type=int, default=10_000_000)
    command = sub.add_parser("strip-planner")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("enable-prompt-attention")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("screen")
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--output-bytes", type=int, default=480)
    command.add_argument("--threads", type=int, default=1)
    command.add_argument("--repetition-penalty", type=float, default=1.0)
    command.add_argument("--no-repeat-ngram-size", type=int, default=0)
    command = sub.add_parser("benchmark")
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--output-bytes", type=int, default=128)
    command.add_argument("--repetitions", type=int, default=20)
    command.add_argument("--threads", type=int, default=1)
    command.add_argument("--repetition-penalty", type=float, default=1.0)
    command.add_argument("--no-repeat-ngram-size", type=int, default=0)
    command = sub.add_parser("certify-finetune-provenance")
    command.add_argument("--checkpoint", type=Path, required=True)
    sub.add_parser("finalize")
    command = sub.add_parser("verify-decision")
    command.add_argument("--decision", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "build-hybrid-tokenizer":
        result = build_hybrid_tokenizer(
            root,
            args.base,
            args.corpus,
            args.output,
            args.training_bytes,
        )
    elif args.command == "strip-planner":
        result = strip_planner_checkpoint(
            root,
            args.source,
            args.output,
        )
    elif args.command == "enable-prompt-attention":
        result = enable_prompt_attention_checkpoint(
            root,
            args.source,
            args.output,
        )
    elif args.command == "screen":
        result = screen_checkpoint(
            root,
            args.checkpoint,
            args.output,
            args.output_bytes,
            args.threads,
            args.repetition_penalty,
            args.no_repeat_ngram_size,
        )
    elif args.command == "benchmark":
        result = benchmark_checkpoint(
            root,
            args.checkpoint,
            args.output,
            args.output_bytes,
            args.repetitions,
            args.threads,
            args.repetition_penalty,
            args.no_repeat_ngram_size,
        )
    elif args.command == "certify-finetune-provenance":
        result = certify_finetune_provenance(
            root,
            args.checkpoint,
        )
    elif args.command == "finalize":
        result = finalize_bakeoff(root)
    elif args.command == "verify-decision":
        result = verify_decision(
            root,
            args.decision,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
