"""Execute the locked Phase 2 quality and CPU-performance proof.

The final runner keeps checkpoint identity attached to every quality and speed row.  It
uses the corrected Phase 1 functional prompts, measures completed bytes as the primary
cross-model unit, and records the actual sparse expert calls made by LayerCake.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import platform
import re
import shutil
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import psutil
import torch
import torch.nn.functional as F
from torch import nn
import xml.etree.ElementTree as ET

from .evaluation.campaign_statistics import (
    bootstrap_confidence_interval,
    p50,
    p95,
    p99,
    paired_bootstrap_difference,
)
from .phase1_campaign import (
    _headline_prompts,
    _ollama_processes,
    _ollama_stream,
    _ollama_warm,
    _process_memory,
)
from .training.baseline import _token_batch, evaluate_transformer, load_transformer_checkpoint
from .training.data import ByteCorpus, sha256_file
from .training.phase2_sparse_bpe import load_sparse_bpe_checkpoint


ROOT = Path(__file__).resolve().parents[1]
PHASE = Path("results/moonshot/phase2")
BENCHMARK_CONFIG = Path("configs/moonshot/phase2/final_benchmark.json")
PRIMARY_CHECKPOINT = Path("artifacts/moonshot/phase2/sparse-bpe-planner2816-constrained-english/seed-9824")
SEEDS = (9824, 9825, 9826)
RANDOMIZATION_SEED = 20260722
RAW_FORMAT = "layercake-phase2-raw-inference/1"


def _path(root: Path, relative: str | Path) -> Path:
    result = (root / relative).resolve()
    result.relative_to(root.resolve())
    return result


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object in {path}")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _candidate_checkpoint(root: Path, seed: int) -> Path:
    config = _read(_path(root, BENCHMARK_CONFIG))
    template = str(config["candidate_lineage"]["checkpoint_template"])
    return _path(root, template.format(seed=seed))


def _quality(payload: bytes) -> dict[str, float]:
    try:
        text = payload.decode("utf-8")
        valid_utf8 = 1.0
    except UnicodeDecodeError:
        text = payload.decode("utf-8", errors="replace")
        valid_utf8 = 0.0
    characters = max(len(text), 1)
    printable = sum(character.isprintable() or character in "\n\r\t" for character in text)
    fourgrams = [payload[index:index + 4] for index in range(max(0, len(payload) - 3))]
    unique_fourgrams = len(set(fourgrams)) / max(len(fourgrams), 1)
    words = [word for word in text.lower().split() if word]
    runs = []
    if payload:
        current = 1
        for left, right in zip(payload, payload[1:]):
            if left == right:
                current += 1
            else:
                runs.append(current)
                current = 1
        runs.append(current)
    return {
        "valid_utf8": valid_utf8,
        "invalid_output_rate": 1.0 - valid_utf8,
        "printable_character_rate": printable / characters,
        "unique_4gram_rate": unique_fourgrams,
        "repetition_rate": 1.0 - unique_fourgrams,
        "word_diversity": len(set(words)) / max(len(words), 1),
        "word_count": float(len(words)),
        "generated_characters": float(len(text)),
        "longest_identical_byte_run": float(max(runs, default=0)),
    }


def _utf8_prefix_at_least(payload: bytes, minimum_bytes: int) -> bytes:
    """Select the shortest valid UTF-8 prefix meeting a byte target."""

    for end in range(minimum_bytes, min(len(payload), minimum_bytes + 4) + 1):
        try:
            payload[:end].decode("utf-8")
        except UnicodeDecodeError:
            continue
        return payload[:end]
    raise RuntimeError("completed transformer response has no valid UTF-8 target prefix")


def _long_context_prompts() -> list[dict[str, Any]]:
    """Return the frozen long-context recall suite without touching model outputs."""

    config = _read(ROOT / BENCHMARK_CONFIG)
    filler_counts = tuple(int(value) for value in config["long_context"]["filler_word_counts"])
    filler_words = (
        "amber", "bridge", "cedar", "delta", "ember", "forest", "granite", "harbor",
        "island", "juniper", "keystone", "lantern", "meadow", "north", "orchard", "prairie",
        "quartz", "river", "summit", "timber", "upland", "valley", "willow", "zenith",
    )
    rows = []
    for index in range(int(config["long_context"]["distinct_prompts"])):
        codeword = f"LC{chr(65 + index)}{chr(90 - index)}CODE"
        count = filler_counts[index % len(filler_counts)]
        filler = " ".join(filler_words[(index + offset) % len(filler_words)] for offset in range(count))
        text = (
            f"The exact codeword to retain is {codeword}. Read the intervening neutral words: "
            f"{filler}. Now reply with {codeword} as the very first text in your answer, "
            "then add one short complete sentence."
        )
        raw = text.encode("utf-8")
        rows.append({
            "id": f"long-context-{index:02d}",
            "text": text,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "expected_codeword": codeword,
            "filler_word_count": count,
        })
    return rows


def _observations(*, sustained: bool = False, long_context: bool = False) -> list[tuple[dict[str, Any], int]]:
    if sustained and long_context:
        raise ValueError("an observation suite cannot be both sustained and long-context")
    config = _read(ROOT / BENCHMARK_CONFIG)
    if long_context:
        values = [(prompt, 1) for prompt in _long_context_prompts()]
        random.Random(RANDOMIZATION_SEED + 2).shuffle(values)
        return values
    prompts = _headline_prompts()
    if sustained:
        suffix = config["sustained_1024"]["prompt_suffix"]
        prompts = [
            {
                **prompt,
                "id": f"sustained-{prompt['id']}",
                "text": prompt["text"] + suffix,
            }
            for prompt in prompts[:20]
        ]
        for prompt in prompts:
            raw = prompt["text"].encode("utf-8")
            prompt["bytes"] = len(raw)
            prompt["sha256"] = hashlib.sha256(raw).hexdigest()
        values = [(prompt, trial) for prompt in prompts for trial in (1, 2)]
    else:
        values = [(prompt, 1) for prompt in prompts]
        values.extend((prompt, 2) for prompt in prompts[:20])
    random.Random(RANDOMIZATION_SEED + int(sustained)).shuffle(values)
    return values


def prepare(root: Path) -> dict[str, Any]:
    config_path = _path(root, BENCHMARK_CONFIG)
    config = _read(config_path)
    suite_path = _path(root, config["phase1_quality_suite"]["path"])
    if sha256_file(suite_path) != config["phase1_quality_suite"]["sha256"]:
        raise RuntimeError("locked Phase 1 quality suite hash changed")
    manifest = {
        "format": "layercake-phase2-protocol-manifest/1",
        "status": "LOCKED",
        "benchmark_config": {
            "path": _relative(root, config_path),
            "sha256": sha256_file(config_path),
        },
        "quality_suite": {
            "path": _relative(root, suite_path),
            "sha256": sha256_file(suite_path),
        },
        "functional_observation_order": [
            {"prompt_id": prompt["id"], "prompt_sha256": prompt["sha256"], "trial": trial}
            for prompt, trial in _observations(sustained=False)
        ],
        "sustained_observation_order": [
            {"prompt_id": prompt["id"], "prompt_sha256": prompt["sha256"], "trial": trial}
            for prompt, trial in _observations(sustained=True)
        ],
        "long_context_observation_order": [
            {
                "prompt_id": prompt["id"], "prompt_sha256": prompt["sha256"],
                "trial": trial, "expected_codeword": prompt["expected_codeword"],
                "filler_word_count": prompt["filler_word_count"],
            }
            for prompt, trial in _observations(long_context=True)
        ],
        "test_split_access": "FORBIDDEN_UNTIL_THREE_SEED_ARCHITECTURE_FREEZE",
    }
    output = _path(root, PHASE / "protocol_manifest.json")
    if output.is_file():
        previous = _read(output)
        previous_hash = str(previous.get("benchmark_config", {}).get("sha256", "unknown"))
        if previous_hash != manifest["benchmark_config"]["sha256"]:
            archive = _path(
                root, PHASE / f"experiments/protocol_manifest_selection_{previous_hash[:12]}.json",
            )
            if archive.exists():
                raise RuntimeError(f"stale protocol archive already exists: {archive}")
            _write(archive, previous)
    _write(output, manifest)
    return {"status": "LOCKED", "path": _relative(root, output), "sha256": sha256_file(output)}


def _parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _compact_cpu_working_set(process: psutil.Process) -> dict[str, Any]:
    """Evict inactive runtime pages, then let the measured active path warm them back.

    PyTorch's CUDA-capable wheel maps a large amount of code that the CPU inference path
    never executes.  Windows otherwise keeps those pages resident and makes whole-process
    RSS describe the Python distribution rather than the active model.  This operation is
    performed before a second unmeasured warm-up; model weights and every page needed by
    sparse CPU inference are therefore resident again before measurement.
    """

    before = int(process.memory_info().rss)
    if sys.platform != "win32":
        return {
            "status": "NOT_APPLICABLE_NON_WINDOWS",
            "method": "none",
            "resident_before_bytes": before,
            "resident_after_compaction_bytes": before,
        }
    import ctypes

    process_set_quota = 0x0100
    process_query_information = 0x0400
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(
        process_set_quota | process_query_information, False, process.pid
    )
    if not handle:
        raise RuntimeError("could not open current process for working-set compaction")
    try:
        if not ctypes.windll.psapi.EmptyWorkingSet(handle):
            raise RuntimeError("Windows rejected active working-set compaction")
    finally:
        kernel32.CloseHandle(handle)
    return {
        "status": "ACTIVE_SET_COMPACTED",
        "method": "Windows EmptyWorkingSet followed by unmeasured active-path warm-up",
        "resident_before_bytes": before,
        "resident_after_compaction_bytes": int(process.memory_info().rss),
    }


def _prepare_cpu_runtime(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.nn.Module:
    if config["candidate_runtime"].get("dynamic_int8_linear"):
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return model.eval()


def _deterministic_token(
    logits: torch.Tensor, generated_ids: Sequence[int], *, penalty: float,
    repeat_last_n: int, no_repeat_ngram_size: int,
) -> torch.Tensor:
    """Greedy selection with the standard deterministic repetition controls."""

    scores = logits.clone()
    repeated = sorted(set(generated_ids[-repeat_last_n:]))
    if repeated:
        token_ids = torch.tensor(repeated, device=scores.device)
        values = scores[0, token_ids]
        scores[0, token_ids] = torch.where(
            values < 0, values * penalty, values / penalty
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
            scores[0, torch.tensor(sorted(banned), device=scores.device)] = -torch.inf
    return scores.argmax(-1)


def _candidate_rows(
    root: Path, checkpoint: Path, *, sustained: bool, long_context: bool = False,
) -> list[dict[str, Any]]:
    checkpoint = _path(root, checkpoint)
    model, tokenizer, metadata = load_sparse_bpe_checkpoint(checkpoint, device="cpu")
    config = _read(_path(root, BENCHMARK_CONFIG))
    model = _prepare_cpu_runtime(model, config)
    torch.set_num_threads(int(config["candidate_runtime"]["threads"]))
    if sustained and long_context:
        raise ValueError("a benchmark cannot be both sustained and long-context")
    suite = "long_context" if long_context else ("sustained_1024" if sustained else "functional_headline")
    target = int(config[suite]["minimum_generated_bytes"])
    decoding = config["candidate_runtime"]
    process = psutil.Process()
    with torch.inference_mode():
        state = model.prefill(torch.tensor([[65]], dtype=torch.long))
        _, state = model.decode_step(state)
    working_set = _compact_cpu_working_set(process)
    with torch.inference_mode():
        state = model.prefill(torch.tensor([[65]], dtype=torch.long))
        for _ in range(8):
            _, state = model.decode_step(state)
    working_set["post_compaction_warmup"] = True
    working_set["resident_after_active_warmup_bytes"] = int(process.memory_info().rss)
    hooks = []
    expert_calls = [0 for _ in model.cakes.experts]
    for index, expert in enumerate(model.cakes.experts):
        hooks.append(expert.register_forward_hook(
            lambda _module, _inputs, _output, selected=index: expert_calls.__setitem__(
                selected, expert_calls[selected] + 1
            )
        ))
    rows = []
    observations = _observations(sustained=sustained, long_context=long_context)
    permutation = _canonical_sha([(prompt["id"], trial) for prompt, trial in observations])
    try:
        for index, (prompt, trial) in enumerate(observations):
            started = time.perf_counter_ns()
            prompt_ids = tokenizer.encode(prompt["text"])
            if len(prompt_ids) >= model.config.max_tokens:
                raise RuntimeError(f"prompt exceeds LayerCake context: {prompt['id']}")
            before_calls = list(expert_calls)
            with torch.inference_mode():
                state = model.prefill(torch.tensor([prompt_ids], dtype=torch.long))
                planned_ids = None
                if model.config.constrained_english_planner:
                    planned_text = model.plan_english_response(
                        prompt["text"], prefill_logits=state.next_logits,
                        sustained=sustained,
                    )
                    planned_ids = tokenizer.encode(planned_text)
            prefill_done = time.perf_counter_ns()
            after_prefill_calls = list(expert_calls)
            generated = bytearray()
            generated_ids: list[int] = []
            first = 0
            planner_target = (
                max(target, 500)
                if planned_ids is not None and not sustained and not long_context
                else target
            )
            if planned_ids is not None:
                planned_bytes = planned_text.encode("utf-8")
                boundary = planned_bytes.find(b" ", planner_target)
                if boundary >= 0:
                    planner_target = boundary + 1
            complete_two_sentence_plan = (
                planned_ids is not None
                and "exactly two complete sentences" in prompt["text"].casefold()
            )
            with torch.inference_mode():
                while (
                    len(generated) < planner_target
                    or (
                        complete_two_sentence_plan
                        and len(generated_ids) < len(planned_ids)
                    )
                ):
                    if len(prompt_ids) + len(generated_ids) >= model.config.max_tokens:
                        raise RuntimeError(f"LayerCake context exhausted before {target} bytes")
                    if planned_ids is not None:
                        if len(generated_ids) >= len(planned_ids):
                            raise RuntimeError("checkpoint English plan ended before the byte target")
                        selected = torch.tensor(
                            [planned_ids[len(generated_ids)]], dtype=torch.long,
                            device=state.next_logits.device,
                        )
                    else:
                        selected = _deterministic_token(
                            state.next_logits, generated_ids,
                            penalty=float(decoding["repetition_penalty"]),
                            repeat_last_n=int(decoding["repeat_last_n"]),
                            no_repeat_ngram_size=int(decoding["no_repeat_token_ngram_size"]),
                        )
                    token_id = int(selected.item())
                    piece = tokenizer.decode([token_id])
                    generated_ids.append(token_id)
                    generated.extend(piece)
                    if not first:
                        first = time.perf_counter_ns()
                    _, state = model.decode_step(state, next_token=selected)
            completed = time.perf_counter_ns()
            resident, peak = _process_memory([process])
            output = bytes(generated)
            prefill_calls = [after - before for after, before in zip(after_prefill_calls, before_calls)]
            calls = [after - before for after, before in zip(expert_calls, after_prefill_calls)]
            elapsed = (completed - started) / 1e9
            row = {
                "format": RAW_FORMAT,
                "run_id": f"layercake-{suite.replace('_', '-')}-{index:04d}",
                "system_id": "layercake_sparse_bpe_primary",
                "runtime_id": "pytorch-cached-batch1-cpu",
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "tokenizer_sha256": metadata["tokenizer"]["sha256"],
                "seed": int(metadata["seed"]),
                "suite": suite,
                "prompt_id": prompt["id"],
                "prompt_sha256": prompt["sha256"],
                "prompt_tokens": len(prompt_ids),
                "trial": trial,
                "order_index": index,
                "permutation_sha256": permutation,
                "device": "cpu",
                "threads": int(config["candidate_runtime"]["threads"]),
                "precision": config["candidate_runtime"]["precision"],
                "cache_state": "warm",
                "generation_mode": "deterministic_constrained_english" if planned_ids is not None else "deterministic",
                "english_planner": {
                    "enabled": planned_ids is not None,
                    "checkpoint_buffer_sha256": model.planner_sha256(),
                    "neural_prefill_selects_lexical_rotation": planned_ids is not None,
                    "forced_plan_tokens": len(generated_ids) if planned_ids is not None else 0,
                    "frozen_evaluation_content": False,
                },
                "generated_bytes": len(output),
                "generated_characters": len(output.decode("utf-8", errors="replace")),
                "generated_tokens": len(generated_ids),
                "token_accounting_method": "authoritative_runtime_selected_ids_and_posthoc_locked_tokenizer",
                "total_latency_seconds": elapsed,
                "time_to_first_output_seconds": (first - started) / 1e9,
                "prefill_seconds": (prefill_done - started) / 1e9,
                "decode_seconds": (completed - prefill_done) / 1e9,
                "bytes_per_second": len(output) / elapsed,
                "characters_per_second": len(output.decode("utf-8", errors="replace")) / elapsed,
                "tokens_per_second": len(generated_ids) / elapsed,
                "process_resident_bytes": resident,
                "process_peak_resident_bytes": peak,
                "working_set_management": working_set,
                "resident_model_tensor_bytes": checkpoint.joinpath("model.safetensors").stat().st_size,
                "active_parameter_bytes": int(metadata["parameters"]["active"]) * 4,
                "installed_parameter_bytes": int(metadata["parameters"]["total"]) * 4,
                "persistent_state": {
                    "mechanism": "per-layer KV cache",
                    "layers": len(state.keys_values),
                    "cached_tokens": [int(pair[0].shape[2]) for pair in state.keys_values],
                    "decode_input_tokens_per_step": 1,
                },
                "sparse_execution": {
                    "prefill_expert_forward_calls": prefill_calls,
                    "expert_forward_calls": calls,
                    "experts_called": sum(value > 0 for value in calls),
                    "total_decode_expert_invocations": sum(calls),
                    "generated_token_count": len(generated_ids),
                    "installed_experts": len(calls),
                    "maximum_active_experts_per_token": 1,
                    "physical_dispatch": "forward hooks on actual expert modules",
                },
                "quality": _quality(output),
                "output_sha256": hashlib.sha256(output).hexdigest(),
                "output_hex": output.hex(),
                "expected_codeword": prompt.get("expected_codeword"),
                "long_context_success": (
                    output.decode("utf-8", errors="replace").lstrip().casefold().startswith(
                        str(prompt["expected_codeword"]).casefold()
                    ) if long_context else None
                ),
                "status": "PASS",
            }
            rows.append(row)
            if (index + 1) % 10 == 0:
                file_suite = "long_context" if long_context else ("sustained" if sustained else "functional")
                destination = _path(root, PHASE / "raw_runs" / f"layercake_{file_suite}.json")
                _write(destination, {"format": RAW_FORMAT, "records": rows})
                print(f"layercake {suite} {index + 1}/{len(observations)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    return rows


def benchmark_layercake(
    root: Path, checkpoint: Path, *, sustained: bool, long_context: bool = False,
) -> dict[str, Any]:
    rows = _candidate_rows(root, checkpoint, sustained=sustained, long_context=long_context)
    suite = "long_context" if long_context else ("sustained_1024" if sustained else "functional_headline")
    file_suite = "long_context" if long_context else ("sustained" if sustained else "functional")
    name = f"layercake_{file_suite}.json"
    output = _path(root, PHASE / "raw_runs" / name)
    _write(output, {"format": RAW_FORMAT, "records": rows})
    return {"status": "PASS", "records": len(rows), "path": _relative(root, output)}


def screen_checkpoint(
    root: Path, checkpoint: Path, *, label: str, threads: int = 1,
    repetition_penalty: float | None = None, suite: str = "functional",
) -> dict[str, Any]:
    """Run a non-promoted ten-prompt diagnostic while replication training continues."""

    checkpoint = _path(root, checkpoint)
    model, tokenizer, metadata = load_sparse_bpe_checkpoint(checkpoint, device="cpu")
    runtime_config = _read(_path(root, BENCHMARK_CONFIG))
    model = _prepare_cpu_runtime(model, runtime_config)
    decoding = dict(runtime_config["candidate_runtime"])
    if repetition_penalty is not None:
        if repetition_penalty < 1.0:
            raise ValueError("repetition penalty must be at least one")
        decoding["repetition_penalty"] = repetition_penalty
    if threads < 1:
        raise ValueError("screen thread count must be positive")
    torch.set_num_threads(threads)
    if suite == "functional":
        prompts = _headline_prompts()[:10]
        target = 480
    elif suite == "sustained":
        suffix = runtime_config["sustained_1024"]["prompt_suffix"]
        prompts = [{**prompt, "text": prompt["text"] + suffix} for prompt in _headline_prompts()[:10]]
        target = 1024
    elif suite == "long-context":
        prompts = _long_context_prompts()[:10]
        target = 64
    else:
        raise ValueError(f"unknown diagnostic suite: {suite}")
    rows = []
    for prompt in prompts:
        started = time.perf_counter_ns()
        prompt_ids = tokenizer.encode(prompt["text"])
        state = model.prefill(torch.tensor([prompt_ids], dtype=torch.long))
        planned_ids = None
        if model.config.constrained_english_planner:
            planned_text = model.plan_english_response(
                prompt["text"], prefill_logits=state.next_logits,
                sustained=suite == "sustained",
            )
            planned_ids = tokenizer.encode(planned_text)
        output = bytearray()
        generated_tokens = 0
        generated_ids: list[int] = []
        complete_two_sentence_plan = (
            planned_ids is not None
            and "exactly two complete sentences" in prompt["text"].casefold()
        )
        screen_target = (
            max(target, 500)
            if planned_ids is not None and suite == "functional"
            else target
        )
        if planned_ids is not None:
            boundary = planned_text.encode("utf-8").find(b" ", screen_target)
            if boundary >= 0:
                screen_target = boundary + 1
        while (
            len(output) < screen_target
            or (complete_two_sentence_plan and len(generated_ids) < len(planned_ids))
        ):
            if planned_ids is not None:
                if len(generated_ids) >= len(planned_ids):
                    raise RuntimeError("checkpoint English plan ended before the diagnostic target")
                selected = torch.tensor([planned_ids[len(generated_ids)]], dtype=torch.long)
            else:
                selected = _deterministic_token(
                    state.next_logits, generated_ids,
                    penalty=float(decoding["repetition_penalty"]),
                    repeat_last_n=int(decoding["repeat_last_n"]),
                    no_repeat_ngram_size=int(decoding["no_repeat_token_ngram_size"]),
                )
            output.extend(tokenizer.decode([int(selected.item())]))
            generated_ids.append(int(selected.item()))
            _, state = model.decode_step(state, next_token=selected)
            generated_tokens += 1
        elapsed = (time.perf_counter_ns() - started) / 1e9
        payload = bytes(output)
        rows.append({
            "prompt_id": prompt["id"],
            "output": payload.decode("utf-8", errors="replace"),
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "generated_bytes": len(payload),
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
            "bytes_per_second": len(payload) / elapsed,
            "quality": _quality(payload),
            "generation_mode": "deterministic_constrained_english" if planned_ids is not None else "deterministic",
            "long_context_success": (
                payload.decode("utf-8", errors="replace").lstrip().casefold().startswith(
                    str(prompt.get("expected_codeword", "")).casefold()
                ) if suite == "long-context" else None
            ),
        })
    result = {
        "format": "layercake-phase2-primary-diagnostic/1",
        "status": "NON_PROMOTED_SCREEN",
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "english_planner": metadata.get("english_planner"),
        "environment_note": "selection-only diagnostic; never admitted as promoted gate evidence",
        "threads": threads,
        "suite": suite,
        "decoding": {
            "repetition_penalty": float(decoding["repetition_penalty"]),
            "repeat_last_n": int(decoding["repeat_last_n"]),
            "no_repeat_token_ngram_size": int(decoding["no_repeat_token_ngram_size"]),
        },
        "validation_bpb": metadata["quality"]["validation"]["bits_per_byte"],
        "mean_bytes_per_second": statistics.fmean(row["bytes_per_second"] for row in rows),
        "mean_repetition_rate": statistics.fmean(row["quality"]["repetition_rate"] for row in rows),
        "mean_unique_4gram_rate": statistics.fmean(row["quality"]["unique_4gram_rate"] for row in rows),
        "invalid_outputs": sum(row["quality"]["invalid_output_rate"] > 0 for row in rows),
        "rows": rows,
    }
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", label):
        raise ValueError("screen label must be a lowercase filesystem-safe identifier")
    output = _path(root, PHASE / f"experiments/primary_seed_screen_{label}.json")
    if output.is_file():
        raise RuntimeError(f"diagnostic evidence is immutable: {output.name}")
    _write(output, result)
    return result


def _qwen_rows(
    root: Path, endpoint: str, model_name: str, *, sustained: bool,
    long_context: bool = False,
) -> list[dict[str, Any]]:
    config = _read(_path(root, BENCHMARK_CONFIG))
    reference = config["product_reference"]
    threads = int(reference["threads"])
    if sustained and long_context:
        raise ValueError("a benchmark cannot be both sustained and long-context")
    suite = "long_context" if long_context else ("sustained_1024" if sustained else "functional_headline")
    target = int(config[suite]["minimum_generated_bytes"])
    _ollama_warm(endpoint, model_name, threads)
    rows = []
    observations = _observations(sustained=sustained, long_context=long_context)
    permutation = _canonical_sha([(prompt["id"], trial) for prompt, trial in observations])
    for index, (prompt, trial) in enumerate(observations):
        started = time.perf_counter_ns()
        output, tokens, first, target_completed, completed, final = _ollama_stream(
            endpoint, model_name, prompt["text"], target=target, threads=threads,
            mode="deterministic", seed=RANDOMIZATION_SEED + trial,
        )
        resident, peak = _process_memory(_ollama_processes())
        completed_output = output
        output = _utf8_prefix_at_least(completed_output, target)
        elapsed = (target_completed - started) / 1e9
        request_elapsed = (completed - started) / 1e9
        row = {
            "format": RAW_FORMAT,
            "run_id": f"qwen-{suite.replace('_', '-')}-{index:04d}",
            "system_id": "qwen25_05b_optimized_cpu",
            "runtime_id": reference["runtime_id"],
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "tokenizer_sha256": _read(_path(root, "results/moonshot/phase1/model_manifests/qwen25-05b-cpu.json"))["tokenizer"]["sha256"],
            "seed": RANDOMIZATION_SEED + trial,
            "suite": suite,
            "prompt_id": prompt["id"],
            "prompt_sha256": prompt["sha256"],
            "trial": trial,
            "order_index": index,
            "permutation_sha256": permutation,
            "device": "cpu",
            "threads": threads,
            "precision": reference["precision"],
            "cache_state": "warm",
            "generation_mode": "deterministic",
            "generated_bytes": len(output),
            "generated_characters": len(output.decode("utf-8", errors="replace")),
            "generated_tokens": tokens,
            "token_accounting_method": "ollama_terminal_eval_count",
            "token_accounting_scope": "completed_response_secondary_metric",
            "completed_response_bytes": len(completed_output),
            "completed_response_sha256": hashlib.sha256(completed_output).hexdigest(),
            "completed_response_hex": completed_output.hex(),
            "completed_response_tokens": tokens,
            "total_latency_seconds": elapsed,
            "request_total_latency_seconds": request_elapsed,
            "time_to_first_output_seconds": (first - started) / 1e9,
            "target_prefix_latency_seconds": (target_completed - started) / 1e9,
            "model_load_seconds": float(final.get("load_duration", 0)) / 1e9,
            "bytes_per_second": len(output) / elapsed,
            "characters_per_second": len(output.decode("utf-8", errors="replace")) / elapsed,
            "tokens_per_second": tokens / request_elapsed,
            "process_resident_bytes": resident,
            "process_peak_resident_bytes": peak,
            "resident_model_tensor_bytes": 484221909,
            "active_parameter_bytes": 484221909,
            "installed_parameter_bytes": 484221909,
            "quality": _quality(output),
            "output_sha256": hashlib.sha256(output).hexdigest(),
            "output_hex": output.hex(),
            "expected_codeword": prompt.get("expected_codeword"),
            "long_context_success": (
                output.decode("utf-8", errors="replace").lstrip().casefold().startswith(
                    str(prompt["expected_codeword"]).casefold()
                ) if long_context else None
            ),
            "status": "PASS",
        }
        rows.append(row)
        if (index + 1) % 10 == 0:
            file_suite = "long_context" if long_context else ("sustained" if sustained else "functional")
            destination = _path(root, PHASE / "raw_runs" / f"qwen_{file_suite}.json")
            _write(destination, {"format": RAW_FORMAT, "records": rows})
            print(f"qwen {suite} {index + 1}/{len(observations)}", flush=True)
    return rows


def benchmark_qwen(
    root: Path, endpoint: str, model_name: str, *, sustained: bool,
    long_context: bool = False,
) -> dict[str, Any]:
    rows = _qwen_rows(
        root, endpoint, model_name, sustained=sustained, long_context=long_context,
    )
    suite = "long_context" if long_context else ("sustained_1024" if sustained else "functional_headline")
    file_suite = "long_context" if long_context else ("sustained" if sustained else "functional")
    name = f"qwen_{file_suite}.json"
    output = _path(root, PHASE / "raw_runs" / name)
    _write(output, {"format": RAW_FORMAT, "records": rows})
    return {"status": "PASS", "records": len(rows), "path": _relative(root, output)}


@torch.inference_mode()
def _calibration_error(model, tokenizer, corpus: ByteCorpus, *, device, bins: int) -> float:
    model.eval()
    confidences = []
    correctness = []
    for rows in corpus.fixed_batches(batch_size=8, sequence_bytes=256, batches=16, device="cpu"):
        tokens, _ = _token_batch(tokenizer, rows, device=device, max_tokens=model.config.max_tokens)
        logits = model(tokens[:, :-1])
        probabilities = F.softmax(logits.float(), dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        confidences.append(confidence.flatten().cpu())
        correctness.append((predicted == tokens[:, 1:]).float().flatten().cpu())
    confidence = torch.cat(confidences)
    correct = torch.cat(correctness)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (confidence >= lower) & (confidence < upper if index + 1 < bins else confidence <= upper)
        if selected.any():
            error += float(selected.float().mean()) * abs(
                float(confidence[selected].mean()) - float(correct[selected].mean())
            )
    model.train()
    return error


def evaluate_quality(root: Path, *, access_test: bool) -> dict[str, Any]:
    records = []
    architectures = set()
    config = _read(_path(root, BENCHMARK_CONFIG))
    calibration_bins = int(config["calibration"]["bins"])
    for seed in SEEDS:
        checkpoint = _candidate_checkpoint(root, seed)
        model, tokenizer, metadata = load_sparse_bpe_checkpoint(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
        architectures.add(_canonical_sha(metadata["architecture"]))
        record = {
            "seed": seed,
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "tokenizer_sha256": metadata["tokenizer"]["sha256"],
            "raw_training_bytes": metadata["training"].get(
                "cumulative_raw_bytes_seen", metadata["training"]["raw_bytes_seen"]
            ),
            "validation_bpb": metadata["quality"]["validation"]["bits_per_byte"],
            "validation_calibration_error": _calibration_error(
                model, tokenizer, ByteCorpus(metadata["data"]["validation"]["path"]),
                device=next(model.parameters()).device, bins=calibration_bins,
            ),
            "architecture_selection_bpb": metadata["quality"]["architecture_selection"]["bits_per_byte"],
            "test_bpb": None,
            "test_accessed": False,
        }
        if access_test:
            test = ByteCorpus(metadata["data"]["test"]["path"])
            device = next(model.parameters()).device
            score = evaluate_transformer(
                model, tokenizer, test,
                config={"batch_size": 8, "sequence_bytes": 256, "batches": 16},
                device=device,
            )
            record["test_bpb"] = score["bits_per_byte"]
            record["test_accessed"] = True
        records.append(record)
    if len(architectures) != 1:
        raise RuntimeError("three-seed quality evidence mixes architectures")
    reference_device = "cuda" if torch.cuda.is_available() else "cpu"
    reference_model, reference_tokenizer, reference_metadata = load_transformer_checkpoint(
        _path(root, "artifacts/final/medium-transformers/seed-9801"), device=reference_device
    )
    reference_calibration = _calibration_error(
        reference_model, reference_tokenizer,
        ByteCorpus(reference_metadata["data"]["validation"]["path"]),
        device=next(reference_model.parameters()).device, bins=calibration_bins,
    )
    output = _path(root, PHASE / "raw_runs/quality_seeds.json")
    _write(output, {
        "format": "layercake-phase2-quality-seeds/1",
        "architecture_hash": next(iter(architectures)),
        "reference": {
            "model_id": "bpe-reference",
            "checkpoint_sha256": reference_metadata["checkpoint"]["sha256"],
            "validation_bpb": reference_metadata["quality"]["validation"]["bits_per_byte"],
            "validation_calibration_error": reference_calibration,
        },
        "test_access_authorized_after_architecture_freeze": bool(access_test),
        "records": records,
    })
    return {"status": "PASS", "records": records, "path": _relative(root, output)}


def _load_rows(root: Path, name: str) -> list[dict[str, Any]]:
    document = _read(_path(root, PHASE / "raw_runs" / name))
    rows = document.get("records")
    if not isinstance(rows, list):
        raise RuntimeError(f"raw evidence {name} has no records")
    return rows


def _paired_means(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(float(row[field]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def _interval(value) -> dict[str, Any]:
    return dict(value.__dict__)


def _functional_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    primary = [row for row in rows if row["trial"] == 1]
    instruction = [row for row in primary if str(row["prompt_id"]).endswith("-04")]
    coherence = [row for row in primary if str(row["prompt_id"]).endswith("-08")]
    topics = (
        "efficient computing", "public libraries", "urban gardens", "coastal weather",
        "scientific replication", "music practice", "safe navigation", "local history",
        "water conservation", "collaborative design",
    )

    def acceptable(row: Mapping[str, Any]) -> bool:
        metrics = row["quality"]
        return (
            metrics["valid_utf8"] == 1.0
            and metrics["printable_character_rate"] >= 0.95
            and metrics["word_count"] >= 80.0
            and metrics["repetition_rate"] <= 0.50
        )

    def exactly_two_sentences(row: Mapping[str, Any]) -> bool:
        text = bytes.fromhex(str(row["output_hex"])).decode("utf-8", errors="replace")
        terminators = sum(text.count(symbol) for symbol in (".", "!", "?"))
        return acceptable(row) and terminators == 2

    def text(row: Mapping[str, Any]) -> str:
        return bytes.fromhex(str(row["output_hex"])).decode("utf-8", errors="replace").lower()

    def relevant(row: Mapping[str, Any]) -> bool:
        topic_index = int(str(row["prompt_id"]).split("-")[1])
        words = topics[topic_index].split()
        observed = text(row)
        return acceptable(row) and any(word in observed for word in words)

    def task_constraint(row: Mapping[str, Any]) -> bool:
        if not relevant(row):
            return False
        task_index = int(str(row["prompt_id"]).split("-")[2])
        observed = text(row)
        if task_index == 2:
            numeric = all(marker in observed for marker in ("1", "2", "3"))
            ordinal = all(marker in observed for marker in ("first", "second", "third"))
            return numeric or ordinal
        if task_index == 3:
            return any(marker in observed for marker in ("tradeoff", "however", "whereas", "while", "versus"))
        if task_index == 4:
            return exactly_two_sentences(row)
        if task_index == 5:
            cause = any(marker in observed for marker in ("cause", "because", "due to"))
            consequence = any(marker in observed for marker in ("consequence", "result", "lead", "therefore"))
            return cause and consequence
        if task_index == 6:
            return not bool(re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s", observed))
        return True

    return {
        "functional_pass_rate": sum(acceptable(row) for row in primary) / len(primary),
        "instruction_following_rate": sum(exactly_two_sentences(row) for row in instruction) / len(instruction),
        "coherence_pass_rate": sum(acceptable(row) for row in coherence) / len(coherence),
        "topic_relevance_rate": sum(relevant(row) for row in primary) / len(primary),
        "task_constraint_success_rate": sum(task_constraint(row) for row in primary) / len(primary),
    }


def finalize(root: Path) -> dict[str, Any]:
    candidate = _load_rows(root, "layercake_functional.json")
    qwen = _load_rows(root, "qwen_functional.json")
    candidate_sustained = _load_rows(root, "layercake_sustained.json")
    qwen_sustained = _load_rows(root, "qwen_sustained.json")
    candidate_long = _load_rows(root, "layercake_long_context.json")
    qwen_long = _load_rows(root, "qwen_long_context.json")
    quality = _read(_path(root, PHASE / "raw_runs/quality_seeds.json"))
    config = _read(_path(root, BENCHMARK_CONFIG))
    statistics_config = config["statistics"]
    confidence = float(statistics_config["confidence"])
    resamples = int(statistics_config["resamples"])
    bootstrap_seed = int(statistics_config["bootstrap_seed"])

    def ci(values: Sequence[float]):
        return _interval(bootstrap_confidence_interval(
            values, confidence=confidence, resamples=resamples, seed=bootstrap_seed
        ))

    pairs = {}
    for field in ("bytes_per_second", "characters_per_second", "total_latency_seconds", "time_to_first_output_seconds"):
        left = _paired_means(candidate, field)
        right = _paired_means(qwen, field)
        pairs[field] = _interval(paired_bootstrap_difference(
            left, right, confidence=confidence, resamples=resamples, seed=bootstrap_seed
        ))
    candidate_bps = [float(row["bytes_per_second"]) for row in candidate]
    qwen_bps = [float(row["bytes_per_second"]) for row in qwen]
    candidate_latency = [float(row["total_latency_seconds"]) for row in candidate]
    qwen_latency = [float(row["total_latency_seconds"]) for row in qwen]
    candidate_ttfo = [float(row["time_to_first_output_seconds"]) for row in candidate]
    qwen_ttfo = [float(row["time_to_first_output_seconds"]) for row in qwen]
    candidate_sustained_bps = [float(row["bytes_per_second"]) for row in candidate_sustained]
    qwen_sustained_bps = [float(row["bytes_per_second"]) for row in qwen_sustained]
    quality_rows = quality["records"]
    validation_bpb = [float(row["validation_bpb"]) for row in quality_rows]
    calibration_errors = [float(row["validation_calibration_error"]) for row in quality_rows]
    reference_calibration = float(quality["reference"]["validation_calibration_error"])
    candidate_long_accuracy = statistics.fmean(float(row["long_context_success"]) for row in candidate_long)
    qwen_long_accuracy = statistics.fmean(float(row["long_context_success"]) for row in qwen_long)

    functional = {}
    for system, rows in (("layercake", candidate), ("qwen", qwen)):
        primary = [row for row in rows if row["trial"] == 1]
        metrics = tuple(primary[0]["quality"])
        functional[system] = {
            metric: sum(float(row["quality"][metric]) for row in primary) / len(primary)
            for metric in metrics
        }
        functional[system].update(_functional_scores(rows))
    threshold = float(config["quality_reference"]["validation_bpb"])
    margin = float(config["quality_reference"]["non_inferiority_margin"])
    aggregates = {
        "validation_bpb_mean": sum(validation_bpb) / len(validation_bpb),
        "validation_bpb_seed_ci": ci(validation_bpb),
        "locked_transformer_validation_bpb": threshold,
        "heldout_bpb_delta": sum(validation_bpb) / len(validation_bpb) - threshold,
        "validation_calibration_error_mean": sum(calibration_errors) / len(calibration_errors),
        "reference_validation_calibration_error": reference_calibration,
        "calibration_error_delta": sum(calibration_errors) / len(calibration_errors) - reference_calibration,
        "cpu_throughput_ratio": (sum(candidate_bps) / len(candidate_bps)) / (sum(qwen_bps) / len(qwen_bps)),
        "cpu_median_latency_ratio": statistics.median(candidate_latency) / statistics.median(qwen_latency),
        "time_to_first_output_ratio": statistics.median(candidate_ttfo) / statistics.median(qwen_ttfo),
        "active_memory_ratio": max(row["active_parameter_bytes"] for row in candidate) / max(row["active_parameter_bytes"] for row in qwen),
        "installed_model_memory_ratio": max(row["installed_parameter_bytes"] for row in candidate) / max(row["installed_parameter_bytes"] for row in qwen),
        "process_resident_memory_ratio": max(row["process_resident_bytes"] for row in candidate) / max(row["process_resident_bytes"] for row in qwen),
        "sustained_1024_byte_throughput_ratio": (sum(candidate_sustained_bps) / len(candidate_sustained_bps)) / (sum(qwen_sustained_bps) / len(qwen_sustained_bps)),
        "long_context_accuracy": candidate_long_accuracy,
        "qwen_long_context_accuracy": qwen_long_accuracy,
        "long_context_accuracy_delta": candidate_long_accuracy - qwen_long_accuracy,
        "physical_sparse_execution": float(all(
            row["sparse_execution"]["maximum_active_experts_per_token"] == 1
            and row["sparse_execution"]["total_decode_expert_invocations"]
            == row["sparse_execution"]["generated_token_count"]
            for row in candidate + candidate_sustained + candidate_long
        )),
        "headline_distributions": {
            system: {
                "latency_seconds": {"p50": p50(latencies), "p95": p95(latencies), "p99": p99(latencies)},
                "bytes_per_second": {"p50": p50(rates), "p95": p95(rates), "p99": p99(rates)},
                "observations": len(latencies),
            }
            for system, latencies, rates in (
                ("layercake", candidate_latency, candidate_bps),
                ("qwen", qwen_latency, qwen_bps),
            )
        },
        "sustained_distributions": {
            system: {
                "bytes_per_second": {"p50": p50(rates), "p95": p95(rates), "p99": p99(rates)},
                "observations": len(rates),
            }
            for system, rates in (
                ("layercake", candidate_sustained_bps), ("qwen", qwen_sustained_bps),
            )
        },
    }
    gates = {
        "heldout_bpb_delta": aggregates["heldout_bpb_delta"] <= margin,
        "calibration_noninferior": aggregates["calibration_error_delta"] <= float(config["calibration"]["candidate_minus_reference_margin"]),
        "cpu_throughput_ratio": aggregates["cpu_throughput_ratio"] >= 2.0,
        "cpu_median_latency_ratio": aggregates["cpu_median_latency_ratio"] <= 0.60,
        "time_to_first_output_ratio": aggregates["time_to_first_output_ratio"] <= 1.0,
        "active_memory_ratio": aggregates["active_memory_ratio"] < 1.0,
        "installed_model_memory_ratio": aggregates["installed_model_memory_ratio"] < 1.0,
        "process_resident_memory_ratio": aggregates["process_resident_memory_ratio"] < 1.0,
        "sustained_1024_byte_throughput_ratio": aggregates["sustained_1024_byte_throughput_ratio"] >= 2.0,
        "long_context_noninferior": aggregates["long_context_accuracy_delta"] >= -0.02,
        "physical_sparse_execution": aggregates["physical_sparse_execution"] == 1.0,
        "three_seed_replication": len({row["seed"] for row in quality_rows}) >= 3,
        "invalid_output_noninferior": functional["layercake"]["invalid_output_rate"] <= functional["qwen"]["invalid_output_rate"] + 0.01,
        "repetition_noninferior": functional["layercake"]["repetition_rate"] <= functional["qwen"]["repetition_rate"] + 0.02,
        "unique_ngram_noninferior": functional["layercake"]["unique_4gram_rate"] >= functional["qwen"]["unique_4gram_rate"] - 0.02,
        "no_entropy_collapse": functional["layercake"]["longest_identical_byte_run"] <= max(8.0, functional["qwen"]["longest_identical_byte_run"] + 2.0),
        "minimum_response_length": functional["layercake"]["word_count"] >= 80.0,
        "functional_task_noninferior": functional["layercake"]["functional_pass_rate"] >= functional["qwen"]["functional_pass_rate"] - 0.02,
        "instruction_following_noninferior": functional["layercake"]["instruction_following_rate"] >= functional["qwen"]["instruction_following_rate"] - 0.02,
        "coherence_noninferior": functional["layercake"]["coherence_pass_rate"] >= functional["qwen"]["coherence_pass_rate"] - 0.02,
        "topic_relevance_noninferior": functional["layercake"]["topic_relevance_rate"] >= functional["qwen"]["topic_relevance_rate"] - 0.02,
        "task_constraint_noninferior": functional["layercake"]["task_constraint_success_rate"] >= functional["qwen"]["task_constraint_success_rate"] - 0.02,
        "same_primary_checkpoint_for_quality_and_speed": (
            candidate[0]["checkpoint_sha256"] == quality_rows[0]["checkpoint_sha256"]
            == candidate_sustained[0]["checkpoint_sha256"] == candidate_long[0]["checkpoint_sha256"]
        ),
    }
    document = {
        "format": "layercake-phase2-derived-evidence/1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "benchmark_config": {"path": BENCHMARK_CONFIG.as_posix(), "sha256": sha256_file(_path(root, BENCHMARK_CONFIG))},
        "primary_checkpoint_sha256": candidate[0]["checkpoint_sha256"],
        "transformer_checkpoint_sha256": qwen[0]["checkpoint_sha256"],
        "aggregates": aggregates,
        "paired_prompt_bootstrap_differences": pairs,
        "functional_quality": functional,
        "gates": gates,
        "failed_gates": sorted(key for key, passed in gates.items() if not passed),
        "raw_artifacts": {
            name: sha256_file(_path(root, PHASE / "raw_runs" / name))
            for name in (
                "layercake_functional.json", "qwen_functional.json",
                "layercake_sustained.json", "qwen_sustained.json", "quality_seeds.json",
                "layercake_long_context.json", "qwen_long_context.json",
            )
        },
    }
    output = _path(root, PHASE / "derived_evidence.json")
    _write(output, document)

    gate_records = [
        {
            "system_id": "layercake_sparse_bpe_primary",
            "suite": "heldout_quality",
            "seed": row["seed"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "heldout_bpb": row["validation_bpb"],
        }
        for row in quality_rows
    ]
    gate_records.append({
        "system_id": "bpe_reference",
        "suite": "heldout_quality",
        "seed": 9801,
        "checkpoint_sha256": config["quality_reference"]["checkpoint_sha256"],
        "heldout_bpb": threshold,
    })
    for row in candidate + qwen + candidate_sustained + qwen_sustained + candidate_long + qwen_long:
        gate_records.append({
            "system_id": row["system_id"],
            "suite": row["suite"],
            "seed": row["seed"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "bytes_per_second": row["bytes_per_second"],
            "total_latency_seconds": row["total_latency_seconds"],
            "time_to_first_output_seconds": row["time_to_first_output_seconds"],
            "resident_model_tensor_bytes": row["resident_model_tensor_bytes"],
            "active_parameter_bytes": row["active_parameter_bytes"],
            "installed_parameter_bytes": row["installed_parameter_bytes"],
            "process_resident_bytes": row["process_resident_bytes"],
            "long_context_success": float(row.get("long_context_success") or 0.0),
            "physical_sparse_execution": float(
                row.get("sparse_execution", {}).get("maximum_active_experts_per_token") == 1
                and row.get("sparse_execution", {}).get("total_decode_expert_invocations")
                == row.get("sparse_execution", {}).get("generated_token_count")
            ) if row["system_id"] == "layercake_sparse_bpe_primary" else 0.0,
        })
    gate_path = _path(root, PHASE / "raw_runs/gate_observations.json")
    _write(gate_path, {
        "format": "layercake-phase2-gate-observations/1",
        "source_artifacts": document["raw_artifacts"],
        "records": gate_records,
    })
    gate_raw = _relative(root, gate_path)
    gate_sha = sha256_file(gate_path)

    def selector(field: str, system_id: str, suite: str) -> dict[str, Any]:
        return {"field": field, "where": {"system_id": system_id, "suite": suite}}

    claims = [
        {
            "gate_id": "heldout_bpb_delta", "value": aggregates["heldout_bpb_delta"],
            "raw_artifact": gate_raw, "raw_sha256": gate_sha,
            "derivation": {"operation": "difference", "reduction": "mean",
                "numerator": selector("heldout_bpb", "layercake_sparse_bpe_primary", "heldout_quality"),
                "denominator": selector("heldout_bpb", "bpe_reference", "heldout_quality")},
        },
        {
            "gate_id": "cpu_throughput_ratio", "value": aggregates["cpu_throughput_ratio"],
            "raw_artifact": gate_raw, "raw_sha256": gate_sha,
            "derivation": {"operation": "ratio", "reduction": "mean",
                "numerator": selector("bytes_per_second", "layercake_sparse_bpe_primary", "functional_headline"),
                "denominator": selector("bytes_per_second", "qwen25_05b_optimized_cpu", "functional_headline")},
        },
        {
            "gate_id": "cpu_median_latency_ratio", "value": aggregates["cpu_median_latency_ratio"],
            "raw_artifact": gate_raw, "raw_sha256": gate_sha,
            "derivation": {"operation": "ratio", "reduction": "median",
                "numerator": selector("total_latency_seconds", "layercake_sparse_bpe_primary", "functional_headline"),
                "denominator": selector("total_latency_seconds", "qwen25_05b_optimized_cpu", "functional_headline")},
        },
        {
            "gate_id": "time_to_first_output_ratio", "value": aggregates["time_to_first_output_ratio"],
            "raw_artifact": gate_raw, "raw_sha256": gate_sha,
            "derivation": {"operation": "ratio", "reduction": "median",
                "numerator": selector("time_to_first_output_seconds", "layercake_sparse_bpe_primary", "functional_headline"),
                "denominator": selector("time_to_first_output_seconds", "qwen25_05b_optimized_cpu", "functional_headline")},
        },
        {
            "gate_id": "active_memory_ratio", "value": aggregates["active_memory_ratio"],
            "raw_artifact": gate_raw, "raw_sha256": gate_sha,
            "derivation": {"operation": "ratio", "reduction": "max",
                "numerator": selector("active_parameter_bytes", "layercake_sparse_bpe_primary", "functional_headline"),
                "denominator": selector("active_parameter_bytes", "qwen25_05b_optimized_cpu", "functional_headline")},
        },
        {
            "gate_id": "sustained_1024_byte_throughput_ratio", "value": aggregates["sustained_1024_byte_throughput_ratio"],
            "raw_artifact": gate_raw, "raw_sha256": gate_sha,
            "derivation": {"operation": "ratio", "reduction": "mean",
                "numerator": selector("bytes_per_second", "layercake_sparse_bpe_primary", "sustained_1024"),
                "denominator": selector("bytes_per_second", "qwen25_05b_optimized_cpu", "sustained_1024")},
        },
        {
            "gate_id": "physical_sparse_execution", "value": aggregates["physical_sparse_execution"],
            "raw_artifact": gate_raw, "raw_sha256": gate_sha,
            "derivation": {"operation": "min", **selector(
                "physical_sparse_execution", "layercake_sparse_bpe_primary", "functional_headline"
            )},
        },
        {
            "gate_id": "long_context_accuracy_delta", "value": aggregates["long_context_accuracy_delta"],
            "raw_artifact": gate_raw, "raw_sha256": gate_sha,
            "derivation": {"operation": "difference", "reduction": "mean",
                "numerator": selector("long_context_success", "layercake_sparse_bpe_primary", "long_context"),
                "denominator": selector("long_context_success", "qwen25_05b_optimized_cpu", "long_context")},
        },
    ]
    for claim in claims:
        if claim["gate_id"] in {
            "cpu_throughput_ratio", "cpu_median_latency_ratio",
            "time_to_first_output_ratio", "sustained_1024_byte_throughput_ratio",
        }:
            claim["kind"] = "speed_ratio"
            claim["promoted"] = True
    payload = {
        "format": "layercake-phase2-certificate-payload/1",
        "status": document["status"],
        "lineage": {
            "architecture_id": "layercake-sparse-bpe-core/3-neural-guided-english-288x3-top1-8cakes",
            "architecture_hash": quality["architecture_hash"],
            "primary_checkpoint_sha256": candidate[0]["checkpoint_sha256"],
            "transformer_checkpoint_sha256": qwen[0]["checkpoint_sha256"],
        },
        "claims": claims,
        "quality_match": {
            "heldout_bpb": gates["heldout_bpb_delta"],
            "functional_task_quality": gates["functional_task_noninferior"],
            "instruction_following": gates["instruction_following_noninferior"],
            "invalid_output_rate": gates["invalid_output_noninferior"],
            "repetition": gates["repetition_noninferior"] and gates["no_entropy_collapse"],
            "coherence": gates["coherence_noninferior"],
            "topic_relevance": gates["topic_relevance_noninferior"],
            "task_constraints": gates["task_constraint_noninferior"],
            "long_context_recall": gates["long_context_noninferior"],
            "domain_success": "NOT_APPLICABLE_PHASE2_CORE_ONLY",
            "layercake_checkpoint_sha256": candidate[0]["checkpoint_sha256"],
            "transformer_checkpoint_sha256": qwen[0]["checkpoint_sha256"],
        },
        "derived_evidence": {"path": _relative(root, output), "sha256": sha256_file(output)},
    }
    _write(_path(root, PHASE / "certificate_payload.json"), payload)
    return document


def adversarial_checks(root: Path) -> dict[str, Any]:
    from .evaluation.phase2_evidence import (
        Phase2EvidenceError,
        _validate_depth,
        validate_inference_records,
    )

    candidate = _load_rows(root, "layercake_functional.json")
    qwen = _load_rows(root, "qwen_functional.json")
    candidate_long = _load_rows(root, "layercake_long_context.json")
    checkpoint = candidate[0]["checkpoint_sha256"]
    checks: list[dict[str, str]] = []

    def validate(rows: Sequence[Mapping[str, Any]]) -> None:
        validate_inference_records(
            rows, system_id="layercake_sparse_bpe_primary",
            suite="functional_headline", checkpoint_sha256=checkpoint,
            minimum_bytes=480,
            planner_sha256=candidate[0]["english_planner"]["checkpoint_buffer_sha256"],
        )

    def validate_qwen(rows: Sequence[Mapping[str, Any]]) -> None:
        validate_inference_records(
            rows, system_id="qwen25_05b_optimized_cpu",
            suite="functional_headline", checkpoint_sha256=qwen[0]["checkpoint_sha256"],
            minimum_bytes=480,
        )

    def validate_long(rows: Sequence[Mapping[str, Any]]) -> None:
        validate_inference_records(
            rows, system_id="layercake_sparse_bpe_primary",
            suite="long_context", checkpoint_sha256=checkpoint, minimum_bytes=64,
            planner_sha256=candidate[0]["english_planner"]["checkpoint_buffer_sha256"],
        )

    def detected(identifier: str, action) -> None:
        try:
            action()
        except Phase2EvidenceError as error:
            checks.append({"id": identifier, "status": "DETECTED", "error": str(error)})
            return
        raise RuntimeError(f"Phase 2 adversarial mutation escaped detection: {identifier}")

    mutations = []
    for identifier, path, value in (
        ("mixed_checkpoint_lineage", (0, "checkpoint_sha256"), "0" * 64),
        ("forged_bytes_per_second", (0, "bytes_per_second"), candidate[0]["bytes_per_second"] * 2),
        ("forged_output_hash", (0, "output_sha256"), "0" * 64),
        ("non_authoritative_token_count", (0, "token_accounting_method"), "estimated"),
        ("missing_incremental_state", (0, "persistent_state"), {}),
        ("dense_expert_execution", (0, "sparse_execution", "total_decode_expert_invocations"), candidate[0]["generated_tokens"] * 8),
        ("short_generation", (0, "generated_bytes"), 1),
    ):
        rows = copy.deepcopy(candidate)
        if len(path) == 2:
            rows[path[0]][path[1]] = value
        else:
            rows[path[0]][path[1]][path[2]] = value
        mutations.append((identifier, lambda rows=rows: validate(rows)))
    for identifier, action in mutations:
        detected(identifier, action)
    detected(
        "shallow_prompt_depth",
        lambda: _validate_depth(candidate[:-1], distinct=100, repeated=20, observations=120),
    )
    qwen_mutation = copy.deepcopy(qwen)
    qwen_mutation[0]["token_accounting_scope"] = "target_prefix"
    detected("mixed_transformer_token_scope", lambda: validate_qwen(qwen_mutation))
    qwen_hash_mutation = copy.deepcopy(qwen)
    qwen_hash_mutation[0]["completed_response_sha256"] = "0" * 64
    detected("forged_completed_transformer_response", lambda: validate_qwen(qwen_hash_mutation))
    long_mutation = copy.deepcopy(candidate_long)
    long_mutation[0]["long_context_success"] = not long_mutation[0]["long_context_success"]
    detected("forged_long_context_success", lambda: validate_long(long_mutation))
    planner_hash_mutation = copy.deepcopy(candidate)
    planner_hash_mutation[0]["english_planner"]["checkpoint_buffer_sha256"] = "0" * 64
    detected("forged_checkpoint_planner_binding", lambda: validate(planner_hash_mutation))
    planner_count_mutation = copy.deepcopy(candidate)
    planner_count_mutation[0]["english_planner"]["forced_plan_tokens"] -= 1
    detected("forged_planner_token_count", lambda: validate(planner_count_mutation))
    result = {
        "format": "layercake-phase2-adversarial-checks/1",
        "status": "PASS",
        "detected": len(checks),
        "checks": checks,
    }
    _write(_path(root, PHASE / "adversarial_checks.json"), result)
    return result


def assemble_release(root: Path) -> dict[str, Any]:
    """Materialize the final core, failure ledger, environment, and hash manifest."""

    derived = _read(_path(root, PHASE / "derived_evidence.json"))
    if derived.get("status") != "PASS":
        raise RuntimeError("refusing to assemble a failed Phase 2 candidate")
    adversarial = _read(_path(root, PHASE / "adversarial_checks.json"))
    if adversarial.get("status") != "PASS" or adversarial.get("detected", 0) < 13:
        raise RuntimeError("Phase 2 adversarial verification is incomplete")

    final_root = _path(root, "artifacts/moonshot/phase2/final_core")
    if final_root.exists():
        raise RuntimeError("final_core already exists; Phase 2 artifacts are immutable")
    final_root.mkdir(parents=True)
    seed_artifacts = []
    for seed in SEEDS:
        source = _candidate_checkpoint(root, seed)
        destination = final_root / f"seed-{seed}"
        destination.mkdir()
        files = []
        for name in ("model.safetensors", "tokenizer.json", "metadata.json"):
            copied = destination / name
            if name == "metadata.json":
                source_metadata_path = source / name
                normalized = _read(source_metadata_path)
                normalized["source_metadata_sha256"] = sha256_file(source_metadata_path)
                normalized["checkpoint"]["path"] = _relative(root, destination / "model.safetensors")
                normalized["tokenizer"]["path"] = _relative(root, destination / "tokenizer.json")
                corpus_name = Path(normalized["instruction_distillation"]["corpus_path"]).name
                normalized["instruction_distillation"]["corpus_path"] = (
                    Path("data/moonshot/phase2") / corpus_name
                ).as_posix()
                _write(copied, normalized)
            else:
                shutil.copy2(source / name, copied)
            files.append({"path": _relative(root, copied), "sha256": sha256_file(copied), "bytes": copied.stat().st_size})
        seed_artifacts.append({"seed": seed, "primary": seed == 9824, "files": files})
    core_manifest = {
        "format": "layercake-phase2-final-core/1",
        "architecture_id": "layercake-sparse-bpe-core/3-neural-guided-english-288x3-top1-8cakes",
        "primary_seed": 9824,
        "seeds": seed_artifacts,
        "same_checkpoint_quality_and_speed": derived["primary_checkpoint_sha256"],
        "installed_cakes": 8,
        "active_cakes_per_token": 1,
        "incremental_state": "per-layer KV cache",
        "english_realization": {
            "mode": "checkpoint-bound neural-guided constrained English",
            "planner_buffer_sha256": _read(_candidate_checkpoint(root, 9824) / "metadata.json")["english_planner"]["checkpoint_buffer_sha256"],
            "planner_preserving_tokenizer_sha256": _read(_candidate_checkpoint(root, 9824) / "metadata.json")["tokenizer"]["sha256"],
            "frozen_evaluation_content": False,
        },
        "instruction_distillation": {
            "corpus_sha256": _read(_candidate_checkpoint(root, 9824) / "metadata.json")["instruction_distillation"]["corpus_sha256"],
            "steps": _read(_candidate_checkpoint(root, 9824) / "metadata.json")["instruction_distillation"]["steps"],
            "same_frozen_corpus_all_seeds": len({
                _read(_candidate_checkpoint(root, seed) / "metadata.json")["instruction_distillation"]["corpus_sha256"]
                for seed in SEEDS
            }) == 1,
        },
    }
    _write(final_root / "manifest.json", core_manifest)

    architecture = _read(_path(root, PHASE / "experiments/architecture_pilot.json"))
    gru = _read(_path(root, PHASE / "experiments/gru_local_pilot.json"))
    shallow = _read(_path(root, "artifacts/moonshot/phase2/shallow-transformer-pilot/seed-9823/metadata.json"))
    sparse_pilot = _read(_path(root, "artifacts/moonshot/phase2/sparse-bpe-288x3-pilot/seed-9824/metadata.json"))
    ledger = []
    for campaign_id, experiment, reason in (
        ("adaptive_attention_10m", architecture, "REJECTED_QUALITY"),
        ("adaptive_gru_10m", gru, "REJECTED_QUALITY"),
    ):
        for row in experiment["runs"]:
            ledger.append({
                "experiment": campaign_id,
                "candidate": row["candidate"],
                "seed": row["seed"],
                "raw_training_bytes": row["raw_bytes_seen"],
                "validation_bpb": row["validation_bpb"],
                "selection_bpb": row["selection_bpb"],
                "decision": reason,
                "test_accessed": row["test_accessed"],
            })
    ledger.extend([
        {
            "experiment": "shallow_transformer_shape_control",
            "candidate": "transformer-d160-l8",
            "seed": shallow["seed"],
            "raw_training_bytes": shallow["training"]["raw_bytes_seen"],
            "validation_bpb": shallow["quality"]["validation"]["bits_per_byte"],
            "selection_bpb": shallow["quality"]["architecture_selection"]["bits_per_byte"],
            "decision": "NON_LAYERCAKE_CONTROL_INFORMED_SHALLOW_SHAPE",
            "test_accessed": shallow["quality"]["test_accessed"],
        },
        {
            "experiment": "integrated_sparse_bpe_10m",
            "candidate": "sparse-bpe-d288-l3-top1-8cakes",
            "seed": sparse_pilot["seed"],
            "raw_training_bytes": sparse_pilot["training"]["raw_bytes_seen"],
            "validation_bpb": sparse_pilot["quality"]["validation"]["bits_per_byte"],
            "selection_bpb": sparse_pilot["quality"]["architecture_selection"]["bits_per_byte"],
            "decision": "PROMOTED_TO_100M_THREE_SEED_VALIDATION",
            "test_accessed": sparse_pilot["quality"]["test_accessed"],
        },
    ])
    speed_screen = _read(_path(root, PHASE / "experiments/speed_screen.json"))
    for row in speed_screen["records"]:
        ledger.append({
            "experiment": "architecture_cpu_speed_screen",
            **row,
            "promoted_gate_evidence": False,
        })
    for screen_path in sorted(_path(root, PHASE / "experiments").glob("primary_seed_screen*.json")):
        screen = _read(screen_path)
        ledger.append({
            "experiment": "primary_checkpoint_decode_screen",
            "screen": screen_path.stem,
            "checkpoint_sha256": screen["checkpoint_sha256"],
            "threads": screen.get("threads", 1),
            "validation_bpb": screen["validation_bpb"],
            "mean_bytes_per_second": screen["mean_bytes_per_second"],
            "mean_repetition_rate": screen["mean_repetition_rate"],
            "mean_unique_4gram_rate": screen["mean_unique_4gram_rate"],
            "invalid_outputs": screen["invalid_outputs"],
            "decision": "NON_PROMOTED_SELECTION_DIAGNOSTIC",
            "promoted_gate_evidence": False,
        })
    for seed in SEEDS:
        metadata = _read(_path(root, f"artifacts/moonshot/phase2/sparse-bpe-288x3-full/seed-{seed}/metadata.json"))
        ledger.append({
            "experiment": "integrated_sparse_bpe_100m",
            "candidate": "sparse-bpe-d288-l3-top1-8cakes",
            "seed": seed,
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "raw_training_bytes": metadata["training"]["raw_bytes_seen"],
            "validation_bpb": metadata["quality"]["validation"]["bits_per_byte"],
            "selection_bpb": metadata["quality"]["architecture_selection"]["bits_per_byte"],
            "decision": "PROMOTED_TO_INSTRUCTION_DISTILLATION",
            "test_accessed_during_selection": metadata["quality"]["test_accessed"],
        })
        word_stage = _path(root, (
            "artifacts/moonshot/phase2/sparse-bpe-word2304-transfer-pilot/seed-9824"
            if seed == 9824 else
            f"artifacts/moonshot/phase2/sparse-bpe-word2304-replication/seed-{seed}"
        ))
        phrase_stage = _path(root, (
            "artifacts/moonshot/phase2/sparse-bpe-planner2816-transfer-pilot/seed-9824"
            if seed == 9824 else
            f"artifacts/moonshot/phase2/sparse-bpe-planner2816-transfer-replication/seed-{seed}"
        ))
        for stage_name, stage_path, decision in (
            ("exact_prefix_word_tokenizer_transfer", word_stage, "PROMOTED_TO_PROMPT_CONDITIONING"),
            ("exact_prefix_planner_tokenizer_transfer", phrase_stage, "PROMOTED_TO_FINAL_DISTILLATION"),
        ):
            stage = _read(stage_path / "metadata.json")
            ledger.append({
                "experiment": stage_name,
                "seed": seed,
                "checkpoint_sha256": stage["checkpoint"]["sha256"],
                "tokenizer_sha256": stage["tokenizer"]["sha256"],
                "stage_raw_training_bytes": stage["training"]["raw_bytes_seen"],
                "cumulative_raw_training_bytes": stage["training"].get(
                    "cumulative_raw_bytes_seen", stage["training"]["raw_bytes_seen"]
                ),
                "validation_bpb": stage["quality"]["validation"]["bits_per_byte"],
                "decision": decision,
                "test_accessed_during_selection": stage["quality"]["test_accessed"],
            })
        distilled = _read(_candidate_checkpoint(root, seed) / "metadata.json")
        ledger.append({
            "experiment": "integrated_sparse_bpe_instruction_distillation",
            "candidate": "sparse-bpe-d288-l3-top1-8cakes-neural-guided-english",
            "seed": seed,
            "checkpoint_sha256": distilled["checkpoint"]["sha256"],
            "parent_checkpoint_sha256": distilled["parent_checkpoint"]["sha256"],
            "raw_pretraining_bytes": distilled["training"].get(
                "cumulative_raw_bytes_seen", distilled["training"]["raw_bytes_seen"]
            ),
            "distillation_steps": distilled["instruction_distillation"]["steps"],
            "validation_bpb": distilled["quality"]["validation"]["bits_per_byte"],
            "selection_bpb": distilled["quality"]["architecture_selection"]["bits_per_byte"],
            "decision": "PROMOTED_REPLICATION" if seed != 9824 else "PROMOTED_PRIMARY",
            "test_accessed_during_selection": distilled["quality"]["test_accessed"],
        })
    ledger_path = _path(root, PHASE / "search_ledger.jsonl")
    ledger_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in ledger
    ), encoding="utf-8")

    junit = _path(root, PHASE / "pytest.xml")
    suites_root = ET.parse(junit).getroot()
    suites = [suites_root] if suites_root.tag == "testsuite" else list(suites_root.findall("testsuite"))
    totals = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    test_results = {
        "format": "layercake-phase2-test-results/1",
        "status": "PASS" if totals["tests"] > 0 and totals["failures"] == totals["errors"] == 0 else "FAIL",
        **totals,
        "passed": totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"],
        "command": f'"{sys.executable}" -m pytest -q --junitxml=results/moonshot/phase2/pytest.xml',
        "junit_path": _relative(root, junit),
        "junit_sha256": sha256_file(junit),
    }
    if test_results["status"] != "PASS":
        raise RuntimeError("complete Phase 2 regression suite is not green")
    _write(_path(root, PHASE / "test_results.json"), test_results)
    _write(_path(root, PHASE / "environment.json"), {
        "format": "layercake-phase2-environment/1",
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "phase1_hardware": {
            "path": "results/moonshot/phase1/hardware.json",
            "sha256": sha256_file(_path(root, "results/moonshot/phase1/hardware.json")),
        },
        "energy": {
            "status": "NOT_MEASURABLE_NO_CALIBRATED_PER_PROCESS_SENSOR",
            "reason": "host exposes no calibrated per-process CPU energy counter to this harness",
        },
    })
    _write(_path(root, PHASE / "execution_commands.json"), {
        "format": "layercake-phase2-execution-commands/1",
        "commands": [
            "python -m layercake.phase2_campaign prepare",
            "python -m layercake.training.phase2_sparse_bpe configs/moonshot/phase2/sparse_bpe_288x3_full_seed_9824.json artifacts/moonshot/phase2/sparse-bpe-288x3-full/seed-9824",
            "python -m layercake.training.phase2_sparse_bpe configs/moonshot/phase2/sparse_bpe_288x3_full_seed_9825.json artifacts/moonshot/phase2/sparse-bpe-288x3-full/seed-9825",
            "python -m layercake.training.phase2_sparse_bpe configs/moonshot/phase2/sparse_bpe_288x3_full_seed_9826.json artifacts/moonshot/phase2/sparse-bpe-288x3-full/seed-9826",
            "python -m layercake.training.phase2_distillation generate --endpoint http://127.0.0.1:11435",
            "python -m layercake.training.phase2_distillation build-clean-curriculum",
            "python -m layercake.training.phase2_tokenizer --base artifacts/final/medium-transformers/seed-9801/tokenizer.json --output data/moonshot/phase2/word_preserving_bpe_2304.json --merges 2048 --training-bytes 10000000",
            "python -m layercake.training.phase2_tokenizer --base data/moonshot/phase2/word_preserving_bpe_2304.json --output data/moonshot/phase2/planner_preserving_bpe_2816.json --planner-extension-merges 512",
            "per-seed exact-prefix word transfer, prompt-conditioned clean distillation, exact-prefix planner-tokenizer transfer, and final 1200-step clean distillation using the recorded configs and search ledger",
            "python -m layercake.phase2_campaign evaluate-quality --access-test",
            "python -m layercake.phase2_campaign benchmark-layercake --suite functional",
            "python -m layercake.phase2_campaign benchmark-qwen --suite functional --endpoint http://127.0.0.1:11435",
            "python -m layercake.phase2_campaign benchmark-layercake --suite sustained",
            "python -m layercake.phase2_campaign benchmark-qwen --suite sustained --endpoint http://127.0.0.1:11435",
            "python -m layercake.phase2_campaign benchmark-layercake --suite long-context",
            "python -m layercake.phase2_campaign benchmark-qwen --suite long-context --endpoint http://127.0.0.1:11435",
            "python -m layercake.phase2_campaign finalize",
            "python -m layercake.phase2_campaign adversarial",
        ],
    })
    excluded = {"candidate.json", "candidate_verification.json", "release_certificate.json", "handoff.json", "seal.json", "evidence_manifest.json"}
    evidence_files = sorted(
        path for path in _path(root, PHASE).rglob("*") if path.is_file() and path.name not in excluded
    )
    artifact_files = sorted(path for path in final_root.rglob("*") if path.is_file())
    distillation_files = sorted(
        path for path in _path(root, "data/moonshot/phase2").glob("*")
        if path.is_file()
    )
    evidence_manifest = {
        "format": "layercake-phase2-evidence-manifest/1",
        "artifacts": [
            {"path": _relative(root, path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in evidence_files + artifact_files + distillation_files
        ],
    }
    evidence_manifest["raw_evidence_manifest_sha256"] = _canonical_sha([
        (row["path"], row["sha256"]) for row in evidence_manifest["artifacts"]
        if "/raw_runs/" in f"/{row['path']}"
    ])
    _write(_path(root, PHASE / "evidence_manifest.json"), evidence_manifest)
    return {
        "status": "PASS",
        "final_core": _relative(root, final_root / "manifest.json"),
        "search_ledger_entries": len(ledger),
        "tests": test_results,
        "evidence_artifacts": len(evidence_manifest["artifacts"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m layercake.phase2_campaign")
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    layercake = sub.add_parser("benchmark-layercake")
    layercake.add_argument("--checkpoint", type=Path, default=PRIMARY_CHECKPOINT)
    layercake.add_argument("--suite", choices=("functional", "sustained", "long-context"), required=True)
    screen = sub.add_parser("screen-checkpoint")
    screen.add_argument("--checkpoint", type=Path, default=PRIMARY_CHECKPOINT)
    screen.add_argument("--label", default="diagnostic")
    screen.add_argument("--threads", type=int, default=1)
    screen.add_argument("--repetition-penalty", type=float)
    screen.add_argument("--suite", choices=("functional", "sustained", "long-context"), default="functional")
    qwen = sub.add_parser("benchmark-qwen")
    qwen.add_argument("--endpoint", default="http://127.0.0.1:11435")
    qwen.add_argument("--model", default="qwen2.5:0.5b")
    qwen.add_argument("--suite", choices=("functional", "sustained", "long-context"), required=True)
    quality = sub.add_parser("evaluate-quality")
    quality.add_argument("--access-test", action="store_true")
    sub.add_parser("finalize")
    sub.add_parser("adversarial")
    sub.add_parser("assemble-release")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare":
        result = prepare(root)
    elif args.command == "benchmark-layercake":
        result = benchmark_layercake(
            root, args.checkpoint, sustained=args.suite == "sustained",
            long_context=args.suite == "long-context",
        )
    elif args.command == "screen-checkpoint":
        result = screen_checkpoint(
            root, args.checkpoint, label=args.label, threads=args.threads,
            repetition_penalty=args.repetition_penalty, suite=args.suite,
        )
    elif args.command == "benchmark-qwen":
        result = benchmark_qwen(
            root, args.endpoint, args.model, sustained=args.suite == "sustained",
            long_context=args.suite == "long-context",
        )
    elif args.command == "evaluate-quality":
        result = evaluate_quality(root, access_test=args.access_test)
    elif args.command == "finalize":
        result = finalize(root)
    elif args.command == "adversarial":
        result = adversarial_checks(root)
    elif args.command == "assemble-release":
        result = assemble_release(root)
    else:  # pragma: no cover
        raise RuntimeError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"PASS", "LOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
