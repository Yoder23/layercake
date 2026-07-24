"""Tokenizer-free Phase 2 recertification research and evidence commands."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import psutil
import torch
from safetensors.torch import load_file, save_file

from .moonshot_campaign import read_document, sha256_file
from .training.data import ByteCorpus
from .training.patch_campaign import (
    _build_model,
    _evaluate,
    _forward,
    run_variable_patch_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE = Path("results/moonshot/phase2_recertification")
ARTIFACTS = Path("artifacts/moonshot/phase2_recertification")
LOCKED_BPB_DELTA = 0.03


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _repetition_rate(payload: bytes, width: int = 4) -> float:
    if len(payload) < width:
        return 0.0
    grams = [payload[index : index + width] for index in range(len(payload) - width + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _output_quality(payload: bytes) -> dict[str, float]:
    try:
        decoded = payload.decode("utf-8")
        valid_utf8 = 1.0
    except UnicodeDecodeError:
        decoded = payload.decode("utf-8", errors="replace")
        valid_utf8 = 0.0
    characters = max(1, len(decoded))
    printable = sum(
        value.isprintable() or value in "\n\r\t" for value in decoded
    ) / characters
    words = [word for word in decoded.lower().split() if word]
    repetition = _repetition_rate(payload)
    return {
        "valid_utf8": valid_utf8,
        "invalid_output": 1.0 - valid_utf8,
        "printable_character_rate": printable,
        "unique_4gram_rate": 1.0 - repetition,
        "repetition_rate": repetition,
        "word_diversity": len(set(words)) / max(1, len(words)),
        "generated_characters": float(len(decoded)),
    }


def _topic_from_prompt(prompt: str, category: str) -> str:
    patterns = {
        "continuation": r"about (.+?)\.",
        "explanation": r"Explain (.+?) to ",
        "planning": r"improving (.+?)\.",
        "comparison": r"approaches to (.+?) and ",
        "instruction_following": r"sentences about (.+?)\.",
        "reasoning": r"involving (.+?)\.",
        "summarization": r"why (.+?) matters",
        "question_answering": r"benefit of (.+?)\?",
        "coherence": r"tools, and (.+?)\.",
        "repetition_control": r"Describe (.+?) with ",
    }
    match = re.search(patterns[category], prompt, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot extract topic from {category!r} prompt")
    return match.group(1).strip()


def _semantic_adherence_metrics(
    prompt: str,
    category: str,
    payload: bytes,
) -> dict[str, Any]:
    decoded = payload.decode("utf-8", errors="replace")
    normalized = " ".join(
        re.findall(r"[a-z0-9]+", decoded.lower())
    )
    topic = _topic_from_prompt(prompt, category)
    topic_words = re.findall(r"[a-z0-9]+", topic.lower())
    output_words = re.findall(r"[a-z0-9]+", decoded.lower())
    output_word_set = set(output_words)
    topic_recall = sum(
        word in output_word_set for word in set(topic_words)
    ) / max(1, len(set(topic_words)))
    topic_phrase = " ".join(topic_words) in normalized
    sentence_count = len(
        re.findall(r"[.!?](?=\s|$)", decoded.strip())
    )
    ordered_steps = len(
        re.findall(
            r"(?:^|\n)\s*(?:[1-3][.)]|first[,:]|second[,:]|third[,:])",
            decoded,
            flags=re.IGNORECASE,
        )
    )
    lower = decoded.lower()
    category_structure_pass = True
    if category == "planning":
        category_structure_pass = ordered_steps >= 3
    elif category == "comparison":
        category_structure_pass = (
            ("approach" in lower or "method" in lower)
            and ("tradeoff" in lower or "trade-off" in lower)
        )
    elif category == "instruction_following":
        category_structure_pass = sentence_count == 2
    elif category == "reasoning":
        category_structure_pass = (
            "cause" in lower and "consequence" in lower
        )
    elif category == "summarization":
        category_structure_pass = not bool(
            re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])", decoded)
        )
    elif category == "repetition_control":
        category_structure_pass = _repetition_rate(payload) <= 0.25
    minimum_words_pass = len(output_words) >= 80
    core_pass = (
        minimum_words_pass
        and topic_phrase
        and category_structure_pass
    )
    return {
        "topic": topic,
        "word_count": len(output_words),
        "minimum_80_words_pass": minimum_words_pass,
        "topic_phrase_present": topic_phrase,
        "topic_token_recall": topic_recall,
        "sentence_count": sentence_count,
        "ordered_step_markers": ordered_steps,
        "category_structure_pass": category_structure_pass,
        "core_adherence_pass": core_pass,
    }


def _distribution_summary(logits: torch.Tensor) -> dict[str, Any]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log2()).sum()
    values, indexes = probabilities.topk(5, dim=-1)
    return {
        "entropy_bits": float(entropy),
        "top_bytes": [int(value) for value in indexes.flatten()],
        "top_probabilities": [float(value) for value in values.flatten()],
    }


def _select_byte(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> int:
    if temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probabilities, sorted_indexes = probabilities.sort(
        dim=-1, descending=True
    )
    cumulative_before = sorted_probabilities.cumsum(dim=-1) - (
        sorted_probabilities
    )
    retained = cumulative_before < top_p
    filtered = sorted_probabilities * retained
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    sampled_rank = torch.multinomial(
        filtered, 1, generator=generator
    )
    return int(sorted_indexes.gather(-1, sampled_rank).item())


def _load_adaptive_checkpoint(run: Mapping[str, Any]) -> tuple[torch.nn.Module, dict]:
    artifact = Path(str(run["artifact"]))
    metadata = read_document(artifact / "metadata.json")
    model = _build_model(metadata["candidate"])
    model.load_state_dict(load_file(str(artifact / "model.safetensors"), device="cpu"))
    model.eval()
    return model, metadata


def _has_prompt_conditioning(model: torch.nn.Module) -> bool:
    return (
        getattr(model, "prompt_memory_adapter", None) is not None
        or getattr(model, "prompt_cross_key", None) is not None
        or getattr(model, "prompt_pointer_key", None) is not None
    )


@torch.inference_mode()
def _screen_incremental(
    run: Mapping[str, Any],
    prompt: bytes,
    output_bytes: int,
    *,
    loaded: tuple[torch.nn.Module, dict] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    sampling_seed: int = 0,
) -> dict[str, Any]:
    if run["kind"] != "adaptive_two_four" or run.get("status") != "PASS":
        return {"status": "NOT_APPLICABLE_NO_INCREMENTAL_RUNTIME"}
    model, metadata = loaded if loaded is not None else _load_adaptive_checkpoint(run)
    requested_threads = 1
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(requested_threads)
    prompt_tensor = torch.tensor([list(prompt)], dtype=torch.long)
    generator = (
        torch.Generator(device="cpu").manual_seed(sampling_seed)
        if temperature > 0.0
        else None
    )

    warm = model.prefill_incremental(prompt_tensor)
    for _ in range(8):
        selected = warm["next_logits"].argmax(dim=-1)
        model.incremental_step(warm, selected)

    process = psutil.Process()
    resident_before = process.memory_info().rss
    started = time.perf_counter_ns()
    state = model.prefill_incremental(prompt_tensor)
    prefill_completed = time.perf_counter_ns()
    generated = bytearray()
    first_output = None
    while len(generated) < output_bytes:
        remaining = output_bytes - len(generated)
        multibyte = (
            remaining >= 2
            and 2 in state["next_future_logits"]
            and int(state["byte_count"]) % 2 == 0
            and len(state["pending_bytes"]) in {0, 2}
            and int(state["byte_count"]) % model.local_window + 2
            <= model.local_window
        )
        logits = state["next_logits"]
        selected = _select_byte(
            logits,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        if first_output is None:
            first_output = time.perf_counter_ns()
        generated.append(selected)
        if multibyte:
            future_logits = model.proposed_future_logits(
                state,
                2,
                torch.tensor([selected], dtype=torch.long),
            )
            future_selected = _select_byte(
                future_logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            generated.append(future_selected)
            model.incremental_step_many(
                state,
                torch.tensor(
                    [selected, future_selected], dtype=torch.long
                ),
            )
        else:
            model.incremental_step(
                state, torch.tensor([selected], dtype=torch.long)
            )
    completed = time.perf_counter_ns()
    resident_after = process.memory_info().rss
    torch.set_num_threads(previous_threads)
    elapsed = (completed - started) / 1e9
    decode = (completed - prefill_completed) / 1e9
    raw = bytes(generated)

    # Trace replay is intentionally outside the benchmark timing boundary.
    # The replay must reproduce the timed greedy continuation byte for byte.
    trace_state = model.prefill_incremental(prompt_tensor)
    trace_generator = (
        torch.Generator(device="cpu").manual_seed(sampling_seed)
        if temperature > 0.0
        else None
    )
    events = []
    traced = bytearray()
    while len(traced) < output_bytes:
        index = len(traced)
        remaining = output_bytes - index
        multibyte = (
            remaining >= 2
            and 2 in trace_state["next_future_logits"]
            and int(trace_state["byte_count"]) % 2 == 0
            and len(trace_state["pending_bytes"]) in {0, 2}
            and int(trace_state["byte_count"]) % model.local_window + 2
            <= model.local_window
        )
        logits = trace_state["next_logits"]
        selected = _select_byte(
            logits,
            temperature=temperature,
            top_p=top_p,
            generator=trace_generator,
        )
        traced.append(selected)
        events.append({
            "index": index,
            "chosen_byte": selected,
            "source": "model_logits",
            "logit_horizon": 1,
            "external_override": False,
            "distribution_summary": _distribution_summary(logits[0]),
        })
        if multibyte:
            future_logits = model.proposed_future_logits(
                trace_state,
                2,
                torch.tensor([selected], dtype=torch.long),
            )
            future_selected = _select_byte(
                future_logits,
                temperature=temperature,
                top_p=top_p,
                generator=trace_generator,
            )
            traced.append(future_selected)
            events.append({
                "index": index + 1,
                "chosen_byte": future_selected,
                "source": "model_logits",
                "logit_horizon": 2,
                "external_override": False,
                "distribution_summary": _distribution_summary(
                    future_logits[0]
                ),
            })
            model.incremental_step_many(
                trace_state,
                torch.tensor(
                    [selected, future_selected], dtype=torch.long
                ),
            )
        else:
            model.incremental_step(
                trace_state, torch.tensor([selected], dtype=torch.long)
            )
    if bytes(traced) != raw:
        raise RuntimeError("unmeasured generation trace replay diverged from timed output")
    try:
        decoded = raw.decode("utf-8")
        valid_utf8 = True
    except UnicodeDecodeError:
        decoded = raw.decode("utf-8", errors="replace")
        valid_utf8 = False
    trace = {
        "format": "layercake-free-neural-byte-trace/1",
        "prompt_bytes_hex": prompt.hex(),
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "generated_bytes_hex": raw.hex(),
        "generated_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_events": events,
        "external_path_counters": {
            "planner_calls": 0,
            "template_calls": 0,
            "retrieval_calls": 0,
            "stored_answer_calls": 0,
        },
        "decoding": {
            "mode": "sampled" if temperature > 0.0 else "greedy",
            "temperature": temperature,
            "top_p": top_p,
            "seed": sampling_seed,
        },
    }
    return {
        "status": "PASS",
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "prompt_bytes": len(prompt),
        "generated_bytes": len(raw),
        "generated_text": decoded,
        "valid_utf8": valid_utf8,
        "repetition_rate_4gram": _repetition_rate(raw),
        "unique_byte_fraction": len(set(raw)) / max(1, len(raw)),
        "prefill_seconds": (prefill_completed - started) / 1e9,
        "time_to_first_output_seconds": (first_output - started) / 1e9,
        "decode_seconds": decode,
        "total_latency_seconds": elapsed,
        "bytes_per_second_decode": len(raw) / decode,
        "bytes_per_second_total": len(raw) / elapsed,
        "resident_memory_bytes_before": resident_before,
        "resident_memory_bytes_after": resident_after,
        "state_patch_count": int(state["patch_count"]),
        "state_byte_count": int(state["byte_count"]),
        "physically_executed_experts": len(state["routed_expert_trace"]),
        "trace": trace,
    }


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _timing_summary(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "bytes_per_second_decode",
        "bytes_per_second_total",
        "prefill_seconds",
        "time_to_first_output_seconds",
        "decode_seconds",
        "total_latency_seconds",
    )
    summary: dict[str, Any] = {"observations": len(observations)}
    for field in fields:
        values = [float(row[field]) for row in observations]
        summary[field] = {
            "minimum": min(values),
            "median": statistics.median(values),
            "p95": _nearest_rank(values, 0.95),
            "p99": _nearest_rank(values, 0.99),
            "maximum": max(values),
        }
    return summary


def _byte_class(value: int) -> str:
    if value in {9, 10, 13, 32}:
        return "whitespace"
    if 65 <= value <= 90 or 97 <= value <= 122:
        return "ascii_letter"
    if 48 <= value <= 57:
        return "ascii_digit"
    if 32 <= value <= 126:
        return "ascii_punctuation"
    if 128 <= value <= 191:
        return "utf8_continuation"
    if 192 <= value <= 247:
        return "utf8_lead"
    return "control_or_other"


def _transition_class(previous: int, target: int) -> str:
    previous_class = _byte_class(previous)
    target_class = _byte_class(target)
    if previous_class == "whitespace" and target_class == "ascii_letter":
        return "ascii_word_start"
    if previous_class == "ascii_letter" and target_class == "ascii_letter":
        return "inside_ascii_word"
    if previous_class == "ascii_letter" and target_class == "whitespace":
        return "ascii_word_end"
    if "utf8" in previous_class or "utf8" in target_class:
        return "utf8_transition"
    return "other_transition"


def _add_group(
    groups: dict[str, dict[str, float | int]],
    key: str,
    losses: torch.Tensor,
    correct: torch.Tensor,
) -> None:
    row = groups.setdefault(key, {"count": 0, "loss_nats": 0.0, "correct": 0})
    row["count"] = int(row["count"]) + int(losses.numel())
    row["loss_nats"] = float(row["loss_nats"]) + float(losses.sum())
    row["correct"] = int(row["correct"]) + int(correct.sum())


def _finalize_groups(
    groups: Mapping[str, Mapping[str, float | int]],
) -> dict[str, Any]:
    result = {}
    for key, row in sorted(groups.items()):
        count = int(row["count"])
        result[key] = {
            "bytes": count,
            "bits_per_byte": float(row["loss_nats"]) / max(1, count) / math.log(2),
            "byte_accuracy": int(row["correct"]) / max(1, count),
        }
    return result


@torch.inference_mode()
def profile_quality_limiter(
    root: Path,
    input_path: Path,
    candidate_name: str,
    output_path: Path,
) -> dict[str, Any]:
    """Locate where the best pre-branch checkpoint loses predictive quality."""

    source = read_document(input_path)
    run = next(
        (
            row for row in source["campaign"]["runs"]
            if row["candidate"] == candidate_name
        ),
        None,
    )
    if run is None:
        raise ValueError(f"candidate not found in search evidence: {candidate_name}")
    config_path = root / source["config_path"]
    config = read_document(config_path)
    model, metadata = _load_adaptive_checkpoint(run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    corpus_path = root / source["campaign"]["data"]["validation"]["path"]
    corpus = ByteCorpus(corpus_path)
    evaluation = config["evaluation"]
    groups: dict[str, dict[str, dict[str, float | int]]] = {
        "patch_offset": {},
        "local_position": {},
        "local_context_depth": {},
        "target_byte_class": {},
        "transition_class": {},
    }
    loss_nats = 0.0
    byte_count = 0
    correct_count = 0
    for row in corpus.fixed_batches(
        batch_size=int(evaluation["batch_size"]),
        sequence_bytes=int(evaluation["sequence_bytes"]),
        batches=int(evaluation["batches"]),
        device=device,
    ):
        inputs, targets = row[:, :-1], row[:, 1:]
        logits, forward_metadata = _forward(model, inputs)
        losses = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(),
            targets.flatten(),
            reduction="none",
        ).reshape_as(targets)
        correct = logits.argmax(dim=-1).eq(targets)
        loss_nats += float(losses.sum())
        byte_count += int(losses.numel())
        correct_count += int(correct.sum())

        patch_offsets = forward_metadata["patch_offsets"][:, : inputs.shape[1]]
        for offset in range(model.max_patch_size):
            mask = patch_offsets.eq(offset)
            _add_group(
                groups["patch_offset"],
                str(offset),
                losses[mask],
                correct[mask],
            )
        for position in range(model.local_window):
            mask = (
                torch.arange(inputs.shape[1], device=device)
                .remainder(model.local_window)
                .eq(position)[None]
                .expand_as(inputs)
            )
            _add_group(
                groups["local_position"],
                str(position),
                losses[mask],
                correct[mask],
            )
            depth = min(position + 1, 8)
            depth_key = str(depth) if depth < 8 else "8_plus"
            _add_group(
                groups["local_context_depth"],
                depth_key,
                losses[mask],
                correct[mask],
            )

        flat_inputs = inputs.flatten().tolist()
        flat_targets = targets.flatten().tolist()
        flat_losses = losses.flatten()
        flat_correct = correct.flatten()
        class_indexes: dict[str, list[int]] = {}
        transition_indexes: dict[str, list[int]] = {}
        for index, (previous, target) in enumerate(
            zip(flat_inputs, flat_targets)
        ):
            class_indexes.setdefault(_byte_class(target), []).append(index)
            transition_indexes.setdefault(
                _transition_class(previous, target), []
            ).append(index)
        for key, indexes in class_indexes.items():
            selected = torch.tensor(indexes, device=device)
            _add_group(
                groups["target_byte_class"],
                key,
                flat_losses[selected],
                flat_correct[selected],
            )
        for key, indexes in transition_indexes.items():
            selected = torch.tensor(indexes, device=device)
            _add_group(
                groups["transition_class"],
                key,
                flat_losses[selected],
                flat_correct[selected],
            )

    finalized = {
        name: _finalize_groups(rows) for name, rows in groups.items()
    }
    local_rows = finalized["local_position"]
    highest_local = max(
        local_rows,
        key=lambda key: local_rows[key]["bits_per_byte"],
    )
    transition_rows = finalized["transition_class"]
    highest_transition = max(
        transition_rows,
        key=lambda key: transition_rows[key]["bits_per_byte"],
    )
    exact_command = " ".join([
        sys.executable,
        "-m",
        "layercake.phase2_recertification",
        "profile-quality",
        "--input",
        input_path.relative_to(root).as_posix(),
        "--candidate",
        candidate_name,
        "--output",
        output_path.relative_to(root).as_posix(),
    ])
    document = {
        "format": "layercake-phase2-quality-limiter-profile/1",
        "status": "PASS",
        "input_path": input_path.relative_to(root).as_posix(),
        "input_sha256": sha256_file(input_path),
        "candidate": candidate_name,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "validation_data_sha256": source["campaign"]["data"]["validation"]["sha256"],
        "test_accessed": False,
        "exact_command": exact_command,
        "overall": {
            "evaluated_bytes": byte_count,
            "bits_per_byte": loss_nats / max(1, byte_count) / math.log(2),
            "byte_accuracy": correct_count / max(1, byte_count),
        },
        "groups": finalized,
        "measured_bottleneck": {
            "highest_loss_local_position": int(highest_local),
            "highest_loss_local_position_bpb": local_rows[highest_local][
                "bits_per_byte"
            ],
            "highest_loss_transition": highest_transition,
            "highest_loss_transition_bpb": transition_rows[
                highest_transition
            ]["bits_per_byte"],
        },
    }
    document["profile_sha256"] = _canonical_sha(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_ledger(root / PHASE / "experiment_ledger.jsonl", [{
        "event": "quality_limiter_profile_completed",
        "candidate": candidate_name,
        "checkpoint_hash": metadata["checkpoint"]["sha256"],
        "exact_command": exact_command,
        "profile": output_path.relative_to(root).as_posix(),
        "profile_sha256": document["profile_sha256"],
        "test_accessed": False,
        "measured_bottleneck": document["measured_bottleneck"],
        "next_experiment": (
            "select one architecture change that directly addresses the "
            "highest measured causal loss stratum without reducing CPU speed"
        ),
    }])
    state_path = root / PHASE / "task_state.json"
    state = read_document(state_path)
    state.update({
        "latest_batch": output_path.relative_to(root).as_posix(),
        "latest_batch_sha256": document["profile_sha256"],
        "phase2_status": "OPEN_QUALITY_LIMITER_PROFILED",
        "continuation_command": "EVIDENCE_GATED_ARCHITECTURE_SELECTION_REQUIRED",
    })
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "candidate": candidate_name,
        "profile_sha256": document["profile_sha256"],
        "measured_bottleneck": document["measured_bottleneck"],
    }


def decide_conditional_branch(
    root: Path,
    search_path: Path,
    rescreen_path: Path,
    prior_search_path: Path,
    prior_rescreen_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Apply the user-locked Pareto and bounded-fusion decision rule."""

    search = read_document(search_path)
    rescreen = read_document(rescreen_path)
    prior_search = read_document(prior_search_path)
    prior_rescreen = read_document(prior_rescreen_path)
    baseline = read_document(root / PHASE / "baseline_audit.json")
    same_scale = next(
        row for row in baseline["branches"]
        if row["id"] == "same_scale_bpe_transformer_control"
    )
    qwen_summary = next(
        row for row in read_document(
            root / "results/moonshot/phase1/baseline_performance.json"
        )["summaries"]
        if row["system_id"] == "transformer_product_headline"
    )
    quality_reference_bpb = float(same_scale["validation_bpb"])
    quality_ceiling = quality_reference_bpb + LOCKED_BPB_DELTA
    qwen_median_bps = float(qwen_summary["bytes_per_second"]["p50"])
    search_runs = {
        row["candidate"]: row for row in search["campaign"]["runs"]
    }
    rows = {}
    for name, timing in rescreen["candidates"].items():
        run = search_runs[name]
        screen = search["cpu_screens"][name]
        median_bps = float(
            timing["summary"]["bytes_per_second_decode"]["median"]
        )
        quality_pass = float(timing["validation_bpb"]) <= quality_ceiling
        speed_ratio = median_bps / qwen_median_bps
        speed_pass = speed_ratio >= 2.0
        observations = timing["raw_observations"]
        rows[name] = {
            "checkpoint_sha256": timing["checkpoint_sha256"],
            "validation_bpb": timing["validation_bpb"],
            "quality_gate_ceiling_bpb": quality_ceiling,
            "quality_pass": quality_pass,
            "autonomous_generation": {
                "generated_text": screen["generated_text"],
                "valid_utf8": screen["valid_utf8"],
                "repetition_rate_4gram": screen["repetition_rate_4gram"],
                "collapse_pass": (
                    screen["valid_utf8"]
                    and screen["repetition_rate_4gram"] < 0.50
                ),
            },
            "cpu": {
                "observations": len(observations),
                "median_bytes_per_second": median_bps,
                "optimized_transformer_median_bytes_per_second": qwen_median_bps,
                "transformer_relative_ratio": speed_ratio,
                "speed_pass": speed_pass,
                "median_time_to_first_output_seconds": timing["summary"][
                    "time_to_first_output_seconds"
                ]["median"],
            },
            "memory": {
                "active_parameters": run["active_parameters"],
                "active_fp32_weight_bytes": int(run["active_parameters"]) * 4,
                "median_process_resident_bytes": statistics.median(
                    row["resident_memory_bytes_after"] for row in observations
                ),
            },
            "promotion_pass": quality_pass and speed_pass,
            "runtime_fusion_eligible": (
                name in {"local4_conditional_h2_w10", "local5_conditional_h2_w10"}
                and quality_pass
                and 1.8 <= speed_ratio < 2.0
            ),
        }
    prior_name = "adaptive_top1_attention_local5"
    prior_timing = prior_rescreen["candidates"][prior_name]
    prior_run = next(
        row for row in prior_search["campaign"]["runs"]
        if row["candidate"] == prior_name
    )
    promoted = [name for name, row in rows.items() if row["promotion_pass"]]
    fusion = [name for name, row in rows.items() if row["runtime_fusion_eligible"]]
    if promoted:
        decision = "PROMOTE_CONDITIONAL_BRANCH_CANDIDATE"
    elif fusion:
        decision = "ONE_PROFILED_RUNTIME_FUSION_ATTEMPT_REQUIRED"
    else:
        decision = "FAILED_BRANCH_RETURN_TO_PRIOR_PARETO"
    document = {
        "format": "layercake-conditional-horizon2-branch-decision/1",
        "status": "PASS",
        "decision": decision,
        "decision_rule": {
            "quality_reference_bpb": quality_reference_bpb,
            "quality_delta_maximum": LOCKED_BPB_DELTA,
            "quality_ceiling_bpb": quality_ceiling,
            "optimized_transformer_median_bytes_per_second": qwen_median_bps,
            "minimum_cpu_ratio": 2.0,
            "fusion_floor_ratio": 1.8,
            "fusion_requires_quality_pass": True,
        },
        "inputs": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in (
                search_path,
                rescreen_path,
                prior_search_path,
                prior_rescreen_path,
            )
        },
        "candidates": rows,
        "promoted_candidates": promoted,
        "fusion_eligible_candidates": fusion,
        "fusion_attempts_authorized": 1 if fusion else 0,
        "fusion_attempts_performed": 0,
        "branch_expansion_authorized": False,
        "best_prior_tokenizer_free_pareto": {
            "candidate": prior_name,
            "checkpoint_sha256": prior_timing["checkpoint_sha256"],
            "validation_bpb": prior_timing["validation_bpb"],
            "median_cpu_bytes_per_second": prior_timing["summary"][
                "bytes_per_second_decode"
            ]["median"],
            "active_parameters": prior_run["active_parameters"],
        },
        "next_action": (
            "profile the returned prior candidate's main-head validation loss "
            "and attack the measured quality limiter"
        ),
        "test_accessed": False,
    }
    document["decision_sha256"] = _canonical_sha(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_ledger(root / PHASE / "experiment_ledger.jsonl", [{
        "event": "conditional_horizon2_branch_closed",
        "decision": decision,
        "decision_path": output_path.relative_to(root).as_posix(),
        "decision_sha256": document["decision_sha256"],
        "negative_evidence_preserved": True,
        "candidate_results": {
            name: {
                "validation_bpb": row["validation_bpb"],
                "quality_pass": row["quality_pass"],
                "cpu_ratio": row["cpu"]["transformer_relative_ratio"],
                "speed_pass": row["cpu"]["speed_pass"],
                "promotion_pass": row["promotion_pass"],
            }
            for name, row in rows.items()
        },
        "fusion_attempts_authorized": document["fusion_attempts_authorized"],
        "fusion_attempts_performed": 0,
        "next_experiment": document["next_action"],
    }])
    state_path = root / PHASE / "task_state.json"
    state = read_document(state_path)
    batches = list(state.get("completed_batches", []))
    label = f"conditional horizon2 closed {document['decision_sha256'][:12]}"
    if label not in batches:
        batches.append(label)
    state.update({
        "completed_batches": batches,
        "active_candidate": prior_name,
        "latest_batch": output_path.relative_to(root).as_posix(),
        "latest_batch_sha256": document["decision_sha256"],
        "phase2_status": "OPEN_CONDITIONAL_HORIZON2_BRANCH_FAILED",
        "continuation_command": (
            "python -m layercake.phase2_recertification profile-quality "
            "--input results/moonshot/phase2_recertification/"
            "architecture_search_100m.json "
            "--candidate adaptive_top1_attention_local5 "
            "--output results/moonshot/phase2_recertification/"
            "quality_limiter_profile_prior_local5.json"
        ),
    })
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "decision": decision,
        "decision_sha256": document["decision_sha256"],
        "fusion_attempts_authorized": document["fusion_attempts_authorized"],
        "promoted_candidates": promoted,
    }


def functional_screen(
    root: Path,
    input_path: Path,
    candidate_name: str,
    output_path: Path,
    output_bytes: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 20260807,
) -> dict[str, Any]:
    """Generate autonomously on the frozen 100-prompt product suite."""

    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    source = read_document(input_path)
    run = next(
        row for row in source["campaign"]["runs"]
        if row["candidate"] == candidate_name
    )
    loaded = _load_adaptive_checkpoint(run)
    manifest_path = root / "results/moonshot/phase1/quality_suite_manifest.json"
    manifest = read_document(manifest_path)
    prompts = manifest["prompts"]
    if len(prompts) < 100 or len({row["id"] for row in prompts}) < 100:
        raise ValueError("functional screen requires 100 distinct frozen prompts")
    records = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_bytes = prompt["text"].encode("utf-8")
        if hashlib.sha256(prompt_bytes).hexdigest() != prompt["sha256"]:
            raise ValueError(f"prompt hash mismatch: {prompt['id']}")
        screen = _screen_incremental(
            run,
            prompt_bytes,
            output_bytes,
            loaded=loaded,
            temperature=temperature,
            top_p=top_p,
            sampling_seed=seed + prompt_index,
        )
        generated = bytes.fromhex(screen["trace"]["generated_bytes_hex"])
        records.append({
            "prompt_id": prompt["id"],
            "prompt_sha256": prompt["sha256"],
            "category": prompt["category"],
            "generated_hex": generated.hex(),
            "generated_sha256": hashlib.sha256(generated).hexdigest(),
            "metrics": _output_quality(generated),
            "timing": {
                "bytes_per_second_decode": screen["bytes_per_second_decode"],
                "time_to_first_output_seconds": screen[
                    "time_to_first_output_seconds"
                ],
                "total_latency_seconds": screen["total_latency_seconds"],
            },
            "trace_external_path_counters": screen["trace"][
                "external_path_counters"
            ],
        })
    metric_names = tuple(records[0]["metrics"])
    aggregates = {
        name: statistics.mean(
            float(record["metrics"][name]) for record in records
        )
        for name in metric_names
    }
    qwen = read_document(
        root / "results/moonshot/phase1/functional_quality.json"
    )["systems"]["qwen25-05b-cpu"]["aggregates"]
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
        "product_functional_noninferiority_pass": (
            aggregates["repetition_rate"] <= qwen["repetition_rate"] + 0.02
            and aggregates["word_diversity"] >= qwen["word_diversity"] - 0.02
            and aggregates["invalid_output"] <= qwen["invalid_output"]
        ),
    }
    exact_command = " ".join([
        sys.executable,
        "-m",
        "layercake.phase2_recertification",
        "functional-screen",
        "--input",
        input_path.relative_to(root).as_posix(),
        "--candidate",
        candidate_name,
        "--output",
        output_path.relative_to(root).as_posix(),
        "--output-bytes",
        str(output_bytes),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--seed",
        str(seed),
    ])
    document = {
        "format": "layercake-phase2-functional-screen/1",
        "status": "PASS",
        "input_path": input_path.relative_to(root).as_posix(),
        "input_sha256": sha256_file(input_path),
        "candidate": candidate_name,
        "checkpoint_sha256": loaded[1]["checkpoint"]["sha256"],
        "prompt_manifest": manifest_path.relative_to(root).as_posix(),
        "prompt_manifest_sha256": sha256_file(manifest_path),
        "distinct_prompts": len(prompts),
        "output_bytes_per_prompt": output_bytes,
        "decoding": {
            "mode": "sampled" if temperature > 0.0 else "greedy",
            "temperature": temperature,
            "top_p": top_p,
            "base_seed": seed,
            "source": "checkpoint neural byte logits",
        },
        "test_accessed": False,
        "exact_command": exact_command,
        "records": records,
        "aggregates": aggregates,
        "qwen_product_reference_aggregates": qwen,
        "comparison": comparison,
    }
    document["screen_sha256"] = _canonical_sha(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_ledger(root / PHASE / "experiment_ledger.jsonl", [{
        "event": "functional_screen_completed",
        "candidate": candidate_name,
        "checkpoint_hash": loaded[1]["checkpoint"]["sha256"],
        "screen": output_path.relative_to(root).as_posix(),
        "screen_sha256": document["screen_sha256"],
        "distinct_prompts": len(prompts),
        "aggregates": aggregates,
        "comparison": comparison,
        "next_experiment": (
            "reject product-quality promotion on functional noninferiority "
            "failure; otherwise continue Phase 2 integrated verification"
        ),
    }])
    return {
        "status": "PASS",
        "candidate": candidate_name,
        "screen_sha256": document["screen_sha256"],
        "aggregates": aggregates,
        "comparison": comparison,
    }


def audit_functional_semantics(
    root: Path,
    screen_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Audit frozen outputs for topic, length, and structural adherence."""

    screen = read_document(screen_path)
    manifest_path = root / "results/moonshot/phase1/quality_suite_manifest.json"
    manifest = read_document(manifest_path)
    prompts = {row["id"]: row for row in manifest["prompts"]}
    layercake_outputs = {
        row["prompt_id"]: bytes.fromhex(row["generated_hex"])
        for row in screen["records"]
    }
    phase1_quality_path = root / "results/moonshot/phase1/functional_quality.json"
    phase1_quality = read_document(phase1_quality_path)
    qwen_quality = phase1_quality["systems"]["qwen25-05b-cpu"]
    raw_qwen_path = (
        root
        / "results/moonshot/phase1/raw_runs/headline_qwen_product.json"
    )
    raw_qwen = read_document(raw_qwen_path)
    raw_outputs = {
        row["output"]["sha256"]: bytes.fromhex(row["output"]["hex"])
        for row in raw_qwen["records"]
    }
    qwen_outputs = {
        row["prompt_id"]: raw_outputs[row["output_sha256"]]
        for row in qwen_quality["records"]
    }
    expected_ids = set(prompts)
    if set(layercake_outputs) != expected_ids:
        raise ValueError("LayerCake screen does not cover the frozen prompt set")
    if set(qwen_outputs) != expected_ids:
        raise ValueError("Qwen evidence does not cover the frozen prompt set")

    systems = {}
    aggregate_names = (
        "minimum_80_words_pass",
        "topic_phrase_present",
        "topic_token_recall",
        "category_structure_pass",
        "core_adherence_pass",
        "word_count",
    )
    for system_id, outputs in (
        ("layercake", layercake_outputs),
        ("qwen25-05b-cpu", qwen_outputs),
    ):
        records = []
        for prompt_id in sorted(expected_ids):
            prompt = prompts[prompt_id]
            metrics = _semantic_adherence_metrics(
                prompt["text"],
                prompt["category"],
                outputs[prompt_id],
            )
            records.append({
                "prompt_id": prompt_id,
                "prompt_sha256": prompt["sha256"],
                "output_sha256": hashlib.sha256(
                    outputs[prompt_id]
                ).hexdigest(),
                "category": prompt["category"],
                "metrics": metrics,
            })
        aggregates = {
            name: statistics.mean(
                float(row["metrics"][name]) for row in records
            )
            for name in aggregate_names
        }
        systems[system_id] = {
            "records": records,
            "aggregates": aggregates,
        }
    layercake = systems["layercake"]["aggregates"]
    qwen = systems["qwen25-05b-cpu"]["aggregates"]
    comparison = {
        f"{name}_delta_layercake_minus_qwen": layercake[name] - qwen[name]
        for name in aggregate_names
    }
    comparison["product_semantic_noninferiority_pass"] = all((
        layercake["minimum_80_words_pass"]
        >= qwen["minimum_80_words_pass"] - 0.02,
        layercake["topic_phrase_present"]
        >= qwen["topic_phrase_present"] - 0.02,
        layercake["topic_token_recall"]
        >= qwen["topic_token_recall"] - 0.02,
        layercake["category_structure_pass"]
        >= qwen["category_structure_pass"] - 0.02,
        layercake["core_adherence_pass"]
        >= qwen["core_adherence_pass"] - 0.02,
    ))
    document = {
        "format": "layercake-phase2-functional-semantic-audit/1",
        "status": "PASS",
        "screen_path": screen_path.relative_to(root).as_posix(),
        "screen_sha256": sha256_file(screen_path),
        "candidate": screen["candidate"],
        "checkpoint_sha256": screen["checkpoint_sha256"],
        "prompt_manifest": manifest_path.relative_to(root).as_posix(),
        "prompt_manifest_sha256": sha256_file(manifest_path),
        "qwen_functional_quality_sha256": sha256_file(
            phase1_quality_path
        ),
        "qwen_raw_evidence_sha256": sha256_file(raw_qwen_path),
        "distinct_prompts": len(expected_ids),
        "systems": systems,
        "comparison": comparison,
        "scope": {
            "kind": "deterministic_adherence_proxy",
            "covers": [
                "minimum requested length",
                "explicit topic phrase and topic-token recall",
                "category-specific surface structure",
            ],
            "does_not_replace": [
                "human semantic judgment",
                "factuality evaluation",
                "reasoning correctness evaluation",
            ],
            "promotion_rule": (
                "a pass is necessary but not sufficient; raw-output "
                "adversarial review remains required"
            ),
        },
    }
    document["audit_sha256"] = _canonical_sha(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_ledger(root / PHASE / "experiment_ledger.jsonl", [{
        "event": "functional_semantic_audit_completed",
        "candidate": screen["candidate"],
        "checkpoint_hash": screen["checkpoint_sha256"],
        "audit": output_path.relative_to(root).as_posix(),
        "audit_sha256": document["audit_sha256"],
        "aggregates": {
            system_id: value["aggregates"]
            for system_id, value in systems.items()
        },
        "comparison": comparison,
        "next_experiment": (
            "reject metric-only promotion when semantic adherence fails; "
            "repair the measured prompt-conditioning and response-horizon "
            "training defects without changing the CPU inference graph"
        ),
    }])
    return {
        "status": "PASS",
        "candidate": screen["candidate"],
        "audit_sha256": document["audit_sha256"],
        "aggregates": {
            system_id: value["aggregates"]
            for system_id, value in systems.items()
        },
        "comparison": comparison,
    }


def _instruction_examples(
    rows: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
    sequence_bytes: int,
    device: torch.device,
    prompt_indexes: Sequence[int] | None = None,
    append_newline: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prompt_indexes is not None and len(prompt_indexes) != len(indexes):
        raise ValueError("prompt_indexes must align with response indexes")
    sequences = []
    masks = []
    for offset, index in enumerate(indexes):
        row = rows[index]
        prompt_row = (
            row
            if prompt_indexes is None
            else rows[prompt_indexes[offset]]
        )
        prefix = (
            str(prompt_row["prompt"]) + ("\n" if append_newline else "")
        ).encode("utf-8")
        response = str(row["response"]).encode("utf-8")
        raw = (prefix + response)[: sequence_bytes + 1]
        valid_targets = max(0, len(raw) - 1)
        if len(raw) < sequence_bytes + 1:
            raw = raw + b" " * (sequence_bytes + 1 - len(raw))
        sequences.append(list(raw))
        target_indexes = torch.arange(sequence_bytes)
        masks.append(
            (target_indexes >= max(0, len(prefix) - 1))
            & (target_indexes < valid_targets)
        )
    return (
        torch.tensor(sequences, dtype=torch.long, device=device),
        torch.stack(masks).to(device=device),
    )


def _negative_prompt_indexes(
    rows: Sequence[Mapping[str, Any]],
) -> list[int]:
    by_task: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_task.setdefault(str(row.get("task", "unknown")), []).append(index)
    result = list(range(len(rows)))
    for indexes in by_task.values():
        if len(indexes) < 2:
            continue
        for offset, index in enumerate(indexes):
            result[index] = indexes[(offset + 1) % len(indexes)]
    return result


def _instruction_focus_masks(
    rows: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
    sequence_bytes: int,
    device: torch.device,
    *,
    prompt_indexes: Sequence[int] | None = None,
    append_newline: bool = True,
) -> torch.Tensor:
    masks = []
    for offset, index in enumerate(indexes):
        row = rows[index]
        prompt_row = (
            row
            if prompt_indexes is None
            else rows[prompt_indexes[offset]]
        )
        prefix_bytes = len((
            str(prompt_row["prompt"])
            + ("\n" if append_newline else "")
        ).encode("utf-8"))
        response = str(row["response"]).encode("utf-8")
        if str(row.get("task")) == "long_context_recall":
            match = re.search(
                r"[A-Z]+RECALL[0-9]+", str(row["prompt"])
            )
            focus = (
                match.group(0).encode("utf-8")
                if match is not None
                else b""
            )
        else:
            focus = str(row.get("topic", "")).encode("utf-8")
        mask = torch.zeros(sequence_bytes, dtype=torch.bool)
        if focus:
            response_lower = response.lower()
            focus_lower = focus.lower()
            start = 0
            while True:
                found = response_lower.find(focus_lower, start)
                if found < 0:
                    break
                target_start = prefix_bytes - 1 + found
                target_end = min(
                    sequence_bytes, target_start + len(focus)
                )
                if target_start < sequence_bytes:
                    mask[max(0, target_start):target_end] = True
                start = found + len(focus)
        masks.append(mask)
    return torch.stack(masks).to(device=device)


def _instruction_sequence_coverage(
    rows: Sequence[Mapping[str, Any]],
    sequence_bytes: int,
    *,
    append_newline: bool = True,
) -> dict[str, Any]:
    covered = []
    full = 0
    zero = 0
    for row in rows:
        prefix_bytes = len((
            str(row["prompt"]) + ("\n" if append_newline else "")
        ).encode("utf-8"))
        response_bytes = len(str(row["response"]).encode("utf-8"))
        target_bytes = max(
            0, min(response_bytes, sequence_bytes + 1 - prefix_bytes)
        )
        covered.append(target_bytes)
        zero += int(target_bytes == 0)
        full += int(target_bytes == response_bytes)
    return {
        "examples": len(rows),
        "sequence_bytes": sequence_bytes,
        "zero_response_target_examples": zero,
        "fully_covered_response_examples": full,
        "mean_response_target_bytes": statistics.mean(covered),
        "minimum_response_target_bytes": min(covered),
        "maximum_response_target_bytes": max(covered),
        "total_response_target_bytes_per_epoch": sum(covered),
    }


@torch.inference_mode()
def _instruction_conditioning_profile(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    sequence_bytes: int,
    device: torch.device,
    append_newline: bool = True,
) -> dict[str, float]:
    model.eval()
    negative_indexes = _negative_prompt_indexes(rows)
    correct_sum = 0.0
    correct_count = 0
    wrong_sum = 0.0
    wrong_count = 0
    correct_lower = 0
    compared = 0
    focused_correct_sum = 0.0
    focused_wrong_sum = 0.0
    focused_correct_count = 0
    focused_wrong_count = 0
    focused_correct_lower = 0
    focused_compared = 0
    for start in range(0, len(rows), batch_size):
        indexes = list(range(start, min(start + batch_size, len(rows))))
        prompts = [negative_indexes[index] for index in indexes]
        correct_batch, correct_mask = _instruction_examples(
            rows,
            indexes,
            sequence_bytes,
            device,
            append_newline=append_newline,
        )
        wrong_batch, wrong_mask = _instruction_examples(
            rows,
            indexes,
            sequence_bytes,
            device,
            prompt_indexes=prompts,
            append_newline=append_newline,
        )
        correct_focus = _instruction_focus_masks(
            rows,
            indexes,
            sequence_bytes,
            device,
            append_newline=append_newline,
        )
        wrong_focus = _instruction_focus_masks(
            rows,
            indexes,
            sequence_bytes,
            device,
            prompt_indexes=prompts,
            append_newline=append_newline,
        )
        correct_boundary = correct_mask.long().argmax(dim=1)
        wrong_boundary = wrong_mask.long().argmax(dim=1)
        use_prompt_memory = _has_prompt_conditioning(model)
        correct_logits, _ = _forward(
            model,
            correct_batch[:, :-1],
            prompt_boundary_indexes=(
                correct_boundary if use_prompt_memory else None
            ),
        )
        wrong_logits, _ = _forward(
            model,
            wrong_batch[:, :-1],
            prompt_boundary_indexes=(
                wrong_boundary if use_prompt_memory else None
            ),
        )
        correct_losses = torch.nn.functional.cross_entropy(
            correct_logits.flatten(0, 1).float(),
            correct_batch[:, 1:].flatten(),
            reduction="none",
        ).reshape_as(correct_mask)
        wrong_losses = torch.nn.functional.cross_entropy(
            wrong_logits.flatten(0, 1).float(),
            wrong_batch[:, 1:].flatten(),
            reduction="none",
        ).reshape_as(wrong_mask)
        correct_sum += float(correct_losses[correct_mask].sum())
        correct_count += int(correct_mask.sum())
        wrong_sum += float(wrong_losses[wrong_mask].sum())
        wrong_count += int(wrong_mask.sum())
        focused_correct_sum += float(
            correct_losses[correct_focus].sum()
        )
        focused_correct_count += int(correct_focus.sum())
        focused_wrong_sum += float(
            wrong_losses[wrong_focus].sum()
        )
        focused_wrong_count += int(wrong_focus.sum())
        correct_per_example = (
            (correct_losses * correct_mask).sum(dim=1)
            / correct_mask.sum(dim=1).clamp_min(1)
        )
        wrong_per_example = (
            (wrong_losses * wrong_mask).sum(dim=1)
            / wrong_mask.sum(dim=1).clamp_min(1)
        )
        valid = correct_mask.any(dim=1) & wrong_mask.any(dim=1)
        correct_lower += int(
            (correct_per_example[valid] < wrong_per_example[valid]).sum()
        )
        compared += int(valid.sum())
        correct_focus_per_example = (
            (correct_losses * correct_focus).sum(dim=1)
            / correct_focus.sum(dim=1).clamp_min(1)
        )
        wrong_focus_per_example = (
            (wrong_losses * wrong_focus).sum(dim=1)
            / wrong_focus.sum(dim=1).clamp_min(1)
        )
        focus_valid = correct_focus.any(dim=1) & wrong_focus.any(dim=1)
        focused_correct_lower += int(
            (
                correct_focus_per_example[focus_valid]
                < wrong_focus_per_example[focus_valid]
            ).sum()
        )
        focused_compared += int(focus_valid.sum())
    model.train()
    correct_bpb = correct_sum / max(1, correct_count) / math.log(2)
    wrong_bpb = wrong_sum / max(1, wrong_count) / math.log(2)
    focused_correct_bpb = (
        focused_correct_sum
        / max(1, focused_correct_count)
        / math.log(2)
    )
    focused_wrong_bpb = (
        focused_wrong_sum
        / max(1, focused_wrong_count)
        / math.log(2)
    )
    return {
        "correct_prompt_response_bpb": correct_bpb,
        "mismatched_prompt_response_bpb": wrong_bpb,
        "mismatched_minus_correct_bpb": wrong_bpb - correct_bpb,
        "correct_prompt_lower_loss_rate": correct_lower / max(1, compared),
        "compared_examples": float(compared),
        "response_target_bytes_correct": float(correct_count),
        "response_target_bytes_mismatched": float(wrong_count),
        "focused_correct_prompt_response_bpb": focused_correct_bpb,
        "focused_mismatched_prompt_response_bpb": focused_wrong_bpb,
        "focused_mismatched_minus_correct_bpb": (
            focused_wrong_bpb - focused_correct_bpb
        ),
        "focused_correct_prompt_lower_loss_rate": (
            focused_correct_lower / max(1, focused_compared)
        ),
        "focused_compared_examples": float(focused_compared),
        "focused_target_bytes_correct": float(
            focused_correct_count
        ),
        "focused_target_bytes_mismatched": float(
            focused_wrong_count
        ),
    }


def profile_instruction_conditioning(
    root: Path,
    input_path: Path,
    candidate_name: str,
    corpus_path: Path,
    output_path: Path,
    sequence_bytes: int,
    prompt_separator: str = "newline",
) -> dict[str, Any]:
    source = read_document(input_path)
    run = next(
        row for row in source["campaign"]["runs"]
        if row["candidate"] == candidate_name
    )
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation_rows = [
        row for row in rows if row["split"] == "instruction_validation"
    ]
    model, metadata = _load_adaptive_checkpoint(run)
    if prompt_separator not in {"newline", "none"}:
        raise ValueError("prompt_separator must be newline or none")
    append_newline = prompt_separator == "newline"
    profile = _instruction_conditioning_profile(
        model,
        validation_rows,
        batch_size=2,
        sequence_bytes=sequence_bytes,
        device=torch.device("cpu"),
        append_newline=append_newline,
    )
    coverage = _instruction_sequence_coverage(
        validation_rows,
        sequence_bytes,
        append_newline=append_newline,
    )
    document = {
        "format": "layercake-phase2-instruction-conditioning-profile/1",
        "status": "PASS",
        "input_path": input_path.relative_to(root).as_posix(),
        "input_sha256": sha256_file(input_path),
        "candidate": candidate_name,
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "corpus_path": corpus_path.relative_to(root).as_posix(),
        "corpus_sha256": sha256_file(corpus_path),
        "split": "instruction_validation",
        "sequence_coverage": coverage,
        "prompt_separator": prompt_separator,
        "profile": profile,
        "interpretation": {
            "desired_direction": (
                "mismatched-minus-correct BPB must be positive and "
                "correct-prompt lower-loss rate must exceed chance"
            ),
            "free_generation_required": True,
        },
    }
    document["profile_sha256"] = _canonical_sha(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_ledger(root / PHASE / "experiment_ledger.jsonl", [{
        "event": "instruction_conditioning_profile_completed",
        "candidate": candidate_name,
        "checkpoint_hash": metadata["checkpoint"]["sha256"],
        "profile_path": output_path.relative_to(root).as_posix(),
        "profile_sha256": document["profile_sha256"],
        "sequence_coverage": coverage,
        "profile": profile,
    }])
    return {
        "status": "PASS",
        "candidate": candidate_name,
        "profile_sha256": document["profile_sha256"],
        "sequence_coverage": coverage,
        "profile": profile,
    }


@torch.inference_mode()
def _instruction_validation_loss(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    sequence_bytes: int,
    device: torch.device,
    append_newline: bool = True,
) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    for start in range(0, len(rows), batch_size):
        indexes = list(range(start, min(start + batch_size, len(rows))))
        batch, mask = _instruction_examples(
            rows,
            indexes,
            sequence_bytes,
            device,
            append_newline=append_newline,
        )
        logits, _ = _forward(
            model,
            batch[:, :-1],
            prompt_boundary_indexes=(
                mask.long().argmax(dim=1)
                if _has_prompt_conditioning(model)
                else None
            ),
        )
        losses = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1).float(),
            batch[:, 1:].flatten(),
            reduction="none",
        ).reshape_as(mask)
        loss_sum += float(losses[mask].sum())
        count += int(mask.sum())
    model.train()
    return loss_sum / max(1, count) / math.log(2)


def instruction_finetune(
    root: Path,
    input_path: Path,
    candidate_name: str,
    config_path: Path,
    output_path: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    """Response-only byte fine-tuning with WikiText replay."""

    source = read_document(input_path)
    source_run = next(
        row for row in source["campaign"]["runs"]
        if row["candidate"] == candidate_name
    )
    config = read_document(config_path)
    corpus_path = root / config["instruction_corpus"]
    manifest_path = corpus_path.with_suffix(".manifest.json")
    manifest = read_document(manifest_path)
    if manifest.get("status") != "PASS" or manifest.get(
        "exact_phase1_prompt_overlap"
    ) != 0:
        raise ValueError("instruction corpus is not verified disjoint")
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [
        row for row in rows if row["split"] == "instruction_validation"
    ]
    source_model, source_metadata = _load_adaptive_checkpoint(source_run)
    architecture_overrides = dict(config.get("model_overrides", {}))
    weight_transfer = {
        "architecture_overrides": architecture_overrides,
        "missing_parameters": [],
        "unexpected_parameters": [],
    }
    if architecture_overrides:
        adapted_candidate = copy.deepcopy(source_metadata["candidate"])
        adapted_candidate["model"].update(architecture_overrides)
        model = _build_model(adapted_candidate)
        incompatibility = model.load_state_dict(
            source_model.state_dict(), strict=False
        )
        missing = list(incompatibility.missing_keys)
        unexpected = list(incompatibility.unexpected_keys)
        allowed_missing_prefixes = (
            "prompt_memory_adapter.",
            "prompt_cross_",
            "prompt_pointer_",
        )
        if unexpected or any(
            not name.startswith(allowed_missing_prefixes)
            for name in missing
        ):
            raise ValueError(
                "architecture weight transfer has unsupported key changes: "
                f"missing={missing}, unexpected={unexpected}"
            )
        source_metadata = copy.deepcopy(source_metadata)
        source_metadata["candidate"] = adapted_candidate
        weight_transfer.update({
            "missing_parameters": missing,
            "unexpected_parameters": unexpected,
        })
    else:
        model = source_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    training = config["training"]
    steps = int(training["steps"])
    batch_size = int(training["batch_size"])
    sequence_bytes = int(training["sequence_bytes"])
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    trainable_scope = str(
        training.get("trainable_parameters", "all")
    )
    if trainable_scope in {
        "prompt_memory_adapter",
        "prompt_cross_attention",
        "prompt_pointer",
        "prompt_conditioning_adapters",
    }:
        if not _has_prompt_conditioning(model):
            raise ValueError(
                "prompt-conditioning trainable scope requires an adapter"
            )
        prefixes = {
            "prompt_memory_adapter": ("prompt_memory_adapter.",),
            "prompt_cross_attention": ("prompt_cross_",),
            "prompt_pointer": ("prompt_pointer_",),
            "prompt_conditioning_adapters": (
                "prompt_memory_adapter.",
                "prompt_cross_",
                "prompt_pointer_",
            ),
        }[trainable_scope]
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith(prefixes)
            )
    elif trainable_scope != "all":
        raise ValueError(
            f"unsupported trainable parameter scope: {trainable_scope}"
        )
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("instruction fine-tune has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    wiki_weight = float(training.get("wiki_weight", 1.0))
    if wiki_weight > 0.0:
        wiki = ByteCorpus(source["campaign"]["data"]["train"]["path"])
        wiki_batches = wiki.batches(
            batch_size=int(training["wiki_batch_size"]),
            sequence_bytes=sequence_bytes,
            steps=steps,
            seed=seed,
            device=device,
        )
    else:
        wiki_batches = (None for _ in range(steps))
    curves = []
    peak_memory = 0
    started = time.perf_counter()
    permutation = list(range(len(train_rows)))
    random.Random(seed).shuffle(permutation)
    negative_prompt_indexes = _negative_prompt_indexes(train_rows)
    contrastive_weight = float(
        training.get("prompt_contrastive_weight", 0.0)
    )
    contrastive_margin = float(
        training.get("prompt_contrastive_margin_nats", 0.10)
    )
    focus_weight = float(
        training.get("prompt_focus_weight", 0.0)
    )
    focus_contrastive_weight = float(
        training.get("prompt_focus_contrastive_weight", 0.0)
    )
    focus_contrastive_margin = float(
        training.get(
            "prompt_focus_contrastive_margin_nats", 0.20
        )
    )
    pointer_gate_supervision_weight = float(
        training.get(
            "pointer_gate_supervision_weight", 0.0
        )
    )
    scheduled_sampling_probability = float(
        training.get("scheduled_sampling_probability", 0.0)
    )
    if not 0.0 <= scheduled_sampling_probability <= 1.0:
        raise ValueError(
            "scheduled_sampling_probability must be in [0, 1]"
        )
    wiki_scheduled_sampling_probability = float(
        training.get(
            "wiki_scheduled_sampling_probability", 0.0
        )
    )
    if not 0.0 <= wiki_scheduled_sampling_probability <= 1.0:
        raise ValueError(
            "wiki_scheduled_sampling_probability must be in [0, 1]"
        )
    prompt_separator = str(
        training.get("prompt_separator", "newline")
    )
    if prompt_separator not in {"newline", "none"}:
        raise ValueError("prompt_separator must be newline or none")
    append_newline = prompt_separator == "newline"
    use_prompt_memory = _has_prompt_conditioning(model)
    for step, wiki_row in enumerate(wiki_batches, start=1):
        start = ((step - 1) * batch_size) % len(permutation)
        indexes = [
            permutation[(start + offset) % len(permutation)]
            for offset in range(batch_size)
        ]
        instruction_row, instruction_mask = _instruction_examples(
            train_rows,
            indexes,
            sequence_bytes,
            device,
            append_newline=append_newline,
        )
        training_instruction_row = instruction_row
        corrupted_input_count = 0
        eligible_input_count = 0
        if scheduled_sampling_probability > 0.0:
            boundary_indexes = instruction_mask.long().argmax(dim=1)
            with torch.inference_mode():
                teacher_logits, _ = _forward(
                    model,
                    instruction_row[:, :-1],
                    prompt_boundary_indexes=(
                        boundary_indexes
                        if use_prompt_memory
                        else None
                    ),
                )
            predicted_inputs = torch.cat(
                [
                    instruction_row[:, :1],
                    teacher_logits[:, :-1].argmax(dim=-1),
                ],
                dim=1,
            )
            input_response_mask = torch.cat(
                [
                    torch.zeros_like(instruction_mask[:, :1]),
                    instruction_mask[:, :-1],
                ],
                dim=1,
            )
            sampled_mask = (
                torch.rand(
                    input_response_mask.shape,
                    device=device,
                )
                < scheduled_sampling_probability
            ) & input_response_mask
            training_instruction_row = instruction_row.clone()
            training_inputs = training_instruction_row[:, :-1]
            training_inputs[sampled_mask] = predicted_inputs[
                sampled_mask
            ]
            corrupted_input_count = int(sampled_mask.sum())
            eligible_input_count = int(input_response_mask.sum())
        optimizer.zero_grad(set_to_none=True)
        instruction_logits, instruction_metadata = _forward(
            model,
            training_instruction_row[:, :-1],
            prompt_boundary_indexes=(
                instruction_mask.long().argmax(dim=1)
                if use_prompt_memory
                else None
            ),
        )
        instruction_losses = torch.nn.functional.cross_entropy(
            instruction_logits.flatten(0, 1),
            instruction_row[:, 1:].flatten(),
            reduction="none",
        ).reshape_as(instruction_mask)
        instruction_loss = instruction_losses[instruction_mask].mean()
        instruction_focus_mask = _instruction_focus_masks(
            train_rows,
            indexes,
            sequence_bytes,
            device,
            append_newline=append_newline,
        )
        instruction_focus_loss = (
            instruction_losses[instruction_focus_mask].mean()
            if bool(instruction_focus_mask.any())
            else instruction_loss.new_zeros(())
        )
        pointer_gate_supervision_loss = instruction_loss.new_zeros(())
        pointer_gate_values = instruction_metadata.get(
            "prompt_pointer_gate_values"
        )
        if (
            pointer_gate_supervision_weight > 0.0
            and pointer_gate_values is not None
        ):
            gates = pointer_gate_values.squeeze(-1).clamp(
                1e-6, 1.0 - 1e-6
            )
            positive = instruction_focus_mask
            negative = instruction_mask & ~positive
            terms = []
            if bool(positive.any()):
                terms.append(-gates[positive].log().mean())
            if bool(negative.any()):
                terms.append(
                    -(1.0 - gates[negative]).log().mean()
                )
            pointer_gate_supervision_loss = torch.stack(
                terms
            ).mean()
        prompt_contrastive_loss = instruction_loss.new_zeros(())
        prompt_focus_contrastive_loss = instruction_loss.new_zeros(())
        contrastive_metadata = None
        if (
            contrastive_weight > 0.0
            or focus_contrastive_weight > 0.0
        ):
            negative_indexes = [
                negative_prompt_indexes[index] for index in indexes
            ]
            wrong_row, wrong_mask = _instruction_examples(
                train_rows,
                indexes,
                sequence_bytes,
                device,
                prompt_indexes=negative_indexes,
                append_newline=append_newline,
            )
            wrong_logits, contrastive_metadata = _forward(
                model,
                wrong_row[:, :-1],
                prompt_boundary_indexes=(
                    wrong_mask.long().argmax(dim=1)
                    if use_prompt_memory
                    else None
                ),
            )
            wrong_losses = torch.nn.functional.cross_entropy(
                wrong_logits.flatten(0, 1),
                wrong_row[:, 1:].flatten(),
                reduction="none",
            ).reshape_as(wrong_mask)
            wrong_focus_mask = _instruction_focus_masks(
                train_rows,
                indexes,
                sequence_bytes,
                device,
                prompt_indexes=negative_indexes,
                append_newline=append_newline,
            )
            correct_per_example = (
                (instruction_losses * instruction_mask).sum(dim=1)
                / instruction_mask.sum(dim=1).clamp_min(1)
            )
            wrong_per_example = (
                (wrong_losses * wrong_mask).sum(dim=1)
                / wrong_mask.sum(dim=1).clamp_min(1)
            )
            valid = instruction_mask.any(dim=1) & wrong_mask.any(dim=1)
            if bool(valid.any()):
                prompt_contrastive_loss = torch.relu(
                    contrastive_margin
                    + correct_per_example[valid]
                    - wrong_per_example[valid]
                ).mean()
            focus_valid = (
                instruction_focus_mask.any(dim=1)
                & wrong_focus_mask.any(dim=1)
            )
            if bool(focus_valid.any()):
                correct_focus_per_example = (
                    (
                        instruction_losses
                        * instruction_focus_mask
                    ).sum(dim=1)
                    / instruction_focus_mask.sum(dim=1).clamp_min(1)
                )
                wrong_focus_per_example = (
                    (wrong_losses * wrong_focus_mask).sum(dim=1)
                    / wrong_focus_mask.sum(dim=1).clamp_min(1)
                )
                prompt_focus_contrastive_loss = torch.relu(
                    focus_contrastive_margin
                    + correct_focus_per_example[focus_valid]
                    - wrong_focus_per_example[focus_valid]
                ).mean()
        wiki_metadata = None
        wiki_loss = instruction_loss.new_zeros(())
        wiki_corrupted_input_count = 0
        wiki_eligible_input_count = 0
        if wiki_weight > 0.0:
            training_wiki_row = wiki_row
            if wiki_scheduled_sampling_probability > 0.0:
                with torch.inference_mode():
                    teacher_wiki_logits, _ = _forward(
                        model, wiki_row[:, :-1]
                    )
                predicted_wiki_inputs = torch.cat(
                    [
                        wiki_row[:, :1],
                        teacher_wiki_logits[:, :-1].argmax(dim=-1),
                    ],
                    dim=1,
                )
                wiki_eligible_mask = torch.ones_like(
                    wiki_row[:, :-1], dtype=torch.bool
                )
                wiki_eligible_mask[:, 0] = False
                wiki_sampled_mask = (
                    torch.rand(
                        wiki_eligible_mask.shape,
                        device=device,
                    )
                    < wiki_scheduled_sampling_probability
                ) & wiki_eligible_mask
                training_wiki_row = wiki_row.clone()
                wiki_inputs = training_wiki_row[:, :-1]
                wiki_inputs[wiki_sampled_mask] = (
                    predicted_wiki_inputs[wiki_sampled_mask]
                )
                wiki_corrupted_input_count = int(
                    wiki_sampled_mask.sum()
                )
                wiki_eligible_input_count = int(
                    wiki_eligible_mask.sum()
                )
            wiki_logits, wiki_metadata = _forward(
                model, training_wiki_row[:, :-1]
            )
            wiki_loss = torch.nn.functional.cross_entropy(
                wiki_logits.flatten(0, 1),
                wiki_row[:, 1:].flatten(),
            )
        loss = (
            float(training.get("instruction_weight", 1.0))
            * instruction_loss
            + focus_weight * instruction_focus_loss
            + (
                pointer_gate_supervision_weight
                * pointer_gate_supervision_loss
            )
            + wiki_weight * wiki_loss
            + contrastive_weight * prompt_contrastive_loss
            + (
                focus_contrastive_weight
                * prompt_focus_contrastive_loss
            )
        )
        for metadata in (
            instruction_metadata,
            wiki_metadata,
            contrastive_metadata,
        ):
            if metadata is None:
                continue
            routing = metadata.get("routing")
            if routing is not None:
                loss = loss + float(
                    training.get("routing_balance_weight", 0.02)
                ) * routing["balance_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training.get("gradient_clip", 1.0))
        )
        optimizer.step()
        if device.type == "cuda":
            peak_memory = max(
                peak_memory, int(torch.cuda.max_memory_allocated())
            )
        if step in {1, steps // 2, steps}:
            validation_bpb = _evaluate(
                model,
                ByteCorpus(source["campaign"]["data"]["validation"]["path"]),
                batch_size=8,
                sequence_bytes=256,
                batches=32,
                device=device,
            )["bits_per_byte"]
            instruction_bpb = _instruction_validation_loss(
                model,
                validation_rows,
                batch_size=8,
                sequence_bytes=sequence_bytes,
                device=device,
                append_newline=append_newline,
            )
            conditioning = _instruction_conditioning_profile(
                model,
                validation_rows,
                batch_size=batch_size,
                sequence_bytes=sequence_bytes,
                device=device,
                append_newline=append_newline,
            )
            curves.append({
                "step": step,
                "instruction_loss": float(instruction_loss.detach()),
                "prompt_contrastive_loss": float(
                    prompt_contrastive_loss.detach()
                ),
                "instruction_focus_loss": float(
                    instruction_focus_loss.detach()
                ),
                "pointer_gate_supervision_loss": float(
                    pointer_gate_supervision_loss.detach()
                ),
                "prompt_focus_contrastive_loss": float(
                    prompt_focus_contrastive_loss.detach()
                ),
                "scheduled_sampling_probability": (
                    scheduled_sampling_probability
                ),
                "scheduled_sampling_corrupted_fraction": (
                    corrupted_input_count / max(1, eligible_input_count)
                ),
                "wiki_scheduled_sampling_probability": (
                    wiki_scheduled_sampling_probability
                ),
                "wiki_scheduled_sampling_corrupted_fraction": (
                    wiki_corrupted_input_count
                    / max(1, wiki_eligible_input_count)
                ),
                "prompt_pointer_mean_gate": (
                    float(
                        instruction_metadata[
                            "prompt_pointer_mean_gate"
                        ].detach()
                    )
                    if instruction_metadata.get(
                        "prompt_pointer_mean_gate"
                    ) is not None
                    else None
                ),
                "wiki_loss": float(wiki_loss.detach()),
                "validation_bpb": validation_bpb,
                "heldout_instruction_response_bpb": instruction_bpb,
                "heldout_prompt_conditioning": conditioning,
                "wall_seconds": time.perf_counter() - started,
            })
    validation_metrics = _evaluate(
        model,
        ByteCorpus(source["campaign"]["data"]["validation"]["path"]),
        batch_size=8,
        sequence_bytes=256,
        batches=32,
        device=device,
    )
    selection_metrics = _evaluate(
        model,
        ByteCorpus(
            source["campaign"]["data"]["architecture_selection"]["path"]
        ),
        batch_size=8,
        sequence_bytes=256,
        batches=32,
        device=device,
    )
    artifact_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_path / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(checkpoint_path),
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    candidate = dict(source_metadata["candidate"])
    candidate["name"] = str(config["candidate_name"])
    frozen_parameter_verification = {
        "scope": trainable_scope,
        "checked_source_parameters": 0,
        "bitwise_equal_source_parameters": 0,
        "excluded_trainable_source_parameters": 0,
        "all_source_parameters_bitwise_equal": None,
    }
    if trainable_scope != "all":
        source_state = load_file(
            str(source_metadata["checkpoint"]["path"]),
            device="cpu",
        )
        current_state = model.state_dict()
        trainable_names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        frozen_source_names = [
            name for name in source_state
            if name not in trainable_names
        ]
        equal = 0
        for name in frozen_source_names:
            source_value = source_state[name]
            if torch.equal(
                source_value,
                current_state[name].detach().cpu(),
            ):
                equal += 1
        frozen_parameter_verification.update({
            "checked_source_parameters": len(frozen_source_names),
            "bitwise_equal_source_parameters": equal,
            "excluded_trainable_source_parameters": (
                len(source_state) - len(frozen_source_names)
            ),
            "all_source_parameters_bitwise_equal": (
                equal == len(frozen_source_names)
            ),
        })
        if equal != len(frozen_source_names):
            raise RuntimeError(
                "frozen base parameters changed during adapter training"
            )
    metadata = {
        "format": "layercake-byte-instruction-finetune/1",
        "status": "PASS",
        "candidate": candidate,
        "seed": seed,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
        },
        "source_checkpoint_sha256": source_metadata["checkpoint"]["sha256"],
        "weight_transfer": weight_transfer,
        "frozen_parameter_verification": frozen_parameter_verification,
        "instruction_corpus_sha256": sha256_file(corpus_path),
        "instruction_manifest_sha256": sha256_file(manifest_path),
        "training": {
            **training,
            "instruction_sequence_coverage": {
                "train": _instruction_sequence_coverage(
                    train_rows,
                    sequence_bytes,
                    append_newline=append_newline,
                ),
                "validation": _instruction_sequence_coverage(
                    validation_rows,
                    sequence_bytes,
                    append_newline=append_newline,
                ),
            },
            "curves": curves,
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": peak_memory,
        },
        "quality": {
            "selection": selection_metrics,
            "validation": validation_metrics,
            "test_accessed": False,
        },
    }
    (artifact_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    routed = model.routed
    inactive_expert_parameters = sum(
        parameter.numel()
        for expert in routed.experts
        for parameter in expert.parameters()
    )
    active_parameters = (
        sum(parameter.numel() for parameter in model.parameters())
        - inactive_expert_parameters
        + max(
            sum(parameter.numel() for parameter in expert.parameters())
            for expert in routed.experts
        )
    )
    run = {
        "candidate": candidate["name"],
        "kind": "adaptive_two_four",
        "status": "PASS",
        "seed": seed,
        "artifact": str(artifact_path),
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "active_parameters": active_parameters,
        "selection_bpb": selection_metrics["bits_per_byte"],
        "validation_bpb": validation_metrics["bits_per_byte"],
        "raw_bytes_seen": source_run["raw_bytes_seen"],
        "test_accessed": False,
        "instruction_finetune": {
            "steps": steps,
            "instruction_examples_seen": steps * batch_size,
            "wiki_replay_bytes": (
                steps
                * int(training["wiki_batch_size"])
                * sequence_bytes
            ),
        },
    }
    screen = _screen_incremental(
        run,
        Path(
            source["campaign"]["data"]["architecture_selection"]["path"]
        ).read_bytes()[:128],
        128,
        loaded=(model.cpu().eval(), metadata),
    )
    exact_command = " ".join([
        sys.executable,
        "-m",
        "layercake.phase2_recertification",
        "instruction-finetune",
        "--input",
        input_path.relative_to(root).as_posix(),
        "--candidate",
        candidate_name,
        "--config",
        config_path.relative_to(root).as_posix(),
        "--output",
        output_path.relative_to(root).as_posix(),
        "--artifacts",
        artifact_path.relative_to(root).as_posix(),
    ])
    document = {
        "format": "layercake-phase2-instruction-finetune-search/1",
        "status": "PASS",
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "exact_command": exact_command,
        "selection_split_only": True,
        "test_accessed": False,
        "campaign": {
            "format": "layercake-byte-instruction-finetune-campaign/1",
            "status": "PASS",
            "runs": [run],
            "data": {
                **source["campaign"]["data"],
                "instruction": {
                    "path": str(corpus_path),
                    "sha256": sha256_file(corpus_path),
                },
            },
            "final_test_accessed": False,
        },
        "cpu_screens": {candidate["name"]: screen},
        "selected_candidate": candidate["name"],
        "selection_policy": "single bounded instruction-behavior repair of passed BPB/speed architecture",
        "quality_promising_speed_blocked": [],
    }
    document["batch_sha256"] = _canonical_sha(document)
    document["output_path"] = output_path.relative_to(root).as_posix()
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_ledger(root / PHASE / "experiment_ledger.jsonl", [{
        "event": "instruction_finetune_completed",
        "candidate": candidate["name"],
        "source_checkpoint_hash": source_metadata["checkpoint"]["sha256"],
        "checkpoint_hash": checkpoint_sha,
        "quality": {
            "selection_bpb": selection_metrics["bits_per_byte"],
            "validation_bpb": validation_metrics["bits_per_byte"],
        },
        "speed": {
            "cpu_bytes_per_second": screen["bytes_per_second_decode"]
        },
        "free_generation": {
            "repetition_rate_4gram": screen["repetition_rate_4gram"],
            "generated_text": screen["generated_text"],
        },
        "exact_command": exact_command,
        "next_experiment": "100-prompt functional screen and repeated CPU timing",
    }])
    return {
        "status": "PASS",
        "candidate": candidate["name"],
        "batch_sha256": document["batch_sha256"],
        "validation_bpb": validation_metrics["bits_per_byte"],
        "cpu_bytes_per_second": screen["bytes_per_second_decode"],
    }


def rescreen(
    root: Path,
    input_path: Path,
    output_path: Path,
    repetitions: int,
    output_bytes: int = 128,
) -> dict[str, Any]:
    """Repeat provisional CPU screens without retraining or timing trace creation."""

    if repetitions < 20:
        raise ValueError("timing rescreen requires at least 20 observations per checkpoint")
    if output_bytes < 1:
        raise ValueError("output_bytes must be positive")
    source = read_document(input_path)
    prompt_path = root / source["campaign"]["data"]["architecture_selection"]["path"]
    prompt = prompt_path.read_bytes()[:128]
    qwen_historical_bps = 508.71529942368556
    exact_command = " ".join([
        sys.executable, "-m", "layercake.phase2_recertification", "rescreen",
        "--input", input_path.relative_to(root).as_posix(),
        "--output", output_path.relative_to(root).as_posix(),
        "--repetitions", str(repetitions),
        "--output-bytes", str(output_bytes),
    ])
    candidates: dict[str, Any] = {}
    ledger = []
    for run in source["campaign"]["runs"]:
        if run["kind"] != "adaptive_two_four" or run.get("status") != "PASS":
            continue
        loaded = _load_adaptive_checkpoint(run)
        rows = []
        trace_verification = None
        for observation_index in range(repetitions):
            screen = _screen_incremental(
                run, prompt, output_bytes, loaded=loaded
            )
            if observation_index == 0:
                trace_verification = screen["trace"]
            rows.append({
                "observation_index": observation_index,
                "bytes_per_second_decode": screen["bytes_per_second_decode"],
                "bytes_per_second_total": screen["bytes_per_second_total"],
                "prefill_seconds": screen["prefill_seconds"],
                "time_to_first_output_seconds": screen["time_to_first_output_seconds"],
                "decode_seconds": screen["decode_seconds"],
                "total_latency_seconds": screen["total_latency_seconds"],
                "resident_memory_bytes_before": screen["resident_memory_bytes_before"],
                "resident_memory_bytes_after": screen["resident_memory_bytes_after"],
                "generated_sha256": screen["trace"]["generated_sha256"],
            })
        summary = _timing_summary(rows)
        median_bps = summary["bytes_per_second_decode"]["median"]
        candidates[run["candidate"]] = {
            "checkpoint_sha256": loaded[1]["checkpoint"]["sha256"],
            "selection_bpb": run["selection_bpb"],
            "validation_bpb": run["validation_bpb"],
            "raw_observations": rows,
            "summary": summary,
            "historical_qwen_ratio_diagnostic": median_bps / qwen_historical_bps,
            "trace_verification": trace_verification,
        }
        ledger.append({
            "event": "timing_rescreen_completed",
            "batch": output_path.relative_to(root).as_posix(),
            "candidate": run["candidate"],
            "architecture_hash": _canonical_sha(loaded[1]["candidate"]),
            "checkpoint_hash": loaded[1]["checkpoint"]["sha256"],
            "data_hashes": {
                name: value["sha256"]
                for name, value in source["campaign"]["data"].items()
            },
            "hypothesis": "replace non-promotable one-shot architecture timing with repeated warm observations",
            "exact_command": exact_command,
            "quality": {
                "selection_bpb": run["selection_bpb"],
                "validation_bpb": run["validation_bpb"],
            },
            "speed": {
                "cpu_bytes_per_second_median": median_bps,
                "historical_qwen_ratio_diagnostic": median_bps / qwen_historical_bps,
                "observations": repetitions,
            },
            "memory": {
                "resident_before_median": statistics.median(
                    row["resident_memory_bytes_before"] for row in rows
                ),
                "resident_after_median": statistics.median(
                    row["resident_memory_bytes_after"] for row in rows
                ),
            },
            "failure_classification": (
                "SEARCH_CANDIDATE_NOT_YET_PROMOTED"
                if median_bps >= 2.0 * qwen_historical_bps
                else "QUALITY_PROMISING_SPEED_BLOCKED"
            ),
            "next_experiment": "promote Pareto quality-speed candidates to 100m bytes",
            "tokenizer_free": True,
            "free_neural_generation_verified": True,
            "cpu_speed_above_required_gate": median_bps >= 2.0 * qwen_historical_bps,
        })
    eligible = [
        name for name, evidence in candidates.items()
        if evidence["historical_qwen_ratio_diagnostic"] >= 2.0
    ]
    selected = min(
        eligible,
        key=lambda name: (
            candidates[name]["selection_bpb"],
            -candidates[name]["summary"]["bytes_per_second_decode"]["median"],
        ),
    ) if eligible else None
    document = {
        "format": "layercake-phase2-recertification-timing-rescreen/1",
        "status": "PASS",
        "input_path": input_path.relative_to(root).as_posix(),
        "input_sha256": sha256_file(input_path),
        "exact_command": exact_command,
        "repetitions_per_checkpoint": repetitions,
        "output_bytes_per_observation": output_bytes,
        "timing_scope": "provisional architecture screen; not headline certification",
        "supersedes_one_shot_timing_for_selection": True,
        "candidates": candidates,
        "eligible_candidates": eligible,
        "selected_candidate": selected,
    }
    document["batch_sha256"] = _canonical_sha(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _append_ledger(root / PHASE / "experiment_ledger.jsonl", ledger)
    state_path = root / PHASE / "task_state.json"
    state = read_document(state_path)
    batches = list(state.get("completed_batches", []))
    label = (
        f"20x timing rescreen {output_bytes}b "
        f"{document['batch_sha256'][:12]}"
    )
    if label not in batches:
        batches.append(label)
    state.update({
        "completed_batches": batches,
        "active_candidate": selected,
        "latest_batch": output_path.relative_to(root).as_posix(),
        "latest_batch_sha256": document["batch_sha256"],
    })
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "selected_candidate": selected,
        "batch_sha256": document["batch_sha256"],
        "output_path": output_path.relative_to(root).as_posix(),
        "candidates": len(candidates),
    }


def _append_ledger(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _update_task_state(root: Path, result: Mapping[str, Any]) -> None:
    path = root / PHASE / "task_state.json"
    state = read_document(path)
    batches = list(state.get("completed_batches", []))
    label = f"10m architecture search {result['batch_sha256'][:12]}"
    if label not in batches:
        batches.append(label)
    state.update({
        "updated_at_utc": "2026-07-23T00:00:00Z",
        "current_stage": "PHASE2_ARCHITECTURE_RESEARCH",
        "phase2_status": "OPEN_RECERTIFICATION_SEARCH_10M_COMPLETE",
        "completed_batches": batches,
        "active_candidate": result.get("selected_candidate"),
        "latest_batch": result["output_path"],
        "latest_batch_sha256": result["batch_sha256"],
        "continuation_command": (
            "python -m layercake.phase2_recertification search "
            "--config configs/moonshot/phase2_recertification/architecture_search_100m.json "
            "--output results/moonshot/phase2_recertification/architecture_search_100m.json "
            "--artifacts artifacts/moonshot/phase2_recertification/search_100m"
        ),
    })
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def search(root: Path, config_path: Path, output_path: Path, artifact_root: Path) -> dict:
    config = read_document(config_path)
    exact_command = " ".join([
        sys.executable, "-m", "layercake.phase2_recertification", "search",
        "--config", config_path.relative_to(root).as_posix(),
        "--output", output_path.relative_to(root).as_posix(),
        "--artifacts", artifact_root.relative_to(root).as_posix(),
    ])
    campaign = run_variable_patch_campaign(
        config_path,
        output_path,
        artifact_root=artifact_root,
    )
    prompt = (root / config["data"]["architecture_selection"]).read_bytes()[:128]
    screens = {}
    ledger = []
    qwen_historical_bps = 508.71529942368556
    for run in campaign["runs"]:
        screen = _screen_incremental(run, prompt, 128)
        screens[run["candidate"]] = screen
        speed = screen.get("bytes_per_second_decode")
        quality = {
            "selection_bpb": run.get("selection_bpb"),
            "validation_bpb": run.get("validation_bpb"),
        }
        checkpoint = None
        architecture_hash = None
        hypothesis = next(
            candidate.get("hypothesis")
            for candidate in config["candidates"]
            if candidate["name"] == run["candidate"]
        )
        if run.get("status") == "PASS":
            metadata = read_document(Path(str(run["artifact"])) / "metadata.json")
            checkpoint = metadata["checkpoint"]["sha256"]
            architecture_hash = _canonical_sha(metadata["candidate"])
        cpu_ratio = speed / qwen_historical_bps if isinstance(speed, (int, float)) else None
        ledger.append({
            "event": "experiment_completed",
            "batch": output_path.relative_to(root).as_posix(),
            "candidate": run["candidate"],
            "architecture_hash": architecture_hash,
            "checkpoint_hash": checkpoint,
            "data_hashes": {
                name: value["sha256"] for name, value in campaign["data"].items()
            },
            "hypothesis": hypothesis,
            "exact_command": exact_command,
            "quality": quality,
            "speed": {
                "cpu_bytes_per_second": speed,
                "historical_qwen_ratio_diagnostic": cpu_ratio,
            },
            "memory": {
                "resident_before": screen.get("resident_memory_bytes_before"),
                "resident_after": screen.get("resident_memory_bytes_after"),
            },
            "failure_classification": (
                run.get("failure", "QUALITY_PROMISING_SPEED_BLOCKED")
                if run.get("status") != "PASS" or not cpu_ratio or cpu_ratio < 2.0
                else "SEARCH_CANDIDATE_NOT_YET_PROMOTED"
            ),
            "next_experiment": "promote Pareto quality-speed candidates to 100m bytes",
            "tokenizer_free": True,
            "free_neural_generation_verified": screen.get("status") == "PASS",
            "cpu_speed_above_required_gate": bool(cpu_ratio and cpu_ratio >= 2.0),
        })
    eligible = [
        run for run in campaign["runs"]
        if run.get("status") == "PASS"
        and screens[run["candidate"]].get("status") == "PASS"
        and screens[run["candidate"]]["bytes_per_second_decode"]
        >= 2.0 * qwen_historical_bps
    ]
    selected = min(
        eligible,
        key=lambda run: (
            run["selection_bpb"],
            -screens[run["candidate"]]["bytes_per_second_decode"],
        ),
    ) if eligible else None
    document = {
        "format": "layercake-phase2-recertification-architecture-search/1",
        "status": campaign["status"],
        "selection_split_only": True,
        "test_accessed": False,
        "exact_command": exact_command,
        "config_path": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "campaign": campaign,
        "cpu_screens": screens,
        "selected_candidate": selected["candidate"] if selected else None,
        "selection_policy": "lowest selection BPB among incrementally executable candidates above the provisional 2x optimized-CPU sentinel",
        "quality_promising_speed_blocked": [
            run["candidate"] for run in campaign["runs"]
            if run.get("status") == "PASS"
            and screens[run["candidate"]].get("status") == "PASS"
            and screens[run["candidate"]]["bytes_per_second_decode"]
            < 2.0 * qwen_historical_bps
        ],
    }
    document["batch_sha256"] = _canonical_sha(document)
    document["output_path"] = output_path.relative_to(root).as_posix()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _append_ledger(root / PHASE / "experiment_ledger.jsonl", ledger)
    _update_task_state(root, document)
    return {
        "status": document["status"],
        "selected_candidate": document["selected_candidate"],
        "batch_sha256": document["batch_sha256"],
        "output_path": document["output_path"],
        "runs": len(campaign["runs"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m layercake.phase2_recertification")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("search")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--artifacts", type=Path, required=True)
    command = sub.add_parser("rescreen")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--repetitions", type=int, default=20)
    command.add_argument("--output-bytes", type=int, default=128)
    command = sub.add_parser("decide-conditional-branch")
    command.add_argument("--search", type=Path, required=True)
    command.add_argument("--rescreen", type=Path, required=True)
    command.add_argument("--prior-search", type=Path, required=True)
    command.add_argument("--prior-rescreen", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("profile-quality")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--candidate", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("functional-screen")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--candidate", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--output-bytes", type=int, default=128)
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--top-p", type=float, default=1.0)
    command.add_argument("--seed", type=int, default=20260807)
    command = sub.add_parser("audit-functional-semantics")
    command.add_argument("--screen", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("profile-instruction-conditioning")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--candidate", required=True)
    command.add_argument("--corpus", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--sequence-bytes", type=int, default=256)
    command.add_argument(
        "--prompt-separator",
        choices=("newline", "none"),
        default="newline",
    )
    command = sub.add_parser("instruction-finetune")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--candidate", required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--artifacts", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.command == "search":
        config = args.config if args.config.is_absolute() else root / args.config
        artifacts = args.artifacts if args.artifacts.is_absolute() else root / args.artifacts
        result = search(root, config.resolve(), output.resolve(), artifacts.resolve())
    elif args.command == "rescreen":
        input_path = args.input if args.input.is_absolute() else root / args.input
        result = rescreen(
            root,
            input_path.resolve(),
            output.resolve(),
            args.repetitions,
            args.output_bytes,
        )
    elif args.command == "decide-conditional-branch":
        def resolve(path: Path) -> Path:
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        result = decide_conditional_branch(
            root,
            resolve(args.search),
            resolve(args.rescreen),
            resolve(args.prior_search),
            resolve(args.prior_rescreen),
            output.resolve(),
        )
    elif args.command == "profile-quality":
        input_path = args.input if args.input.is_absolute() else root / args.input
        result = profile_quality_limiter(
            root,
            input_path.resolve(),
            args.candidate,
            output.resolve(),
        )
    elif args.command == "functional-screen":
        input_path = args.input if args.input.is_absolute() else root / args.input
        result = functional_screen(
            root,
            input_path.resolve(),
            args.candidate,
            output.resolve(),
            args.output_bytes,
            args.temperature,
            args.top_p,
            args.seed,
        )
    elif args.command == "audit-functional-semantics":
        screen_path = (
            args.screen
            if args.screen.is_absolute()
            else root / args.screen
        )
        result = audit_functional_semantics(
            root,
            screen_path.resolve(),
            output.resolve(),
        )
    elif args.command == "profile-instruction-conditioning":
        input_path = (
            args.input if args.input.is_absolute() else root / args.input
        )
        corpus_path = (
            args.corpus if args.corpus.is_absolute() else root / args.corpus
        )
        result = profile_instruction_conditioning(
            root,
            input_path.resolve(),
            args.candidate,
            corpus_path.resolve(),
            output.resolve(),
            args.sequence_bytes,
            args.prompt_separator,
        )
    elif args.command == "instruction-finetune":
        input_path = args.input if args.input.is_absolute() else root / args.input
        config_path = (
            args.config if args.config.is_absolute() else root / args.config
        )
        artifact_path = (
            args.artifacts
            if args.artifacts.is_absolute()
            else root / args.artifacts
        )
        result = instruction_finetune(
            root,
            input_path.resolve(),
            args.candidate,
            config_path.resolve(),
            output.resolve(),
            artifact_path.resolve(),
        )
    else:  # pragma: no cover
        raise RuntimeError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
