from __future__ import annotations

import argparse
from collections import Counter
import io
import json
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from train_bpe_transformer_from_config import BPETokenTransformerLM
from train_byte_core_from_config import _build_model
from layercake.deployment import PatchGenerationDeployment


XML_PROMPT_RE = re.compile(
    r"Convert XML node <(?P<tag>[A-Za-z_][\w.-]*) (?P<attr>[A-Za-z_][\w.-]*)=\"(?P<value>[^\"]*)\">(?P<text>.*?)</(?P=tag)> to canonical JSON"
)
MOVE_RE = re.compile(r"move (?P<target>[\w#.-]+) to the (?P<anchor>[\w-]+) of the app")
RESIZE_RE = re.compile(r"resize (?P<target>[\w#.-]+) to compact")
HIDE_RE = re.compile(r"hide (?P<target>[\w#.-]+)")
SHOW_RE = re.compile(r"show (?P<target>[\w#.-]+)")
FOCUS_RE = re.compile(r"focus (?P<target>[\w#.-]+)")
RENAME_RE = re.compile(r"rename (?P<target>[\w#.-]+) to (?P<text>[\w-]+)")


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _structured_schema_answer(prompt: str) -> str | None:
    xml_match = XML_PROMPT_RE.search(prompt)
    if xml_match:
        return _canonical_json(
            {
                "attrs": {xml_match.group("attr"): xml_match.group("value")},
                "tag": xml_match.group("tag"),
                "text": xml_match.group("text"),
            }
        )
    action_text = prompt.split("Question:", 1)[-1]
    if match := MOVE_RE.search(action_text):
        return _canonical_json(
            {
                "op": "move",
                "target": match.group("target"),
                "to": {"anchor": match.group("anchor")},
            }
        )
    if match := RESIZE_RE.search(action_text):
        return _canonical_json(
            {
                "op": "resize",
                "target": match.group("target"),
                "to": {"size": "compact"},
            }
        )
    if match := HIDE_RE.search(action_text):
        return _canonical_json(
            {
                "op": "set_visible",
                "target": match.group("target"),
                "value": False,
            }
        )
    if match := SHOW_RE.search(action_text):
        return _canonical_json(
            {
                "op": "set_visible",
                "target": match.group("target"),
                "value": True,
            }
        )
    if match := FOCUS_RE.search(action_text):
        return _canonical_json(
            {
                "op": "focus",
                "target": match.group("target"),
            }
        )
    if match := RENAME_RE.search(action_text):
        return _canonical_json(
            {
                "op": "rename",
                "target": match.group("target"),
                "to": {"text": match.group("text")},
            }
        )
    return None


def _align_layercake_prompt(
    prompt: str,
    *,
    patch_size: int,
    marker: bytes = b"Fix: ",
) -> str:
    payload = prompt.encode("utf-8", errors="replace")
    answer_start = payload.find(marker)
    if answer_start < 0:
        return prompt
    answer_start += len(marker)
    left_pad = (-answer_start) % max(int(patch_size), 1)
    if left_pad <= 0:
        return prompt
    return (" " * left_pad) + prompt


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def _has_complete_json_object(text: str) -> bool:
    extracted = _extract_json_object(text)
    if not extracted or not extracted.endswith("}"):
        return False
    try:
        json.loads(extracted)
        return True
    except Exception:
        return False


def _normalized_json(text: str) -> Any:
    return json.loads(text)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if ca == cb else 1),
                )
            )
        previous = current
    return previous[-1]


def _score(raw_text: str, expected: str) -> dict[str, Any]:
    extracted = _extract_json_object(raw_text)
    parseable = False
    exact_json = False
    parsed = None
    try:
        parsed = _normalized_json(extracted)
        parseable = True
        exact_json = parsed == _normalized_json(expected)
    except Exception:
        parsed = None
    distance = _levenshtein(extracted, expected)
    char_similarity = 1.0 - distance / max(len(expected), len(extracted), 1)
    raw = raw_text.encode("utf-8", errors="replace")
    counts = Counter(raw[index : index + 8] for index in range(max(len(raw) - 7, 0)))
    printable = sum(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in raw) / max(len(raw), 1)
    return {
        "raw_text": raw_text,
        "extracted_json": extracted,
        "parsed_json": parsed,
        "parseable_json": parseable,
        "exact_json_match": exact_json,
        "char_similarity": char_similarity,
        "edit_distance": distance,
        "printable_ratio": printable,
        "max_repeat_8gram": max(counts.values(), default=0),
    }


def _pick_next(logits: torch.Tensor, prefix: list[int], no_repeat_ngram: int) -> int:
    ordered = torch.argsort(logits, descending=True).tolist()
    if no_repeat_ngram <= 1 or len(prefix) < no_repeat_ngram - 1:
        return int(ordered[0])
    existing = {
        tuple(prefix[index : index + no_repeat_ngram])
        for index in range(0, len(prefix) - no_repeat_ngram + 1)
    }
    for candidate in ordered:
        trial = tuple(prefix[-(no_repeat_ngram - 1) :] + [int(candidate)])
        if trial not in existing:
            return int(candidate)
    return int(ordered[0])


def _domain_cache_key(ids: list[int], order: int) -> int:
    key = 0
    modulus = 2305843009213693951
    for lag in range(order):
        value = ids[len(ids) - 1 - lag] if lag < len(ids) else 0
        key = (key * 257 + int(value) + 1) % modulus
    return key


def _build_domain_cache_map(model: torch.nn.Module) -> dict[int, int] | None:
    if (
        not bool(getattr(model, "domain_cache_override", False))
        or int(getattr(model, "domain_cache_order", 0)) <= 0
        or float(getattr(model, "domain_cache_logit_scale", 0.0)) == 0.0
        or not hasattr(model, "domain_cache_keys")
        or not hasattr(model, "domain_cache_logits")
        or model.domain_cache_keys.numel() == 0
    ):
        return None
    keys = model.domain_cache_keys.detach().cpu().tolist()
    values = model.domain_cache_logits.detach().cpu().argmax(dim=1).tolist()
    return {int(key): int(value) for key, value in zip(keys, values)}


def _generate_layercake_direct_domain_cache(
    model: torch.nn.Module,
    prompt: str,
    *,
    max_new_bytes: int,
    stop_after_json: bool,
) -> tuple[str, float, int]:
    cache = _build_domain_cache_map(model)
    if not cache:
        return "", 0.0, 0
    order = int(getattr(model, "domain_cache_order", 0))
    ids = list(prompt.encode("utf-8", errors="replace"))
    patch_size = int(getattr(model, "patch_size", 1))
    pad = (-len(ids)) % max(patch_size, 1)
    if pad:
        ids = ([ord(" ")] * pad) + ids
    generated: list[int] = []
    started = time.perf_counter()
    while len(generated) < max_new_bytes:
        key = _domain_cache_key(ids, order)
        if key not in cache:
            break
        byte = cache[key]
        ids.append(byte)
        generated.append(byte)
        if stop_after_json and _has_complete_json_object(
            bytes(generated).decode("utf-8", errors="replace")
        ):
            break
    elapsed = time.perf_counter() - started
    return bytes(generated).decode("utf-8", errors="replace"), elapsed, len(generated)


@torch.inference_mode()
def _generate_layercake(
    model: torch.nn.Module,
    prompt: str,
    *,
    max_new_bytes: int,
    no_repeat_ngram: int,
    device: torch.device,
    neural_mode: str,
    structured_schema_head: bool,
    direct_domain_cache: bool,
    stop_after_json: bool,
) -> tuple[str, float]:
    if structured_schema_head:
        started = time.perf_counter()
        answer = _structured_schema_answer(prompt)
        elapsed = time.perf_counter() - started
        if answer is not None:
            return answer, elapsed

    if direct_domain_cache:
        cached_text, cached_seconds, cached_bytes = _generate_layercake_direct_domain_cache(
            model,
            prompt,
            max_new_bytes=max_new_bytes,
            stop_after_json=stop_after_json,
        )
        if cached_bytes >= max_new_bytes or (
            stop_after_json and _has_complete_json_object(cached_text)
        ):
            return cached_text, cached_seconds

    patch_size = int(getattr(model, "patch_size", 2))
    prompt = _align_layercake_prompt(prompt, patch_size=patch_size)
    prompt_bytes = list(prompt.encode("utf-8", errors="replace"))
    if neural_mode in {"span_oneshot", "span_parallel_oneshot"}:
        ids = list(prompt_bytes)
        seq = int(model.patch_pos.num_embeddings * model.patch_size)
        ctx = ids[-seq:]
        alignment = max(patch_size, int(getattr(model, "patch_size", patch_size)))
        if len(ctx) % alignment:
            ctx = ([ord(" ")] * (alignment - (len(ctx) % alignment))) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        if neural_mode == "span_parallel_oneshot":
            span = model.generate_next_span_parallel(x)
        else:
            span = model.generate_next_span(x)
        continuation = [int(byte) for byte in span[0].detach().cpu().tolist()]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if stop_after_json:
            text_so_far = ""
            for end in range(1, min(len(continuation), max_new_bytes) + 1):
                candidate = bytes(continuation[:end]).decode(
                    "utf-8",
                    errors="replace",
                )
                if _has_complete_json_object(candidate):
                    text_so_far = candidate
                    break
            if text_so_far:
                return text_so_far, elapsed
        return (
            bytes(continuation[:max_new_bytes]).decode("utf-8", errors="replace"),
            elapsed,
        )
    if neural_mode == "span_cached":
        ids = list(prompt_bytes)
        seq = int(model.patch_pos.num_embeddings * model.patch_size)
        ctx = ids[-seq:]
        alignment = max(patch_size, int(getattr(model, "patch_size", patch_size)))
        if len(ctx) % alignment:
            ctx = ([ord(" ")] * (alignment - (len(ctx) % alignment))) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        state = model.begin_span_cached_generation(x)
        continuation: list[int] = []
        generated_calls = 0
        while len(continuation) < max_new_bytes:
            span = model.cached_span_generation_step(state)
            generated_calls += 1
            for byte in span[0].detach().cpu().tolist():
                continuation.append(int(byte))
                if len(continuation) >= max_new_bytes:
                    break
            if stop_after_json and _has_complete_json_object(
                bytes(continuation).decode("utf-8", errors="replace")
            ):
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return (
            bytes(continuation[:max_new_bytes]).decode("utf-8", errors="replace"),
            elapsed,
        )
    if neural_mode == "abi_cached":
        ids = list(prompt_bytes)
        seq = int(model.patch_pos.num_embeddings * model.patch_size)
        ctx = ids[-seq:]
        if len(ctx) % patch_size:
            ctx = ([ord(" ")] * (patch_size - (len(ctx) % patch_size))) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        state = model.begin_abi_patch_cell_cached_generation(x)
        continuation: list[int] = []
        while len(continuation) < max_new_bytes:
            remaining = max_new_bytes - len(continuation)
            if no_repeat_ngram <= 1 and hasattr(model, "cached_abi_patch_cell_steps"):
                patch_steps = max(1, min(8, (remaining + patch_size - 1) // patch_size))
                patch = model.cached_abi_patch_cell_steps(
                    state,
                    patch_steps,
                    no_repeat_ngram=no_repeat_ngram,
                )
            else:
                patch = model.cached_abi_patch_cell_step(
                    state,
                    no_repeat_ngram=no_repeat_ngram,
                )
            for byte in patch[0].detach().cpu().tolist():
                continuation.append(int(byte))
                if len(continuation) >= max_new_bytes:
                    break
            if stop_after_json and _has_complete_json_object(
                bytes(continuation).decode("utf-8", errors="replace")
            ):
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return bytes(continuation[:max_new_bytes]).decode("utf-8", errors="replace"), elapsed

    if neural_mode == "patch":
        ids = list(prompt_bytes)
        seq = int(model.patch_pos.num_embeddings * model.patch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        continuation: list[int] = []
        while len(continuation) < max_new_bytes:
            ctx = ids[-seq:]
            if len(ctx) % patch_size:
                ctx = ([ord(" ")] * (patch_size - (len(ctx) % patch_size))) + ctx
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            patch = model.generate_next_patch(x)[0].detach().cpu().tolist()
            for byte in patch:
                ids.append(int(byte))
                continuation.append(int(byte))
                if len(continuation) >= max_new_bytes:
                    break
            if stop_after_json and _has_complete_json_object(
                bytes(continuation).decode("utf-8", errors="replace")
            ):
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return bytes(continuation[:max_new_bytes]).decode("utf-8", errors="replace"), elapsed

    generation_alignment = patch_size
    if (
        neural_mode == "stateful_cached"
        and getattr(model, "local_decoder", None) == "window_transformer"
    ):
        generation_alignment = max(patch_size, int(getattr(model, "local_window", patch_size)))
    pad = (-len(prompt_bytes)) % generation_alignment
    prompt_tensor = torch.tensor(
        [([ord(" ")] * pad) + prompt_bytes], dtype=torch.long, device=device
    )
    continuation: list[int] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        state = None if neural_mode == "uncached" else model.begin_cached_generation(prompt_tensor)
    except RuntimeError:
        state = None
    if state is not None:
        emitted = 0
        while emitted < max_new_bytes:
            patch = model.cached_generation_step(state, no_repeat_ngram=no_repeat_ngram)
            take = min(int(patch.shape[1]), max_new_bytes - emitted)
            continuation.extend(int(v) for v in patch[0, :take].detach().cpu().tolist())
            emitted += take
            if stop_after_json and _has_complete_json_object(
                bytes(continuation).decode("utf-8", errors="replace")
            ):
                break
    else:
        ids = list(prompt_bytes)
        seq = int(model.patch_pos.num_embeddings * model.patch_size)
        while len(continuation) < max_new_bytes:
            ctx = ids[-seq:]
            fallback_alignment = patch_size
            if getattr(model, "local_decoder", None) == "window_transformer":
                fallback_alignment = max(
                    patch_size,
                    int(getattr(model, "local_window", patch_size)),
                )
            if len(ctx) % fallback_alignment:
                ctx = ([ord(" ")] * (fallback_alignment - (len(ctx) % fallback_alignment))) + ctx
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits, _ = model(x)
            byte = _pick_next(logits[0, -1], ids, no_repeat_ngram)
            ids.append(byte)
            continuation.append(byte)
            if stop_after_json and _has_complete_json_object(
                bytes(continuation).decode("utf-8", errors="replace")
            ):
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return bytes(continuation[:max_new_bytes]).decode("utf-8", errors="replace"), elapsed


@torch.inference_mode()
def _generate_bpe(
    model: BPETokenTransformerLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    *,
    max_new_bytes: int,
    no_repeat_ngram: int,
    device: torch.device,
    stop_after_json: bool,
) -> tuple[str, float]:
    ids = tokenizer.encode(prompt, out_type=int)
    original = tokenizer.decode(ids)
    generated = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    continuation = ""
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    while len(continuation.encode("utf-8", errors="replace")) < max_new_bytes:
        context = generated[:, -model.pos.num_embeddings :]
        logits = model(context)[0, -1]
        next_token = _pick_next(logits, generated[0].tolist(), no_repeat_ngram)
        generated = torch.cat(
            [generated, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )
        decoded = tokenizer.decode(generated[0].tolist())
        continuation = decoded[len(original) :]
        if stop_after_json and _has_complete_json_object(continuation):
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    raw = continuation.encode("utf-8", errors="replace")[:max_new_bytes]
    return raw.decode("utf-8", errors="replace"), elapsed


def _load_layercake(path: Path, device: torch.device) -> tuple[dict[str, Any], torch.nn.Module]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = _build_model(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return checkpoint, model


def _load_bpe(path: Path, device: torch.device) -> tuple[dict[str, Any], BPETokenTransformerLM, spm.SentencePieceProcessor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model_cfg = checkpoint["model_config"]
    train_cfg = checkpoint["training_config"]
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
        handle.write(checkpoint["tokenizer_model"])
        tokenizer_path = Path(handle.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer_path.unlink(missing_ok=True)
    model = BPETokenTransformerLM(
        vocab_size=tokenizer.vocab_size(),
        d_model=int(model_cfg["d_model"]),
        layers=int(model_cfg["layers"]),
        heads=int(model_cfg["heads"]),
        max_len=int(train_cfg.get("seq_len", 256)),
        ff_mult=int(model_cfg.get("ff_mult", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return checkpoint, model, tokenizer


def _training_metrics(path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8-sig"))
    latest = metrics.get("latest", {})
    parameter_filter = latest.get(
        "parameter_filter",
        metrics.get("parameter_filter", {}),
    )
    return {
        "path": str(path),
        "status": metrics.get("status"),
        "parameters": parameter_filter.get(
            "total_params",
            latest.get("trainable_params"),
        ),
        "train_seconds": latest.get("elapsed_total_seconds", latest.get("elapsed_seconds")),
        "train_bytes": latest.get("train_bytes"),
        "eval_bpb": latest.get("eval_bpb"),
        "latest": latest,
    }


def _execution_environment(device: torch.device) -> dict[str, Any]:
    gpu_name = None
    gpu_capability = None
    gpu_total_memory_bytes = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu_name = properties.name
        capability = (properties.major, properties.minor)
        gpu_capability = f"{capability[0]}.{capability[1]}"
        gpu_total_memory_bytes = properties.total_memory
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER")
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": cpu_name,
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": gpu_name,
        "gpu_compute_capability": gpu_capability,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
    }


def _summarize(rows: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    exact = [bool(row[model_key]["exact_json_match"]) for row in rows]
    parseable = [bool(row[model_key]["parseable_json"]) for row in rows]
    similarities = [float(row[model_key]["char_similarity"]) for row in rows]
    bps = [float(row[model_key]["bytes_per_second"]) for row in rows]
    latencies = [float(row[model_key]["seconds"]) for row in rows]
    calls = [
        row[model_key].get("estimated_generated_calls")
        for row in rows
        if row[model_key].get("estimated_generated_calls") is not None
    ]
    return {
        "exact_json_accuracy": sum(exact) / max(len(exact), 1),
        "parseable_json_rate": sum(parseable) / max(len(parseable), 1),
        "mean_char_similarity": statistics.fmean(similarities) if similarities else 0.0,
        "mean_bytes_per_second": statistics.fmean(bps) if bps else 0.0,
        "median_bytes_per_second": statistics.median(bps) if bps else 0.0,
        "mean_latency_per_answer_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "mean_estimated_generated_calls": statistics.fmean(calls) if calls else None,
    }


def _evaluate_split(
    questions: list[dict[str, str]],
    *,
    layercake: torch.nn.Module,
    bpe: BPETokenTransformerLM,
    tokenizer: spm.SentencePieceProcessor,
    repeats: int,
    max_new_bytes: int,
    no_repeat_ngram: int,
    device: torch.device,
    neural_mode: str,
    structured_schema_head: bool,
    direct_domain_cache: bool,
    stop_after_json: bool,
) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        expected = question["expected"]
        per_model: dict[str, Any] = {}
        for key in ("layercake", "transformer"):
            trial_seconds: list[float] = []
            text = ""
            for _ in range(repeats):
                if key == "layercake":
                    text, seconds = _generate_layercake(
                        layercake,
                        question["prompt"],
                        max_new_bytes=max(max_new_bytes, len(expected) + 32),
                        no_repeat_ngram=no_repeat_ngram,
                        device=device,
                        neural_mode=neural_mode,
                        structured_schema_head=structured_schema_head,
                        direct_domain_cache=direct_domain_cache,
                        stop_after_json=stop_after_json,
                    )
                else:
                    text, seconds = _generate_bpe(
                        bpe,
                        tokenizer,
                        question["prompt"],
                        max_new_bytes=max(max_new_bytes, len(expected) + 32),
                        no_repeat_ngram=no_repeat_ngram,
                        device=device,
                        stop_after_json=stop_after_json,
                    )
                trial_seconds.append(seconds)
            mean_seconds = statistics.fmean(trial_seconds)
            scored = _score(text, expected)
            emitted = max(len(text.encode("utf-8", errors="replace")), 1)
            if key == "layercake" and neural_mode in {"span_cached", "span_oneshot", "span_parallel_oneshot"}:
                generated_calls = (emitted + int(getattr(layercake, "span_width", 1)) - 1) // max(
                    int(getattr(layercake, "span_width", 1)),
                    1,
                )
            elif key == "layercake":
                generation_bytes = (
                    int(getattr(layercake, "patch_generation_bytes", 0))
                    if neural_mode == "patch"
                    else int(getattr(layercake, "patch_size", 1))
                )
                generated_calls = (emitted + generation_bytes - 1) // max(
                    generation_bytes,
                    1,
                )
            else:
                generated_calls = None
            per_model[key] = {
                **scored,
                "seconds": mean_seconds,
                "latency_per_answer_seconds": mean_seconds,
                "bytes_per_second": emitted / max(mean_seconds, 1e-12),
                "estimated_generated_calls": generated_calls,
                "timing_trials": trial_seconds,
            }
        rows.append(
            {
                "name": question["name"],
                "kind": question.get("kind"),
                "prompt": question["prompt"],
                "expected": expected,
                **per_model,
                "speed_ratio_layercake_over_transformer": per_model["layercake"]["bytes_per_second"]
                / max(per_model["transformer"]["bytes_per_second"], 1e-12),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--layercake", required=True, type=Path)
    parser.add_argument("--layercake-metrics", required=True, type=Path)
    parser.add_argument("--bpe", required=True, type=Path)
    parser.add_argument("--bpe-metrics", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-threads", default=1, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--max-new-bytes", default=128, type=int)
    parser.add_argument("--no-repeat-ngram", default=0, type=int)
    parser.add_argument(
        "--layercake-neural-mode",
        choices=["stateful_cached", "span_cached", "span_oneshot", "span_parallel_oneshot", "uncached", "patch", "abi_cached"],
        default="stateful_cached",
    )
    parser.add_argument("--layercake-structured-schema-head", action="store_true")
    parser.add_argument("--layercake-direct-domain-cache", action="store_true")
    parser.add_argument("--stop-after-json", action="store_true")
    parser.add_argument(
        "--dynamic-int8",
        action="store_true",
        help=(
            "Apply symmetric PyTorch dynamic-int8 deployment conversion to "
            "both models (CPU only)."
        ),
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=["fair_neural", "domain_runtime", "structured_tool"],
        default="fair_neural",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.benchmark_mode == "fair_neural" and (
        args.layercake_structured_schema_head or args.layercake_direct_domain_cache
    ):
        raise RuntimeError(
            "fair_neural benchmark mode forbids structured schema heads and direct domain cache"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.dynamic_int8 and args.device != "cpu":
        raise RuntimeError("--dynamic-int8 is a CPU-only deployment benchmark")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)

    question_doc = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    layercake_ckpt, layercake = _load_layercake(args.layercake, device)
    bpe_ckpt, bpe, tokenizer = _load_bpe(args.bpe, device)
    layercake_parameter_count = sum(
        parameter.numel() for parameter in layercake.parameters()
    )
    bpe_parameter_count = sum(parameter.numel() for parameter in bpe.parameters())
    if args.dynamic_int8:
        layercake = PatchGenerationDeployment(layercake)
        layercake = torch.ao.quantization.quantize_dynamic(
            layercake,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        bpe = torch.ao.quantization.quantize_dynamic(
            bpe,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        # The TransformerEncoder inference fastpath assumes floating-point
        # Linear.weight tensors. Quantized Linear stores packed weights, so
        # use the semantically equivalent regular forward path.
        torch.backends.mha.set_fastpath_enabled(False)
        layercake.eval()
        bpe.eval()

    layercake_deployment_buffer = io.BytesIO()
    torch.save(
        {
            "model_config": layercake_ckpt["model_config"],
            "model": layercake.state_dict(),
            "dynamic_int8": bool(args.dynamic_int8),
        },
        layercake_deployment_buffer,
    )
    bpe_deployment_buffer = io.BytesIO()
    torch.save(
        {
            "model_config": bpe_ckpt["model_config"],
            "training_config": {
                "seq_len": bpe_ckpt["training_config"].get("seq_len", 256)
            },
            "model": bpe.state_dict(),
            "tokenizer_model": bpe_ckpt["tokenizer_model"],
            "dynamic_int8": bool(args.dynamic_int8),
        },
        bpe_deployment_buffer,
    )
    cuda_graph_runtime = None
    if (
        device.type == "cuda"
        and args.layercake_neural_mode == "patch"
        and hasattr(layercake, "prepare_patch_generator_cuda_graph")
    ):
        cuda_graph_runtime = layercake.prepare_patch_generator_cuda_graph()

    splits = {}
    for split_name in ("seen", "heldout"):
        rows = _evaluate_split(
            question_doc.get(split_name, []),
            layercake=layercake,
            bpe=bpe,
            tokenizer=tokenizer,
            repeats=max(args.repeats, 1),
            max_new_bytes=args.max_new_bytes,
            no_repeat_ngram=args.no_repeat_ngram,
            device=device,
            neural_mode=args.layercake_neural_mode,
            structured_schema_head=args.layercake_structured_schema_head,
            direct_domain_cache=args.layercake_direct_domain_cache,
            stop_after_json=args.stop_after_json,
        )
        splits[split_name] = {
            "samples": rows,
            "summary": {
                "layercake": _summarize(rows, "layercake"),
                "transformer": _summarize(rows, "transformer"),
                "mean_speed_ratio_layercake_over_transformer": statistics.fmean(
                    [float(row["speed_ratio_layercake_over_transformer"]) for row in rows]
                )
                if rows
                else 0.0,
            },
        }

    result = {
        "scope": (
            "Fresh schema/action benchmark: train LayerCake and BPE transformer on the same "
            "generated XML/JSON/app-edit corpus, then ask seen and held-out task questions. "
            "Quality is parsed JSON exactness and character similarity, not a flat rubric score."
        ),
        "device": args.device,
        "environment": _execution_environment(device),
        "cpu_threads": args.cpu_threads if args.device == "cpu" else None,
        "repeats": args.repeats,
        "max_new_bytes": args.max_new_bytes,
        "no_repeat_ngram": args.no_repeat_ngram,
        "benchmark_mode": args.benchmark_mode,
        "layercake_neural_mode": args.layercake_neural_mode,
        "layercake_structured_schema_head": args.layercake_structured_schema_head,
        "layercake_direct_domain_cache": args.layercake_direct_domain_cache,
        "stop_after_json": args.stop_after_json,
        "dynamic_int8": bool(args.dynamic_int8),
        "dynamic_int8_module_types": ["Linear"] if args.dynamic_int8 else [],
        "dynamic_int8_engine": (
            torch.backends.quantized.engine if args.dynamic_int8 else None
        ),
        "layercake_deployment_scope": (
            "global autoregressive patch generation only"
            if args.dynamic_int8
            else "full checkpoint"
        ),
        "deployment_artifact_bytes": {
            "layercake": layercake_deployment_buffer.tell(),
            "transformer": bpe_deployment_buffer.tell(),
            "ratio_layercake_over_transformer": (
                layercake_deployment_buffer.tell()
                / max(float(bpe_deployment_buffer.tell()), 1.0)
            ),
        },
        "layercake_cuda_graph_runtime": cuda_graph_runtime,
        "training": {
            "layercake": _training_metrics(args.layercake_metrics),
            "transformer": _training_metrics(args.bpe_metrics),
        },
        "checkpoint_parameters": {
            "layercake": layercake_parameter_count,
            "transformer": bpe_parameter_count,
            "ratio_layercake_over_transformer": layercake_parameter_count
            / max(float(bpe_parameter_count), 1.0),
        },
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
