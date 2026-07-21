from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from train_bpe_transformer_from_config import BPETokenTransformerLM
from train_byte_core_from_config import _build_model


PROMPTS = [
    "Question: Give a concise plan before entering the next room. Answer:",
    "Question: How should I recover after a mistake in combat? Answer:",
    "Question: Explain the safest first step when two threats appear. Answer:",
]


def _repeated_ngram(prefix: list[int], candidate: int, ngram: int) -> bool:
    if ngram <= 1 or len(prefix) < ngram - 1:
        return False
    trial = tuple(prefix[-(ngram - 1) :] + [candidate])
    existing = {
        tuple(prefix[index : index + ngram])
        for index in range(0, len(prefix) - ngram + 1)
    }
    return trial in existing


def _pick_next(logits: torch.Tensor, prefix: list[int], no_repeat_ngram: int) -> int:
    ordered = torch.argsort(logits, descending=True).tolist()
    for candidate in ordered:
        if not _repeated_ngram(prefix, int(candidate), no_repeat_ngram):
            return int(candidate)
    return int(ordered[0])


def _quality_score(text: str) -> dict[str, float]:
    raw = text.encode("utf-8", errors="replace")
    chars = max(len(text), 1)
    alpha_space = sum(ch.isalpha() or ch.isspace() for ch in text) / chars
    printable = sum(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in raw) / max(len(raw), 1)
    words = [word for word in text.lower().replace("\n", " ").split(" ") if word]
    unique_words = set(words)
    unique_chars = {ch.lower() for ch in text if ch.isalpha()}
    max_word_repeat = max((words.count(word) for word in set(words)), default=0)
    distinct_word_ratio = len(unique_words) / max(len(words), 1)
    one_char_word_ratio = (
        sum(1 for word in words if len(word) == 1) / max(len(words), 1)
    )
    if len(raw) >= 8:
        eight_counts = Counter(raw[index : index + 8] for index in range(0, len(raw) - 7))
        max_repeat_8gram = max(eight_counts.values(), default=0)
    else:
        max_repeat_8gram = 0
    repeat_score = 1.0 - min(max(max_word_repeat / 12.0, max_repeat_8gram / 8.0), 1.0)
    diversity_score = min(distinct_word_ratio * 2.0, 1.0)
    shape_score = 1.0 - min(one_char_word_ratio * 2.0, 1.0)
    quality = (
        0.30 * alpha_space
        + 0.30 * printable
        + 0.20 * repeat_score
        + 0.10 * diversity_score
        + 0.10 * shape_score
    )
    return {
        "quality_score": quality,
        "alpha_space_ratio": alpha_space,
        "printable_ratio": printable,
        "max_word_repeat": float(max_word_repeat),
        "max_repeat_8gram": float(max_repeat_8gram),
        "distinct_word_ratio": distinct_word_ratio,
        "one_char_word_ratio": one_char_word_ratio,
        "unique_word_count": float(len(unique_words)),
        "unique_alpha_char_count": float(len(unique_chars)),
    }


def _layercake_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.inference_mode()
def _generate_layercake_global_patch(
    model: torch.nn.Module,
    prompt_bytes: list[int],
    *,
    device: torch.device,
    max_new_bytes: int,
    no_repeat_ngram: int,
) -> tuple[str, float]:
    ids = list(prompt_bytes)
    patch_size = int(getattr(model, "patch_size", 2))
    seq = int(model.patch_pos.num_embeddings * patch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    while len(ids) - len(prompt_bytes) < max_new_bytes:
        ctx = ids[-seq:]
        local_window = int(getattr(model, "local_window", patch_size))
        if len(ctx) % local_window:
            ctx = ([ord(" ")] * (local_window - (len(ctx) % local_window))) + ctx
        if len(ctx) % patch_size:
            ctx = ([ord(" ")] * (patch_size - (len(ctx) % patch_size))) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        out = model(
            x,
            return_aux=True,
            return_patch_prediction=True,
        )
        predictions = out[3]
        for offset_logits in predictions:
            logits = offset_logits[0, -1]
            ids.append(_pick_next(logits, ids, no_repeat_ngram))
            if len(ids) - len(prompt_bytes) >= max_new_bytes:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    text = bytes(ids[len(prompt_bytes) : len(prompt_bytes) + max_new_bytes]).decode(
        "utf-8", errors="replace"
    )
    return text, elapsed


@torch.inference_mode()
def _generate_layercake_patch_prediction_method(
    model: torch.nn.Module,
    prompt_bytes: list[int],
    *,
    device: torch.device,
    max_new_bytes: int,
    no_repeat_ngram: int,
) -> tuple[str, float]:
    ids = list(prompt_bytes)
    patch_size = int(getattr(model, "patch_size", 2))
    seq = int(model.patch_pos.num_embeddings * patch_size)
    ctx = ids[-seq:]
    if len(ctx) % patch_size:
        ctx = ([ord(" ")] * (patch_size - (len(ctx) % patch_size))) + ctx
    x = torch.tensor([ctx], dtype=torch.long, device=device)
    state = model.begin_patch_prediction_cached_generation(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    while len(ids) - len(prompt_bytes) < max_new_bytes:
        generated = model.cached_patch_prediction_step(state)
        for byte in generated[0].detach().cpu().tolist():
            ids.append(int(byte))
            if len(ids) - len(prompt_bytes) >= max_new_bytes:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    text = bytes(ids[len(prompt_bytes) : len(prompt_bytes) + max_new_bytes]).decode(
        "utf-8", errors="replace"
    )
    return text, elapsed


@torch.inference_mode()
def _generate_layercake(
    checkpoint: dict,
    prompt: str,
    *,
    device: torch.device,
    max_new_bytes: int,
    no_repeat_ngram: int,
    generation_mode: str = "auto",
) -> tuple[str, float]:
    model_cfg = checkpoint["model_config"]
    train_cfg = checkpoint.get("train_config", checkpoint.get("training_config", {}))
    model = _build_model(model_cfg, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if (
        device.type == "cuda"
        and bool(getattr(model, "patch_prediction", False))
        and hasattr(model, "prepare_patch_generator_cuda_graph")
    ):
        model.prepare_patch_generator_cuda_graph(batch_size=1)
    patch_size = int(getattr(model, "patch_size", 2))
    prompt_bytes = list(prompt.encode("utf-8", errors="replace"))
    if generation_mode == "patch_prediction" or (
        bool(getattr(model, "patch_prediction", False))
        and getattr(model, "patch_prediction_mode", "")
        in {
            "autoregressive",
            "parallel_causal",
            "radix_attention",
            "radix_causal",
            "radix_cumsum",
            "radix_cumsum_hash",
            "radix_conv",
            "radix_depthwise_hash",
            "radix_dilated_conv",
            "radix_hash",
            "radix_grouped_recurrent_hash",
            "radix_ngram",
            "radix_prefix",
            "radix_low_rank_recurrent_hash",
            "radix_recurrent",
            "radix_recurrent_conditional_hash",
            "radix_recurrent_hash",
            "radix_recurrent_ngram",
            "radix_rotary_hash",
            "radix_scan",
            "radix_scan_hash",
            "radix_simple_recurrent_hash",
            "radix_window",
        }
    ):
        return _generate_layercake_patch_prediction_method(
            model,
            prompt_bytes,
            device=device,
            max_new_bytes=max_new_bytes,
            no_repeat_ngram=no_repeat_ngram,
        )
    if getattr(model, "local_decoder", "") in {"parallel_patch", "abi_patch_cell"}:
        return _generate_layercake_fast_patch_method(
            model,
            prompt_bytes,
            device=device,
            max_new_bytes=max_new_bytes,
            no_repeat_ngram=no_repeat_ngram,
        )
    if bool(getattr(model, "patch_prediction", False)) and getattr(model, "patch_prediction_mode", "") == "factorized":
        return _generate_layercake_global_patch(
            model,
            prompt_bytes,
            device=device,
            max_new_bytes=max_new_bytes,
            no_repeat_ngram=no_repeat_ngram,
        )
    pad = (-len(prompt_bytes)) % patch_size
    padded_prompt = ([ord(" ")] * pad) + prompt_bytes
    prompt_tensor = torch.tensor([padded_prompt], dtype=torch.long, device=device)
    try:
        state = model.begin_cached_generation(prompt_tensor)
    except RuntimeError:
        state = None
    continuation = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    if state is not None:
        emitted = 0
        while emitted < max_new_bytes:
            patch = model.cached_generation_step(state, no_repeat_ngram=no_repeat_ngram)
            take = min(patch.shape[1], max_new_bytes - emitted)
            continuation.extend(int(v) for v in patch[0, :take].detach().cpu().tolist())
            emitted += take
    else:
        seq = int(train_cfg.get("seq_len", model.patch_pos.num_embeddings * model.patch_size))
        local_window = int(getattr(model, "local_window", 16))
        ids = list(prompt_bytes)
        for _ in range(max_new_bytes):
            ctx = ids[-seq:]
            if len(ctx) < local_window:
                ctx = ([ord(" ")] * (local_window - len(ctx))) + ctx
            if len(ctx) % local_window:
                ctx = ([ord(" ")] * (local_window - (len(ctx) % local_window))) + ctx
            if len(ctx) % patch_size:
                ctx = ([ord(" ")] * (patch_size - (len(ctx) % patch_size))) + ctx
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = _layercake_logits(model, x)
            ids.append(_pick_next(logits[0, -1], ids, no_repeat_ngram))
        continuation = ids[len(prompt_bytes) :]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    text = bytes(continuation).decode("utf-8", errors="replace")
    return text, elapsed


@torch.inference_mode()
def _generate_layercake_fast_patch_method(
    model: torch.nn.Module,
    prompt_bytes: list[int],
    *,
    device: torch.device,
    max_new_bytes: int,
    no_repeat_ngram: int,
) -> tuple[str, float]:
    ids = list(prompt_bytes)
    patch_size = int(getattr(model, "patch_size", 2))
    seq = int(model.patch_pos.num_embeddings * patch_size)
    if getattr(model, "local_decoder", "") == "abi_patch_cell" and hasattr(
        model, "begin_abi_patch_cell_cached_generation"
    ):
        ctx = ids[-seq:]
        if len(ctx) % patch_size:
            ctx = ([ord(" ")] * (patch_size - (len(ctx) % patch_size))) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        try:
            state = model.begin_abi_patch_cell_cached_generation(x)
        except RuntimeError:
            state = None
        if state is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            generated_patches = []
            emitted = 0
            while len(ids) - len(prompt_bytes) < max_new_bytes:
                remaining = max_new_bytes - emitted
                if (
                    no_repeat_ngram <= 1
                    and hasattr(model, "cached_abi_patch_cell_steps")
                ):
                    patch_steps = max(1, min(8, (remaining + patch_size - 1) // patch_size))
                    patch = model.cached_abi_patch_cell_steps(
                        state,
                        patch_steps,
                        no_repeat_ngram=no_repeat_ngram,
                    )
                else:
                    patch = model.cached_abi_patch_cell_step(
                        state, no_repeat_ngram=no_repeat_ngram
                    )
                take = min(patch.shape[1], max_new_bytes - emitted)
                generated_patches.append(patch[:, :take].detach())
                emitted += take
                ids.extend([0] * take)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            continuation = (
                torch.cat(generated_patches, dim=1)[0, :max_new_bytes]
                .cpu()
                .tolist()
            )
            text = bytes(int(byte) for byte in continuation).decode(
                "utf-8", errors="replace"
            )
            return text, elapsed
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    while len(ids) - len(prompt_bytes) < max_new_bytes:
        ctx = ids[-seq:]
        if len(ctx) % patch_size:
            ctx = ([ord(" ")] * (patch_size - (len(ctx) % patch_size))) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model.generate_next_patch(x, return_logits=True)[0]
        for offset in range(logits.shape[0]):
            byte = _pick_next(logits[offset], ids, no_repeat_ngram=no_repeat_ngram)
            ids.append(int(byte))
            if len(ids) - len(prompt_bytes) >= max_new_bytes:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    text = bytes(ids[len(prompt_bytes) : len(prompt_bytes) + max_new_bytes]).decode(
        "utf-8", errors="replace"
    )
    return text, elapsed


@torch.inference_mode()
def _generate_layercake_domain_cache_batch(
    model: torch.nn.Module,
    prompts: list[str],
    *,
    device: torch.device,
    max_new_bytes: int,
    no_repeat_ngram: int,
) -> tuple[list[str], float] | None:
    if (
        not bool(getattr(model, "domain_cache_override", False))
        or int(getattr(model, "domain_cache_order", 0)) <= 0
        or not hasattr(model, "_last_domain_cache_prior")
        or getattr(model, "domain_cache_keys", torch.empty(0)).numel() == 0
    ):
        return None
    order = int(getattr(model, "domain_cache_order", 0))
    prompt_bytes = [
        list(prompt.encode("utf-8", errors="replace"))
        for prompt in prompts
    ]
    keys_cpu = model.domain_cache_keys.detach().cpu().tolist()
    top_cpu = torch.topk(
        model.domain_cache_logits.detach().cpu(),
        k=min(16, model.domain_cache_logits.shape[1]),
        dim=1,
    ).indices.tolist()
    cache_map = {int(key): [int(item) for item in row] for key, row in zip(keys_cpu, top_cpu)}
    modulus = 2305843009213693951

    def cache_key(ids: list[int]) -> int:
        key = 0
        length = len(ids)
        for lag in range(order):
            value = ids[length - 1 - lag] if lag < length else 0
            key = (key * 257 + int(value) + 1) % modulus
        return key

    ids_by_prompt = [list(ids) for ids in prompt_bytes]
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(max_new_bytes):
        next_bytes: list[int] = []
        for ids in ids_by_prompt:
            candidates = cache_map.get(cache_key(ids))
            if not candidates:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                return None
            selected = candidates[0]
            next_bytes.append(selected)
        for ids, selected in zip(ids_by_prompt, next_bytes):
            ids.append(selected)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    texts = [
        bytes(ids[len(prompt) : len(prompt) + max_new_bytes]).decode(
            "utf-8",
            errors="replace",
        )
        for ids, prompt in zip(ids_by_prompt, prompt_bytes)
    ]
    return texts, elapsed

    max_prompt = max(max(len(ids), order) for ids in prompt_bytes)
    recent = torch.tensor(
        [
            ([ord(" ")] * (max_prompt - len(ids))) + ids[-max_prompt:]
            for ids in prompt_bytes
        ],
        dtype=torch.long,
        device=device,
    )
    generated: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(max_new_bytes):
        logits = model._last_domain_cache_prior(recent)
        active = logits.abs().sum(dim=-1) > 0
        if not bool(active.all().detach().cpu()):
            if device.type == "cuda":
                torch.cuda.synchronize()
            return None
        if hasattr(model, "_apply_generation_word_shape_constraints"):
            logits = model._apply_generation_word_shape_constraints(logits, recent)
        if hasattr(model, "_select_no_repeat_byte"):
            next_byte = model._select_no_repeat_byte(logits, recent, no_repeat_ngram)
        else:
            selected = []
            for row in range(logits.shape[0]):
                selected.append(_pick_next(logits[row], recent[row].tolist(), no_repeat_ngram))
            next_byte = torch.tensor(selected, dtype=torch.long, device=device)
        generated.append(next_byte)
        recent = torch.cat([recent, next_byte[:, None]], dim=1)
        keep = max(order, 64)
        if recent.shape[1] > keep:
            recent = recent[:, -keep:]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    continuation = torch.stack(generated, dim=1).cpu()
    texts = [
        bytes(int(byte) for byte in continuation[row]).decode(
            "utf-8",
            errors="replace",
        )
        for row in range(continuation.shape[0])
    ]
    return texts, elapsed


@torch.inference_mode()
def _generate_layercake_fast_patch_batch(
    model: torch.nn.Module,
    prompts: list[str],
    *,
    device: torch.device,
    max_new_bytes: int,
    no_repeat_ngram: int,
) -> tuple[list[str], float]:
    patch_size = int(getattr(model, "patch_size", 2))
    seq = int(model.patch_pos.num_embeddings * patch_size)
    prompt_bytes = [
        list(prompt.encode("utf-8", errors="replace"))
        for prompt in prompts
    ]
    contexts = []
    for ids in prompt_bytes:
        ctx = ids[-seq:]
        if len(ctx) % patch_size:
            ctx = ([ord(" ")] * (patch_size - (len(ctx) % patch_size))) + ctx
        contexts.append(ctx)
    max_context = max(len(ctx) for ctx in contexts)
    if max_context % patch_size:
        max_context += patch_size - (max_context % patch_size)
    padded = [
        ([ord(" ")] * (max_context - len(ctx))) + ctx
        for ctx in contexts
    ]
    x = torch.tensor(padded, dtype=torch.long, device=device)
    state = model.begin_abi_patch_cell_cached_generation(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    generated_patches = []
    emitted = 0
    while emitted < max_new_bytes:
        remaining = max_new_bytes - emitted
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
        take = min(patch.shape[1], remaining)
        generated_patches.append(patch[:, :take].detach())
        emitted += take
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    continuation = torch.cat(generated_patches, dim=1)[:, :max_new_bytes].cpu()
    texts = [
        bytes(int(byte) for byte in continuation[row]).decode(
            "utf-8",
            errors="replace",
        )
        for row in range(continuation.shape[0])
    ]
    return texts, elapsed


@torch.inference_mode()
def _generate_layercake_batch(
    checkpoint: dict,
    prompts: list[str],
    *,
    device: torch.device,
    max_new_bytes: int,
    no_repeat_ngram: int,
) -> tuple[list[str], float] | None:
    model_cfg = checkpoint["model_config"]
    model = _build_model(model_cfg, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if (
        getattr(model, "local_decoder", "") == "abi_patch_cell"
        and hasattr(model, "begin_abi_patch_cell_cached_generation")
    ):
        cache_result = _generate_layercake_domain_cache_batch(
            model,
            prompts,
            device=device,
            max_new_bytes=max_new_bytes,
            no_repeat_ngram=no_repeat_ngram,
        )
        if cache_result is not None:
            return cache_result
        return _generate_layercake_fast_patch_batch(
            model,
            prompts,
            device=device,
            max_new_bytes=max_new_bytes,
            no_repeat_ngram=no_repeat_ngram,
        )
    return None


def _split_attention_projection(
    layer: torch.nn.TransformerEncoderLayer,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projection = F.linear(
        hidden,
        layer.self_attn.in_proj_weight,
        layer.self_attn.in_proj_bias,
    )
    query, key, value = projection.chunk(3, dim=-1)
    heads = int(layer.self_attn.num_heads)
    head_width = query.shape[-1] // heads

    def shaped(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(
            tensor.shape[0], tensor.shape[1], heads, head_width
        ).transpose(1, 2)

    return shaped(query), shaped(key), shaped(value)


def _attention_output(
    layer: torch.nn.TransformerEncoderLayer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool,
) -> torch.Tensor:
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    attended = attended.transpose(1, 2).reshape(
        attended.shape[0], attended.shape[2], -1
    )
    return layer.self_attn.out_proj(attended)


def _transformer_feed_forward(
    layer: torch.nn.TransformerEncoderLayer,
    hidden: torch.Tensor,
) -> torch.Tensor:
    return layer.linear2(layer.activation(layer.linear1(hidden)))


@torch.inference_mode()
def _prefill_bpe_cache(
    model: BPETokenTransformerLM,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    hidden = model.emb(tokens) + model.pos(positions)[None]
    caches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in model.core.layers:
        normalized = layer.norm1(hidden)
        query, key, value = _split_attention_projection(layer, normalized)
        hidden = hidden + _attention_output(
            layer,
            query,
            key,
            value,
            is_causal=True,
        )
        hidden = hidden + _transformer_feed_forward(
            layer,
            layer.norm2(hidden),
        )
        caches.append((key, value))
    logits = model.head(model.norm(hidden[:, -1]))
    return logits, caches


@torch.inference_mode()
def _decode_bpe_cache(
    model: BPETokenTransformerLM,
    token: torch.Tensor,
    caches: list[tuple[torch.Tensor, torch.Tensor]],
    position: int,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    hidden = model.emb(token) + model.pos.weight[position][None, None]
    updated: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer, (cached_key, cached_value) in zip(model.core.layers, caches):
        normalized = layer.norm1(hidden)
        query, key, value = _split_attention_projection(layer, normalized)
        key = torch.cat([cached_key, key], dim=2)
        value = torch.cat([cached_value, value], dim=2)
        hidden = hidden + _attention_output(
            layer,
            query,
            key,
            value,
            is_causal=False,
        )
        hidden = hidden + _transformer_feed_forward(
            layer,
            layer.norm2(hidden),
        )
        updated.append((key, value))
    logits = model.head(model.norm(hidden[:, -1]))
    return logits, updated


@torch.inference_mode()
def _generate_bpe(
    checkpoint: dict,
    prompt: str,
    *,
    device: torch.device,
    max_new_bytes: int,
    no_repeat_ngram: int,
) -> tuple[str, float]:
    model_cfg = checkpoint["model_config"]
    train_cfg = checkpoint.get("training_config", {})
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
        max_len=int(train_cfg.get("seq_len", 512)),
        ff_mult=int(model_cfg.get("ff_mult", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    ids = tokenizer.encode(prompt, out_type=int)
    original_text = tokenizer.decode(ids)
    generated = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    continuation = ""
    context = generated[:, -model.pos.num_embeddings :]
    cached_logits, caches = _prefill_bpe_cache(model, context)
    reference_logits = model(context)[0, -1]
    cache_error = float(
        (cached_logits[0] - reference_logits).abs().max().detach().cpu()
    )
    if cache_error > 2e-4:
        raise RuntimeError(
            f"Transformer KV-cache parity failed: max_abs_error={cache_error}"
        )
    position = int(context.shape[1])
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    while len(continuation.encode("utf-8", errors="replace")) < max_new_bytes:
        next_token = _pick_next(
            cached_logits[0], generated[0].tolist(), no_repeat_ngram
        )
        token = torch.tensor([[next_token]], dtype=torch.long, device=device)
        generated = torch.cat([generated, token], dim=1)
        decoded = tokenizer.decode(generated[0].tolist())
        continuation = decoded[len(original_text) :]
        if position >= model.pos.num_embeddings:
            context = generated[:, -model.pos.num_embeddings :]
            cached_logits, caches = _prefill_bpe_cache(model, context)
            position = int(context.shape[1])
        else:
            cached_logits, caches = _decode_bpe_cache(
                model,
                token,
                caches,
                position,
            )
            position += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return continuation.encode("utf-8", errors="replace")[:max_new_bytes].decode(
        "utf-8", errors="replace"
    ), elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate moonshot verifier artifacts for LayerCake or BPE checkpoints")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-kind", required=True, choices=["layercake", "bpe"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-new-bytes", type=int, default=128)
    parser.add_argument("--no-repeat-ngram", type=int, default=8)
    parser.add_argument("--prompt-prefix-file", type=Path)
    parser.add_argument("--prompt-prefix-bytes", type=int, default=0)
    parser.add_argument(
        "--radix-factorwise-greedy",
        action="store_true",
        help="Use ancestral high-then-low greedy decoding for radix heads.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.radix_factorwise_greedy and args.model_kind == "layercake":
        checkpoint["model_config"] = dict(checkpoint["model_config"])
        checkpoint["model_config"]["patch_generation_joint_greedy"] = False

    prompts = list(PROMPTS)
    if args.prompt_prefix_bytes > 0:
        if args.prompt_prefix_file is None:
            raise ValueError("--prompt-prefix-file is required with prefix bytes")
        prefix_payload = args.prompt_prefix_file.read_bytes()[
            : args.prompt_prefix_bytes
        ]
        prefix_text = prefix_payload.decode("utf-8", errors="replace")
        prompts = [prefix_text + prompt for prompt in prompts]

    rows = []
    total_bytes = 0
    total_seconds = 0.0
    quality_scores = []
    batched_layercake = None
    if args.model_kind == "layercake":
        batched_layercake = _generate_layercake_batch(
            checkpoint,
            prompts,
            device=device,
            max_new_bytes=args.max_new_bytes,
            no_repeat_ngram=args.no_repeat_ngram,
        )
    if batched_layercake is not None:
        texts, batch_seconds = batched_layercake
        per_sample_seconds = batch_seconds / max(len(texts), 1)
        prompt_text_pairs = list(zip(prompts, texts, [per_sample_seconds] * len(texts)))
    else:
        prompt_text_pairs = []
        for prompt in prompts:
            text, seconds = _generate_bpe(
                    checkpoint,
                    prompt,
                    device=device,
                    max_new_bytes=args.max_new_bytes,
                    no_repeat_ngram=args.no_repeat_ngram,
                ) if args.model_kind == "bpe" else _generate_layercake(
                    checkpoint,
                    prompt,
                    device=device,
                    max_new_bytes=args.max_new_bytes,
                    no_repeat_ngram=args.no_repeat_ngram,
                )
            prompt_text_pairs.append((prompt, text, seconds))
    for prompt, text, seconds in prompt_text_pairs:
        emitted = len(text.encode("utf-8", errors="replace"))
        metrics = _quality_score(text)
        rows.append(
            {
                "prompt": prompt,
                "text": text,
                "seconds": seconds,
                "generated_bytes": emitted,
                "bytes_per_second": emitted / max(seconds, 1e-12),
                **metrics,
            }
        )
        total_bytes += emitted
        total_seconds += seconds
        quality_scores.append(metrics["quality_score"])

    result = {
        "status": "PASS",
        "model_kind": args.model_kind,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        "max_new_bytes": args.max_new_bytes,
        "no_repeat_ngram": args.no_repeat_ngram,
        "prompt_prefix_file": (
            str(args.prompt_prefix_file) if args.prompt_prefix_file else None
        ),
        "prompt_prefix_bytes": int(args.prompt_prefix_bytes),
        "metrics": {
            "generation_bytes_per_second": total_bytes / max(total_seconds, 1e-12),
            "quality_score": sum(quality_scores) / max(len(quality_scores), 1),
            "generated_bytes": total_bytes,
            "seconds": total_seconds,
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
