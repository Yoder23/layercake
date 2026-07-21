from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import math
import os
import sys
import time
from collections import deque
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.causal_byte_models import CausalBytePatchLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _iter_text_bytes_from_jsonl(path: Path) -> Iterable[bytes]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    text = payload.get("text") or payload.get("content") or ""
                else:
                    text = str(payload)
            except json.JSONDecodeError:
                text = line
            if text:
                yield text.encode("utf-8", errors="replace") + b"\n"


def _iter_file_payload(path: Path, read_block_bytes: int = 1 << 20) -> Iterable[bytes]:
    if path.suffix.lower() == ".jsonl":
        yield from _iter_text_bytes_from_jsonl(path)
        return

    try:
        with path.open("rb") as handle:
            while True:
                payload = handle.read(read_block_bytes)
                if not payload:
                    break
                yield payload
    except OSError:
        return


def _iter_file_payload_forever(path: Path, read_block_bytes: int) -> Iterable[bytes]:
    while True:
        any_payload = False
        for payload in _iter_file_payload(path, read_block_bytes=read_block_bytes):
            any_payload = True
            yield payload
        if not any_payload:
            return
        yield b"\n"


def _collect_corpus_files(roots: list[Path], include_suffixes: set[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in include_suffixes:
            files.append(root)
            continue
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in include_suffixes:
                files.append(path)
    return files


class ByteCorpusDataset(IterableDataset):
    """Streams raw UTF-8 bytes from mixed corpus files without tokenization."""

    def __init__(self, files: list[Path], seq_len: int, read_block_bytes: int = 1 << 20):
        self.files = files
        self.seq_len = seq_len
        self.read_block_bytes = read_block_bytes

    def __iter__(self):
        streams = deque(
            (idx, _iter_file_payload_forever(path, read_block_bytes=self.read_block_bytes))
            for idx, path in enumerate(self.files)
        )
        if not streams:
            return

        buffers = [bytearray() for _ in self.files]

        while streams:
            stream_idx, stream = streams.popleft()
            try:
                payload = next(stream)
            except StopIteration:
                continue

            streams.append((stream_idx, stream))
            buffer = buffers[stream_idx]
            buffer.extend(payload)
            while len(buffer) >= self.seq_len:
                chunk = bytes(buffer[: self.seq_len])
                del buffer[: self.seq_len]
                yield torch.tensor(list(chunk), dtype=torch.long)


class JsonlRowByteDataset(IterableDataset):
    """Yields one JSONL text row per sample, with answer start patch-aligned."""

    def __init__(
        self,
        files: list[Path],
        seq_len: int,
        *,
        patch_size: int,
        answer_marker: bytes = b"Answer: ",
        read_block_bytes: int = 1 << 20,
    ):
        self.files = files
        self.seq_len = seq_len
        self.patch_size = max(int(patch_size), 1)
        self.answer_marker = answer_marker
        self.read_block_bytes = read_block_bytes

    def __iter__(self):
        if not self.files:
            return
        while True:
            emitted = False
            for path in self.files:
                for payload in _iter_file_payload(path, read_block_bytes=self.read_block_bytes):
                    emitted = True
                    marker = self.answer_marker
                    marker_start = payload.find(marker)
                    if marker_start >= 0:
                        answer_start = marker_start + len(marker)
                        left_pad = (-answer_start) % self.patch_size
                    else:
                        left_pad = 0
                    row = (b" " * left_pad) + payload
                    if len(row) < self.seq_len:
                        row = row + (b" " * (self.seq_len - len(row)))
                    else:
                        row = row[: self.seq_len]
                    yield torch.tensor(list(row), dtype=torch.long)
            if not emitted:
                return


class MixedByteComponent:
    def __init__(
        self,
        *,
        name: str,
        weight: float,
        dataset: IterableDataset,
        files: list[Path],
        row_preserve_jsonl_examples: bool,
    ):
        self.name = name
        self.weight = weight
        self.dataset = dataset
        self.files = files
        self.row_preserve_jsonl_examples = row_preserve_jsonl_examples


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def _weighted_schedule(weights: list[float], *, max_slots: int = 128) -> list[int]:
    if not weights:
        return []
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("data_mix component weights must be positive")
    fractions = [Fraction(str(weight)).limit_denominator(100) for weight in weights]
    denominator = reduce(_lcm, (fraction.denominator for fraction in fractions), 1)
    counts = [
        max(1, int(fraction.numerator * denominator // fraction.denominator))
        for fraction in fractions
    ]
    total = sum(counts)
    if total > max_slots:
        scale = total / float(max_slots)
        counts = [max(1, int(round(count / scale))) for count in counts]

    schedule: list[int] = []
    remaining = counts[:]
    emitted = [0 for _ in counts]
    total = sum(counts)
    for _ in range(total):
        best_index = max(
            range(len(counts)),
            key=lambda idx: (
                remaining[idx] / counts[idx],
                -emitted[idx],
                -idx,
            ),
        )
        schedule.append(best_index)
        remaining[best_index] -= 1
        emitted[best_index] += 1
    return schedule


class WeightedMixedByteDataset(IterableDataset):
    """Deterministically interleaves row-preserved tasks with raw LM replay."""

    def __init__(self, components: list[MixedByteComponent]):
        if not components:
            raise ValueError("WeightedMixedByteDataset requires at least one component")
        self.components = components
        self.schedule = _weighted_schedule([component.weight for component in components])

    def __iter__(self):
        iterators = [iter(component.dataset) for component in self.components]
        empty_components: set[int] = set()
        while len(empty_components) < len(self.components):
            emitted_this_cycle = False
            for component_index in self.schedule:
                if component_index in empty_components:
                    continue
                try:
                    yield next(iterators[component_index])
                    emitted_this_cycle = True
                    continue
                except StopIteration:
                    iterators[component_index] = iter(
                        self.components[component_index].dataset
                    )
                try:
                    yield next(iterators[component_index])
                    emitted_this_cycle = True
                except StopIteration:
                    empty_components.add(component_index)
            if not emitted_this_cycle:
                return


def _load_eval_byte_stream(
    files: list[Path],
    *,
    max_bytes: int,
    read_block_bytes: int,
) -> torch.Tensor:
    collected = bytearray()
    for path in files:
        for payload in _iter_file_payload(path, read_block_bytes=read_block_bytes):
            remaining = max_bytes - len(collected)
            if remaining <= 0:
                break
            collected.extend(payload[:remaining])
            if len(collected) >= max_bytes:
                break
        if len(collected) >= max_bytes:
            break
    if len(collected) < 4096:
        raise RuntimeError("Not enough held-out eval bytes collected")
    return torch.tensor(list(collected), dtype=torch.long)


def _answer_span_weights(
    rows: torch.Tensor,
    *,
    target_len: int,
    answer_weight: float,
    base_weight: float = 1.0,
    answer_marker: bytes = b"Answer: ",
) -> torch.Tensor:
    if answer_weight <= 1.0 and base_weight == 1.0:
        return torch.ones(
            rows.shape[0],
            target_len,
            dtype=torch.float32,
            device=rows.device,
        )
    weights = torch.ones(
        rows.shape[0],
        target_len,
        dtype=torch.float32,
        device=rows.device,
    ) * float(base_weight)
    marker = answer_marker
    terminator = b"\n###"
    rows_cpu = rows.detach().cpu().tolist()
    for row_index, values in enumerate(rows_cpu):
        payload = bytes(int(v) for v in values)
        search_from = 0
        while True:
            marker_start = payload.find(marker, search_from)
            if marker_start < 0:
                break
            answer_start = marker_start + len(marker)
            answer_end = payload.find(terminator, answer_start)
            if answer_end < 0:
                answer_end = len(payload)
            target_start = max(answer_start - 1, 0)
            target_end = min(answer_end - 1, target_len)
            if target_start < target_end:
                weights[row_index, target_start:target_end] = answer_weight
            search_from = max(answer_end + len(terminator), answer_start + 1)
    return weights


def _answer_position_weights(
    rows: torch.Tensor,
    absolute_positions: torch.Tensor,
    *,
    answer_weight: float,
    base_weight: float = 1.0,
    answer_marker: bytes = b"Answer: ",
) -> torch.Tensor:
    max_position = (
        int(absolute_positions.max().item()) + 1
        if absolute_positions.numel()
        else 1
    )
    position_weights = torch.ones(
        rows.shape[0],
        max_position,
        dtype=torch.float32,
        device=rows.device,
    ) * float(base_weight)
    marker = answer_marker
    terminator = b"\n###"
    rows_cpu = rows.detach().cpu().tolist()
    for row_index, values in enumerate(rows_cpu):
        payload = bytes(int(v) for v in values)
        search_from = 0
        while True:
            marker_start = payload.find(marker, search_from)
            if marker_start < 0:
                break
            answer_start = marker_start + len(marker)
            answer_end = payload.find(terminator, answer_start)
            if answer_end < 0:
                answer_end = len(payload)
            bounded_start = min(max(answer_start, 0), max_position)
            bounded_end = min(max(answer_end, 0), max_position)
            if bounded_start < bounded_end:
                position_weights[
                    row_index,
                    bounded_start:bounded_end,
                ] = float(answer_weight)
            search_from = max(answer_end + len(terminator), answer_start + 1)
    positions = absolute_positions.to(device=rows.device, dtype=torch.long)
    if positions.ndim > 1 and positions.shape[0] == rows.shape[0]:
        flat_positions = positions.reshape(rows.shape[0], -1)
        return position_weights.gather(1, flat_positions).reshape_as(positions)
    return position_weights.index_select(1, positions)


def _answer_start_positions(
    rows: torch.Tensor,
    *,
    answer_marker: bytes = b"Answer: ",
) -> torch.Tensor:
    """Locate the first answer byte in every row, or -1 when absent."""
    starts = []
    for values in rows.detach().cpu().tolist():
        payload = bytes(int(value) for value in values)
        marker_start = payload.find(answer_marker)
        starts.append(
            marker_start + len(answer_marker)
            if marker_start >= 0
            else -1
        )
    return torch.tensor(starts, device=rows.device, dtype=torch.long)


def _copy_alignment_labels(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    source_len: int,
    source_end_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    source_len = min(int(source_len), x.shape[1])
    source = x[:, -source_len:]
    source_start = x.shape[1] - source_len
    source_positions = (
        torch.arange(source_len, device=x.device, dtype=torch.long)
        + source_start
    )
    target_positions = torch.arange(y.shape[1], device=y.device, dtype=torch.long)
    causal = source_positions[None, :] <= target_positions[:, None]
    if source_end_positions is not None:
        source_end_positions = source_end_positions.to(
            device=x.device,
            dtype=torch.long,
        )
        prompt_bounded = source_positions.view(1, 1, -1) < source_end_positions.view(
            -1,
            1,
            1,
        )
        causal = causal[None] & prompt_bounded
    else:
        causal = causal[None]
    matches = (source[:, None, :] == y[:, :, None]) & causal
    indexes = torch.arange(source_len, device=x.device, dtype=torch.long)
    labeled = torch.where(matches, indexes.view(1, 1, -1), torch.full_like(matches, -1, dtype=torch.long))
    labels = labeled.max(dim=-1).values
    return torch.where(labels >= 0, labels, torch.full_like(labels, -100))


def _copy_alignment_labels_at_positions(
    x: torch.Tensor,
    targets: torch.Tensor,
    absolute_positions: torch.Tensor,
    *,
    source_len: int,
    source_end_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    source_len = min(int(source_len), x.shape[1])
    source = x[:, -source_len:]
    source_start = x.shape[1] - source_len
    source_positions = (
        torch.arange(source_len, device=x.device, dtype=torch.long)
        + source_start
    )
    causal = source_positions.view(1, 1, -1) <= absolute_positions[:, :, None]
    if source_end_positions is not None:
        source_end_positions = source_end_positions.to(
            device=x.device,
            dtype=torch.long,
        )
        prompt_bounded = source_positions.view(1, 1, -1) < source_end_positions.view(
            -1,
            1,
            1,
        )
        causal = causal & prompt_bounded
    matches = (source[:, None, :] == targets[:, :, None]) & causal
    indexes = torch.arange(source_len, device=x.device, dtype=torch.long)
    labeled = torch.where(
        matches,
        indexes.view(1, 1, -1),
        torch.full_like(matches, -1, dtype=torch.long),
    )
    labels = labeled.max(dim=-1).values
    return torch.where(labels >= 0, labels, torch.full_like(labels, -100))


@torch.inference_mode()
def _eval_byte_bpb(
    model: CausalBytePatchLM,
    stream: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    max_start = stream.numel() - seq_len - 1
    if max_start <= 0:
        raise RuntimeError("held-out byte stream is too short for eval")
    losses: list[float] = []
    for _ in range(batches):
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        rows = torch.stack([stream[start : start + seq_len + 1] for start in starts]).to(device)
        x = rows[:, :-1]
        y = rows[:, 1:]
        logits, _ = model(x)
        logits = logits[:, : y.shape[1], :]
        losses.append(float(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()))
    if was_training:
        model.train()
    return (sum(losses) / max(len(losses), 1)) / math.log(2)


def _load_prior_bytes(
    files: list[Path],
    *,
    read_block_bytes: int,
    max_bytes: int,
) -> torch.Tensor:
    collected = bytearray()
    for path in files:
        for payload in _iter_file_payload(path, read_block_bytes=read_block_bytes):
            remaining = max_bytes - len(collected)
            if remaining <= 0:
                break
            collected.extend(payload[:remaining])
        if len(collected) >= max_bytes:
            break
    if len(collected) < 2:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(list(collected), dtype=torch.long)


def _context_ids_1d(x: torch.Tensor, buckets: int, order: int) -> torch.Tensor:
    ids = torch.zeros_like(x)
    for lag in range(order):
        if lag == 0:
            shifted = x
        else:
            shifted = F.pad(x[:-lag], (lag, 0))
        ids = (ids * 257 + shifted + 1) % buckets
    return ids


def _initialize_byte_priors_from_corpus(
    model: CausalBytePatchLM,
    files: list[Path],
    *,
    read_block_bytes: int,
    max_bytes: int,
    smoothing: float,
) -> dict[str, Any]:
    prior_bytes = _load_prior_bytes(
        files,
        read_block_bytes=read_block_bytes,
        max_bytes=max_bytes,
    )
    if prior_bytes.numel() < 2:
        return {"prior_bytes": int(prior_bytes.numel()), "status": "SKIPPED"}
    x = prior_bytes[:-1]
    y = prior_bytes[1:]
    with torch.no_grad():
        transition_counts = torch.full((256, 256), float(smoothing))
        transition_counts.index_put_(
            (x, y),
            torch.ones_like(y, dtype=transition_counts.dtype),
            accumulate=True,
        )
        transition_logits = transition_counts.log()
        transition_logits = transition_logits - transition_logits.mean(dim=1, keepdim=True)
        model.transition_head.weight.copy_(transition_logits.to(model.transition_head.weight.device))

        context_buckets = int(getattr(model, "context_buckets", 0))
        if context_buckets and hasattr(model, "context_head"):
            context_ids = _context_ids_1d(
                x,
                context_buckets,
                int(getattr(model, "context_order", 2)),
            )
            context_counts = torch.full((context_buckets, 256), float(smoothing))
            context_counts.index_put_(
                (context_ids, y),
                torch.ones_like(y, dtype=context_counts.dtype),
                accumulate=True,
            )
            context_logits = context_counts.log()
            context_logits = context_logits - context_logits.mean(dim=1, keepdim=True)
            model.context_head.weight.copy_(context_logits.to(model.context_head.weight.device))
    return {
        "prior_bytes": int(prior_bytes.numel()),
        "transition_rows": 256,
        "context_buckets": int(getattr(model, "context_buckets", 0)),
        "status": "INITIALIZED",
    }


def _domain_context_keys_1d(x: torch.Tensor, order: int) -> torch.Tensor:
    ids = torch.zeros_like(x, dtype=torch.long)
    modulus = 2305843009213693951
    for lag in range(order):
        if lag == 0:
            shifted = x.to(torch.long)
        else:
            shifted = F.pad(x[:-lag], (lag, 0)).to(torch.long)
        ids = torch.remainder(ids * 257 + shifted + 1, modulus)
    return ids


def _initialize_domain_cache_from_corpus(
    model: CausalBytePatchLM,
    files: list[Path],
    *,
    read_block_bytes: int,
    max_bytes: int,
    order: int,
    smoothing: float,
    max_entries: int,
    min_confidence: float,
    high_confidence_boost: float,
) -> dict[str, Any]:
    if order <= 0 or not hasattr(model, "set_domain_cache"):
        return {"status": "SKIPPED"}
    cache_bytes = _load_prior_bytes(
        files,
        read_block_bytes=read_block_bytes,
        max_bytes=max_bytes,
    )
    if cache_bytes.numel() <= order + 1:
        return {"status": "SKIPPED", "cache_bytes": int(cache_bytes.numel())}
    x = cache_bytes[:-1]
    y = cache_bytes[1:]
    keys = _domain_context_keys_1d(x, order)
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    counts = torch.full(
        (unique_keys.numel(), 256),
        float(smoothing),
        dtype=torch.float32,
    )
    counts.index_put_(
        (inverse, y),
        torch.ones_like(y, dtype=counts.dtype),
        accumulate=True,
    )
    if max_entries > 0 and unique_keys.numel() > max_entries:
        totals = counts.sum(dim=1)
        keep = torch.topk(totals, k=max_entries).indices.sort().values
        unique_keys = unique_keys[keep]
        counts = counts[keep]
    totals = counts.sum(dim=1, keepdim=True)
    confidence = counts.max(dim=1, keepdim=True).values / totals.clamp_min(1e-12)
    logits = counts.log()
    logits = logits - logits.mean(dim=1, keepdim=True)
    if min_confidence > 0.0:
        high_confidence = confidence >= float(min_confidence)
        if high_confidence_boost > 0.0:
            logits = torch.where(
                high_confidence,
                logits * float(high_confidence_boost),
                logits,
            )
    model.set_domain_cache(
        unique_keys,
        logits,
        order=order,
        logit_scale=float(getattr(model, "domain_cache_logit_scale", 1.0)),
    )
    return {
        "status": "INITIALIZED",
        "cache_bytes": int(cache_bytes.numel()),
        "order": int(order),
        "entries": int(unique_keys.numel()),
        "min_confidence": float(min_confidence),
        "high_confidence_boost": float(high_confidence_boost),
        "active_entries": int((confidence[:, 0] >= float(min_confidence)).sum().item()) if min_confidence > 0.0 else int(unique_keys.numel()),
    }


def _build_model(model_cfg: dict, device: torch.device) -> CausalBytePatchLM:
    model = CausalBytePatchLM(
        patch_size=model_cfg.get("patch_size", 2),
        d_byte=model_cfg.get("d_byte", 96),
        d_model=model_cfg.get("d_model", 1280),
        d_abi=model_cfg.get("d_abi", 320),
        layers=model_cfg.get("layers", 24),
        heads=model_cfg.get("heads", 16),
        max_patches=model_cfg.get("max_patches", 1024),
        continuous_local=model_cfg.get("continuous_local", False),
        direct_global_context=model_cfg.get("direct_global_context", True),
        local_decoder=model_cfg.get("local_decoder", "window_transformer"),
        local_layers=model_cfg.get("local_layers", 4),
        local_width=model_cfg.get("local_width", 768),
        conv_layers=model_cfg.get("conv_layers", 4),
        modern_blocks=model_cfg.get("modern_blocks", True),
        fused_attention=model_cfg.get("fused_attention", True),
        local_window=model_cfg.get("local_window", 64),
        ngram_buckets=model_cfg.get("ngram_buckets", 0),
        patch_unit_buckets=model_cfg.get("patch_unit_buckets", 0),
        dropout=model_cfg.get("dropout", 0.1),
        qk_norm=model_cfg.get("qk_norm", True),
        global_block=model_cfg.get("global_block", "attention"),
        routed_cake_experts=model_cfg.get("routed_cake_experts", 0),
        shared_cake_layers=model_cfg.get("shared_cake_layers", 0),
        default_cake_route=model_cfg.get("default_cake_route"),
        sparse_state_local_window=model_cfg.get("sparse_state_local_window", 32),
        sparse_state_dilated_offsets=tuple(
            model_cfg.get("sparse_state_dilated_offsets", [32, 48, 64, 96])
        ),
        sparse_state_chunk_size=model_cfg.get("sparse_state_chunk_size", 16),
        context_buckets=model_cfg.get("context_buckets", 0),
        context_order=model_cfg.get("context_order", 2),
        transition_logit_scale=model_cfg.get("transition_logit_scale", 1.0),
        context_logit_scale=model_cfg.get("context_logit_scale", 1.0),
        trainable_prior_gates=model_cfg.get("trainable_prior_gates", False),
        trainable_transition_head=model_cfg.get("trainable_transition_head", True),
        trainable_context_head=model_cfg.get("trainable_context_head", True),
        abi_patch_cell_static_generation=model_cfg.get("abi_patch_cell_static_generation", False),
        abi_patch_cell_global_update_interval=model_cfg.get("abi_patch_cell_global_update_interval", 1),
        abi_patch_cell_fast_global_decode=model_cfg.get("abi_patch_cell_fast_global_decode", False),
        abi_patch_cell_fast_local_runtime=model_cfg.get("abi_patch_cell_fast_local_runtime", False),
        abi_patch_cell_lightweight_context_update=model_cfg.get("abi_patch_cell_lightweight_context_update", False),
        abi_patch_cell_lightweight_context_blend=model_cfg.get("abi_patch_cell_lightweight_context_blend", 0.15),
        generation_min_word_chars=model_cfg.get("generation_min_word_chars", 0),
        generation_repeat_suppression_window=model_cfg.get("generation_repeat_suppression_window", 0),
        generation_repeat_suppression_scale=model_cfg.get("generation_repeat_suppression_scale", 0.0),
        domain_cache_order=model_cfg.get("domain_cache_order", 0),
        domain_cache_logit_scale=model_cfg.get("domain_cache_logit_scale", 0.0),
        domain_cache_override=model_cfg.get("domain_cache_override", False),
        copy_attention=model_cfg.get("copy_attention", False),
        copy_attention_dim=model_cfg.get("copy_attention_dim", 32),
        copy_attention_scale=model_cfg.get("copy_attention_scale", 4.0),
        copy_attention_window=model_cfg.get("copy_attention_window", 128),
        copy_transducer=model_cfg.get("copy_transducer", False),
        copy_transducer_dim=model_cfg.get("copy_transducer_dim", 32),
        copy_transducer_scale=model_cfg.get("copy_transducer_scale", 4.0),
        copy_transducer_window=model_cfg.get("copy_transducer_window", 128),
        copy_transducer_logit_mode=model_cfg.get("copy_transducer_logit_mode", "prob"),
        copy_transducer_projection=model_cfg.get("copy_transducer_projection", "soft"),
        span_width=model_cfg.get("span_width", 4),
        span_verifier=model_cfg.get("span_verifier", False),
        span_prefix_conditioning=model_cfg.get("span_prefix_conditioning", True),
        dynamic_prior_gates=model_cfg.get("dynamic_prior_gates", False),
        prior_dropout=model_cfg.get("prior_dropout", 0.0),
        repeat_suppression_window=model_cfg.get("repeat_suppression_window", 0),
        repeat_suppression_scale=model_cfg.get("repeat_suppression_scale", 0.0),
        trainable_repeat_suppression=model_cfg.get("trainable_repeat_suppression", False),
        patch_prediction=model_cfg.get("patch_prediction", False),
        patch_prediction_stride=model_cfg.get("patch_prediction_stride", 1),
        patch_prediction_mode=model_cfg.get("patch_prediction_mode", "factorized"),
        patch_prediction_context=model_cfg.get("patch_prediction_context", "global"),
        patch_prediction_detach_context=model_cfg.get("patch_prediction_detach_context", False),
        patch_generation_width=model_cfg.get("patch_generation_width", 96),
        patch_generation_bytes=model_cfg.get("patch_generation_bytes", 0),
        patch_prediction_rollout_training=model_cfg.get(
            "patch_prediction_rollout_training", False
        ),
        patch_prediction_rollout_mix=model_cfg.get(
            "patch_prediction_rollout_mix", 1.0
        ),
        patch_generation_context=model_cfg.get("patch_generation_context", 0),
        patch_generation_copy_window=model_cfg.get("patch_generation_copy_window", 0),
        patch_generation_copy_dim=model_cfg.get("patch_generation_copy_dim", 32),
        patch_generation_copy_scale=model_cfg.get("patch_generation_copy_scale", 4.0),
        patch_generation_position_copy=model_cfg.get(
            "patch_generation_position_copy", False
        ),
        patch_generation_contextual_copy=model_cfg.get(
            "patch_generation_contextual_copy", False
        ),
        patch_generation_lowercase_copy=model_cfg.get(
            "patch_generation_lowercase_copy", False
        ),
        patch_generation_semantic_copy=model_cfg.get(
            "patch_generation_semantic_copy", False
        ),
        tie_byte_embeddings=model_cfg.get("tie_byte_embeddings", False),
    ).to(device)
    return model


def _apply_parameter_training_filter(
    model: torch.nn.Module,
    *,
    trainable_patterns: list[str] | None = None,
    frozen_patterns: list[str] | None = None,
) -> dict[str, Any]:
    trainable_patterns = list(trainable_patterns or [])
    frozen_patterns = list(frozen_patterns or [])
    if not trainable_patterns and not frozen_patterns:
        total = sum(parameter.numel() for parameter in model.parameters())
        return {
            "enabled": False,
            "total_params": total,
            "trainable_params_after_filter": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "frozen_params_after_filter": sum(
                parameter.numel()
                for parameter in model.parameters()
                if not parameter.requires_grad
            ),
            "trainable_patterns": [],
            "frozen_patterns": [],
            "trainable_parameter_names": [],
        }

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        if trainable_patterns:
            should_train = any(
                fnmatch.fnmatchcase(name, pattern)
                for pattern in trainable_patterns
            )
        else:
            should_train = True
        if frozen_patterns and any(
            fnmatch.fnmatchcase(name, pattern)
            for pattern in frozen_patterns
        ):
            should_train = False
        parameter.requires_grad_(should_train)
        if should_train:
            trainable_names.append(name)
        else:
            frozen_names.append(name)

    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_params = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    if trainable_params <= 0:
        raise RuntimeError("parameter filter froze every model parameter")
    return {
        "enabled": True,
        "total_params": trainable_params + frozen_params,
        "trainable_params_after_filter": trainable_params,
        "frozen_params_after_filter": frozen_params,
        "trainable_patterns": trainable_patterns,
        "frozen_patterns": frozen_patterns,
        "trainable_parameter_names": trainable_names[:64],
        "frozen_parameter_count": len(frozen_names),
    }


def _resolve_config_paths(root: Path, items: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        paths.append((root / path).resolve() if not path.is_absolute() else path)
    return paths


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _train(config: dict):
    root = Path(__file__).resolve().parents[1]

    train_cfg = config["training"]
    model_cfg = config["model"]

    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    seed = int(train_cfg.get("seed", 1234))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(train_cfg.get("matmul_precision", "high"))

    include_suffixes = set(train_cfg.get("include_suffixes", [".jsonl", ".json", ".txt", ".md", ".csv"]))
    read_block_bytes = int(train_cfg.get("read_block_bytes", 1 << 20))
    data_mix_cfg = train_cfg.get("data_mix") or train_cfg.get("data_mixes") or []
    mix_component_specs: list[dict[str, Any]] = []
    files: list[Path] = []
    if data_mix_cfg:
        if not isinstance(data_mix_cfg, list):
            raise TypeError("training.data_mix must be a list of component objects")
        for index, component_cfg in enumerate(data_mix_cfg):
            if not isinstance(component_cfg, dict):
                raise TypeError("each training.data_mix component must be an object")
            component_name = str(component_cfg.get("name", f"component_{index}"))
            component_weight = float(component_cfg.get("weight", 1.0))
            component_include_suffixes = set(
                component_cfg.get("include_suffixes", include_suffixes)
            )
            component_roots = _resolve_config_paths(
                root,
                component_cfg.get("data_roots", []),
            )
            component_files = _collect_corpus_files(
                component_roots,
                component_include_suffixes,
            )
            if not component_files:
                raise RuntimeError(
                    "No corpus files found for data_mix component "
                    f"{component_name!r}"
                )
            mix_component_specs.append(
                {
                    "name": component_name,
                    "weight": component_weight,
                    "files": component_files,
                    "row_preserve_jsonl_examples": bool(
                        component_cfg.get(
                            "row_preserve_jsonl_examples",
                            train_cfg.get("row_preserve_jsonl_examples", False),
                        )
                    ),
                }
            )
            files.extend(component_files)
        _weighted_schedule(
            [component["weight"] for component in mix_component_specs]
        )
    else:
        data_roots = _resolve_config_paths(root, train_cfg.get("data_roots", []))
        files = _collect_corpus_files(data_roots, include_suffixes)
        if not files:
            raise RuntimeError("No corpus files found for configured data_roots/include_suffixes")

    data_source_summary: dict[str, Any]
    if mix_component_specs:
        data_source_summary = {
            "mode": "weighted_mix",
            "components": [
                {
                    "name": component["name"],
                    "weight": component["weight"],
                    "row_preserve_jsonl_examples": component[
                        "row_preserve_jsonl_examples"
                    ],
                    "file_count": len(component["files"]),
                    "files": [
                        _relative_path(root, path)
                        for path in component["files"][:8]
                    ],
                }
                for component in mix_component_specs
            ],
            "schedule": _weighted_schedule(
                [component["weight"] for component in mix_component_specs]
            ),
        }
    else:
        data_source_summary = {
            "mode": "single_stream",
            "file_count": len(files),
            "row_preserve_jsonl_examples": bool(
                train_cfg.get("row_preserve_jsonl_examples", False)
            ),
            "files": [_relative_path(root, path) for path in files[:8]],
        }

    logger.info("Using %d corpus files", len(files))
    eval_files: list[Path] = []
    if train_cfg.get("eval_data_roots"):
        eval_roots = [
            (root / Path(item)).resolve() if not Path(item).is_absolute() else Path(item)
            for item in train_cfg.get("eval_data_roots", [])
        ]
        eval_files = _collect_corpus_files(eval_roots, include_suffixes)
        if not eval_files:
            raise RuntimeError("No held-out eval corpus files found for eval_data_roots/include_suffixes")
        logger.info("Using %d held-out eval corpus files", len(eval_files))

    seq_len = int(train_cfg.get("seq_len", 2048))
    chunk_len = seq_len + 1
    micro_batch_size = int(train_cfg.get("micro_batch_size", 2))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 16))
    steps = int(train_cfg.get("steps", 100000))
    lr = float(train_cfg.get("lr", 1e-4))
    min_lr = float(train_cfg.get("min_lr", lr))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    lr_decay_steps = int(train_cfg.get("lr_decay_steps", 0))
    lr_step_offset = int(train_cfg.get("lr_step_offset", 0))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    patch_prediction_loss_weight = float(train_cfg.get("patch_prediction_loss_weight", 0.0))
    patch_prediction_answer_loss_weight = float(
        train_cfg.get("patch_prediction_answer_loss_weight", 1.0)
    )
    patch_prediction_answer_only_loss = bool(
        train_cfg.get("patch_prediction_answer_only_loss", False)
    )
    patch_prediction_copy_loss_weight = float(
        train_cfg.get("patch_prediction_copy_loss_weight", 0.0)
    )
    patch_prediction_answer_start_only = bool(
        train_cfg.get("patch_prediction_answer_start_only", False)
    )
    domain_cake_training_only = bool(
        train_cfg.get("domain_cake_training_only", False)
    )
    if domain_cake_training_only and patch_prediction_loss_weight <= 0.0:
        raise ValueError(
            "domain_cake_training_only requires patch_prediction_loss_weight > 0"
        )
    answer_loss_weight = float(train_cfg.get("answer_loss_weight", 1.0))
    answer_only_loss = bool(train_cfg.get("answer_only_loss", False))
    answer_marker = str(train_cfg.get("answer_marker", "Answer: ")).encode("utf-8")
    copy_loss_weight = float(train_cfg.get("copy_loss_weight", 0.0))
    span_loss_weight = float(train_cfg.get("span_loss_weight", 0.0))
    answer_aligned_span_loss = bool(train_cfg.get("answer_aligned_span_loss", False))
    log_interval = int(train_cfg.get("log_interval", 50))
    save_interval = int(train_cfg.get("save_interval", 1000))
    resume_from = train_cfg.get("resume_from")
    keep_last_n = int(train_cfg.get("keep_last_n", 2))
    save_optimizer = bool(train_cfg.get("save_optimizer", True))
    dataloader_workers = int(train_cfg.get("dataloader_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(
        train_cfg.get("persistent_workers", dataloader_workers > 0)
    )
    if dataloader_workers <= 0:
        persistent_workers = False
    compile_model = bool(train_cfg.get("compile_model", False))
    optimizer_fused = bool(train_cfg.get("optimizer_fused", device.type == "cuda"))
    cake_route_value = train_cfg.get("cake_route")
    cake_route = None if cake_route_value is None else int(cake_route_value)
    cake_sparse_optimizer = bool(train_cfg.get("cake_sparse_optimizer", False))
    cake_optimizer_include_router = bool(
        train_cfg.get("cake_optimizer_include_router", False)
    )
    if cake_sparse_optimizer and cake_route is None:
        raise ValueError("cake_sparse_optimizer requires an explicit cake_route")
    if cake_optimizer_include_router and not cake_sparse_optimizer:
        raise ValueError(
            "cake_optimizer_include_router requires cake_sparse_optimizer"
        )
    if cake_sparse_optimizer and compile_model:
        raise ValueError(
            "cake_sparse_optimizer currently requires compile_model=false"
        )
    throughput_guard = train_cfg.get("throughput_guard", {})
    metrics_path_name = train_cfg.get("metrics_path", "training_metrics.json")

    out_dir = (root / Path(train_cfg.get("out_dir", "runs_experiment/byte_core"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if mix_component_specs:
        components: list[MixedByteComponent] = []
        for component in mix_component_specs:
            if component["row_preserve_jsonl_examples"]:
                component_dataset: IterableDataset = JsonlRowByteDataset(
                    component["files"],
                    seq_len=chunk_len,
                    patch_size=int(model_cfg.get("patch_size", 1)),
                    answer_marker=answer_marker,
                    read_block_bytes=read_block_bytes,
                )
            else:
                component_dataset = ByteCorpusDataset(
                    component["files"],
                    seq_len=chunk_len,
                    read_block_bytes=read_block_bytes,
                )
            components.append(
                MixedByteComponent(
                    name=component["name"],
                    weight=component["weight"],
                    dataset=component_dataset,
                    files=component["files"],
                    row_preserve_jsonl_examples=component[
                        "row_preserve_jsonl_examples"
                    ],
                )
            )
        dataset = WeightedMixedByteDataset(components)
    elif bool(train_cfg.get("row_preserve_jsonl_examples", False)):
        dataset = JsonlRowByteDataset(
            files,
            seq_len=chunk_len,
            patch_size=int(model_cfg.get("patch_size", 1)),
            answer_marker=answer_marker,
            read_block_bytes=read_block_bytes,
        )
    else:
        dataset = ByteCorpusDataset(
            files,
            seq_len=chunk_len,
            read_block_bytes=read_block_bytes,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=dataloader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    pretrain_started = time.perf_counter()
    model = _build_model(model_cfg, device=device)
    prior_init: dict[str, Any] = {"status": "DISABLED"}
    if bool(train_cfg.get("initialize_byte_priors_from_corpus", False)):
        prior_init = _initialize_byte_priors_from_corpus(
            model,
            files,
            read_block_bytes=read_block_bytes,
            max_bytes=int(train_cfg.get("byte_prior_max_bytes", 4 * 1024 * 1024)),
            smoothing=float(train_cfg.get("byte_prior_smoothing", 0.25)),
        )
        logger.info("Initialized byte priors: %s", prior_init)
    domain_cache_init: dict[str, Any] = {"status": "DISABLED"}
    if bool(train_cfg.get("initialize_domain_cache_from_corpus", False)):
        domain_cache_init = _initialize_domain_cache_from_corpus(
            model,
            files,
            read_block_bytes=read_block_bytes,
            max_bytes=int(
                train_cfg.get(
                    "domain_cache_max_bytes",
                    train_cfg.get("byte_prior_max_bytes", 4 * 1024 * 1024),
                )
            ),
            order=int(model_cfg.get("domain_cache_order", 0)),
            smoothing=float(train_cfg.get("domain_cache_smoothing", 0.01)),
            max_entries=int(train_cfg.get("domain_cache_max_entries", 0)),
            min_confidence=float(train_cfg.get("domain_cache_min_confidence", 0.0)),
            high_confidence_boost=float(train_cfg.get("domain_cache_high_confidence_boost", 0.0)),
        )
        logger.info("Initialized domain cache: %s", domain_cache_init)
    counted_pretrain_seconds = time.perf_counter() - pretrain_started
    teacher_model = None
    distill_loss_weight = float(train_cfg.get("distill_loss_weight", 0.0))
    distill_interval = max(1, int(train_cfg.get("distill_interval", 4)))
    distill_until_step_ratio = float(train_cfg.get("distill_until_step_ratio", 0.25))
    teacher_local_decoder = train_cfg.get("teacher_local_decoder")
    if distill_loss_weight > 0.0 and teacher_local_decoder:
        teacher_cfg = dict(model_cfg)
        teacher_cfg["local_decoder"] = teacher_local_decoder
        teacher_cfg["patch_prediction"] = False
        teacher_model = _build_model(teacher_cfg, device=device)
        teacher_checkpoint = train_cfg.get("teacher_checkpoint")
        if teacher_checkpoint:
            teacher_path = Path(teacher_checkpoint)
            if not teacher_path.is_absolute():
                teacher_path = (root / teacher_path).resolve()
            ckpt = torch.load(teacher_path, map_location=device)
            teacher_model.load_state_dict(ckpt["model"], strict=True)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

    global_step = 0
    resume_optimizer_state = None
    resume_load: dict[str, Any] = {"status": "DISABLED"}
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = (root / resume_path).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)
        resume_strict = bool(train_cfg.get("resume_strict", True))
        mismatched_keys: list[str] = []
        if "model" in ckpt:
            resume_state = ckpt["model"]
            if bool(train_cfg.get("resume_ignore_shape_mismatch", False)):
                target_state = model.state_dict()
                mismatched_keys = [
                    key
                    for key, value in resume_state.items()
                    if key in target_state and target_state[key].shape != value.shape
                ]
                resume_state = {
                    key: value
                    for key, value in resume_state.items()
                    if key in target_state and target_state[key].shape == value.shape
                }
                resume_strict = False
            load_result = model.load_state_dict(resume_state, strict=resume_strict)
        elif "patch_model" in ckpt:
            load_result = model.load_state_dict(
                ckpt["patch_model"],
                strict=resume_strict,
            )
        else:
            raise KeyError("resume checkpoint must contain 'model' or 'patch_model'")
        resume_load = {
            "status": "LOADED",
            "path": str(resume_path),
            "strict": resume_strict,
            "missing_keys": list(getattr(load_result, "missing_keys", [])),
            "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
            "mismatched_keys": mismatched_keys,
        }
        resume_optimizer_state = ckpt.get("optimizer")
        global_step = int(ckpt.get("step", 0))
        logger.info("Resumed from %s at step=%d", resume_path, global_step)

    parameter_filter = _apply_parameter_training_filter(
        model,
        trainable_patterns=train_cfg.get("trainable_parameter_patterns"),
        frozen_patterns=train_cfg.get("frozen_parameter_patterns"),
    )
    if cake_route is not None:
        model.set_cake_route(cake_route)
    if cake_sparse_optimizer:
        optimizer_params = list(
            model.sparse_cake_parameters(
                cake_route,
                include_router=cake_optimizer_include_router,
            )
        )
    else:
        optimizer_params = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
    if compile_model:
        logger.info("Compiling model with torch.compile")
        model = torch.compile(
            model,
            mode=train_cfg.get("compile_mode", "default"),
            fullgraph=bool(train_cfg.get("compile_fullgraph", False)),
        )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_param_count = sum(parameter.numel() for parameter in optimizer_params)
    cake_routing = {
        "route": cake_route,
        "sparse_optimizer": cake_sparse_optimizer,
        "optimizer_includes_router": cake_optimizer_include_router,
        "optimizer_params": optimizer_param_count,
        "trainable_params": trainable_params,
        "optimizer_fraction_of_trainable": (
            optimizer_param_count / trainable_params if trainable_params else 0.0
        ),
    }
    logger.info(
        "Model params: %.3fM trainable; %.3fM in optimizer",
        trainable_params / 1e6,
        optimizer_param_count / 1e6,
    )

    optimizer_kwargs: dict[str, Any] = {
        "lr": lr,
        "betas": tuple(train_cfg.get("betas", [0.9, 0.95])),
        "weight_decay": weight_decay,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = optimizer_fused
    if not optimizer_params:
        raise RuntimeError("No trainable parameters are available for the optimizer")
    optimizer = torch.optim.AdamW(optimizer_params, **optimizer_kwargs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if resume_optimizer_state is not None:
        if parameter_filter["enabled"]:
            logger.warning(
                "Skipping optimizer state restore because parameter filters are active"
            )
        else:
            optimizer.load_state_dict(resume_optimizer_state)

    def _current_lr(step: int) -> float:
        phase_step = max(step - lr_step_offset, 0)
        if warmup_steps > 0 and phase_step < warmup_steps:
            return lr * float(phase_step + 1) / float(warmup_steps)
        if lr_decay_steps <= 0:
            return lr
        decay_pos = min(max(phase_step - warmup_steps, 0), lr_decay_steps)
        decay_ratio = decay_pos / max(lr_decay_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + (lr - min_lr) * cosine

    model.train()
    running_loss = 0.0
    running_count = 0
    running_data_seconds = 0.0
    running_train_seconds = 0.0
    running_opt_seconds = 0.0
    start = time.perf_counter()
    bytes_per_step = seq_len * micro_batch_size * grad_accum_steps
    last_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []

    iterator = iter(dataloader)

    while global_step < steps:
        step_started = time.perf_counter()
        current_lr = _current_lr(global_step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(grad_accum_steps):
            try:
                data_started = time.perf_counter()
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                data_started = time.perf_counter()
                batch = next(iterator)
            running_data_seconds += time.perf_counter() - data_started

            if batch.shape[0] < micro_batch_size:
                continue

            batch = batch.to(device, non_blocking=pin_memory)
            x = batch[:, :-1]
            y = batch[:, 1:]
            patch_context_indices = None
            if patch_prediction_answer_start_only:
                answer_starts = _answer_start_positions(
                    batch,
                    answer_marker=answer_marker,
                )
                source_patch_indices = answer_starts.div(
                    model.patch_size,
                    rounding_mode="floor",
                ) - 1
                patch_context_indices = source_patch_indices.div(
                    model.patch_prediction_stride,
                    rounding_mode="floor",
                ).clamp_min(0)

            train_started = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                wants_copy_aux = copy_loss_weight > 0.0 and bool(model_cfg.get("copy_transducer", False))
                wants_span_aux = (
                    span_loss_weight > 0.0
                    and model_cfg.get("local_decoder") == "span_patch_decoder"
                )
                next_patches = None
                if domain_cake_training_only:
                    patch_predictions, next_patches = (
                        model.domain_cake_patch_predictions(
                            x,
                            context_indices=patch_context_indices,
                        )
                    )
                    logits = None
                    auxiliary = []
                elif (
                    patch_prediction_loss_weight > 0.0
                    and bool(model_cfg.get("patch_prediction", False))
                ) or wants_copy_aux or wants_span_aux:
                    output = model(
                        x,
                        return_aux=True,
                        return_patch_prediction=True,
                        patch_prediction_context_indices=patch_context_indices,
                    )
                    logits = output[0]
                    auxiliary = output[2]
                    patch_predictions = output[3] if len(output) > 3 and bool(model_cfg.get("patch_prediction", False)) else None
                else:
                    logits, _ = model(x)
                    auxiliary = []
                    patch_predictions = None
                if domain_cake_training_only:
                    loss = x.new_zeros((), dtype=torch.float32)
                elif answer_loss_weight > 1.0 or answer_only_loss:
                    logits = logits[:, : y.shape[1], :]
                    token_loss = F.cross_entropy(
                        logits.flatten(0, 1),
                        y.flatten(),
                        reduction="none",
                    ).view_as(y)
                    answer_weights = _answer_span_weights(
                        batch,
                        target_len=y.shape[1],
                        answer_weight=answer_loss_weight,
                        base_weight=0.0 if answer_only_loss else 1.0,
                        answer_marker=answer_marker,
                    ).to(dtype=token_loss.dtype)
                    loss = (token_loss * answer_weights).sum() / answer_weights.sum().clamp_min(1.0)
                else:
                    logits = logits[:, : y.shape[1], :]
                    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                if patch_predictions is not None:
                    if next_patches is None:
                        next_patches = model.patch_prediction_targets(x)[
                            :, :: model.patch_prediction_stride
                        ]
                    if (
                        patch_context_indices is not None
                        and not domain_cake_training_only
                    ):
                        next_patches = next_patches.gather(
                            1,
                            patch_context_indices[:, None, None].expand(
                                -1,
                                1,
                                next_patches.shape[-1],
                            ),
                        )
                    prediction_tensor = torch.stack(patch_predictions, dim=2)
                    target_tensor = next_patches[
                        :, : prediction_tensor.shape[1], : prediction_tensor.shape[2]
                    ]
                    byte_offsets = torch.arange(
                        prediction_tensor.shape[2],
                        device=x.device,
                        dtype=torch.long,
                    )
                    if patch_context_indices is None:
                        patch_indices = (
                            torch.arange(
                                prediction_tensor.shape[1],
                                device=x.device,
                                dtype=torch.long,
                            )
                            * model.patch_prediction_stride
                        )
                        absolute_positions = (
                            (patch_indices[:, None] + 1) * model.patch_size
                            + byte_offsets[None, :]
                        )
                    else:
                        patch_indices = (
                            patch_context_indices * model.patch_prediction_stride
                        )
                        absolute_positions = (
                            (patch_indices[:, None, None] + 1) * model.patch_size
                            + byte_offsets[None, None, :]
                        )
                    valid_positions = absolute_positions < x.shape[1]
                    per_byte_patch_loss = F.cross_entropy(
                        prediction_tensor.reshape(-1, prediction_tensor.shape[-1]),
                        target_tensor.reshape(-1),
                        reduction="none",
                    ).view_as(target_tensor)
                    if (
                        patch_prediction_answer_loss_weight > 1.0
                        or patch_prediction_answer_only_loss
                    ):
                        if patch_context_indices is None:
                            patch_weights = _answer_position_weights(
                                batch,
                                absolute_positions.flatten().clamp(
                                    max=batch.shape[1] - 1
                                ),
                                answer_weight=patch_prediction_answer_loss_weight,
                                base_weight=0.0
                                if patch_prediction_answer_only_loss
                                else 1.0,
                                answer_marker=answer_marker,
                            ).view(
                                batch.shape[0],
                                prediction_tensor.shape[1],
                                prediction_tensor.shape[2],
                            )
                            patch_weights = (
                                patch_weights * valid_positions.unsqueeze(0)
                            )
                        else:
                            patch_weights = _answer_position_weights(
                                batch,
                                absolute_positions.clamp(
                                    max=batch.shape[1] - 1
                                ),
                                answer_weight=patch_prediction_answer_loss_weight,
                                base_weight=0.0
                                if patch_prediction_answer_only_loss
                                else 1.0,
                                answer_marker=answer_marker,
                            )
                            patch_weights = patch_weights * valid_positions
                        patch_weights = patch_weights.to(
                            dtype=prediction_tensor.dtype
                        )
                    else:
                        patch_weights = (
                            valid_positions.unsqueeze(0)
                            if patch_context_indices is None
                            else valid_positions
                        ).to(dtype=prediction_tensor.dtype)
                    offset_numerators = (
                        per_byte_patch_loss * patch_weights
                    ).sum(dim=(0, 1))
                    offset_denominators = patch_weights.sum(dim=(0, 1)).clamp_min(1.0)
                    patch_loss = (offset_numerators / offset_denominators).mean()
                    loss = loss + patch_prediction_loss_weight * patch_loss
                    patch_generator = getattr(model, "patch_generator", None)
                    copy_logits = getattr(
                        patch_generator,
                        "last_copy_logits",
                        None,
                    )
                    if (
                        patch_prediction_copy_loss_weight > 0.0
                        and copy_logits is not None
                    ):
                        copy_sources = model._patch_generation_copy_sources(x)[
                            :, :: model.patch_prediction_stride
                        ]
                        if patch_context_indices is not None:
                            copy_sources = copy_sources.gather(
                                1,
                                patch_context_indices[:, None, None].expand(
                                    -1,
                                    1,
                                    copy_sources.shape[-1],
                                ),
                            )
                        copy_source_ids = copy_sources
                        if bool(
                            getattr(
                                model.patch_generator,
                                "lowercase_copy",
                                False,
                            )
                        ):
                            copy_source_ids = torch.where(
                                (copy_source_ids >= ord("A"))
                                & (copy_source_ids <= ord("Z")),
                                copy_source_ids + (ord("a") - ord("A")),
                                copy_source_ids,
                            )
                        copyable = (
                            copy_source_ids.unsqueeze(2)
                            == target_tensor.unsqueeze(-1)
                        ).any(dim=-1)
                        copy_weights = patch_weights * copyable.to(
                            dtype=patch_weights.dtype
                        )
                        per_byte_copy_loss = F.cross_entropy(
                            copy_logits.reshape(-1, copy_logits.shape[-1]),
                            target_tensor.reshape(-1),
                            reduction="none",
                        ).view_as(target_tensor)
                        copy_numerators = (
                            per_byte_copy_loss * copy_weights
                        ).sum(dim=(0, 1))
                        raw_copy_denominators = copy_weights.sum(dim=(0, 1))
                        active_copy_offsets = raw_copy_denominators > 0
                        copy_losses = copy_numerators / raw_copy_denominators.clamp_min(1.0)
                        copy_loss = copy_losses[active_copy_offsets].mean()
                        loss = loss + (
                            patch_prediction_copy_loss_weight * copy_loss
                        )
                if copy_loss_weight > 0.0 and auxiliary:
                    copy_scores = next(
                        (item for item in reversed(auxiliary) if item.ndim == 3),
                        None,
                    )
                    if copy_scores is not None:
                        source_end_positions = None
                        if answer_only_loss or answer_loss_weight > 1.0:
                            marker_starts: list[int] = []
                            for values in batch.detach().cpu().tolist():
                                payload = bytes(int(v) for v in values)
                                marker_starts.append(payload.find(answer_marker))
                            if all(position >= 0 for position in marker_starts):
                                source_end_positions = torch.tensor(
                                    marker_starts,
                                    device=x.device,
                                    dtype=torch.long,
                                )
                        copy_labels = _copy_alignment_labels(
                            x,
                            y,
                            source_len=copy_scores.shape[-1],
                            source_end_positions=source_end_positions,
                        )
                        copy_loss = F.cross_entropy(
                            copy_scores.reshape(-1, copy_scores.shape[-1]),
                            copy_labels.reshape(-1),
                            ignore_index=-100,
                        )
                        if torch.isfinite(copy_loss):
                            loss = loss + copy_loss_weight * copy_loss
                if span_loss_weight > 0.0 and hasattr(model, "span_width"):
                    span = max(int(getattr(model, "span_width", 1)), 1)
                    span_future_logits = next(
                        (
                            item
                            for item in auxiliary
                            if item.ndim == 4
                            and item.shape[2] == span
                            and item.shape[-1] == logits.shape[-1]
                        ),
                        None,
                    )
                    if span > 1 and span_future_logits is not None and answer_aligned_span_loss:
                        span_copy_scores = next(
                            (
                                item
                                for item in auxiliary
                                if item.ndim == 4
                                and item.shape[2] == span
                                and item.shape[-1] != logits.shape[-1]
                            ),
                            None,
                        )
                        rows_cpu = batch.detach().cpu().tolist()
                        marker = answer_marker
                        terminator = b"\n###"
                        selected_logits: list[torch.Tensor] = []
                        selected_targets: list[torch.Tensor] = []
                        selected_weights: list[torch.Tensor] = []
                        selected_copy_scores: list[torch.Tensor] = []
                        selected_abs_positions: list[torch.Tensor] = []
                        selected_source_rows: list[torch.Tensor] = []
                        selected_source_end_positions: list[torch.Tensor] = []
                        for row_index, values in enumerate(rows_cpu):
                            payload = bytes(int(v) for v in values)
                            marker_start = payload.find(marker)
                            if marker_start < 0:
                                continue
                            answer_start = marker_start + len(marker)
                            answer_end = payload.find(terminator, answer_start)
                            if answer_end < 0:
                                answer_end = min(answer_start + span, x.shape[1])
                            patch_index = answer_start // model.patch_size - 1
                            if patch_index < 0 or patch_index >= span_future_logits.shape[1]:
                                continue
                            target_start = (patch_index + 1) * model.patch_size
                            if target_start + span > x.shape[1]:
                                continue
                            absolute_positions = torch.arange(
                                target_start,
                                target_start + span,
                                device=x.device,
                            )
                            weights = (
                                (absolute_positions >= answer_start)
                                & (absolute_positions < min(answer_end, x.shape[1]))
                            ).to(dtype=span_future_logits.dtype)
                            if not bool(weights.any()):
                                continue
                            selected_logits.append(span_future_logits[row_index, patch_index])
                            selected_targets.append(x[row_index, target_start : target_start + span])
                            selected_weights.append(weights)
                            if span_copy_scores is not None:
                                selected_copy_scores.append(
                                    span_copy_scores[row_index, patch_index]
                                )
                                selected_abs_positions.append(absolute_positions)
                                selected_source_rows.append(x[row_index])
                                selected_source_end_positions.append(
                                    torch.tensor(marker_start, device=x.device)
                                )
                        if selected_logits:
                            span_logits = torch.stack(selected_logits, dim=0)
                            span_targets = torch.stack(selected_targets, dim=0)
                            span_weights = torch.stack(selected_weights, dim=0)
                            per_byte_span_loss = F.cross_entropy(
                                span_logits.reshape(-1, span_logits.shape[-1]),
                                span_targets.reshape(-1),
                                reduction="none",
                            ).view_as(span_targets)
                            span_loss = (
                                per_byte_span_loss * span_weights
                            ).sum() / span_weights.sum().clamp_min(1.0)
                            loss = loss + span_loss_weight * span_loss
                            if (
                                copy_loss_weight > 0.0
                                and selected_copy_scores
                                and selected_abs_positions
                            ):
                                copy_score_tensor = torch.stack(
                                    selected_copy_scores,
                                    dim=0,
                                )
                                abs_position_tensor = torch.stack(
                                    selected_abs_positions,
                                    dim=0,
                                )
                                source_rows = torch.stack(
                                    selected_source_rows,
                                    dim=0,
                                )
                                source_end_positions = torch.stack(
                                    selected_source_end_positions,
                                    dim=0,
                                )
                                copy_labels = _copy_alignment_labels_at_positions(
                                    source_rows,
                                    span_targets,
                                    abs_position_tensor,
                                    source_len=copy_score_tensor.shape[-1],
                                    source_end_positions=source_end_positions,
                                )
                                span_copy_loss = F.cross_entropy(
                                    copy_score_tensor.reshape(
                                        -1,
                                        copy_score_tensor.shape[-1],
                                    ),
                                    copy_labels.reshape(-1),
                                    ignore_index=-100,
                                    reduction="none",
                                ).view_as(span_targets)
                                span_copy_loss = (
                                    span_copy_loss * span_weights
                                ).sum() / span_weights.sum().clamp_min(1.0)
                                if torch.isfinite(span_copy_loss):
                                    loss = loss + copy_loss_weight * span_copy_loss
                    elif span > 1 and span_future_logits is not None:
                        usable = x.shape[1] // model.patch_size * model.patch_size
                        starts = (
                            (torch.arange(span_future_logits.shape[1], device=x.device) + 1)
                            * model.patch_size
                        )
                        valid = starts + span <= usable
                        if valid.any():
                            valid_starts = starts[valid]
                            target_rows = [
                                x[:, int(start.item()) : int(start.item()) + span]
                                for start in valid_starts
                            ]
                            span_targets = torch.stack(target_rows, dim=1)
                            span_logits = span_future_logits[:, : span_targets.shape[1]]
                            if answer_only_loss:
                                answer_weights_for_span = _answer_span_weights(
                                    batch,
                                    target_len=y.shape[1],
                                    answer_weight=1.0,
                                    base_weight=0.0,
                                    answer_marker=answer_marker,
                                ).to(dtype=span_logits.dtype)
                                absolute_positions = (
                                    valid_starts[:, None]
                                    + torch.arange(span, device=x.device)[None, :]
                                )
                                target_positions = (absolute_positions - 1).clamp(
                                    min=0,
                                    max=y.shape[1] - 1,
                                )
                                span_weights = answer_weights_for_span[
                                    :,
                                    target_positions,
                                ]
                                per_byte_span_loss = F.cross_entropy(
                                    span_logits.reshape(-1, span_logits.shape[-1]),
                                    span_targets.reshape(-1),
                                    reduction="none",
                                ).view_as(span_targets)
                                span_loss = (
                                    per_byte_span_loss * span_weights
                                ).sum() / span_weights.sum().clamp_min(1.0)
                            else:
                                span_loss = F.cross_entropy(
                                    span_logits.reshape(-1, span_logits.shape[-1]),
                                    span_targets.reshape(-1),
                                )
                            loss = loss + span_loss_weight * span_loss
                distill_active = (
                    teacher_model is not None
                    and global_step % distill_interval == 0
                    and global_step < int(steps * distill_until_step_ratio)
                )
                if distill_active:
                    with torch.no_grad():
                        teacher_logits, _ = teacher_model(x)
                    teacher_logits = teacher_logits[:, : logits.shape[1], :]
                    distill_loss = F.kl_div(
                        F.log_softmax(logits.float(), dim=-1),
                        F.softmax(teacher_logits.float(), dim=-1),
                        reduction="batchmean",
                    ) / max(logits.shape[1], 1)
                    loss = loss + distill_loss_weight * distill_loss
                loss = loss / grad_accum_steps
                if not torch.isfinite(loss.detach()):
                    raise RuntimeError(
                        f"Non-finite loss at step {global_step + 1}; "
                        "aborting run before writing a promoted checkpoint."
                    )

            scaler.scale(loss).backward()
            running_train_seconds += time.perf_counter() - train_started
            step_loss += float(loss.item())

        opt_started = time.perf_counter()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(optimizer_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        running_opt_seconds += time.perf_counter() - opt_started

        global_step += 1
        running_loss += step_loss
        running_count += 1

        if global_step % log_interval == 0 or global_step == 1:
            mean_loss = running_loss / max(running_count, 1)
            elapsed = max(time.perf_counter() - start + counted_pretrain_seconds, 1e-6)
            steps_per_sec = global_step / elapsed
            bpb = mean_loss / math.log(2)
            gib_per_hour = (bytes_per_step * steps_per_sec * 3600.0) / (1024.0**3)
            projected_total_hours = (
                steps / max(steps_per_sec, 1e-12)
            ) / 3600.0
            remaining_hours = (
                (steps - global_step) / max(steps_per_sec, 1e-12)
            ) / 3600.0
            interval_count = max(running_count, 1)
            metrics = {
                "step": global_step,
                "steps": steps,
                "loss": mean_loss,
                "bpb": bpb,
                "lr": current_lr,
                "elapsed_seconds": elapsed,
                "steps_per_second": steps_per_sec,
                "bytes_per_step": bytes_per_step,
                "train_bytes": bytes_per_step * global_step,
                "gib_per_hour": gib_per_hour,
                "projected_total_hours": projected_total_hours,
                "remaining_hours": remaining_hours,
                "data_seconds_per_step": running_data_seconds / interval_count,
                "forward_backward_seconds_per_step": (
                    running_train_seconds / interval_count
                ),
                "optimizer_seconds_per_step": running_opt_seconds / interval_count,
                "trainable_params": trainable_params,
                "optimizer_params": optimizer_param_count,
                "cake_routing": cake_routing,
                "counted_pretrain_seconds": counted_pretrain_seconds,
                "prior_initialization": prior_init,
                "domain_cache_initialization": domain_cache_init,
                "parameter_filter": parameter_filter,
                "resume_load": resume_load,
            }
            last_metrics = metrics
            history.append(metrics)
            logger.info(
                (
                    "step=%d/%d loss=%.5f bpb=%.5f steps_per_sec=%.3f "
                    "lr=%.6g gib_per_hour=%.2f eta_h=%.2f "
                    "data/train/opt_ms=%.1f/%.1f/%.1f"
                ),
                global_step,
                steps,
                mean_loss,
                bpb,
                steps_per_sec,
                current_lr,
                gib_per_hour,
                remaining_hours,
                metrics["data_seconds_per_step"] * 1000.0,
                metrics["forward_backward_seconds_per_step"] * 1000.0,
                metrics["optimizer_seconds_per_step"] * 1000.0,
            )
            metrics_output = {
                "status": "RUNNING",
                "config_name": config.get("name"),
                "device": str(device),
                "model_config": model_cfg,
                "training_config": train_cfg,
                "data_source_summary": data_source_summary,
                "parameter_filter": parameter_filter,
                "cake_routing": cake_routing,
                "resume_load": resume_load,
                "prior_initialization": prior_init,
                "domain_cache_initialization": domain_cache_init,
                "latest": metrics,
                "history": history[-200:],
            }
            (out_dir / metrics_path_name).write_text(
                json.dumps(metrics_output, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            guard_warmup = int(throughput_guard.get("warmup_steps", log_interval))
            if throughput_guard and global_step >= guard_warmup:
                failed_guard = []
                min_gib_per_hour = throughput_guard.get("min_gib_per_hour")
                max_projected_hours = throughput_guard.get("max_projected_hours")
                if (
                    min_gib_per_hour is not None
                    and gib_per_hour < float(min_gib_per_hour)
                ):
                    failed_guard.append("min_gib_per_hour")
                if (
                    max_projected_hours is not None
                    and projected_total_hours > float(max_projected_hours)
                ):
                    failed_guard.append("max_projected_hours")
                if failed_guard:
                    guard_result = {
                        **metrics_output,
                        "status": "ABORTED_THROUGHPUT_GUARD",
                        "failed_guard": failed_guard,
                        "guard": throughput_guard,
                    }
                    (out_dir / metrics_path_name).write_text(
                        json.dumps(guard_result, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    message = (
                        "Throughput guard failed at step "
                        f"{global_step}: {failed_guard}. "
                        f"Projected total hours={projected_total_hours:.2f}, "
                        f"GiB/hour={gib_per_hour:.2f}."
                    )
                    if throughput_guard.get("abort_on_fail", True):
                        raise RuntimeError(message)
                    logger.warning(message)
            running_loss = 0.0
            running_count = 0
            running_data_seconds = 0.0
            running_train_seconds = 0.0
            running_opt_seconds = 0.0

        if global_step % save_interval == 0 or global_step == steps:
            ckpt = {
                "step": global_step,
                "model": model.state_dict(),
                "model_config": model_cfg,
                "train_config": train_cfg,
                "parameter_filter": parameter_filter,
                "resume_load": resume_load,
                "trainable_params": trainable_params,
                "optimizer_params": optimizer_param_count,
                "cake_routing": cake_routing,
                "pid": os.getpid(),
            }
            if save_optimizer:
                ckpt["optimizer"] = optimizer.state_dict()
            path = out_dir / f"step_{global_step}.pt"
            torch.save(ckpt, path)
            torch.save(ckpt, out_dir / "latest.pt")
            logger.info("Saved checkpoint: %s", path)

            if keep_last_n > 0:
                checkpoint_history = sorted(
                    out_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime
                )
                while len(checkpoint_history) > keep_last_n:
                    victim = checkpoint_history.pop(0)
                    if victim.exists():
                        victim.unlink()
                        logger.info("Removed old checkpoint: %s", victim)

    if eval_files:
        eval_stream = _load_eval_byte_stream(
            eval_files,
            max_bytes=int(train_cfg.get("eval_bytes", 100_000)),
            read_block_bytes=read_block_bytes,
        )
        eval_bpb = _eval_byte_bpb(
            model,
            eval_stream,
            seq_len=seq_len,
            batch_size=micro_batch_size,
            batches=int(train_cfg.get("eval_batches", 8)),
            seed=int(train_cfg.get("eval_seed", seed + 1)),
            device=device,
        )
        last_metrics = {
            **last_metrics,
            "eval_bpb": eval_bpb,
            "eval_bytes": int(eval_stream.numel()),
        }

    final_output = {
        "status": "COMPLETE",
        "config_name": config.get("name"),
        "device": str(device),
        "model_config": model_cfg,
        "training_config": train_cfg,
        "data_source_summary": data_source_summary,
        "parameter_filter": parameter_filter,
        "cake_routing": cake_routing,
        "resume_load": resume_load,
        "prior_initialization": prior_init,
        "domain_cache_initialization": domain_cache_init,
        "latest": last_metrics,
        "history": history[-200:],
    }
    (out_dir / metrics_path_name).write_text(
        json.dumps(final_output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Training complete. Output: %s", out_dir)


def _merge_config_dicts(base: dict, override: dict) -> dict:
    combined = dict(base)
    for key, value in override.items():
        if (
            key in combined
            and isinstance(combined[key], dict)
            and isinstance(value, dict)
        ):
            combined[key] = _merge_config_dicts(combined[key], value)
        else:
            combined[key] = value
    return combined


def _load_config_with_extends(
    config_path: Path,
    *,
    seen: set[Path] | None = None,
) -> dict:
    config_path = config_path.resolve()
    seen = set() if seen is None else set(seen)
    if config_path in seen:
        raise ValueError(f"Cyclic config inheritance at {config_path}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    extends = config.pop("extends", None)
    if extends is None:
        return config
    base_path = (config_path.parent / str(extends)).resolve()
    base_config = _load_config_with_extends(base_path, seen=seen)
    return _merge_config_dicts(base_config, config)


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer-free LayerCake byte core from JSON config")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parents[1] / config_path).resolve()

    config = _load_config_with_extends(config_path)

    _train(config)


if __name__ == "__main__":
    main()
