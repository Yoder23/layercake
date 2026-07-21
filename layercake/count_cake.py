"""Pruned backoff byte cakes and a hierarchical recurrent LayerCake host.

The count cake learns normalized byte distributions by counting a real corpus.
Its sparse state is capped explicitly, so a comparison can match learned-state
entries instead of hiding an unbounded language table behind a neural model.
The optional hierarchical host adds patch-scale recurrent context and composes
that context through the portable LayerCake ABI before predicting bytes.
"""

from __future__ import annotations

import math
import json
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_BACKOFF_STRENGTHS = (128.0, 32.0, 64.0) + (32.0,) * 29
DEFAULT_ONLINE_CACHE_SPECS = ((5, 2.0), (3, 8.0))
DEFAULT_CACHE_WINDOW = 768
DEFAULT_RECENT_CACHE_SPECS = (
    (24, 0.6024011669848757),
    (16, 2.571387785013007),
    (12, 4.801966956201964),
    (10, 5.461094199851411),
)
DEFAULT_NORMALIZED_CACHE_SPECS = (
    (5, 18.735277374458377),
    (3, 29.653365201856612),
)
HASH_MASK = (1 << 63) - 1
ASCII_CASEFOLD_TABLE = bytes.maketrans(
    bytes(range(256)),
    bytes([value + 32 if 65 <= value <= 90 else value for value in range(256)]),
)
ASCII_CLASS_TABLE = bytes.maketrans(
    bytes(range(256)),
    bytes(
        [
            value + 32
            if 65 <= value <= 90
            else 48
            if 48 <= value <= 57
            else 32
            if value in (9, 10, 13, 32)
            else value
            for value in range(256)
        ]
    ),
)


class CausalOnlineByteCache:
    """Exact document-local byte counts mixed into a normalized base model.

    The cache starts empty for every independent row/document and is updated
    only after scoring or emitting a byte.  Each ``(order, strength)`` stage
    applies ``(counts + strength * base) / (total + strength)``.  Cascading
    stages therefore remains exactly normalized while using no future bytes.
    """

    def __init__(
        self,
        specs: Iterable[tuple[int, float]] = DEFAULT_ONLINE_CACHE_SPECS,
        window: int | None = None,
        normalization: str = "exact",
    ) -> None:
        parsed = tuple((int(order), float(strength)) for order, strength in specs)
        if not parsed:
            raise ValueError("online cache requires at least one stage")
        if any(order <= 0 or strength <= 0 for order, strength in parsed):
            raise ValueError("online cache orders and strengths must be positive")
        self.specs = parsed
        self.max_order = max(order for order, _ in parsed)
        self.window = None if window is None else int(window)
        if self.window is not None and self.window <= 0:
            raise ValueError("online cache window must be positive")
        if normalization not in {"exact", "casefold", "classes"}:
            raise ValueError("unsupported online cache normalization")
        self.normalization = normalization
        self._counts: list[dict[bytes, dict[int, int]]] = [
            {} for _ in parsed
        ]
        self._events = [deque() for _ in parsed]

    def _context(self, history: bytes | bytearray, order: int) -> bytes:
        context = bytes(history[-order:])
        if self.normalization == "casefold":
            return context.translate(ASCII_CASEFOLD_TABLE)
        if self.normalization == "classes":
            return context.translate(ASCII_CLASS_TABLE)
        return context

    def prefill(self, payload: bytes | bytearray) -> bytearray:
        """Observe a prompt from left to right and return mutable history."""
        history = bytearray()
        for target in payload:
            self.update(history, int(target))
            history.append(int(target))
        return history

    def update(self, history: bytes | bytearray, target: int) -> None:
        target = int(target)
        if target < 0 or target > 255:
            raise ValueError("online cache targets must be bytes")
        for stage, events, (order, _) in zip(
            self._counts, self._events, self.specs
        ):
            if len(history) < order:
                continue
            context = self._context(history, order)
            continuations = stage.setdefault(context, {})
            continuations[target] = continuations.get(target, 0) + 1
            events.append((context, target))
            if self.window is not None and len(events) > self.window:
                old_context, old_target = events.popleft()
                old_continuations = stage[old_context]
                remaining = old_continuations[old_target] - 1
                if remaining:
                    old_continuations[old_target] = remaining
                else:
                    del old_continuations[old_target]
                if not old_continuations:
                    del stage[old_context]

    def observed_probability(
        self,
        base_probability: float,
        history: bytes | bytearray,
        target: int,
    ) -> float:
        probability = float(base_probability)
        target = int(target)
        for stage, (order, strength) in zip(self._counts, self.specs):
            if len(history) < order:
                continue
            continuations = stage.get(self._context(history, order))
            if not continuations:
                continue
            total = sum(continuations.values())
            probability = (
                continuations.get(target, 0) + strength * probability
            ) / (total + strength)
        return probability

    def probabilities(
        self,
        base_probability: torch.Tensor,
        history: bytes | bytearray,
    ) -> torch.Tensor:
        if base_probability.shape != (256,):
            raise ValueError("base probability must contain 256 bytes")
        probability = base_probability
        for stage, (order, strength) in zip(self._counts, self.specs):
            if len(history) < order:
                continue
            continuations = stage.get(self._context(history, order))
            if not continuations:
                continue
            counts = torch.zeros_like(probability)
            indices = torch.tensor(
                tuple(continuations),
                device=probability.device,
                dtype=torch.long,
            )
            values = torch.tensor(
                tuple(continuations.values()),
                device=probability.device,
                dtype=probability.dtype,
            )
            counts.scatter_(0, indices, values)
            probability = (counts + strength * probability) / (
                float(values.sum()) + strength
            )
        return probability / probability.sum()

    def probabilities_numpy(
        self,
        base_probability: np.ndarray,
        history: bytes | bytearray,
    ) -> np.ndarray:
        """NumPy equivalent used by the allocation-conscious CPU decoder."""
        probability = np.asarray(base_probability).copy()
        if probability.shape != (256,):
            raise ValueError("base probability must contain 256 bytes")
        for stage, (order, strength) in zip(self._counts, self.specs):
            if len(history) < order:
                continue
            continuations = stage.get(self._context(history, order))
            if not continuations:
                continue
            total = sum(continuations.values())
            probability *= strength / (total + strength)
            for target, count in continuations.items():
                probability[target] += count / (total + strength)
        return probability / probability.sum()


class CausalRecentByteCache:
    """Bounded last-continuation predictor for repeated byte contexts."""

    def __init__(
        self,
        specs: Iterable[tuple[int, float]],
        *,
        window: int,
    ) -> None:
        self.specs = tuple((int(order), float(strength)) for order, strength in specs)
        if not self.specs or window <= 0:
            raise ValueError("recent cache requires specs and a positive window")
        self.window = int(window)
        self.position = 0
        self._recent: list[dict[bytes, tuple[int, int]]] = [
            {} for _ in self.specs
        ]

    def observed_probability(
        self,
        base_probability: float,
        history: bytes | bytearray,
        target: int,
    ) -> float:
        probability = float(base_probability)
        for recent, (order, strength) in zip(self._recent, self.specs):
            if len(history) < order:
                continue
            match = recent.get(bytes(history[-order:]))
            if match is None or self.position - match[1] > self.window:
                continue
            probability = (
                float(match[0] == int(target)) + strength * probability
            ) / (1.0 + strength)
        return probability

    def probabilities(
        self,
        base_probability: torch.Tensor,
        history: bytes | bytearray,
    ) -> torch.Tensor:
        if base_probability.shape != (256,):
            raise ValueError("base probability must contain 256 bytes")
        probability = base_probability
        for recent, (order, strength) in zip(self._recent, self.specs):
            if len(history) < order:
                continue
            match = recent.get(bytes(history[-order:]))
            if match is None or self.position - match[1] > self.window:
                continue
            probability = probability * (strength / (1.0 + strength))
            probability = probability.clone()
            probability[int(match[0])] += 1.0 / (1.0 + strength)
        return probability / probability.sum()

    def probabilities_numpy(
        self,
        base_probability: np.ndarray,
        history: bytes | bytearray,
    ) -> np.ndarray:
        probability = np.asarray(base_probability).copy()
        if probability.shape != (256,):
            raise ValueError("base probability must contain 256 bytes")
        for recent, (order, strength) in zip(self._recent, self.specs):
            if len(history) < order:
                continue
            match = recent.get(bytes(history[-order:]))
            if match is None or self.position - match[1] > self.window:
                continue
            probability *= strength / (1.0 + strength)
            probability[int(match[0])] += 1.0 / (1.0 + strength)
        return probability / probability.sum()

    def update(self, history: bytes | bytearray, target: int) -> None:
        for recent, (order, _) in zip(self._recent, self.specs):
            if len(history) >= order:
                recent[bytes(history[-order:])] = (int(target), self.position)
        self.position += 1


class CausalCompositeByteCache:
    """One bounded causal decoder memory shared by evaluation and generation."""

    def __init__(
        self,
        *,
        exact_specs: Iterable[tuple[int, float]] = (),
        recent_specs: Iterable[tuple[int, float]] = (),
        normalized_specs: Iterable[tuple[int, float]] = (),
        window: int | None = DEFAULT_CACHE_WINDOW,
        normalization: str = "classes",
    ) -> None:
        exact_specs = tuple(exact_specs)
        recent_specs = tuple(recent_specs)
        normalized_specs = tuple(normalized_specs)
        if not (exact_specs or recent_specs or normalized_specs):
            raise ValueError("composite cache requires at least one stage")
        self.window = None if window is None else int(window)
        if self.window is not None and self.window <= 0:
            raise ValueError("composite cache window must be positive")
        if recent_specs and self.window is None:
            raise ValueError("recent cache requires a bounded window")
        self.exact = (
            CausalOnlineByteCache(exact_specs, window=self.window)
            if exact_specs
            else None
        )
        self.recent = (
            CausalRecentByteCache(recent_specs, window=int(self.window))
            if recent_specs
            else None
        )
        self.normalized = (
            CausalOnlineByteCache(
                normalized_specs,
                window=self.window,
                normalization=normalization,
            )
            if normalized_specs
            else None
        )
        self.max_order = max(
            order
            for specs in (exact_specs, recent_specs, normalized_specs)
            for order, _ in specs
        )

    def prefill(self, payload: bytes | bytearray) -> bytearray:
        history = bytearray()
        for target in payload:
            self.update(history, int(target))
            history.append(int(target))
            if len(history) > self.max_order:
                del history[: len(history) - self.max_order]
        return history

    def observed_probability(
        self,
        base_probability: float,
        history: bytes | bytearray,
        target: int,
    ) -> float:
        probability = float(base_probability)
        if self.exact is not None:
            probability = self.exact.observed_probability(
                probability, history, target
            )
        if self.recent is not None:
            probability = self.recent.observed_probability(
                probability, history, target
            )
        if self.normalized is not None:
            probability = self.normalized.observed_probability(
                probability, history, target
            )
        return probability

    def probabilities(
        self,
        base_probability: torch.Tensor,
        history: bytes | bytearray,
    ) -> torch.Tensor:
        probability = base_probability
        if self.exact is not None:
            probability = self.exact.probabilities(probability, history)
        if self.recent is not None:
            probability = self.recent.probabilities(probability, history)
        if self.normalized is not None:
            probability = self.normalized.probabilities(probability, history)
        return probability / probability.sum()

    def probabilities_numpy(
        self,
        base_probability: np.ndarray,
        history: bytes | bytearray,
    ) -> np.ndarray:
        probability = np.asarray(base_probability)
        if self.exact is not None:
            probability = self.exact.probabilities_numpy(probability, history)
        if self.recent is not None:
            probability = self.recent.probabilities_numpy(probability, history)
        if self.normalized is not None:
            probability = self.normalized.probabilities_numpy(probability, history)
        return probability / probability.sum()

    def update(self, history: bytes | bytearray, target: int) -> None:
        if self.exact is not None:
            self.exact.update(history, target)
        if self.recent is not None:
            self.recent.update(history, target)
        if self.normalized is not None:
            self.normalized.update(history, target)

def apply_causal_online_cache_to_observed(
    base_probabilities: np.ndarray,
    rows: np.ndarray,
    *,
    start: int,
    specs: Iterable[tuple[int, float]] = DEFAULT_ONLINE_CACHE_SPECS,
    reset_each_row: bool = True,
    window: int | None = None,
    recent_specs: Iterable[tuple[int, float]] = (),
    normalized_specs: Iterable[tuple[int, float]] = (),
    normalization: str = "casefold",
) -> np.ndarray:
    """Apply a fresh, strictly-causal cache to each row's observed targets."""
    base = np.asarray(base_probabilities, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.uint8)
    if rows.ndim != 2 or base.shape != (rows.shape[0], rows.shape[1] - start):
        raise ValueError("base probabilities and rows do not align")
    result = np.empty_like(base)
    cache = None
    history = None
    for row_index, row in enumerate(rows):
        if reset_each_row or cache is None:
            cache = CausalCompositeByteCache(
                exact_specs=specs,
                recent_specs=recent_specs,
                normalized_specs=normalized_specs,
                window=window,
                normalization=normalization,
            )
            history = bytearray()
        # Prefill uses only bytes preceding the first scored target.  In
        # stream scope these are the next contiguous bytes, not a reset.
        for target in row[:start]:
            cache.update(history, int(target))
            history.append(int(target))
        for offset, target in enumerate(row[start:]):
            result[row_index, offset] = cache.observed_probability(
                base[row_index, offset], history, int(target)
            )
            cache.update(history, int(target))
            history.append(int(target))
    return result


@dataclass(frozen=True)
class CountCakeTrainingSummary:
    state_budget: int
    state_entries: int
    max_order: int
    trained_order: int
    order_entries: tuple[int, ...]
    corpus_bytes: int


def _parallel_affine_scan(
    decay: torch.Tensor,
    injection: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``state = decay * state + injection`` with a stable scan.

    A cumulative-product/division formulation loses the early state once the
    product underflows (and the former 1e-12 clamp did so much sooner).  Affine
    recurrence pairs compose associatively: ``(a, b) o (c, d)`` is
    ``(a*c, b + a*d)``.  Hillis-Steele doubling therefore needs only
    ``ceil(log2(sequence_length))`` parallel compositions, never divides by a
    small product, and remains an ordinary differentiable PyTorch graph.
    """
    if decay.shape != injection.shape or decay.ndim < 2:
        raise ValueError("decay and injection must share a sequence tensor shape")
    coefficients = decay
    offsets = injection
    distance = 1
    length = decay.shape[1]
    while distance < length:
        identity_coefficients = torch.ones_like(coefficients[:, :distance])
        identity_offsets = torch.zeros_like(offsets[:, :distance])
        preceding_coefficients = torch.cat(
            [identity_coefficients, coefficients[:, :-distance]], dim=1
        )
        preceding_offsets = torch.cat(
            [identity_offsets, offsets[:, :-distance]], dim=1
        )
        offsets = offsets + coefficients * preceding_offsets
        coefficients = coefficients * preceding_coefficients
        distance *= 2
    return offsets


class ParallelSelectiveScanBlock(nn.Module):
    """Causal input-dependent decay scan evaluated in parallel over patches."""

    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = max(64, round((4 * width / 3) / 64) * 64)
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=3, groups=width, bias=False
        )
        self.selective = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width, bias=False)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_gate = nn.Linear(width, hidden, bias=False)
        self.ffn_value = nn.Linear(width, hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, width, bias=False)
        # Start with a genuinely long-lived patch state.  The previous default
        # linear bias placed decay near 0.52, so information lost half its
        # magnitude every patch and the nominal global core behaved locally.
        # A 0.95 initial decay retains about 18% across 32 patches while still
        # leaving the input-dependent decay free to learn shorter time scales.
        initial_decay = 0.95
        raw_decay = math.log(
            ((initial_decay - 0.05) / 0.945)
            / (1.0 - (initial_decay - 0.05) / 0.945)
        )
        with torch.no_grad():
            self.selective.bias[2 * width :].fill_(raw_decay)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        filtered = self.depthwise(
            F.pad(self.norm(hidden).transpose(1, 2), (2, 0))
        ).transpose(1, 2)
        gate_raw, proposal_raw, decay_raw = self.selective(filtered).chunk(
            3, dim=-1
        )
        # A nonzero floor prevents exact resets; the upper margin preserves a
        # usable gradient for long-lived state.  The associative affine scan
        # remains stable even when this core runs directly over raw bytes.
        decay = 0.05 + 0.945 * torch.sigmoid(decay_raw.float())
        proposal = torch.tanh(proposal_raw.float())
        state = _parallel_affine_scan(
            decay, (1.0 - decay) * proposal
        )
        scanned = torch.sigmoid(gate_raw.float()) * state
        hidden = residual + self.out(scanned.to(dtype=hidden.dtype))
        normalized = self.ffn_norm(hidden)
        return hidden + self.ffn_out(
            F.silu(self.ffn_gate(normalized)) * self.ffn_value(normalized)
        )


class ParallelSelectivePatchCore(nn.Module):
    """Stacked selective scans with the GRU-compatible forward signature."""

    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ParallelSelectiveScanBlock(width) for _ in range(layers)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if recurrent_state is not None:
            raise NotImplementedError(
                "incremental selective-scan state is pending architecture promotion"
            )
        for block in self.blocks:
            hidden = block(hidden)
        return hidden, None


class LowRankSelectiveScanBlock(nn.Module):
    """Wide elementwise state with low-rank learned channel mixing."""

    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        if rank <= 0 or rank > width:
            raise ValueError("selective rank must be in [1, width]")
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=3, groups=width, bias=False
        )
        self.feature_down = nn.Linear(width, rank, bias=False)
        self.selective = nn.Linear(rank, width * 3)
        self.output_down = nn.Linear(width, rank, bias=False)
        self.output_up = nn.Linear(rank, width, bias=False)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_gate = nn.Linear(width, rank, bias=False)
        self.ffn_value = nn.Linear(width, rank, bias=False)
        self.ffn_out = nn.Linear(rank, width, bias=False)
        initial_decay = 0.95
        raw_decay = math.log(
            ((initial_decay - 0.05) / 0.945)
            / (1.0 - (initial_decay - 0.05) / 0.945)
        )
        with torch.no_grad():
            self.selective.bias[2 * width :].fill_(raw_decay)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        filtered = self.depthwise(
            F.pad(self.norm(hidden).transpose(1, 2), (2, 0))
        ).transpose(1, 2)
        features = F.silu(self.feature_down(filtered))
        gate_raw, proposal_raw, decay_raw = self.selective(features).chunk(
            3, dim=-1
        )
        decay = 0.05 + 0.945 * torch.sigmoid(decay_raw.float())
        proposal = torch.tanh(proposal_raw.float())
        state = _parallel_affine_scan(
            decay, (1.0 - decay) * proposal
        )
        scanned = torch.sigmoid(gate_raw.float()) * state
        update = self.output_up(
            self.output_down(scanned.to(dtype=hidden.dtype))
        )
        hidden = residual + update
        normalized = self.ffn_norm(hidden)
        return hidden + self.ffn_out(
            F.silu(self.ffn_gate(normalized)) * self.ffn_value(normalized)
        )


class LowRankSelectivePatchCore(nn.Module):
    """Stacked low-rank selective scans with a GRU-compatible signature."""

    def __init__(self, width: int, layers: int, rank: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            LowRankSelectiveScanBlock(width, rank) for _ in range(layers)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if recurrent_state is not None:
            raise NotImplementedError(
                "incremental low-rank selective state is pending promotion"
            )
        for block in self.blocks:
            hidden = block(hidden)
        return hidden, None


class CausalAttentionPatchBlock(nn.Module):
    """Modern causal latent-patch attention block using fused SDPA kernels."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("patch attention width must be divisible by heads")
        hidden = max(64, round((8 * width / 3) / 64) * 64)
        self.heads = int(heads)
        self.head_width = width // heads
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_gate = nn.Linear(width, hidden, bias=False)
        self.ffn_value = nn.Linear(width, hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(self.attn_norm(hidden)).reshape(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(dim=2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2).reshape(batch, length, width)
        hidden = hidden + self.attn_out(attended)
        normalized = self.ffn_norm(hidden)
        return hidden + self.ffn_out(
            F.silu(self.ffn_gate(normalized)) * self.ffn_value(normalized)
        )


class CausalAttentionPatchCore(nn.Module):
    """Stacked latent attention with the GRU-compatible forward signature."""

    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            CausalAttentionPatchBlock(width, heads) for _ in range(layers)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if recurrent_state is not None:
            raise NotImplementedError(
                "incremental latent-attention cache is pending architecture promotion"
            )
        for block in self.blocks:
            hidden = block(hidden)
        return hidden, None


class PrunedBackoffByteCake(nn.Module):
    """Normalized byte n-gram cake with a hard learned-state budget."""

    def __init__(
        self,
        *,
        unigram_counts: torch.Tensor,
        order_tables: Iterable[tuple[torch.Tensor, ...]],
        backoff_strengths: Iterable[float],
        backoff_mode: str = "fixed",
        discount: float = 0.75,
        state_budget: int,
        corpus_bytes: int,
        context_hash_bits: Iterable[int] = (),
    ) -> None:
        super().__init__()
        unigram_counts = unigram_counts.to(dtype=torch.float32).clone()
        if unigram_counts.shape != (256,):
            raise ValueError("unigram_counts must contain exactly 256 entries")
        self.register_buffer("unigram_counts", unigram_counts.contiguous())
        self.state_budget = int(state_budget)
        self.corpus_bytes = int(corpus_bytes)
        self.max_order = 0
        strengths = tuple(float(value) for value in backoff_strengths)
        self.backoff_strengths = strengths
        if backoff_mode not in {"fixed", "distinct", "discount"}:
            raise ValueError("backoff_mode must be fixed, distinct, or discount")
        self.backoff_mode = backoff_mode
        self.discount = float(discount)
        if not 0.0 < self.discount < 1.0:
            raise ValueError("discount must be strictly between zero and one")
        order_entries: list[int] = []
        order_encodings: list[str] = []
        configured_hash_bits = tuple(int(value) for value in context_hash_bits)
        resolved_hash_bits: list[int] = []
        for order, table in enumerate(order_tables, start=1):
            if len(table) not in {2, 3, 4}:
                raise ValueError("order tables must contain two, three, or four tensors")
            keys, counts = table[:2]
            # clone() is intentional: torch.unique/search views can retain the
            # complete corpus-sized storage and silently bloat checkpoints.
            keys = keys.to(dtype=torch.int64).clone().contiguous()
            counts = counts.to(dtype=torch.float32).clone().contiguous()
            if keys.ndim != 1 or counts.shape != keys.shape:
                raise ValueError("each count-cake table must be paired 1D tensors")
            if keys.numel() and not bool(torch.all(keys[1:] > keys[:-1])):
                raise ValueError("count-cake keys must be strictly increasing")
            if len(table) in {2, 3}:
                encoding = "packed"
                hash_bits = 0
                contexts = keys.bitwise_right_shift(8)
                context_keys, inverse = torch.unique_consecutive(
                    contexts,
                    return_inverse=True,
                )
                context_keys = context_keys.clone().contiguous()
                if len(table) == 3:
                    context_totals = (
                        table[2].to(dtype=torch.float32).clone().contiguous()
                    )
                    if context_totals.shape != context_keys.shape:
                        raise ValueError("packed context totals must align")
                else:
                    context_totals = torch.zeros(
                        context_keys.shape,
                        device=counts.device,
                        dtype=torch.float32,
                    )
                    context_totals.scatter_add_(0, inverse, counts)
            else:
                encoding = "hashed_index"
                context_keys = table[2].to(dtype=torch.int64).clone().contiguous()
                context_totals = table[3].to(dtype=torch.float32).clone().contiguous()
                if context_keys.shape != context_totals.shape:
                    raise ValueError("hashed context keys and totals must align")
                if context_keys.numel() and not bool(
                    torch.all(context_keys[1:] > context_keys[:-1])
                ):
                    raise ValueError("hashed context keys must be increasing")
                if order <= len(configured_hash_bits):
                    hash_bits = configured_hash_bits[order - 1]
                else:
                    # Bundles predating the explicit hash ABI are recoverable:
                    # streaming keys were capped at 55 bits, while full-table
                    # training used the positive signed-int64 range.
                    maximum = int(context_keys[-1]) if context_keys.numel() else 0
                    hash_bits = 55 if maximum < (1 << 55) else 63
                if hash_bits not in {55, 63}:
                    raise ValueError("hashed contexts require 55 or 63 hash bits")
            self.register_buffer(f"keys_{order}", keys)
            self.register_buffer(f"counts_{order}", counts)
            self.register_buffer(f"context_keys_{order}", context_keys)
            self.register_buffer(f"context_totals_{order}", context_totals)
            _, context_distinct = torch.unique_consecutive(
                keys.bitwise_right_shift(8),
                return_counts=True,
            )
            if context_distinct.shape != context_keys.shape:
                raise ValueError("context continuation counts do not align")
            self.register_buffer(
                f"context_distinct_{order}",
                context_distinct.to(torch.float32),
            )
            context_max = torch.zeros(
                context_keys.shape,
                device=counts.device,
                dtype=torch.float32,
            )
            if encoding == "packed":
                dense_context_index = inverse
            else:
                dense_context_index = keys.bitwise_right_shift(8)
            context_max.scatter_reduce_(
                0,
                dense_context_index,
                counts,
                reduce="amax",
                include_self=False,
            )
            self.register_buffer(f"context_max_{order}", context_max)
            self.max_order = order
            order_entries.append(int(keys.numel()))
            order_encodings.append(encoding)
            resolved_hash_bits.append(hash_bits)
        self.order_entries = tuple(order_entries)
        self.order_encodings = tuple(order_encodings)
        self.context_hash_bits = tuple(resolved_hash_bits)
        if self.state_entries > self.state_budget:
            raise ValueError("count-cake state exceeds its declared budget")

    @property
    def state_entries(self) -> int:
        return 256 + sum(self.order_entries)

    @property
    def summary(self) -> CountCakeTrainingSummary:
        return CountCakeTrainingSummary(
            state_budget=self.state_budget,
            state_entries=self.state_entries,
            max_order=self.max_order,
            trained_order=self.max_order,
            order_entries=self.order_entries,
            corpus_bytes=self.corpus_bytes,
        )

    @staticmethod
    def _joint_keys(data: torch.Tensor, order: int) -> torch.Tensor:
        if data.ndim != 1:
            raise ValueError("count-cake training data must be one-dimensional")
        if data.numel() <= order:
            raise ValueError("training data is shorter than the requested order")
        keys = data[order:].to(dtype=torch.int64).clone()
        for lag in range(order):
            keys.add_(
                data[order - 1 - lag : -1 - lag].to(dtype=torch.int64)
                << (8 * (lag + 1))
            )
        return keys

    @staticmethod
    def _hashed_contexts(data: torch.Tensor, order: int) -> torch.Tensor:
        """Hash ordered contexts into a positive 63-bit generation key."""
        contexts = torch.zeros(
            data.numel() - order,
            device=data.device,
            dtype=torch.int64,
        )
        for offset in range(order):
            contexts.mul_(257).add_(
                data[offset : data.numel() - order + offset].to(torch.int64)
                + 1
            ).bitwise_and_(HASH_MASK)
        return contexts

    @classmethod
    def train_from_bytes(
        cls,
        data: torch.Tensor,
        *,
        state_budget: int,
        max_order: int = 4,
        backoff_strengths: Iterable[float] = DEFAULT_BACKOFF_STRENGTHS,
        backoff_mode: str = "fixed",
        discount: float = 0.75,
        budget_mode: str = "sequential",
    ) -> "PrunedBackoffByteCake":
        if state_budget < 256:
            raise ValueError("state_budget must reserve the 256 unigram entries")
        if max_order < 1 or max_order > len(DEFAULT_BACKOFF_STRENGTHS):
            raise ValueError(
                f"max_order must be between 1 and {len(DEFAULT_BACKOFF_STRENGTHS)}"
            )
        data = data.to(dtype=torch.int64)
        if budget_mode not in {
            "sequential",
            "balanced",
            "hybrid",
            "hybrid3",
            "hybrid2",
            "information",
        }:
            raise ValueError(
                "budget_mode must be sequential, balanced, hybrid, hybrid3, "
                "hybrid2, or information"
            )
        if data.ndim != 1 or data.numel() <= max_order:
            raise ValueError("insufficient one-dimensional byte training data")
        if int(data.min()) < 0 or int(data.max()) > 255:
            raise ValueError("count-cake training values must be bytes")
        unigram_counts = torch.bincount(data, minlength=256).to(torch.float32)
        used = 256
        tables: list[tuple[torch.Tensor, ...]] = []
        for order in range(1, max_order + 1):
            hashed = order > 6
            if hashed:
                contexts = cls._hashed_contexts(data, order)
                context_keys, context_inverse = torch.unique(
                    contexts,
                    sorted=True,
                    return_inverse=True,
                )
                joint_keys = context_inverse * 256 + data[order:]
            else:
                joint_keys = cls._joint_keys(data, order)
            keys, counts = torch.unique(
                joint_keys,
                sorted=True,
                return_counts=True,
            )
            remaining = state_budget - used
            if remaining <= 0:
                break
            order_limit = remaining
            if budget_mode == "balanced" or (
                budget_mode == "hybrid" and order > 4
            ) or (
                budget_mode in {"hybrid3", "information"} and order > 3
            ) or (
                budget_mode == "hybrid2" and order > 2
            ):
                orders_left = max_order - order + 1
                order_limit = max(1, remaining // orders_left)
            if keys.numel() > order_limit:
                # torch.unique returns ascending keys.  Stable count sorting
                # therefore makes equal-frequency pruning deterministic.
                ranked = torch.argsort(
                    counts,
                    descending=True,
                    stable=True,
                )[:order_limit]
                ranked = torch.sort(ranked).values
                keys = keys[ranked]
                counts = counts[ranked]
            if hashed:
                old_context_ids = keys.bitwise_right_shift(8)
                used_context_ids, remapped = torch.unique_consecutive(
                    old_context_ids,
                    return_inverse=True,
                )
                keys = remapped * 256 + keys.bitwise_and(255)
                context_keys = context_keys[used_context_ids]
                context_totals = torch.zeros(
                    context_keys.shape,
                    device=counts.device,
                    dtype=counts.dtype,
                )
                context_totals.scatter_add_(0, remapped, counts)
                tables.append((keys, counts, context_keys, context_totals))
            else:
                tables.append((keys, counts))
            used += int(keys.numel())
            del joint_keys
            if hashed:
                del contexts, context_inverse
            if used >= state_budget:
                break
        return cls(
            unigram_counts=unigram_counts,
            order_tables=tables,
            backoff_strengths=backoff_strengths,
            backoff_mode=backoff_mode,
            discount=discount,
            state_budget=state_budget,
            corpus_bytes=int(data.numel()),
            context_hash_bits=(0,) * min(6, len(tables))
            + (63,) * max(0, len(tables) - 6),
        )

    @classmethod
    def train_streaming_from_bytes(
        cls,
        data: torch.Tensor,
        *,
        device: torch.device,
        state_budget: int,
        max_order: int = 12,
        chunk_bytes: int = 24_000_000,
        backoff_strengths: Iterable[float] = DEFAULT_BACKOFF_STRENGTHS,
        backoff_mode: str = "fixed",
        discount: float = 0.75,
        budget_mode: str = "balanced",
        candidate_multiplier: int = 2,
    ) -> "PrunedBackoffByteCake":
        """Bounded-memory deterministic heavy-hitter training over a byte stream."""
        if data.ndim != 1 or data.numel() <= max_order:
            raise ValueError("insufficient one-dimensional byte training data")
        if data.device.type != "cpu":
            raise ValueError("streaming source bytes must remain on CPU")
        if chunk_bytes <= max_order or candidate_multiplier < 1:
            raise ValueError("invalid streaming count-cake settings")
        if budget_mode not in {
            "sequential",
            "balanced",
            "hybrid",
            "hybrid3",
            "hybrid2",
            "information",
        }:
            raise ValueError("unsupported streaming budget mode")
        if max_order > len(tuple(backoff_strengths)):
            raise ValueError("backoff strengths are shorter than max_order")
        unigram_counts = torch.zeros(256, device=device, dtype=torch.float32)
        for start in range(0, data.numel(), chunk_bytes):
            chunk = data[start : start + chunk_bytes].to(device, torch.long)
            unigram_counts.add_(torch.bincount(chunk, minlength=256))
        used = 256
        tables: list[tuple[torch.Tensor, ...]] = []
        hash_mask = (1 << 55) - 1
        for order in range(1, max_order + 1):
            remaining = state_budget - used
            if remaining <= 0:
                break
            order_limit = remaining
            if budget_mode == "balanced" or (
                budget_mode == "hybrid" and order > 4
            ) or (
                budget_mode in {"hybrid3", "information"} and order > 3
            ) or (
                budget_mode == "hybrid2" and order > 2
            ):
                order_limit = max(1, remaining // (max_order - order + 1))
            retained_keys = torch.empty(0, device=device, dtype=torch.int64)
            retained_counts = torch.empty(0, device=device, dtype=torch.int64)
            retention_limit = (
                order_limit * candidate_multiplier
                if budget_mode == "information" and order <= 6
                else order_limit
            )
            for start in range(0, data.numel(), chunk_bytes):
                end = min(start + chunk_bytes, data.numel())
                source_start = max(0, start - order)
                chunk = data[source_start:end].to(device, torch.long)
                if source_start == 0 and start == 0:
                    if chunk.numel() <= order:
                        continue
                    targets = chunk[order:]
                else:
                    targets = chunk[order:]
                if order <= 6:
                    chunk_keys = cls._joint_keys(chunk, order)
                else:
                    context = cls._hashed_contexts(chunk, order) & hash_mask
                    chunk_keys = (context << 8) | targets
                chunk_keys, chunk_counts = torch.unique(
                    chunk_keys, sorted=True, return_counts=True
                )
                candidate_limit = min(
                    chunk_keys.numel(), retention_limit * candidate_multiplier
                )
                if chunk_keys.numel() > candidate_limit:
                    selected = torch.argsort(
                        chunk_counts, descending=True, stable=True
                    )[:candidate_limit]
                    selected = torch.sort(selected).values
                    chunk_keys = chunk_keys[selected]
                    chunk_counts = chunk_counts[selected]
                if retained_keys.numel():
                    merged_keys = torch.cat([retained_keys, chunk_keys])
                    merged_counts = torch.cat([retained_counts, chunk_counts])
                    order_index = torch.argsort(merged_keys, stable=True)
                    merged_keys = merged_keys[order_index]
                    merged_counts = merged_counts[order_index]
                    unique_keys, inverse = torch.unique_consecutive(
                        merged_keys, return_inverse=True
                    )
                    unique_counts = torch.zeros(
                        unique_keys.shape, device=device, dtype=torch.int64
                    )
                    unique_counts.scatter_add_(0, inverse, merged_counts)
                    retained_keys, retained_counts = unique_keys, unique_counts
                else:
                    retained_keys, retained_counts = chunk_keys, chunk_counts
                if retained_keys.numel() > retention_limit:
                    selected = torch.argsort(
                        retained_counts, descending=True, stable=True
                    )[:retention_limit]
                    selected = torch.sort(selected).values
                    retained_keys = retained_keys[selected]
                    retained_counts = retained_counts[selected]
            if retained_keys.numel() > order_limit:
                if budget_mode == "information" and order <= 6:
                    contexts = retained_keys.bitwise_right_shift(8)
                    _, context_inverse = torch.unique_consecutive(
                        contexts, return_inverse=True
                    )
                    context_totals = torch.zeros(
                        int(context_inverse[-1]) + 1,
                        device=device,
                        dtype=torch.float32,
                    )
                    context_totals.scatter_add_(
                        0, context_inverse, retained_counts.to(torch.float32)
                    )
                    empirical = retained_counts.to(torch.float32) / context_totals[
                        context_inverse
                    ].clamp_min(1.0)
                    targets = retained_keys.bitwise_and(255)
                    unigram_probability = unigram_counts[targets] / unigram_counts.sum()
                    lower_probability = unigram_probability
                    if order > 1 and tables:
                        lower_keys = retained_keys.bitwise_and(
                            (1 << (8 * order)) - 1
                        )
                        lower_table = tables[-1]
                        lower_counts = cls._lookup(
                            lower_table[0], lower_table[1].to(torch.float32), lower_keys
                        )
                        lower_contexts = lower_keys.bitwise_right_shift(8)
                        if len(lower_table) >= 3:
                            lower_context_keys = torch.unique_consecutive(
                                lower_table[0].bitwise_right_shift(8)
                            )
                            lower_totals_table = lower_table[2].to(torch.float32)
                        else:
                            lower_context_keys, lower_inverse = torch.unique_consecutive(
                                lower_table[0].bitwise_right_shift(8),
                                return_inverse=True,
                            )
                            lower_totals_table = torch.zeros(
                                lower_context_keys.shape,
                                device=device,
                                dtype=torch.float32,
                            )
                            lower_totals_table.scatter_add_(
                                0, lower_inverse, lower_table[1].to(torch.float32)
                            )
                        lower_totals = cls._lookup(
                            lower_context_keys, lower_totals_table, lower_contexts
                        )
                        strength = float(tuple(backoff_strengths)[order - 2])
                        lower_probability = (
                            lower_counts + strength * unigram_probability
                        ) / (lower_totals + strength)
                    information_gain = retained_counts.to(torch.float32) * torch.log(
                        empirical.clamp_min(1e-12)
                        / lower_probability.clamp_min(1e-12)
                    )
                    selected = torch.argsort(
                        information_gain, descending=True, stable=True
                    )[:order_limit]
                else:
                    selected = torch.argsort(
                        retained_counts, descending=True, stable=True
                    )[:order_limit]
                selected = torch.sort(selected).values
                retained_keys = retained_keys[selected]
                retained_counts = retained_counts[selected]
            if order > 6:
                context_keys, remapped = torch.unique_consecutive(
                    retained_keys >> 8, return_inverse=True
                )
                dense_keys = (remapped << 8) | (retained_keys & 255)
                context_totals = torch.zeros(
                    context_keys.shape, device=device, dtype=torch.int64
                )
                context_totals.scatter_add_(0, remapped, retained_counts)
                tables.append(
                    (dense_keys, retained_counts, context_keys, context_totals)
                )
            else:
                tables.append((retained_keys, retained_counts))
            used += int(retained_keys.numel())
        return cls(
            unigram_counts=unigram_counts,
            order_tables=tables,
            backoff_strengths=backoff_strengths,
            backoff_mode=backoff_mode,
            discount=discount,
            state_budget=state_budget,
            corpus_bytes=int(data.numel()),
            context_hash_bits=(0,) * min(6, len(tables))
            + (55,) * max(0, len(tables) - 6),
        )

    @staticmethod
    def _lookup(
        sorted_keys: torch.Tensor,
        values: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if sorted_keys.numel() == 0:
            return torch.zeros(query.shape, device=query.device, dtype=values.dtype)
        indices = torch.searchsorted(sorted_keys, query)
        safe = indices.clamp(max=sorted_keys.numel() - 1)
        found = (indices < sorted_keys.numel()) & (sorted_keys[safe] == query)
        return torch.where(found, values[safe], torch.zeros_like(values[safe]))

    def target_log_probs(
        self,
        rows: torch.Tensor,
        *,
        start: int,
        return_features: bool = False,
        return_stages: bool = False,
        return_stage_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Return normalized log p(observed byte | preceding bytes)."""
        if rows.ndim != 2:
            raise ValueError("rows must have shape [batch, bytes]")
        if start < self.max_order or start >= rows.shape[1]:
            raise ValueError("start must leave enough history and at least one target")
        rows = rows.to(dtype=torch.int64)
        targets = rows[:, start:]
        smoothed = self.unigram_counts + 0.5
        probability = smoothed[targets] / smoothed.sum()
        stage_probabilities = [probability]
        matched_order = torch.zeros_like(probability)
        matched_total = torch.zeros_like(probability)
        matched_density = torch.zeros_like(probability)
        matched_peak = torch.zeros_like(probability)
        stage_features = [
            torch.stack(
                [matched_order, matched_total, matched_density, matched_peak],
                dim=-1,
            )
        ]
        for order in range(1, self.max_order + 1):
            context = torch.zeros_like(targets)
            if self.order_encodings[order - 1] == "packed":
                for lag in range(order):
                    context.add_(
                        rows[:, start - 1 - lag : rows.shape[1] - 1 - lag]
                        << (8 * lag)
                    )
                joint = targets + context.bitwise_left_shift(8)
                joint_counts = self._lookup(
                    getattr(self, f"keys_{order}"),
                    getattr(self, f"counts_{order}"),
                    joint,
                )
                totals = self._lookup(
                    getattr(self, f"context_keys_{order}"),
                    getattr(self, f"context_totals_{order}"),
                    context,
                )
                distinct = self._lookup(
                    getattr(self, f"context_keys_{order}"),
                    getattr(self, f"context_distinct_{order}"),
                    context,
                )
                maximum = self._lookup(
                    getattr(self, f"context_keys_{order}"),
                    getattr(self, f"context_max_{order}"),
                    context,
                )
            else:
                hash_mask = (1 << self.context_hash_bits[order - 1]) - 1
                for offset in range(order):
                    context.mul_(257).add_(
                        rows[
                            :,
                            start - order + offset : rows.shape[1]
                            - order
                            + offset,
                        ]
                        + 1
                    ).bitwise_and_(hash_mask)
                context_keys = getattr(self, f"context_keys_{order}")
                context_indices = torch.searchsorted(context_keys, context)
                safe = context_indices.clamp(max=context_keys.numel() - 1)
                found = (context_indices < context_keys.numel()) & (
                    context_keys[safe] == context
                )
                joint = targets + safe * 256
                joint_counts = torch.where(
                    found,
                    self._lookup(
                        getattr(self, f"keys_{order}"),
                        getattr(self, f"counts_{order}"),
                        joint,
                    ),
                    torch.zeros_like(probability),
                )
                totals = torch.where(
                    found,
                    getattr(self, f"context_totals_{order}")[safe],
                    torch.zeros_like(probability),
                )
                distinct = torch.where(
                    found,
                    getattr(self, f"context_distinct_{order}")[safe],
                    torch.zeros_like(probability),
                )
                maximum = torch.where(
                    found,
                    getattr(self, f"context_max_{order}")[safe],
                    torch.zeros_like(probability),
                )
            matched = totals > 0
            matched_order = torch.where(
                matched,
                torch.full_like(matched_order, order / max(self.max_order, 1)),
                matched_order,
            )
            matched_total = torch.where(
                matched,
                torch.log1p(totals) / 16.0,
                matched_total,
            )
            matched_density = torch.where(
                matched,
                distinct / totals.clamp_min(1.0),
                matched_density,
            )
            matched_peak = torch.where(
                matched,
                maximum / totals.clamp_min(1.0),
                matched_peak,
            )
            if self.backoff_mode == "discount":
                discounted = (joint_counts - self.discount).clamp_min(0.0)
                escape = self.discount * distinct
                updated = (discounted + escape * probability) / totals.clamp_min(1.0)
                probability = torch.where(totals > 0, updated, probability)
            elif self.backoff_mode == "distinct":
                updated = (
                    joint_counts + distinct * probability
                ) / (totals + distinct).clamp_min(1.0)
                probability = torch.where(totals > 0, updated, probability)
            else:
                strength = self.backoff_strengths[order - 1]
                probability = (
                    joint_counts + strength * probability
                ) / (totals + strength)
            stage_probabilities.append(probability)
            stage_features.append(
                torch.stack(
                    [matched_order, matched_total, matched_density, matched_peak],
                    dim=-1,
                )
            )
        log_probability = probability.clamp_min(1e-30).log()
        outputs: list[torch.Tensor] = [log_probability]
        if return_features:
            outputs.append(torch.stack(
                [matched_order, matched_total, matched_density, matched_peak],
                dim=-1,
            ))
        if return_stages:
            outputs.append(
                torch.stack(stage_probabilities, dim=-1).clamp_min(1e-30).log()
            )
        if return_stage_features:
            outputs.append(torch.stack(stage_features, dim=-2))
        if len(outputs) > 1:
            return tuple(outputs)
        return log_probability

    def all_probabilities(
        self,
        rows: torch.Tensor,
        *,
        start: int,
    ) -> torch.Tensor:
        """Return normalized causal distributions for every scored position."""
        if rows.ndim != 2:
            raise ValueError("rows must have shape [batch, bytes]")
        if start < self.max_order or start >= rows.shape[1]:
            raise ValueError("start must leave enough history and at least one target")
        rows = rows.to(dtype=torch.int64)
        target_count = rows.shape[1] - start
        vocabulary = torch.arange(256, device=rows.device, dtype=torch.int64)
        smoothed = self.unigram_counts + 0.5
        unigram = smoothed / smoothed.sum()
        probability = unigram.reshape(1, 1, 256).expand(
            rows.shape[0], target_count, 256
        ).clone()
        for order in range(1, self.max_order + 1):
            context = torch.zeros(
                (rows.shape[0], target_count),
                device=rows.device,
                dtype=torch.int64,
            )
            if self.order_encodings[order - 1] == "packed":
                for lag in range(order):
                    context.add_(
                        rows[:, start - 1 - lag : rows.shape[1] - 1 - lag]
                        << (8 * lag)
                    )
                joint = context.bitwise_left_shift(8).unsqueeze(-1) + vocabulary
                joint_counts = self._lookup(
                    getattr(self, f"keys_{order}"),
                    getattr(self, f"counts_{order}"),
                    joint,
                )
                totals = self._lookup(
                    getattr(self, f"context_keys_{order}"),
                    getattr(self, f"context_totals_{order}"),
                    context,
                )
                distinct = self._lookup(
                    getattr(self, f"context_keys_{order}"),
                    getattr(self, f"context_distinct_{order}"),
                    context,
                )
            else:
                hash_mask = (1 << self.context_hash_bits[order - 1]) - 1
                for offset in range(order):
                    context.mul_(257).add_(
                        rows[
                            :,
                            start - order + offset : rows.shape[1]
                            - order
                            + offset,
                        ]
                        + 1
                    ).bitwise_and_(hash_mask)
                context_keys = getattr(self, f"context_keys_{order}")
                context_indices = torch.searchsorted(context_keys, context)
                safe = context_indices.clamp(max=context_keys.numel() - 1)
                found = (context_indices < context_keys.numel()) & (
                    context_keys[safe] == context
                )
                joint = safe.unsqueeze(-1) * 256 + vocabulary
                joint_counts = torch.where(
                    found.unsqueeze(-1),
                    self._lookup(
                        getattr(self, f"keys_{order}"),
                        getattr(self, f"counts_{order}"),
                        joint,
                    ),
                    torch.zeros_like(probability),
                )
                totals = torch.where(
                    found,
                    getattr(self, f"context_totals_{order}")[safe],
                    torch.zeros_like(context, dtype=probability.dtype),
                )
                distinct = torch.where(
                    found,
                    getattr(self, f"context_distinct_{order}")[safe],
                    torch.zeros_like(context, dtype=probability.dtype),
                )
            if self.backoff_mode == "discount":
                discounted = (joint_counts - self.discount).clamp_min(0.0)
                escape = self.discount * distinct
                updated = (
                    discounted + escape.unsqueeze(-1) * probability
                ) / totals.clamp_min(1.0).unsqueeze(-1)
                probability = torch.where(
                    (totals > 0).unsqueeze(-1), updated, probability
                )
            elif self.backoff_mode == "distinct":
                updated = (
                    joint_counts + distinct.unsqueeze(-1) * probability
                ) / (totals + distinct).clamp_min(1.0).unsqueeze(-1)
                probability = torch.where(
                    (totals > 0).unsqueeze(-1), updated, probability
                )
            else:
                strength = self.backoff_strengths[order - 1]
                probability = (
                    joint_counts + strength * probability
                ) / (totals + strength).unsqueeze(-1)
        return probability / probability.sum(dim=-1, keepdim=True)

    def next_probabilities(
        self,
        history: torch.Tensor,
        *,
        return_features: bool = False,
        return_stages: bool = False,
        return_stage_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Return the complete normalized next-byte distribution."""
        if history.ndim != 1:
            raise ValueError("history must be a one-dimensional byte tensor")
        if history.numel() and (int(history.min()) < 0 or int(history.max()) > 255):
            raise ValueError("history values must be bytes")
        smoothed = self.unigram_counts + 0.5
        probability = smoothed / smoothed.sum()
        stage_probabilities = [probability]
        targets = torch.arange(256, device=history.device, dtype=torch.int64)
        available_order = min(self.max_order, int(history.numel()))
        matched_order = 0.0
        matched_total = torch.zeros((), device=history.device)
        matched_density = torch.zeros((), device=history.device)
        matched_peak = torch.zeros((), device=history.device)
        stage_features = [
            torch.stack(
                [
                    torch.as_tensor(
                        matched_order,
                        device=history.device,
                        dtype=probability.dtype,
                    ),
                    matched_total.to(probability.dtype),
                    matched_density.to(probability.dtype),
                    matched_peak.to(probability.dtype),
                ]
            )
        ]
        for order in range(1, available_order + 1):
            context = torch.zeros((), device=history.device, dtype=torch.int64)
            if self.order_encodings[order - 1] == "packed":
                for lag in range(order):
                    context.add_(
                        history[-1 - lag].to(torch.int64) << (8 * lag)
                    )
                joint = targets + context.bitwise_left_shift(8)
                joint_counts = self._lookup(
                    getattr(self, f"keys_{order}"),
                    getattr(self, f"counts_{order}"),
                    joint,
                )
                total = self._lookup(
                    getattr(self, f"context_keys_{order}"),
                    getattr(self, f"context_totals_{order}"),
                    context,
                )
                distinct = self._lookup(
                    getattr(self, f"context_keys_{order}"),
                    getattr(self, f"context_distinct_{order}"),
                    context,
                )
                maximum = self._lookup(
                    getattr(self, f"context_keys_{order}"),
                    getattr(self, f"context_max_{order}"),
                    context,
                )
            else:
                hash_mask = (1 << self.context_hash_bits[order - 1]) - 1
                for byte in history[-order:]:
                    context.mul_(257).add_(
                        byte.to(torch.int64) + 1
                    ).bitwise_and_(hash_mask)
                context_keys = getattr(self, f"context_keys_{order}")
                context_index = torch.searchsorted(context_keys, context)
                safe = context_index.clamp(max=context_keys.numel() - 1)
                found = (context_index < context_keys.numel()) & (
                    context_keys[safe] == context
                )
                joint = targets + safe * 256
                joint_counts = torch.where(
                    found,
                    self._lookup(
                        getattr(self, f"keys_{order}"),
                        getattr(self, f"counts_{order}"),
                        joint,
                    ),
                    torch.zeros_like(probability),
                )
                total = torch.where(
                    found,
                    getattr(self, f"context_totals_{order}")[safe],
                    torch.zeros((), device=history.device),
                )
                distinct = torch.where(
                    found,
                    getattr(self, f"context_distinct_{order}")[safe],
                    torch.zeros((), device=history.device),
                )
                maximum = torch.where(
                    found,
                    getattr(self, f"context_max_{order}")[safe],
                    torch.zeros((), device=history.device),
                )
            if bool(total > 0):
                matched_order = order / max(self.max_order, 1)
                matched_total = torch.log1p(total) / 16.0
                matched_density = distinct / total.clamp_min(1.0)
                matched_peak = maximum / total.clamp_min(1.0)
            if self.backoff_mode == "discount":
                discounted = (joint_counts - self.discount).clamp_min(0.0)
                escape = self.discount * distinct
                updated = (discounted + escape * probability) / total.clamp_min(1.0)
                probability = torch.where(total > 0, updated, probability)
            elif self.backoff_mode == "distinct":
                updated = (
                    joint_counts + distinct * probability
                ) / (total + distinct).clamp_min(1.0)
                probability = torch.where(total > 0, updated, probability)
            else:
                strength = self.backoff_strengths[order - 1]
                probability = (
                    joint_counts + strength * probability
                ) / (total + strength)
            stage_probabilities.append(probability)
            stage_features.append(
                torch.stack(
                    [
                        torch.as_tensor(
                            matched_order,
                            device=history.device,
                            dtype=probability.dtype,
                        ),
                        matched_total.to(probability.dtype),
                        matched_density.to(probability.dtype),
                        matched_peak.to(probability.dtype),
                    ]
                )
            )
        probability = probability / probability.sum()
        outputs: list[torch.Tensor] = [probability]
        if return_features:
            features = torch.stack(
                [
                    torch.as_tensor(
                        matched_order,
                        device=history.device,
                        dtype=probability.dtype,
                    ),
                    matched_total.to(probability.dtype),
                    matched_density.to(probability.dtype),
                    matched_peak.to(probability.dtype),
                ]
            )
            outputs.append(features)
        if return_stages:
            outputs.append(torch.stack(stage_probabilities, dim=0))
        if return_stage_features:
            outputs.append(torch.stack(stage_features, dim=0))
        if len(outputs) > 1:
            return tuple(outputs)
        return probability


class HierarchicalCountCakeLM(nn.Module):
    """Patch-scale recurrent host mixed with a normalized local count cake."""

    def __init__(
        self,
        count_cake: PrunedBackoffByteCake,
        *,
        patch_size: int = 32,
        chunking_mode: str = "fixed",
        d_byte: int = 8,
        d_model: int = 128,
        d_abi: int = 64,
        patch_layers: int = 1,
        patch_core_type: str = "gru",
        patch_selective_rank: int = 128,
        patch_attention_heads: int = 8,
        scratchpad_stride: int = 0,
        dynamic_hash_buckets: int = 0,
        dynamic_hash_width: int = 64,
        dynamic_hash_tables: int = 1,
        dynamic_hash_sparse: bool = False,
        neural_context_buckets: int = 0,
        neural_context_order: int = 3,
        neural_context_sparse: bool = False,
        local_width: int = 32,
        local_recurrent: bool = False,
        local_continuous: bool = False,
        local_decoder: str | None = None,
        local_layers: int = 5,
        local_dilation_growth: int = 2,
        local_gru_layers: int = 1,
        local_rank: int = 64,
        byte_head: str = "radix",
        online_cache_specs: Iterable[tuple[int, float]] = (),
        online_cache_window: int | None = None,
        recent_cache_specs: Iterable[tuple[int, float]] = (),
        normalized_cache_specs: Iterable[tuple[int, float]] = (),
        cache_normalization: str = "classes",
        prediction_start: int | None = None,
        confidence_gate: bool = False,
        expert_confidence_gate: bool = False,
        count_distribution_gate: bool = False,
        count_order_routing: bool = False,
        count_order_stage_features: bool = True,
        count_order_router_hidden: int = 0,
        gate_hidden_width: int = 0,
        initial_neural_fraction: float = 0.02,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.count_cake = count_cake
        self.patch_size = int(patch_size)
        if chunking_mode not in {"fixed", "delimiter"}:
            raise ValueError("chunking mode must be fixed or delimiter")
        self.chunking_mode = str(chunking_mode)
        self.prediction_start = int(
            self.patch_size if prediction_start is None else prediction_start
        )
        if (
            self.prediction_start < self.patch_size
            or self.prediction_start % self.patch_size
        ):
            raise ValueError(
                "prediction_start must be a positive multiple of patch_size"
            )
        self.d_abi = int(d_abi)
        self.patch_layers = int(patch_layers)
        if self.patch_layers <= 0:
            raise ValueError("patch_layers must be positive")
        if patch_core_type not in {
            "gru",
            "selective_scan",
            "low_rank_selective_scan",
            "attention",
        }:
            raise ValueError(
                "patch_core_type must be gru, selective_scan, "
                "low_rank_selective_scan, or attention"
            )
        self.patch_core_type = str(patch_core_type)
        self.patch_selective_rank = int(patch_selective_rank)
        if (
            self.patch_core_type == "low_rank_selective_scan"
            and (
                self.patch_selective_rank <= 0
                or self.patch_selective_rank > d_model
            )
        ):
            raise ValueError("patch selective rank must be in [1, d_model]")
        self.patch_attention_heads = int(patch_attention_heads)
        if self.patch_attention_heads <= 0:
            raise ValueError("patch_attention_heads must be positive")
        self.scratchpad_stride = int(scratchpad_stride)
        if self.scratchpad_stride:
            if self.chunking_mode != "fixed":
                raise ValueError("scratchpads currently require fixed patches")
            if self.patch_size % self.scratchpad_stride:
                raise ValueError("scratchpad stride must divide patch size")
            if not 0 < self.scratchpad_stride < self.patch_size:
                raise ValueError("scratchpad stride must be inside each patch")
            if self.patch_core_type != "gru" or self.patch_layers != 1:
                raise ValueError("scratchpads currently require a one-layer GRU trunk")
        self.dynamic_hash_buckets = int(dynamic_hash_buckets)
        self.dynamic_hash_width = int(dynamic_hash_width)
        self.dynamic_hash_tables = int(dynamic_hash_tables)
        self.dynamic_hash_sparse = bool(dynamic_hash_sparse)
        self.neural_context_buckets = int(neural_context_buckets)
        self.neural_context_order = int(neural_context_order)
        self.neural_context_sparse = bool(neural_context_sparse)
        if self.dynamic_hash_buckets < 0 or (
            self.dynamic_hash_buckets
            and self.dynamic_hash_buckets & (self.dynamic_hash_buckets - 1)
        ):
            raise ValueError("dynamic hash buckets must be zero or a power of two")
        if self.dynamic_hash_width <= 0:
            raise ValueError("dynamic hash width must be positive")
        if self.dynamic_hash_tables <= 0:
            raise ValueError("dynamic hash tables must be positive")
        if self.neural_context_buckets < 0 or (
            self.neural_context_buckets
            and self.neural_context_buckets & (self.neural_context_buckets - 1)
        ):
            raise ValueError(
                "neural context buckets must be zero or a power of two"
            )
        if self.neural_context_order <= 0:
            raise ValueError("neural context order must be positive")
        if self.neural_context_sparse and not self.neural_context_buckets:
            raise ValueError(
                "sparse neural contexts require a nonzero bucket count"
            )
        if self.neural_context_sparse and self.patch_size != 1:
            raise ValueError(
                "sparse neural context residuals currently require raw-byte patches"
            )
        if self.dynamic_hash_buckets and self.chunking_mode != "delimiter":
            raise ValueError("dynamic span hashing requires delimiter chunking")
        if local_decoder is None:
            local_decoder = "gru" if local_recurrent else "position"
        if local_decoder not in {
            "position",
            "gru",
            "lstm",
            "scan",
            "dilated_conv",
        }:
            raise ValueError(
                "local_decoder must be position, gru, lstm, scan, or dilated_conv"
            )
        self.local_decoder = str(local_decoder)
        self.local_layers = int(local_layers)
        self.local_dilation_growth = int(local_dilation_growth)
        self.local_gru_layers = int(local_gru_layers)
        self.local_rank = int(local_rank)
        self.byte_head = str(byte_head)
        if self.byte_head not in {"radix", "direct"}:
            raise ValueError("byte_head must be radix or direct")
        if self.local_layers <= 0:
            raise ValueError("local_layers must be positive")
        if self.local_dilation_growth < 2:
            raise ValueError("local_dilation_growth must be at least two")
        if self.local_gru_layers <= 0:
            raise ValueError("local GRU layers must be positive")
        if self.local_rank <= 0:
            raise ValueError("local_rank must be positive")
        self.local_recurrent = self.local_decoder in {"gru", "lstm"}
        self.local_continuous = bool(local_continuous)
        if self.local_continuous and self.local_decoder not in {
            "gru",
            "lstm",
            "dilated_conv",
        }:
            raise ValueError(
                "continuous local state requires a GRU, LSTM, or dilated convolution"
            )
        if self.scratchpad_stride and (
            self.local_decoder != "gru" or not self.local_continuous
        ):
            raise ValueError(
                "scratchpads require a continuous GRU local decoder"
            )
        self.online_cache_specs = tuple(
            (int(order), float(strength))
            for order, strength in online_cache_specs
        )
        self.online_cache_window = (
            None if online_cache_window is None else int(online_cache_window)
        )
        self.recent_cache_specs = tuple(
            (int(order), float(strength))
            for order, strength in recent_cache_specs
        )
        self.normalized_cache_specs = tuple(
            (int(order), float(strength))
            for order, strength in normalized_cache_specs
        )
        self.cache_normalization = str(cache_normalization)
        if self.cache_enabled:
            self._new_causal_cache()
        self.confidence_gate_enabled = bool(confidence_gate)
        self.expert_confidence_gate_enabled = bool(expert_confidence_gate)
        self.count_distribution_gate_enabled = bool(count_distribution_gate)
        self.count_order_routing_enabled = bool(count_order_routing)
        self.count_order_stage_features = bool(count_order_stage_features)
        self.count_order_router_hidden = int(count_order_router_hidden)
        if self.count_order_router_hidden < 0:
            raise ValueError("count order router hidden width cannot be negative")
        self.gate_hidden_width = 0
        self.d_model = int(d_model)
        self.byte_embedding = nn.Embedding(256, d_byte)
        if self.neural_context_buckets:
            self.neural_context_embedding = nn.Embedding(
                self.neural_context_buckets,
                d_byte,
                sparse=self.neural_context_sparse,
            )
            nn.init.zeros_(self.neural_context_embedding.weight)
        if self.chunking_mode == "delimiter" and self.local_decoder == "lstm":
            raise ValueError("delimiter chunking currently requires the GRU decoder")
        if self.chunking_mode == "fixed":
            self.patch_projection = nn.Linear(patch_size * d_byte, d_model)
        else:
            delimiter_table = torch.zeros(256, dtype=torch.bool)
            delimiter_table[:33] = True
            delimiter_table[
                torch.tensor(list(b".,;:!?()[]{}\"'`/\\|-"), dtype=torch.long)
            ] = True
            self.register_buffer("chunk_delimiter_table", delimiter_table)
            self.dynamic_position_scale = nn.Embedding(patch_size, d_byte)
            self.dynamic_position_bias = nn.Embedding(patch_size, d_byte)
            self.dynamic_byte_projection = nn.Linear(d_byte, d_model, bias=False)
            self.dynamic_chunk_norm = nn.LayerNorm(d_model)
            if self.dynamic_hash_buckets:
                if self.dynamic_hash_tables == 1:
                    self.dynamic_hash_embedding = nn.Embedding(
                        self.dynamic_hash_buckets,
                        self.dynamic_hash_width,
                        sparse=self.dynamic_hash_sparse,
                    )
                else:
                    self.dynamic_hash_embeddings = nn.ModuleList(
                        nn.Embedding(
                            self.dynamic_hash_buckets,
                            self.dynamic_hash_width,
                            sparse=self.dynamic_hash_sparse,
                        )
                        for _ in range(self.dynamic_hash_tables)
                    )
                self.dynamic_hash_projection = nn.Linear(
                    self.dynamic_hash_width * self.dynamic_hash_tables,
                    d_model,
                    bias=False,
                )
                powers = []
                for table_index in range(self.dynamic_hash_tables):
                    table_powers = []
                    value = 1
                    multiplier = 257 + 6 * table_index
                    for _ in range(self.patch_size):
                        table_powers.append(value)
                        value = (value * multiplier) & (
                            self.dynamic_hash_buckets - 1
                        )
                    powers.append(table_powers)
                self.register_buffer(
                    "dynamic_hash_powers",
                    torch.tensor(powers, dtype=torch.int64).squeeze(0),
                )
            nn.init.zeros_(self.dynamic_position_scale.weight)
            nn.init.zeros_(self.dynamic_position_bias.weight)
        if self.patch_core_type == "gru":
            self.patch_core = nn.GRU(
                d_model,
                d_model,
                num_layers=self.patch_layers,
                batch_first=True,
            )
        elif self.patch_core_type == "selective_scan":
            self.patch_core = ParallelSelectivePatchCore(
                d_model, self.patch_layers
            )
        elif self.patch_core_type == "low_rank_selective_scan":
            self.patch_core = LowRankSelectivePatchCore(
                d_model, self.patch_layers, self.patch_selective_rank
            )
        else:
            self.patch_core = CausalAttentionPatchCore(
                d_model,
                self.patch_layers,
                self.patch_attention_heads,
            )
        self.to_abi = nn.Linear(d_model, d_abi)
        self.from_abi = nn.Linear(d_abi, d_model)
        self.local_projection = nn.Linear(d_model, local_width)
        if self.neural_context_sparse:
            self.neural_context_local_projection = nn.Linear(
                d_byte, local_width, bias=False
            )
        if (
            self.chunking_mode == "fixed"
            and self.local_decoder in {"gru", "lstm", "scan", "dilated_conv"}
        ):
            self.local_bos = nn.Parameter(torch.zeros(d_byte))
        if self.local_decoder in {"gru", "lstm"}:
            recurrent_type = nn.GRU if self.local_decoder == "gru" else nn.LSTM
            self.local_core = recurrent_type(
                d_byte,
                local_width,
                num_layers=self.local_gru_layers,
                batch_first=True,
            )
            if self.local_decoder == "lstm":
                self.local_cell_projection = nn.Linear(d_model, local_width)
            if self.local_continuous:
                self.local_context_input = nn.Linear(d_model, d_byte)
        elif self.local_decoder == "scan":
            if self.local_continuous:
                raise ValueError(
                    "scan is patch-causal and cannot use continuous local state"
                )
            self.local_context_gates = nn.Linear(
                d_model, local_width * 3
            )
            self.local_input_conv = nn.Conv1d(
                d_byte, local_width * 3, kernel_size=2
            )
            with torch.no_grad():
                self.local_context_gates.bias.zero_()
                self.local_context_gates.bias[
                    local_width : 2 * local_width
                ].fill_(2.0)
                self.local_input_conv.bias.zero_()
        elif self.local_decoder == "dilated_conv":
            self.local_input_projection = nn.Linear(
                d_byte, local_width, bias=False
            )
            # A continuous decoder operates on the complete scored byte stream,
            # so its receptive field must not be capped by the presentation
            # patch size.  Patch-local decoders retain the old cap because
            # larger dilations would only read left padding.
            dilations = tuple(
                self.local_dilation_growth**index
                for index in range(self.local_layers)
                if self.local_continuous
                or self.local_dilation_growth**index < self.patch_size
            )
            if not dilations:
                raise ValueError("dilated_conv requires a patch larger than one byte")
            self.local_dilations = dilations
            self.local_block_norms = nn.ModuleList(
                nn.LayerNorm(local_width) for _ in dilations
            )
            self.local_depthwise = nn.ModuleList(
                nn.Conv1d(
                    local_width,
                    local_width,
                    kernel_size=3,
                    dilation=dilation,
                    groups=local_width,
                )
                for dilation in dilations
            )
            self.local_channel_mixers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(local_width, self.local_rank, bias=False),
                    nn.SiLU(),
                    nn.Linear(self.local_rank, local_width * 2),
                )
                for _ in dilations
            )
        self.local_positions = nn.Embedding(patch_size, local_width)
        self.local_norm = nn.LayerNorm(local_width)
        if self.byte_head == "direct":
            self.direct_head = nn.Linear(local_width, 256)
        else:
            self.high_head = nn.Linear(local_width, 16)
            self.high_embedding = nn.Embedding(16, local_width)
            self.high_scale = nn.Embedding(16, local_width)
            nn.init.zeros_(self.high_scale.weight)
            self.low_norm = nn.LayerNorm(local_width)
            self.low_head = nn.Linear(local_width, 16)
        self.mixture_gate = nn.Linear(local_width, 1)
        if self.count_order_routing_enabled:
            router_input = local_width + (
                4 * (self.count_cake.max_order + 1)
                if self.count_order_stage_features
                else 4
            )
            if self.count_order_router_hidden:
                self.count_order_router = nn.Sequential(
                    nn.Linear(router_input, self.count_order_router_hidden),
                    nn.SiLU(),
                    nn.Linear(
                        self.count_order_router_hidden,
                        self.count_cake.max_order + 1,
                    ),
                )
            else:
                self.count_order_router = nn.Linear(
                    router_input, self.count_cake.max_order + 1
                )
        if self.confidence_gate_enabled:
            self.confidence_gate = nn.Linear(3, 1, bias=False)
        if self.expert_confidence_gate_enabled:
            self.expert_confidence_gate = nn.Linear(3, 1, bias=False)
        if self.count_distribution_gate_enabled:
            self.count_distribution_gate = nn.Linear(1, 1, bias=False)
        if int(gate_hidden_width) > 0:
            self.enable_nonlinear_gate(int(gate_hidden_width))
        with torch.no_grad():
            self.mixture_gate.weight.zero_()
            if self.count_order_routing_enabled:
                router_output = (
                    self.count_order_router[-1]
                    if self.count_order_router_hidden
                    else self.count_order_router
                )
                router_output.weight.zero_()
                router_output.bias.fill_(-4.0)
                router_output.bias[-1] = 4.0
            if self.confidence_gate_enabled:
                self.confidence_gate.weight.zero_()
            if self.expert_confidence_gate_enabled:
                self.expert_confidence_gate.weight.zero_()
            if self.count_distribution_gate_enabled:
                self.count_distribution_gate.weight.zero_()
            fraction = torch.tensor(float(initial_neural_fraction)).clamp(1e-5, 1 - 1e-5)
            self.mixture_gate.bias.fill_(torch.logit(fraction).item())

    def enable_nonlinear_gate(self, hidden_width: int) -> None:
        """Add a zero-initialized nonlinear residual to the expert gate."""
        hidden_width = int(hidden_width)
        if hidden_width <= 0:
            raise ValueError("gate hidden width must be positive")
        if self.gate_hidden_width:
            if self.gate_hidden_width != hidden_width:
                raise ValueError(
                    "nonlinear gate is already configured with a different width"
                )
            return
        local_width = self.mixture_gate.in_features
        self.gate_mlp = nn.Sequential(
            nn.Linear(local_width + 3, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 1),
        ).to(
            device=self.mixture_gate.weight.device,
            dtype=self.mixture_gate.weight.dtype,
        )
        with torch.no_grad():
            self.gate_mlp[-1].weight.zero_()
            self.gate_mlp[-1].bias.zero_()
        self.gate_hidden_width = hidden_width

    def _gate_logits(
        self,
        neural_hidden: torch.Tensor,
        count_features: torch.Tensor,
        expert_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_logit = self.mixture_gate(neural_hidden)
        if self.gate_hidden_width:
            gate_logit = gate_logit + self.gate_mlp(
                torch.cat([neural_hidden, count_features[..., :3]], dim=-1)
            )
        if self.confidence_gate_enabled:
            gate_logit = gate_logit + self.confidence_gate(count_features[..., :3])
        if self.count_distribution_gate_enabled:
            gate_logit = gate_logit + self.count_distribution_gate(
                count_features[..., 3:4]
            )
        if self.expert_confidence_gate_enabled:
            if expert_confidence is None:
                raise ValueError("expert-confidence gate requires causal features")
            gate_logit = gate_logit + self.expert_confidence_gate(
                expert_confidence
            )
        return gate_logit

    @staticmethod
    def _expert_confidence_features(
        neural_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Return target-independent confidence signals for expert routing."""
        probability = neural_probability.float().clamp_min(1e-30)
        log_probability = probability.log()
        entropy = -(probability * log_probability).sum(dim=-1) / math.log(256.0)
        top_two = probability.topk(2, dim=-1).values
        margin = top_two[..., 0] - top_two[..., 1]
        return torch.stack([entropy, top_two[..., 0], margin], dim=-1).to(
            dtype=neural_probability.dtype
        )

    @property
    def cache_enabled(self) -> bool:
        return bool(
            self.online_cache_specs
            or self.recent_cache_specs
            or self.normalized_cache_specs
        )

    def _new_causal_cache(self) -> CausalCompositeByteCache:
        return CausalCompositeByteCache(
            exact_specs=self.online_cache_specs,
            recent_specs=self.recent_cache_specs,
            normalized_specs=self.normalized_cache_specs,
            window=self.online_cache_window,
            normalization=self.cache_normalization,
        )

    @property
    def neural_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def logical_total_parameters(self) -> int:
        return self.neural_parameters + self.count_cake.state_entries

    def _neural_probabilities(
        self,
        local: torch.Tensor,
        high_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.byte_head == "direct":
            return F.softmax(self.direct_head(local), dim=-1)
        if high_values is None:
            high_values = torch.arange(16, device=local.device)
        high_log_probability = F.log_softmax(
            self.high_head(local), dim=-1
        )
        low_hidden = self.low_norm(
            local.unsqueeze(-2)
            * (1.0 + self.high_scale(high_values))
            + self.high_embedding(high_values)
        )
        low_log_probability = F.log_softmax(
            self.low_head(low_hidden), dim=-1
        )
        return (
            high_log_probability.unsqueeze(-1) + low_log_probability
        ).flatten(-2).exp()

    def _patch_context(self, rows: torch.Tensor) -> torch.Tensor:
        if self.chunking_mode != "fixed":
            raise RuntimeError("fixed patch context is unavailable in delimiter mode")
        usable = rows.shape[1] // self.patch_size * self.patch_size
        patches = rows[:, :usable].reshape(rows.shape[0], -1, self.patch_size)
        if self.neural_context_sparse:
            byte_features = self._causal_contextual_embedding(
                rows[:, :usable]
            ).reshape(
                rows.shape[0],
                -1,
                self.patch_size,
                self.byte_embedding.embedding_dim,
            )
        else:
            byte_features = self.byte_embedding(patches)
        embedded = byte_features.flatten(-2)
        features = torch.tanh(self.patch_projection(embedded))
        hidden, _ = self.patch_core(features)
        # State after patch i predicts patch i+1.  Every prediction traverses
        # the ABI bridge, preserving a real composition point for domain cakes.
        first_context = self.prediction_start // self.patch_size - 1
        context = hidden[:, first_context:-1]
        if not self.scratchpad_stride:
            abi = self.to_abi(context)
            return context + self.from_abi(abi)

        if not self.local_recurrent or not self.local_continuous:
            raise RuntimeError(
                "scratchpads require a continuous recurrent local decoder"
            )
        target_patch_features = byte_features[
            :, first_context + 1 : first_context + 1 + context.shape[1]
        ]
        scratchpad_contexts = [context]
        for cut in range(
            self.scratchpad_stride,
            self.patch_size,
            self.scratchpad_stride,
        ):
            partial = target_patch_features.clone()
            partial[:, :, cut:] = 0
            partial_features = torch.tanh(
                self.patch_projection(partial.flatten(-2))
            )
            input_gates = F.linear(
                partial_features,
                self.patch_core.weight_ih_l0,
                self.patch_core.bias_ih_l0,
            )
            hidden_gates = F.linear(
                context,
                self.patch_core.weight_hh_l0,
                self.patch_core.bias_hh_l0,
            )
            input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            candidate = torch.tanh(input_new + reset * hidden_new)
            scratchpad_contexts.append(
                candidate + update * (context - candidate)
            )
        segment_context = torch.stack(scratchpad_contexts, dim=-2)
        byte_context = segment_context.repeat_interleave(
            self.scratchpad_stride, dim=-2
        )
        byte_context = byte_context[..., : self.patch_size, :]
        abi = self.to_abi(byte_context)
        return byte_context + self.from_abi(abi)

    def _dynamic_neural_log_probs(
        self,
        rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal next-byte scores from delimiter-aware variable chunks."""
        if not self.local_recurrent or not self.local_continuous:
            raise RuntimeError(
                "delimiter chunking requires a continuous recurrent local decoder"
            )
        batch, length = rows.shape
        positions = torch.arange(length, device=rows.device).unsqueeze(0)
        base_boundary = torch.zeros(
            (batch, length), device=rows.device, dtype=torch.bool
        )
        base_boundary[:, 0] = True
        base_boundary[:, 1:] = self.chunk_delimiter_table[rows[:, :-1]]
        base_index = torch.where(
            base_boundary,
            positions,
            torch.zeros_like(positions),
        )
        last_base = torch.cummax(base_index, dim=1).values
        capped_boundary = (positions - last_base).remainder(self.patch_size) == 0
        boundary = base_boundary | capped_boundary
        boundary_index = torch.where(
            boundary,
            positions,
            torch.zeros_like(positions),
        )
        last_boundary = torch.cummax(boundary_index, dim=1).values
        chunk_position = positions - last_boundary
        chunk_ids = boundary.to(torch.int64).cumsum(dim=1) - 1
        chunk_count = int(chunk_ids[:, -1].max().item()) + 1

        embedded = self.byte_embedding(rows)
        position_scale = self.dynamic_position_scale(chunk_position)
        position_bias = self.dynamic_position_bias(chunk_position)
        contribution = self.dynamic_byte_projection(
            embedded * (1.0 + position_scale) + position_bias
        )
        chunk_sum = torch.zeros(
            batch,
            chunk_count,
            self.d_model,
            device=rows.device,
            dtype=contribution.dtype,
        )
        chunk_sum.scatter_add_(
            1,
            chunk_ids.unsqueeze(-1).expand(-1, -1, self.d_model),
            contribution,
        )
        chunk_length = torch.zeros(
            batch,
            chunk_count,
            device=rows.device,
            dtype=contribution.dtype,
        )
        chunk_length.scatter_add_(
            1,
            chunk_ids,
            torch.ones_like(chunk_ids, dtype=contribution.dtype),
        )
        chunk_features = torch.tanh(
            self.dynamic_chunk_norm(
                chunk_sum / chunk_length.clamp_min(1.0).sqrt().unsqueeze(-1)
            )
        )
        if self.dynamic_hash_buckets:
            # A deterministic polynomial hash preserves span identity without
            # a tokenizer or vocabulary.  The learned table is ordinary model
            # state and is included in the exact logical parameter budget.
            if self.dynamic_hash_tables == 1:
                hash_contribution = (
                    (rows.to(torch.int64) + 1)
                    * self.dynamic_hash_powers[chunk_position]
                ).bitwise_and(self.dynamic_hash_buckets - 1)
                chunk_hash = torch.zeros(
                    batch, chunk_count, device=rows.device, dtype=torch.int64
                )
                chunk_hash.scatter_add_(1, chunk_ids, hash_contribution)
                chunk_hash.bitwise_and_(self.dynamic_hash_buckets - 1)
                hash_features = self.dynamic_hash_embedding(chunk_hash)
            else:
                hash_features = []
                for table_index, embedding in enumerate(
                    self.dynamic_hash_embeddings
                ):
                    hash_contribution = (
                        (rows.to(torch.int64) + 1)
                        * self.dynamic_hash_powers[
                            table_index, chunk_position
                        ]
                    ).bitwise_and(self.dynamic_hash_buckets - 1)
                    chunk_hash = torch.zeros(
                        batch,
                        chunk_count,
                        device=rows.device,
                        dtype=torch.int64,
                    )
                    chunk_hash.scatter_add_(1, chunk_ids, hash_contribution)
                    chunk_hash.bitwise_and_(self.dynamic_hash_buckets - 1)
                    hash_features.append(embedding(chunk_hash))
                hash_features = torch.cat(hash_features, dim=-1)
            chunk_features = chunk_features + self.dynamic_hash_projection(
                hash_features
            )
        chunk_hidden, _ = self.patch_core(chunk_features)
        zero = torch.zeros_like(chunk_hidden[:, :1])
        causal_chunk_context = torch.cat(
            [zero, chunk_hidden[:, :-1]], dim=1
        )
        causal_chunk_context = causal_chunk_context + self.from_abi(
            self.to_abi(causal_chunk_context)
        )
        byte_context = causal_chunk_context.gather(
            1,
            chunk_ids.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        targets = rows[:, self.prediction_start :]
        teacher = rows[:, self.prediction_start - 1 : -1]
        target_context = byte_context[:, self.prediction_start :]
        recurrent_input = self.byte_embedding(teacher) + self.local_context_input(
            target_context
        )
        initial = self.local_projection(
            byte_context[:, self.prediction_start - 1]
        ).unsqueeze(0).expand(self.local_gru_layers, -1, -1).contiguous()
        hidden, _ = self.local_core(recurrent_input, initial)
        hidden = self.local_norm(
            hidden + self.local_positions(chunk_position[:, self.prediction_start :])
        )
        if self.byte_head == "direct":
            log_probability = F.log_softmax(self.direct_head(hidden), dim=-1)
            observed = log_probability.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            return observed, hidden
        high_target = targets.bitwise_right_shift(4)
        low_target = targets.bitwise_and(15)
        high_log_probability = F.log_softmax(self.high_head(hidden), dim=-1)
        low_hidden = self.low_norm(
            hidden * (1.0 + self.high_scale(high_target))
            + self.high_embedding(high_target)
        )
        low_log_probability = F.log_softmax(self.low_head(low_hidden), dim=-1)
        observed = high_log_probability.gather(
            -1, high_target.unsqueeze(-1)
        ).squeeze(-1) + low_log_probability.gather(
            -1, low_target.unsqueeze(-1)
        ).squeeze(-1)
        return observed, hidden

    def _neural_log_probs(
        self,
        context: torch.Tensor,
        targets: torch.Tensor,
        rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(self.patch_size, device=context.device)
        initial = self.local_projection(context)
        if self.neural_context_sparse:
            if rows is None:
                raise ValueError(
                    "sparse neural context residuals require source rows"
                )
            preceding_context = self._causal_contextual_embedding(
                rows[:, :-1]
            )[:, self.prediction_start - 1 :]
            if preceding_context.shape[1] != initial.shape[1]:
                raise ValueError(
                    "sparse neural context residuals do not align with targets"
                )
            initial = initial + self.neural_context_local_projection(
                preceding_context
            )
        if self.local_decoder == "scan":
            bos = self.local_bos.expand(*targets.shape[:-1], 1, -1)
            teacher = torch.cat(
                [bos, self.byte_embedding(targets[..., :-1])], dim=-2
            )
            leading = context.shape[:-1]
            flat_teacher = teacher.reshape(
                -1, self.patch_size, teacher.shape[-1]
            )
            gates = self.local_input_conv(
                F.pad(flat_teacher.transpose(1, 2), (1, 0))
            ).transpose(1, 2)
            gates = gates + self.local_context_gates(context).reshape(
                -1, 1, initial.shape[-1] * 3
            )
            proposal_logits, forget_logits, output_logits = gates.chunk(3, dim=-1)
            proposal = torch.tanh(proposal_logits.float())
            forget = torch.sigmoid(forget_logits.float())
            output = torch.sigmoid(output_logits.float())
            product = torch.cumprod(forget, dim=1)
            weighted = (1.0 - forget) * proposal / product.clamp_min(1e-7)
            initial_state = torch.tanh(initial).reshape(
                -1, 1, initial.shape[-1]
            ).float()
            state = product * (initial_state + torch.cumsum(weighted, dim=1))
            hidden = (output * state).to(dtype=gates.dtype).reshape(
                *leading, self.patch_size, initial.shape[-1]
            )
        elif self.local_decoder == "dilated_conv":
            bos = self.local_bos.expand(*targets.shape[:-1], 1, -1)
            if self.local_continuous:
                # Keep one causal byte stream per row.  The former
                # patch-fragmented implementation launched thousands of tiny
                # convolutions and also reset its receptive field at every
                # patch boundary.  Context is still constant within each
                # patch, but the learned byte path now crosses boundaries.
                flat_targets = targets.reshape(targets.shape[0], -1)
                flat_teacher = torch.cat(
                    [
                        bos[:, 0],
                        self.byte_embedding(flat_targets[:, :-1]),
                    ],
                    dim=1,
                )
                expanded_initial = initial.unsqueeze(-2).expand(
                    *initial.shape[:-1], self.patch_size, initial.shape[-1]
                )
                hidden = expanded_initial.reshape(
                    targets.shape[0], -1, initial.shape[-1]
                ) + self.local_input_projection(flat_teacher)
                leading = targets.shape[:-1]
            else:
                teacher = torch.cat(
                    [bos, self.byte_embedding(targets[..., :-1])], dim=-2
                )
                hidden = (
                    initial.unsqueeze(-2)
                    + self.local_input_projection(teacher)
                )
                leading = hidden.shape[:-2]
                hidden = hidden.reshape(-1, self.patch_size, hidden.shape[-1])
            for dilation, norm, convolution, mixer in zip(
                self.local_dilations,
                self.local_block_norms,
                self.local_depthwise,
                self.local_channel_mixers,
            ):
                normalized = norm(hidden)
                mixed = convolution(
                    F.pad(normalized.transpose(1, 2), (2 * dilation, 0))
                ).transpose(1, 2)
                gate, value = mixer(mixed).chunk(2, dim=-1)
                hidden = hidden + torch.sigmoid(gate) * F.silu(value)
            hidden = hidden.reshape(
                *leading, self.patch_size, hidden.shape[-1]
            )
        elif self.local_recurrent:
            bos = self.local_bos.expand(*targets.shape[:-1], 1, -1)
            teacher = torch.cat(
                [bos, self.byte_embedding(targets[..., :-1])],
                dim=-2,
            )
            if self.local_continuous:
                if rows is None:
                    flat_teacher = teacher.reshape(
                        targets.shape[0], -1, teacher.shape[-1]
                    )
                else:
                    # A continuous recurrent stream must consume the actual
                    # byte immediately preceding every target.  Constructing
                    # teacher input patch-by-patch inserted a synthetic BOS at
                    # each boundary and created a train/generate mismatch.
                    expected = targets.shape[-2] * targets.shape[-1]
                    preceding = rows[
                        :, self.prediction_start - 1 : -1
                    ]
                    if preceding.shape[1] != expected:
                        raise ValueError(
                            "continuous teacher bytes do not align with targets"
                        )
                    flat_teacher = self._contextual_teacher_embedding(rows)
                if context.ndim == 4:
                    byte_context = context.reshape(
                        targets.shape[0], -1, context.shape[-1]
                    )
                    repeated_context = self.local_context_input(byte_context)
                    initial_sequence = initial.reshape(
                        targets.shape[0], -1, initial.shape[-1]
                    )
                else:
                    repeated_context = self.local_context_input(
                        context
                    ).unsqueeze(-2).expand(
                        *context.shape[:-1],
                        self.patch_size,
                        self.byte_embedding.embedding_dim,
                    ).reshape(
                        targets.shape[0],
                        -1,
                        self.byte_embedding.embedding_dim,
                    )
                    initial_sequence = initial
                flat_teacher = flat_teacher + repeated_context
                continuous_initial = initial_sequence[:, 0].unsqueeze(0).expand(
                    self.local_gru_layers, -1, -1
                ).contiguous()
                if self.local_decoder == "lstm":
                    continuous_cell = self.local_cell_projection(
                        context[:, 0]
                    ).unsqueeze(0).expand(
                        self.local_gru_layers, -1, -1
                    ).contiguous()
                    recurrent_initial = (
                        continuous_initial,
                        continuous_cell,
                    )
                else:
                    recurrent_initial = continuous_initial
                hidden, _ = self.local_core(flat_teacher, recurrent_initial)
                hidden = hidden.reshape(*targets.shape, hidden.shape[-1])
            else:
                flat_teacher = teacher.reshape(
                    -1, self.patch_size, teacher.shape[-1]
                )
                flat_initial = initial.reshape(
                    -1, initial.shape[-1]
                ).unsqueeze(0).expand(
                    self.local_gru_layers, -1, -1
                ).contiguous()
                if self.local_decoder == "lstm":
                    flat_cell = self.local_cell_projection(context).reshape(
                        -1, initial.shape[-1]
                    ).unsqueeze(0).expand(
                        self.local_gru_layers, -1, -1
                    ).contiguous()
                    recurrent_initial = (flat_initial, flat_cell)
                else:
                    recurrent_initial = flat_initial
                hidden, _ = self.local_core(flat_teacher, recurrent_initial)
                hidden = hidden.reshape(
                    *targets.shape,
                    hidden.shape[-1],
                )
        else:
            hidden = initial.unsqueeze(-2)
        hidden = self.local_norm(hidden + self.local_positions(positions))
        if self.byte_head == "direct":
            log_probability = F.log_softmax(self.direct_head(hidden), dim=-1)
            observed = log_probability.gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            return observed, hidden
        high_target = targets.bitwise_right_shift(4)
        low_target = targets.bitwise_and(15)
        high_log_probability = F.log_softmax(self.high_head(hidden), dim=-1)
        low_hidden = self.low_norm(
            hidden * (1.0 + self.high_scale(high_target))
            + self.high_embedding(high_target)
        )
        low_log_probability = F.log_softmax(self.low_head(low_hidden), dim=-1)
        observed = high_log_probability.gather(
            -1, high_target.unsqueeze(-1)
        ).squeeze(-1) + low_log_probability.gather(
            -1, low_target.unsqueeze(-1)
        ).squeeze(-1)
        return observed, hidden

    def _contextual_teacher_embedding(self, rows: torch.Tensor) -> torch.Tensor:
        """Embed the exact causal context immediately preceding each target."""
        preceding = rows[:, self.prediction_start - 1 : -1]
        embedded = self.byte_embedding(preceding)
        if not self.neural_context_buckets:
            return embedded
        mask = self.neural_context_buckets - 1
        context_ids = torch.zeros_like(preceding, dtype=torch.int64)
        target_count = preceding.shape[1]
        for offset in range(self.neural_context_order):
            source_start = self.prediction_start - self.neural_context_order + offset
            source = rows[:, source_start : source_start + target_count]
            context_ids.mul_(257).add_(source + 1).bitwise_and_(mask)
        return embedded + self.neural_context_embedding(context_ids)

    def _causal_contextual_embedding(self, rows: torch.Tensor) -> torch.Tensor:
        """Embed each known byte plus a hash of its causal suffix."""
        embedded = self.byte_embedding(rows)
        if not self.neural_context_buckets:
            return embedded
        mask = self.neural_context_buckets - 1
        width = rows.shape[1]
        padded = F.pad(rows + 1, (self.neural_context_order - 1, 0))
        context_ids = torch.zeros_like(rows, dtype=torch.int64)
        for offset in range(self.neural_context_order):
            context_ids.mul_(257).add_(
                padded[:, offset : offset + width]
            ).bitwise_and_(mask)
        return embedded + self.neural_context_embedding(context_ids)

    def loss(
        self,
        rows: torch.Tensor,
        *,
        neural_auxiliary_weight: float = 0.1,
    ) -> torch.Tensor:
        mixture_log_probability, neural_log_probability = self.target_log_probs(
            rows,
            return_neural=True,
        )
        return -mixture_log_probability.mean() - float(
            neural_auxiliary_weight
        ) * neural_log_probability.mean()

    def neural_loss(self, rows: torch.Tensor) -> torch.Tensor:
        """Train the neural host without re-evaluating the frozen sparse cake."""
        if rows.ndim != 2 or rows.shape[1] < self.prediction_start + self.patch_size:
            raise ValueError("rows must contain at least two complete patches")
        usable = rows.shape[1] // self.patch_size * self.patch_size
        rows = rows[:, :usable].to(dtype=torch.int64)
        if self.chunking_mode == "delimiter":
            neural_log_probability, _ = self._dynamic_neural_log_probs(rows)
            return -neural_log_probability.mean()
        context = self._patch_context(rows)
        targets = rows[:, self.prediction_start :].reshape(
            rows.shape[0], -1, self.patch_size
        )
        neural_log_probability, _ = self._neural_log_probs(
            context, targets, rows=rows
        )
        return -neural_log_probability.mean()

    def target_log_probs(
        self,
        rows: torch.Tensor,
        *,
        return_neural: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return causal observed-byte log probabilities for complete rows."""
        if rows.ndim != 2 or rows.shape[1] < self.prediction_start + self.patch_size:
            raise ValueError("rows must contain at least two complete patches")
        usable = rows.shape[1] // self.patch_size * self.patch_size
        rows = rows[:, :usable].to(dtype=torch.int64)
        if self.chunking_mode == "delimiter":
            neural_log_probability, neural_hidden = self._dynamic_neural_log_probs(
                rows
            )
        else:
            context = self._patch_context(rows)
            targets = rows[:, self.prediction_start :].reshape(
                rows.shape[0], -1, self.patch_size
            )
            neural_log_probability, neural_hidden = self._neural_log_probs(
                context, targets, rows=rows
            )
        count_result = self.count_cake.target_log_probs(
            rows,
            start=self.prediction_start,
            return_features=True,
            return_stages=self.count_order_routing_enabled,
            return_stage_features=(
                self.count_order_routing_enabled
                and self.count_order_stage_features
            ),
        )
        if self.count_order_routing_enabled:
            if self.count_order_stage_features:
                (
                    count_log_probability,
                    count_features,
                    count_stage_log_probability,
                    count_stage_features,
                ) = count_result
            else:
                (
                    count_log_probability,
                    count_features,
                    count_stage_log_probability,
                ) = count_result
        else:
            count_log_probability, count_features = count_result
        count_log_probability = count_log_probability.reshape_as(
            neural_log_probability
        )
        count_features = count_features.reshape(
            *neural_log_probability.shape, 4
        )
        if self.count_order_routing_enabled:
            count_stage_log_probability = count_stage_log_probability.reshape(
                *neural_log_probability.shape,
                self.count_cake.max_order + 1,
            )
            route_features = count_features
            if self.count_order_stage_features:
                route_features = count_stage_features.reshape(
                    *neural_log_probability.shape,
                    (self.count_cake.max_order + 1) * 4,
                )
            route_log_probability = F.log_softmax(
                self.count_order_router(
                    torch.cat([neural_hidden, route_features], dim=-1)
                ),
                dim=-1,
            )
            count_log_probability = torch.logsumexp(
                route_log_probability + count_stage_log_probability,
                dim=-1,
            )
        expert_confidence = None
        if self.expert_confidence_gate_enabled:
            expert_confidence = self._expert_confidence_features(
                self._neural_probabilities(neural_hidden)
            )
        gate_logit = self._gate_logits(
            neural_hidden, count_features, expert_confidence
        )
        gate = torch.sigmoid(gate_logit).squeeze(-1)
        mixture_log_probability = torch.logaddexp(
            torch.log1p(-gate.clamp(max=1 - 1e-6)) + count_log_probability,
            gate.clamp_min(1e-6).log() + neural_log_probability,
        ).reshape(rows.shape[0], -1)
        if return_neural:
            return mixture_log_probability, neural_log_probability.reshape(
                rows.shape[0], -1
            )
        return mixture_log_probability

    @torch.no_grad()
    def _sample_patch(
        self,
        context: torch.Tensor,
        history: torch.Tensor,
        *,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
        online_cache: CausalCompositeByteCache | None = None,
        online_history: bytearray | None = None,
        continuous_state: dict | None = None,
    ) -> torch.Tensor:
        context = context + self.from_abi(self.to_abi(context))
        positions = torch.arange(self.patch_size, device=history.device)
        high_values = torch.arange(16, device=history.device)
        if self.local_decoder == "scan":
            scan_state = torch.tanh(self.local_projection(context))
            scan_context_gates = self.local_context_gates(context)
            scan_input = self.local_bos.reshape(1, -1)
            scan_previous = torch.zeros_like(scan_input)
            neural_probability = None
            gates = None
        elif self.local_decoder == "dilated_conv":
            local_base = self.local_projection(context)
            neural_probability = None
            gates = None
        elif self.local_recurrent:
            if self.local_continuous and continuous_state is not None and "hidden" in continuous_state:
                recurrent_state = continuous_state["hidden"]
            else:
                recurrent_state = self.local_projection(context).unsqueeze(0).expand(
                    self.local_gru_layers, -1, -1
                ).contiguous()
            recurrent_input = self.local_bos.reshape(1, 1, -1)
            context_input = (
                self.local_context_input(context).reshape(1, 1, -1)
                if self.local_continuous
                else 0.0
            )
            neural_probability = None
            gates = None
        else:
            position_local = self.local_norm(
                self.local_projection(context).unsqueeze(-2)
                + self.local_positions(positions).unsqueeze(0)
            )[0]
            neural_probability = self._neural_probabilities(
                position_local, high_values
            )
        generated: list[torch.Tensor] = []
        for offset in range(self.patch_size):
            if self.local_decoder == "scan":
                weights = self.local_input_conv.weight
                scan_gates = (
                    F.linear(scan_previous, weights[:, :, 0])
                    + F.linear(scan_input, weights[:, :, 1])
                    + self.local_input_conv.bias
                    + scan_context_gates
                )
                proposal_logits, forget_logits, output_logits = scan_gates.chunk(
                    3, dim=-1
                )
                proposal = torch.tanh(proposal_logits)
                forget = torch.sigmoid(forget_logits)
                scan_state = forget * scan_state + (1.0 - forget) * proposal
                local = self.local_norm(
                    torch.sigmoid(output_logits) * scan_state
                    + self.local_positions.weight[offset]
                )[0]
                neural = self._neural_probabilities(local, high_values)
            elif self.local_decoder == "dilated_conv":
                bos = self.local_bos.reshape(1, 1, -1)
                if generated:
                    prefix = torch.stack(generated).reshape(1, -1)
                    teacher = torch.cat(
                        [bos, self.byte_embedding(prefix)], dim=1
                    )
                else:
                    teacher = bos
                sequence = (
                    local_base.unsqueeze(1)
                    + self.local_input_projection(teacher)
                )
                for dilation, norm, convolution, mixer in zip(
                    self.local_dilations,
                    self.local_block_norms,
                    self.local_depthwise,
                    self.local_channel_mixers,
                ):
                    normalized = norm(sequence)
                    mixed = convolution(
                        F.pad(
                            normalized.transpose(1, 2),
                            (2 * dilation, 0),
                        )
                    ).transpose(1, 2)
                    block_gate, value = mixer(mixed).chunk(2, dim=-1)
                    sequence = sequence + torch.sigmoid(block_gate) * F.silu(
                        value
                    )
                local = self.local_norm(
                    sequence[:, -1] + self.local_positions.weight[offset]
                )[0]
                neural = self._neural_probabilities(local, high_values)
            elif self.local_recurrent:
                _, recurrent_state = self.local_core(
                    recurrent_input + context_input,
                    recurrent_state,
                )
                local = self.local_norm(
                    recurrent_state[-1]
                    + self.local_positions.weight[offset]
                )[0]
                neural = self._neural_probabilities(local, high_values)
            else:
                neural = neural_probability[offset]
                local = position_local[offset]
            count_result = self.count_cake.next_probabilities(
                history,
                return_features=True,
                return_stages=self.count_order_routing_enabled,
                return_stage_features=(
                    self.count_order_routing_enabled
                    and self.count_order_stage_features
                ),
            )
            if self.count_order_routing_enabled:
                if self.count_order_stage_features:
                    count, count_features, count_stages, count_stage_features = (
                        count_result
                    )
                    route_features = count_stage_features.flatten()
                else:
                    count, count_features, count_stages = count_result
                    route_features = count_features
                route = F.softmax(
                    self.count_order_router(
                        torch.cat(
                            [local, route_features], dim=-1
                        )
                    ),
                    dim=-1,
                )
                count = (route.unsqueeze(-1) * count_stages).sum(dim=0)
            else:
                count, count_features = count_result
            expert_confidence = (
                self._expert_confidence_features(neural)
                if self.expert_confidence_gate_enabled
                else None
            )
            gate_logit = self._gate_logits(
                local, count_features, expert_confidence
            ).squeeze()
            gate = torch.sigmoid(gate_logit)
            probability = (
                (1.0 - gate) * count
                + gate * neural
            )
            if online_cache is not None:
                if online_history is None:
                    raise ValueError("online cache requires its complete history")
                probability = online_cache.probabilities(
                    probability,
                    online_history,
                )
            if temperature <= 0:
                next_byte = probability.argmax()
            else:
                logits = probability.clamp_min(1e-30).log() / temperature
                next_byte = torch.multinomial(
                    logits.softmax(dim=-1),
                    1,
                    generator=generator,
                ).squeeze(0)
            generated.append(next_byte)
            history = torch.cat([history, next_byte.view(1)])
            if online_cache is not None:
                online_cache.update(online_history, int(next_byte))
                online_history.append(int(next_byte))
                if len(online_history) > online_cache.max_order:
                    del online_history[
                        : len(online_history) - online_cache.max_order
                    ]
            if self.local_recurrent:
                recurrent_input = self.byte_embedding(next_byte).reshape(
                    1, 1, -1
                )
            elif self.local_decoder == "scan":
                scan_previous = scan_input
                scan_input = self.byte_embedding(next_byte).reshape(1, -1)
        if self.local_continuous and continuous_state is not None:
            continuous_state["hidden"] = recurrent_state
        return torch.stack(generated).unsqueeze(0)

    @torch.no_grad()
    def generate_next_patch(
        self,
        rows: torch.Tensor,
        *,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.chunking_mode != "fixed":
            raise NotImplementedError(
                "delimiter-chunk generation is pending architecture promotion"
            )
        if self.patch_core_type != "gru":
            raise NotImplementedError(
                "selective-scan generation is pending architecture promotion"
            )
        if rows.ndim != 2 or rows.shape[1] < self.patch_size:
            raise ValueError("generation requires at least one complete patch")
        if rows.shape[0] != 1:
            raise ValueError("the reference generator currently supports batch size one")
        usable = rows.shape[1] // self.patch_size * self.patch_size
        rows = rows[:, :usable].to(dtype=torch.int64)
        patches = rows.reshape(1, -1, self.patch_size)
        features = torch.tanh(
            self.patch_projection(self.byte_embedding(patches).flatten(-2))
        )
        _, recurrent_state = self.patch_core(features)
        online_cache = None
        online_history = None
        continuous_state = {} if self.local_continuous else None
        if self.cache_enabled:
            online_cache = self._new_causal_cache()
            online_history = online_cache.prefill(
                bytes(rows[0].detach().cpu().tolist())
            )
        return self._sample_patch(
            recurrent_state[-1],
            rows[0],
            temperature=temperature,
            generator=generator,
            online_cache=online_cache,
            online_history=online_history,
            continuous_state=continuous_state,
        )

    @torch.no_grad()
    def begin_cached_generation(self, rows: torch.Tensor) -> dict:
        if self.chunking_mode != "fixed":
            raise NotImplementedError(
                "delimiter-chunk cached generation is pending architecture promotion"
            )
        if self.patch_core_type != "gru":
            raise NotImplementedError(
                "selective-scan cached generation is pending architecture promotion"
            )
        """Prefill the patch recurrence once for incremental generation."""
        if rows.ndim != 2 or rows.shape[0] != 1:
            raise ValueError("cached generation requires shape [1, bytes]")
        usable = rows.shape[1] // self.patch_size * self.patch_size
        if usable < self.patch_size:
            raise ValueError("cached generation requires one complete patch")
        rows = rows[:, :usable].to(dtype=torch.int64)
        patches = rows.reshape(1, -1, self.patch_size)
        features = torch.tanh(
            self.patch_projection(self.byte_embedding(patches).flatten(-2))
        )
        _, recurrent_state = self.patch_core(features)
        state = {
            "recurrent_state": recurrent_state,
            "history": rows[0, -self.count_cake.max_order :].clone(),
        }
        if self.cache_enabled:
            online_cache = self._new_causal_cache()
            online_history = online_cache.prefill(
                bytes(rows[0].detach().cpu().tolist())
            )
            state["online_cache"] = online_cache
            state["online_history"] = online_history
            # Accelerated backends retain the same prompt on-device and replay
            # the bounded causal recipe without Python dictionary lookups.
            state["full_history"] = rows[0].clone()
        if self.local_continuous:
            state["continuous_state"] = {}
        return state

    @torch.no_grad()
    def generate_cached(
        self,
        state: dict,
        *,
        patches: int = 1,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate patches while updating only one recurrent patch state."""
        if patches <= 0:
            raise ValueError("patches must be positive")
        outputs: list[torch.Tensor] = []
        for _ in range(patches):
            generated = self._sample_patch(
                state["recurrent_state"][-1],
                state["history"],
                temperature=temperature,
                generator=generator,
                online_cache=state.get("online_cache"),
                online_history=state.get("online_history"),
                continuous_state=state.get("continuous_state"),
            )
            outputs.append(generated)
            state["history"] = torch.cat(
                [state["history"], generated[0]],
            )[-self.count_cake.max_order :]
            feature = torch.tanh(
                self.patch_projection(
                    self.byte_embedding(generated).flatten(-2)
                )
            ).unsqueeze(1)
            _, state["recurrent_state"] = self.patch_core(
                feature,
                state["recurrent_state"],
            )
        return torch.cat(outputs, dim=-1)


def assert_parameter_budget(
    model: HierarchicalCountCakeLM,
    *,
    target: int,
    relative_tolerance: float = 0.01,
) -> None:
    difference = abs(model.logical_total_parameters - int(target))
    if difference > math.ceil(int(target) * float(relative_tolerance)):
        raise ValueError(
            "logical parameter mismatch: "
            f"model={model.logical_total_parameters}, target={target}, "
            f"tolerance={relative_tolerance:.3%}"
        )


def save_count_cake_bundle(
    model: HierarchicalCountCakeLM,
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> dict:
    """Write a portable, compressed, pickle-free CountCake bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cake = model.count_cake
    manifest = {
        "format": "layercake-count-cake-bundle/1",
        "model": {
            "patch_size": model.patch_size,
            "chunking_mode": model.chunking_mode,
            "prediction_start": model.prediction_start,
            "d_byte": model.byte_embedding.embedding_dim,
            "d_model": model.d_model,
            "d_abi": model.to_abi.out_features,
            "patch_layers": model.patch_layers,
            "patch_core_type": model.patch_core_type,
            "patch_selective_rank": model.patch_selective_rank,
            "patch_attention_heads": model.patch_attention_heads,
            "scratchpad_stride": model.scratchpad_stride,
            "dynamic_hash_buckets": model.dynamic_hash_buckets,
            "dynamic_hash_width": model.dynamic_hash_width,
            "dynamic_hash_tables": model.dynamic_hash_tables,
            "dynamic_hash_sparse": model.dynamic_hash_sparse,
            "neural_context_buckets": model.neural_context_buckets,
            "neural_context_order": model.neural_context_order,
            "neural_context_sparse": model.neural_context_sparse,
            "local_width": model.mixture_gate.in_features,
            "local_decoder": model.local_decoder,
            "local_layers": model.local_layers,
            "local_dilation_growth": model.local_dilation_growth,
            "local_gru_layers": model.local_gru_layers,
            "local_rank": model.local_rank,
            "byte_head": model.byte_head,
            "local_recurrent": model.local_recurrent,
            "local_continuous": model.local_continuous,
            "confidence_gate": model.confidence_gate_enabled,
            "expert_confidence_gate": model.expert_confidence_gate_enabled,
            "count_distribution_gate": model.count_distribution_gate_enabled,
            "count_order_routing": model.count_order_routing_enabled,
            "count_order_stage_features": model.count_order_stage_features,
            "count_order_router_hidden": model.count_order_router_hidden,
            "gate_hidden_width": model.gate_hidden_width,
            "online_cache_specs": [list(spec) for spec in model.online_cache_specs],
            "online_cache_window": model.online_cache_window,
            "recent_cache_specs": [list(spec) for spec in model.recent_cache_specs],
            "normalized_cache_specs": [
                list(spec) for spec in model.normalized_cache_specs
            ],
            "cache_normalization": model.cache_normalization,
        },
        "count_cake": {
            "state_budget": cake.state_budget,
            "state_entries": cake.state_entries,
            "corpus_bytes": cake.corpus_bytes,
            "max_order": cake.max_order,
            "order_entries": list(cake.order_entries),
            "order_encodings": list(cake.order_encodings),
            "context_hash_bits": list(cake.context_hash_bits),
            "backoff_strengths": list(cake.backoff_strengths),
            "backoff_mode": cake.backoff_mode,
            "discount": cake.discount,
        },
        "parameters": {
            "logical_total": model.logical_total_parameters,
            "neural": model.neural_parameters,
        },
        "metadata": metadata or {},
    }
    arrays: dict[str, np.ndarray] = {
        "manifest_json": np.frombuffer(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            ),
            dtype=np.uint8,
        ).copy(),
        "count_unigram": cake.unigram_counts.detach()
        .round()
        .to(torch.int64)
        .cpu()
        .numpy()
        .astype(np.uint32),
    }
    for order in range(1, cake.max_order + 1):
        key_dtype = np.uint16 if order == 1 else np.uint32 if order <= 3 else np.uint64
        arrays[f"count_keys_{order}"] = (
            getattr(cake, f"keys_{order}").detach().cpu().numpy().astype(key_dtype)
        )
        arrays[f"count_values_{order}"] = (
            getattr(cake, f"counts_{order}")
            .detach()
            .round()
            .to(torch.int64)
            .cpu()
            .numpy()
            .astype(np.uint32)
        )
        # Persist totals even for packed orders.  Recomputing large float32
        # reductions with CUDA atomics is nondeterministic across receivers.
        arrays[f"count_context_totals_{order}"] = (
            getattr(cake, f"context_totals_{order}")
            .detach()
            .round()
            .to(torch.int64)
            .cpu()
            .numpy()
            .astype(np.uint32)
        )
        if cake.order_encodings[order - 1] == "hashed_index":
            arrays[f"count_context_keys_{order}"] = (
                getattr(cake, f"context_keys_{order}")
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint64)
            )
    for name, tensor in model.state_dict().items():
        if name.startswith("count_cake."):
            continue
        arrays[f"neural__{name}"] = tensor.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)
    manifest["serialized_bytes"] = path.stat().st_size
    return manifest


def load_count_cake_bundle(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[HierarchicalCountCakeLM, dict]:
    """Load a portable CountCake bundle without executing serialized code."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest_json"].tobytes().decode("utf-8"))
        if manifest.get("format") != "layercake-count-cake-bundle/1":
            raise ValueError("unsupported CountCake bundle format")
        count_config = manifest["count_cake"]
        target_device = torch.device(device)
        order_tables = []
        for order in range(1, int(count_config["max_order"]) + 1):
            table = (
                torch.from_numpy(
                    archive[f"count_keys_{order}"].astype(np.int64)
                ).to(target_device),
                torch.from_numpy(
                    archive[f"count_values_{order}"].astype(np.float32)
                ).to(target_device),
            )
            encodings = count_config.get(
                "order_encodings",
                ["packed"] * int(count_config["max_order"]),
            )
            if encodings[order - 1] == "hashed_index":
                table = table + (
                    torch.from_numpy(
                        archive[f"count_context_keys_{order}"].astype(np.int64)
                    ).to(target_device),
                    torch.from_numpy(
                        archive[f"count_context_totals_{order}"].astype(np.float32)
                    ).to(target_device),
                )
            elif f"count_context_totals_{order}" in archive.files:
                table = table + (
                    torch.from_numpy(
                        archive[f"count_context_totals_{order}"].astype(
                            np.float32
                        )
                    ).to(target_device),
                )
            order_tables.append(table)
        cake = PrunedBackoffByteCake(
            unigram_counts=torch.from_numpy(
                archive["count_unigram"].astype(np.float32)
            ).to(target_device),
            order_tables=order_tables,
            backoff_strengths=count_config["backoff_strengths"],
            backoff_mode=count_config.get("backoff_mode", "fixed"),
            discount=float(count_config.get("discount", 0.75)),
            state_budget=int(count_config["state_budget"]),
            corpus_bytes=int(count_config["corpus_bytes"]),
            context_hash_bits=count_config.get("context_hash_bits", ()),
        )
        model_config = manifest["model"]
        model = HierarchicalCountCakeLM(
            cake,
            patch_size=int(model_config["patch_size"]),
            chunking_mode=model_config.get("chunking_mode", "fixed"),
            prediction_start=int(
                model_config.get("prediction_start", model_config["patch_size"])
            ),
            d_byte=int(model_config["d_byte"]),
            d_model=int(model_config["d_model"]),
            d_abi=int(model_config["d_abi"]),
            patch_layers=int(model_config.get("patch_layers", 1)),
            patch_core_type=model_config.get("patch_core_type", "gru"),
            patch_selective_rank=int(
                model_config.get("patch_selective_rank", 128)
            ),
            patch_attention_heads=int(
                model_config.get("patch_attention_heads", 8)
            ),
            scratchpad_stride=int(model_config.get("scratchpad_stride", 0)),
            dynamic_hash_buckets=int(
                model_config.get("dynamic_hash_buckets", 0)
            ),
            dynamic_hash_width=int(model_config.get("dynamic_hash_width", 64)),
            dynamic_hash_tables=int(model_config.get("dynamic_hash_tables", 1)),
            dynamic_hash_sparse=bool(
                model_config.get("dynamic_hash_sparse", False)
            ),
            neural_context_buckets=int(
                model_config.get("neural_context_buckets", 0)
            ),
            neural_context_order=int(
                model_config.get("neural_context_order", 3)
            ),
            neural_context_sparse=bool(
                model_config.get("neural_context_sparse", False)
            ),
            local_width=int(model_config["local_width"]),
            local_decoder=model_config.get("local_decoder"),
            local_layers=int(model_config.get("local_layers", 5)),
            local_dilation_growth=int(
                model_config.get("local_dilation_growth", 2)
            ),
            local_gru_layers=int(model_config.get("local_gru_layers", 1)),
            local_rank=int(model_config.get("local_rank", 64)),
            byte_head=model_config.get("byte_head", "radix"),
            local_recurrent=bool(model_config.get("local_recurrent", False)),
            local_continuous=bool(model_config.get("local_continuous", False)),
            online_cache_specs=model_config.get("online_cache_specs", ()),
            online_cache_window=model_config.get("online_cache_window"),
            recent_cache_specs=model_config.get("recent_cache_specs", ()),
            normalized_cache_specs=model_config.get(
                "normalized_cache_specs", ()
            ),
            cache_normalization=model_config.get("cache_normalization", "classes"),
            confidence_gate=bool(model_config.get("confidence_gate", False)),
            expert_confidence_gate=bool(
                model_config.get("expert_confidence_gate", False)
            ),
            count_distribution_gate=bool(
                model_config.get("count_distribution_gate", False)
            ),
            count_order_routing=bool(
                model_config.get("count_order_routing", False)
            ),
            count_order_stage_features=bool(
                model_config.get("count_order_stage_features", False)
            ),
            count_order_router_hidden=int(
                model_config.get("count_order_router_hidden", 0)
            ),
            gate_hidden_width=int(model_config.get("gate_hidden_width", 0)),
        ).to(target_device)
        neural_state = {
            name.removeprefix("neural__"): torch.from_numpy(
                archive[name].copy()
            ).to(target_device)
            for name in archive.files
            if name.startswith("neural__")
        }
        incompatible = model.load_state_dict(neural_state, strict=False)
        missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("count_cake.")
        ]
        if missing or incompatible.unexpected_keys:
            raise ValueError(
                f"invalid CountCake neural state: missing={missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    if model.logical_total_parameters != int(
        manifest["parameters"]["logical_total"]
    ):
        raise ValueError("CountCake bundle parameter total does not match manifest")
    manifest["serialized_bytes"] = path.stat().st_size
    return model, manifest
