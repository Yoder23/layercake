"""Strictly causal byte and byte-patch models for measured experiments."""

from __future__ import annotations

import time
import math

import torch
from torch import nn
import torch.nn.functional as F


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), 1)


def canonical_brick_head(d_abi: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260622)
    return torch.randn(d_abi, 256, generator=generator) / (d_abi ** 0.5)


class CausalConvBlock(nn.Module):
    def __init__(self, width: int, dilation: int, kernel_size: int = 5):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size,
            groups=width,
            dilation=dilation,
        )
        self.mix = nn.Linear(width, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        z = self.norm(h).transpose(1, 2)
        z = self.depthwise(F.pad(z, (self.left_padding, 0))).transpose(1, 2)
        return residual + self.mix(F.gelu(z))


class GatedCausalConvBlock(nn.Module):
    """Parallel causal global mixer with gated depthwise convolutions."""

    def __init__(self, width: int, dilation: int, kernel_size: int = 5):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, 2 * width, bias=False)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size,
            groups=width,
            dilation=dilation,
        )
        self.out_proj = nn.Linear(width, width, bias=False)
        hidden = round((8 * width / 3) / 64) * 64
        self.ffn_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        projected = self.in_proj(self.norm(h))
        value, gate = projected.chunk(2, dim=-1)
        value = self.depthwise(
            F.pad(value.transpose(1, 2), (self.left_padding, 0))
        ).transpose(1, 2)
        h = h + self.out_proj(value * torch.sigmoid(gate))
        normalized = self.ffn_norm(h)
        return h + self.down(
            F.silu(self.gate(normalized)) * self.up(normalized)
        )


class ResidualCausalGRUBlock(nn.Module):
    """cuDNN-backed recurrent global mixer with a residual SwiGLU."""

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.out_norm = nn.LayerNorm(width)
        hidden = round((8 * width / 3) / 64) * 64
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        recurrent, _ = self.gru(self.norm(h))
        h = h + recurrent
        normalized = self.out_norm(h)
        return h + self.down(
            F.silu(self.gate(normalized)) * self.up(normalized)
        )


class ModernCausalBlock(nn.Module):
    """Pre-norm causal attention with parameter-matched SwiGLU."""

    def __init__(
        self,
        width: int,
        heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        hidden = round((8 * width / 3) / 64) * 64
        self.attn_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(h)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False,
        )
        h = h + self.dropout(attended)
        normalized = self.ffn_norm(h)
        return h + self.dropout(
            self.down(
                F.silu(self.gate(normalized)) * self.up(normalized)
            )
        )


class FusedModernCausalBlock(nn.Module):
    """SwiGLU block using fused scaled-dot-product causal attention."""

    def __init__(
        self,
        width: int,
        heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        fused_swiglu: bool = False,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        hidden = round((8 * width / 3) / 64) * 64
        self.heads = heads
        self.head_dim = width // heads
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.attn_out = nn.Linear(width, width, bias=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.fused_swiglu = bool(fused_swiglu)
        if self.fused_swiglu:
            self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        else:
            self.gate = nn.Linear(width, hidden, bias=False)
            self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.qk_norm = qk_norm

    def _ffn(self, normalized: torch.Tensor) -> torch.Tensor:
        if self.fused_swiglu:
            if normalized.device.type == "cpu":
                midpoint = self.gate_up.weight.shape[0] // 2
                gate = F.linear(normalized, self.gate_up.weight[:midpoint])
                up = F.linear(normalized, self.gate_up.weight[midpoint:])
            else:
                gate, up = self.gate_up(normalized).chunk(2, dim=-1)
        else:
            gate = self.gate(normalized)
            up = self.up(normalized)
        return self.down(F.silu(gate) * up)

    def _normalize_qk(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.qk_norm:
            return query, key
        return (
            F.rms_norm(query, (self.head_dim,)),
            F.rms_norm(key, (self.head_dim,)),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None):
        batch, length, width = h.shape
        normalized = self.attn_norm(h)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self._normalize_qk(query, key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        h = h + self.dropout(self.attn_out(attended))
        normalized = self.ffn_norm(h)
        return h + self.dropout(self._ffn(normalized))

    def prefill_with_cache(self, h: torch.Tensor):
        batch, length, width = h.shape
        normalized = self.attn_norm(h)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self._normalize_qk(query, key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        h = h + self.dropout(self.attn_out(attended))
        normalized = self.ffn_norm(h)
        h = h + self.dropout(self._ffn(normalized))
        return h, (key, value)

    def decode_with_cache(
        self,
        h: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | dict,
    ):
        batch, length, width = h.shape
        if length != 1:
            raise ValueError("cached decode expects exactly one token")
        normalized = self.attn_norm(h)
        qkv = self.qkv(normalized).reshape(
            batch, 1, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self._normalize_qk(query, key)
        if isinstance(cache, dict):
            cached_key = cache["key"]
            cached_value = cache["value"]
            cache_length = int(cache["length"])
            cache_capacity = cached_key.shape[2]
            if cache_length >= cache_capacity:
                grow_by = max(cache_capacity, 1)
                new_capacity = cache_capacity + grow_by
                expanded_key = cached_key.new_empty(
                    cached_key.shape[0],
                    cached_key.shape[1],
                    new_capacity,
                    cached_key.shape[3],
                )
                expanded_value = cached_value.new_empty(
                    cached_value.shape[0],
                    cached_value.shape[1],
                    new_capacity,
                    cached_value.shape[3],
                )
                expanded_key[:, :, :cache_length].copy_(
                    cached_key[:, :, :cache_length]
                )
                expanded_value[:, :, :cache_length].copy_(
                    cached_value[:, :, :cache_length]
                )
                cache["key"] = expanded_key
                cache["value"] = expanded_value
                cached_key = expanded_key
                cached_value = expanded_value
            cached_key[:, :, cache_length : cache_length + 1].copy_(key)
            cached_value[:, :, cache_length : cache_length + 1].copy_(value)
            all_key = cached_key[:, :, : cache_length + 1]
            all_value = cached_value[:, :, : cache_length + 1]
            cache["length"] = cache_length + 1
        else:
            cached_key, cached_value = cache
            all_key = torch.cat([cached_key, key], dim=2)
            all_value = torch.cat([cached_value, value], dim=2)
        attended = F.scaled_dot_product_attention(
            query,
            all_key,
            all_value,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, 1, width)
        h = h + self.dropout(self.attn_out(attended))
        normalized = self.ffn_norm(h)
        h = h + self.dropout(self._ffn(normalized))
        return h, cache if isinstance(cache, dict) else (all_key, all_value)


class Top1RoutedCakeBlock(nn.Module):
    """Batch-routed expert cakes with exactly one active dense block per row.

    The expert bank preserves total parameter capacity while a hard top-1
    route bounds forward/backward work.  Training may pin a route for a
    domain-homogeneous batch, allowing the optimizer to allocate state only
    for that expert.  Without an override, a small neural router selects an
    expert from the mean causal representation.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        experts: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if experts < 2:
            raise ValueError("routed cake blocks require at least two experts")
        self.width = int(width)
        self.expert_count = int(experts)
        self.router_norm = nn.LayerNorm(width)
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(
            FusedModernCausalBlock(
                width,
                heads,
                dropout,
                qk_norm,
                fused_swiglu=expert_index == 0,
            )
            for expert_index in range(experts)
        )
        self.route_override: int | None = None
        self.last_routes: torch.Tensor | None = None

    def set_route(self, route: int | None) -> None:
        if route is not None and not 0 <= int(route) < self.expert_count:
            raise ValueError(
                f"route must be in [0, {self.expert_count - 1}]"
            )
        self.route_override = None if route is None else int(route)

    def active_expert_parameters(self, route: int | None = None):
        selected = self.route_override if route is None else route
        if selected is None:
            raise RuntimeError("an explicit route is required for sparse optimizer state")
        yield from self.experts[int(selected)].parameters()

    def _routes(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.route_override is not None:
            routes = torch.full(
                (h.shape[0],),
                self.route_override,
                device=h.device,
                dtype=torch.long,
            )
            return routes, None
        router_input = self.router_norm(h).mean(dim=1)
        probabilities = torch.softmax(self.router(router_input), dim=-1)
        routes = probabilities.argmax(dim=-1)
        selected_probability = probabilities.gather(
            1, routes[:, None]
        ).squeeze(1)
        return routes, selected_probability

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.route_override is not None:
            self.last_routes = torch.full(
                (h.shape[0],),
                self.route_override,
                device=h.device,
                dtype=torch.long,
            )
            return self.experts[self.route_override](h, mask)
        routes, selected_probability = self._routes(h)
        self.last_routes = routes.detach()
        output = torch.empty_like(h)
        for expert_index, expert in enumerate(self.experts):
            row_indexes = torch.nonzero(
                routes == expert_index, as_tuple=False
            ).flatten()
            if row_indexes.numel() == 0:
                continue
            selected = h.index_select(0, row_indexes)
            transformed = expert(selected, mask)
            output.index_copy_(0, row_indexes, transformed)
        if selected_probability is not None:
            # Straight-through scale: numerically one in the forward pass,
            # while preserving a learning signal for the neural router.
            gate = 1.0 + selected_probability - selected_probability.detach()
            output = h + (output - h) * gate[:, None, None]
        return output

    def prefill_with_cache(self, h: torch.Tensor):
        routes, _ = self._routes(h)
        if not bool((routes == routes[0]).all()):
            raise RuntimeError(
                "cached routed prefill requires a uniform batch route"
            )
        route = int(routes[0].item())
        output, cache = self.experts[route].prefill_with_cache(h)
        self.last_routes = routes.detach()
        return output, {"route": route, "expert": cache}

    def decode_with_cache(self, h: torch.Tensor, cache: dict):
        route = int(cache["route"])
        output, expert_cache = self.experts[route].decode_with_cache(
            h, cache["expert"]
        )
        cache["expert"] = expert_cache
        return output, cache


class SparseStatePatchBlock(nn.Module):
    """Causal gathered sparse patch attention plus recurrent chunk state.

    Each patch attends to a bounded recent window, fixed dilated prior
    offsets, and summaries of completed causal chunks. This is intentionally
    pure PyTorch and stores enough key/value state for exact cached decode.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        local_window: int = 32,
        dilated_offsets: tuple[int, ...] = (32, 48, 64, 96),
        chunk_size: int = 16,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if local_window <= 0:
            raise ValueError("sparse local window must be positive")
        if chunk_size <= 0:
            raise ValueError("sparse chunk size must be positive")
        hidden = round((8 * width / 3) / 64) * 64
        self.heads = heads
        self.head_dim = width // heads
        self.local_window = local_window
        self.dilated_offsets = tuple(int(offset) for offset in dilated_offsets)
        self.chunk_size = chunk_size
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.attn_out = nn.Linear(width, width, bias=True)
        self.state_gate_scale = nn.Parameter(torch.zeros(width))
        self.state_gate_bias = nn.Parameter(torch.zeros(width))
        self.state_update_scale = nn.Parameter(torch.zeros(width))
        self.ffn_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.qk_norm = qk_norm

    def _normalize_qk(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.qk_norm:
            return query, key
        return (
            F.rms_norm(query, (self.head_dim,)),
            F.rms_norm(key, (self.head_dim,)),
        )

    def _token_indices(
        self,
        source_length: int,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = positions.device
        local_offsets = torch.arange(self.local_window, device=device)
        local = positions[:, None] - (self.local_window - 1 - local_offsets)[None]
        dilated = positions[:, None] - torch.tensor(
            self.dilated_offsets, device=device
        )[None]
        indices = torch.cat([local, dilated], dim=1)
        valid = (
            (indices >= 0)
            & (indices <= positions[:, None])
            & (indices < source_length)
        )
        return indices.clamp_min(0), valid

    def _summary_indices(
        self,
        source_length: int,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = positions.device
        length = positions.numel()
        chunks = (source_length + self.chunk_size - 1) // self.chunk_size
        if chunks <= 1:
            return (
                torch.zeros(length, 1, dtype=torch.long, device=device),
                torch.zeros(length, 1, dtype=torch.bool, device=device),
            )
        current_chunk = positions // self.chunk_size
        summary_slots = torch.arange(chunks - 1, device=device)
        valid = summary_slots[None] < current_chunk[:, None]
        return summary_slots.expand(length, -1).clamp_min(0), valid

    def _chunk_summaries(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, heads, length, width = tensor.shape
        chunks = (length + self.chunk_size - 1) // self.chunk_size
        padded_length = chunks * self.chunk_size
        padded = F.pad(tensor, (0, 0, 0, padded_length - length))
        grouped = padded.reshape(
            batch, heads, chunks, self.chunk_size, width
        )
        counts = torch.full(
            (chunks,),
            self.chunk_size,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if length % self.chunk_size:
            counts[-1] = length % self.chunk_size
        return grouped.sum(dim=3) / counts[None, None, :, None]

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, heads, target_length, width = query.shape
        source_length = key.shape[2]
        if query_positions is None:
            query_positions = torch.arange(target_length, device=query.device)
        token_indices, token_valid = self._token_indices(
            source_length, query_positions
        )
        flat_token = token_indices.reshape(-1)
        token_key = key[:, :, flat_token, :].reshape(
            batch, heads, target_length, token_indices.shape[1], width
        )
        token_value = value[:, :, flat_token, :].reshape(
            batch, heads, target_length, token_indices.shape[1], width
        )
        summary_key = self._chunk_summaries(key)
        summary_value = self._chunk_summaries(value)
        summary_indices, summary_valid = self._summary_indices(
            source_length, query_positions
        )
        flat_summary = summary_indices.reshape(-1)
        gathered_summary_key = summary_key[:, :, flat_summary, :].reshape(
            batch, heads, target_length, summary_indices.shape[1], width
        )
        gathered_summary_value = summary_value[:, :, flat_summary, :].reshape(
            batch, heads, target_length, summary_indices.shape[1], width
        )
        gathered_key = torch.cat([token_key, gathered_summary_key], dim=3)
        gathered_value = torch.cat([token_value, gathered_summary_value], dim=3)
        valid = torch.cat([token_valid, summary_valid], dim=1)
        scores = (query.unsqueeze(3) * gathered_key).sum(dim=-1)
        scores = scores / (width ** 0.5)
        scores = scores.masked_fill(~valid[None, None], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(
            weights, p=self.dropout_p, training=self.training
        )
        return (weights.unsqueeze(-1) * gathered_value).sum(dim=3)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None):
        batch, length, width = h.shape
        normalized = self.attn_norm(h)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self._normalize_qk(query, key)
        attended = self._attend(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        h = h + self.dropout(self.attn_out(attended))
        gate = torch.sigmoid(h * self.state_gate_scale + self.state_gate_bias)
        h = h + gate * torch.tanh(h * self.state_update_scale)
        normalized = self.ffn_norm(h)
        return h + self.dropout(
            self.down(
                F.silu(self.gate(normalized)) * self.up(normalized)
            )
        )

    def prefill_with_cache(self, h: torch.Tensor):
        batch, length, width = h.shape
        normalized = self.attn_norm(h)
        qkv = self.qkv(normalized).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self._normalize_qk(query, key)
        attended = self._attend(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        h = h + self.dropout(self.attn_out(attended))
        gate = torch.sigmoid(h * self.state_gate_scale + self.state_gate_bias)
        h = h + gate * torch.tanh(h * self.state_update_scale)
        normalized = self.ffn_norm(h)
        h = h + self.dropout(
            self.down(
                F.silu(self.gate(normalized)) * self.up(normalized)
            )
        )
        return h, (key, value)

    def decode_with_cache(
        self,
        h: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ):
        batch, length, width = h.shape
        if length != 1:
            raise ValueError("cached decode expects exactly one token")
        normalized = self.attn_norm(h)
        qkv = self.qkv(normalized).reshape(
            batch, 1, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self._normalize_qk(query, key)
        if isinstance(cache, dict):
            cached_key = cache["key"]
            cached_value = cache["value"]
            cache_length = int(cache["length"])
            cache_capacity = cached_key.shape[2]
            if cache_length >= cache_capacity:
                grow_by = max(cache_capacity, 1)
                new_capacity = cache_capacity + grow_by
                expanded_key = cached_key.new_empty(
                    cached_key.shape[0],
                    cached_key.shape[1],
                    new_capacity,
                    cached_key.shape[3],
                )
                expanded_value = cached_value.new_empty(
                    cached_value.shape[0],
                    cached_value.shape[1],
                    new_capacity,
                    cached_value.shape[3],
                )
                expanded_key[:, :, :cache_length].copy_(
                    cached_key[:, :, :cache_length]
                )
                expanded_value[:, :, :cache_length].copy_(
                    cached_value[:, :, :cache_length]
                )
                cache["key"] = expanded_key
                cache["value"] = expanded_value
                cached_key = expanded_key
                cached_value = expanded_value
            cached_key[:, :, cache_length : cache_length + 1].copy_(key)
            cached_value[:, :, cache_length : cache_length + 1].copy_(value)
            all_key = cached_key[:, :, : cache_length + 1]
            all_value = cached_value[:, :, : cache_length + 1]
            cache["length"] = cache_length + 1
        else:
            cached_key, cached_value = cache
            all_key = torch.cat([cached_key, key], dim=2)
            all_value = torch.cat([cached_value, value], dim=2)
        position = torch.tensor(
            [all_key.shape[2] - 1], device=query.device, dtype=torch.long
        )
        attended = self._attend(
            query, all_key, all_value, query_positions=position
        )
        attended = attended.transpose(1, 2).reshape(batch, 1, width)
        h = h + self.dropout(self.attn_out(attended))
        gate = torch.sigmoid(h * self.state_gate_scale + self.state_gate_bias)
        h = h + gate * torch.tanh(h * self.state_update_scale)
        normalized = self.ffn_norm(h)
        h = h + self.dropout(
            self.down(
                F.silu(self.gate(normalized)) * self.up(normalized)
            )
        )
        return h, cache if isinstance(cache, dict) else (all_key, all_value)


class SelectiveStatePatchBlock(nn.Module):
    """Pure-PyTorch causal selective-state patch mixer.

    This block is the CPU/mobile-oriented global core for ABI Patch Cell v2.
    It replaces quadratic attention with a depthwise causal prefilter, a
    per-channel recurrent state scan, and a compact SwiGLU residual. The scan is
    strictly left-to-right, so patch t cannot read patch t+1.
    """

    def __init__(
        self,
        width: int,
        heads: int | None = None,
        dropout: float = 0.0,
        qk_norm: bool = False,
        conv_kernel: int = 5,
    ):
        super().__init__()
        if conv_kernel <= 0:
            raise ValueError("selective_state conv kernel must be positive")
        self.width = width
        self.left_padding = conv_kernel - 1
        hidden = max(64, round((4 * width / 3) / 64) * 64)
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            conv_kernel,
            groups=width,
            bias=False,
        )
        self.selective_proj = nn.Linear(width, 3 * width, bias=True)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.ffn_norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _scan(
        self,
        z: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        initial_count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_raw, update_raw, decay_raw = self.selective_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        update = torch.tanh(update_raw)
        update = update * torch.sigmoid(-decay_raw)
        if initial_state is None:
            cumulative = torch.cumsum(update, dim=1)
            count = torch.arange(
                1,
                z.shape[1] + 1,
                dtype=z.dtype,
                device=z.device,
            ).view(1, -1, 1)
            state_seq = cumulative / count
            state = cumulative[:, -1]
            total_count = count[:, -1].expand(z.shape[0], 1)
        else:
            if initial_count is None:
                raise ValueError("initial_count is required with initial_state")
            cumulative = torch.cumsum(update, dim=1) + initial_state[:, None, :]
            count_offsets = torch.arange(
                1,
                z.shape[1] + 1,
                dtype=z.dtype,
                device=z.device,
            ).view(1, -1, 1)
            counts = initial_count[:, None, :] + count_offsets
            state_seq = cumulative / counts
            state = cumulative[:, -1]
            total_count = counts[:, -1]
        return gate * state_seq, state, total_count

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None):
        residual = h
        z = self.norm(h)
        z = self.depthwise(
            F.pad(z.transpose(1, 2), (self.left_padding, 0))
        ).transpose(1, 2)
        scanned, _, _ = self._scan(z)
        h = residual + self.dropout(self.out_proj(scanned))
        normalized = self.ffn_norm(h)
        return h + self.dropout(
            self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        )

    def prefill_with_cache(self, h: torch.Tensor):
        z = self.norm(h)
        conv_in = z.transpose(1, 2)
        filtered = self.depthwise(
            F.pad(conv_in, (self.left_padding, 0))
        ).transpose(1, 2)
        scanned, state, count = self._scan(filtered)
        out = h + self.dropout(self.out_proj(scanned))
        normalized = self.ffn_norm(out)
        out = out + self.dropout(
            self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        )
        history = conv_in[:, :, -self.left_padding :].detach() if self.left_padding else conv_in[:, :, :0]
        return out, {"state": state.detach(), "count": count.detach(), "history": history}

    def decode_with_cache(self, h: torch.Tensor, cache: dict):
        if h.shape[1] != 1:
            raise ValueError("cached decode expects exactly one token")
        z = self.norm(h).transpose(1, 2)
        history = cache["history"]
        conv_input = torch.cat([history, z], dim=2)
        filtered = self.depthwise(conv_input).transpose(1, 2)
        scanned, state, count = self._scan(filtered, cache["state"], cache["count"])
        out = h + self.dropout(self.out_proj(scanned))
        normalized = self.ffn_norm(out)
        out = out + self.dropout(
            self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        )
        if self.left_padding:
            cache["history"] = conv_input[:, :, -self.left_padding :].detach()
        cache["state"] = state.detach()
        cache["count"] = count.detach()
        return out, cache

    def decode_state_only_with_cache(self, h: torch.Tensor, cache: dict):
        """Cheaper cached decode: update selective state without FFN expansion."""
        if h.shape[1] != 1:
            raise ValueError("cached decode expects exactly one token")
        z = self.norm(h).transpose(1, 2)
        history = cache["history"]
        conv_input = torch.cat([history, z], dim=2)
        filtered = self.depthwise(conv_input).transpose(1, 2)
        scanned, state, count = self._scan(filtered, cache["state"], cache["count"])
        out = h + self.out_proj(scanned)
        if self.left_padding:
            cache["history"] = conv_input[:, :, -self.left_padding :].detach()
        cache["state"] = state.detach()
        cache["count"] = count.detach()
        return out, cache


def run_modern_stack(
    blocks: nn.ModuleList, h: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    for block in blocks:
        h = block(h, mask)
    return h


class MixtureOfDepthRefinement(nn.Module):
    """Fixed-capacity causal refinement while preserving every patch position."""

    def __init__(
        self,
        width: int,
        heads: int,
        layers: int,
        capacity_ratio: float,
        group_size: int = 8,
        share_weights: bool = False,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if not 0 < capacity_ratio <= 1:
            raise ValueError("capacity_ratio must be in (0, 1]")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.capacity_ratio = capacity_ratio
        self.group_size = group_size
        self.layers = layers
        self.share_weights = share_weights
        self.router = nn.Linear(width, 1, bias=False)
        self.blocks = nn.ModuleList(
            FusedModernCausalBlock(
                width, heads, dropout, qk_norm
            )
            for _ in range(1 if share_weights else layers)
        )

    def route_mask(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.router(h).squeeze(-1)
        batch, length = scores.shape
        padded_length = (
            (length + self.group_size - 1)
            // self.group_size
            * self.group_size
        )
        padded = F.pad(
            scores,
            (0, padded_length - length),
            value=float("-inf"),
        ).reshape(batch, -1, self.group_size)
        capacity = max(
            1, round(self.group_size * self.capacity_ratio)
        )
        selected_offsets = padded.topk(
            capacity, dim=-1
        ).indices
        # Group g routes group g+1. The first group uses a fixed bootstrap
        # pattern, so no position can depend on future routing scores.
        bootstrap = torch.arange(
            capacity, device=h.device
        ).view(1, 1, -1).expand(batch, 1, -1)
        routed_offsets = torch.cat(
            [bootstrap, selected_offsets[:, :-1]], dim=1
        )
        grouped_mask = torch.zeros_like(
            padded, dtype=torch.bool
        ).scatter_(2, routed_offsets, True)
        return grouped_mask.reshape(batch, padded_length)[:, :length]

    def forward(
        self, h: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        route_mask = self.route_mask(h)
        hard_weights = route_mask.to(h.dtype)
        soft_weights = torch.sigmoid(
            self.router(h).squeeze(-1)
        )
        route_weights = (
            hard_weights
            + soft_weights
            - soft_weights.detach()
        ).unsqueeze(-1)
        for index in range(self.layers):
            block = self.blocks[0] if self.share_weights else self.blocks[index]
            refined = block(h, mask)
            h = h + route_weights * (refined - h)
        return h, route_mask


class AutoregressivePatchHead(nn.Module):
    """Decode a short byte patch causally from one global patch state."""

    def __init__(
        self,
        context_width: int,
        byte_embedding: nn.Embedding,
        hidden_width: int,
        patch_size: int,
        copy_window: int = 0,
        copy_dim: int = 32,
        copy_scale: float = 4.0,
        position_copy: bool = False,
        contextual_copy: bool = False,
        lowercase_copy: bool = False,
        semantic_copy: bool = False,
    ):
        super().__init__()
        self.byte_embedding = byte_embedding
        self.patch_size = patch_size
        self.copy_window = int(copy_window)
        self.copy_dim = int(copy_dim)
        self.copy_scale = float(copy_scale)
        self.position_copy = bool(position_copy)
        self.contextual_copy = bool(contextual_copy)
        self.lowercase_copy = bool(lowercase_copy)
        self.semantic_copy = bool(semantic_copy)
        self.initial_state = nn.Linear(context_width, hidden_width)
        self.bos = nn.Parameter(torch.zeros(byte_embedding.embedding_dim))
        self.cell = nn.GRUCell(byte_embedding.embedding_dim, hidden_width)
        self.output = nn.Linear(hidden_width, 256)
        if self.copy_window > 0:
            self.copy_query = nn.Linear(hidden_width, self.copy_dim, bias=False)
            self.copy_key = nn.Linear(byte_embedding.embedding_dim, self.copy_dim, bias=False)
            self.copy_gate = nn.Linear(hidden_width, 1)
            if self.position_copy:
                self.copy_position_key = nn.Embedding(self.copy_window, self.copy_dim)
                self.copy_previous_key = nn.Linear(
                    byte_embedding.embedding_dim,
                    self.copy_dim,
                    bias=False,
                )
                nn.init.zeros_(self.copy_position_key.weight)
                nn.init.zeros_(self.copy_previous_key.weight)
            if self.contextual_copy:
                self.copy_next_key = nn.Linear(
                    byte_embedding.embedding_dim,
                    self.copy_dim,
                    bias=False,
                )
                self.copy_next2_key = nn.Linear(
                    byte_embedding.embedding_dim,
                    self.copy_dim,
                    bias=False,
                )
                nn.init.zeros_(self.copy_next_key.weight)
                nn.init.zeros_(self.copy_next2_key.weight)
            if self.semantic_copy:
                self.copy_context_key = nn.Conv1d(
                    byte_embedding.embedding_dim,
                    self.copy_dim,
                    kernel_size=33,
                    padding=16,
                    bias=False,
                )
                nn.init.zeros_(self.copy_context_key.weight)

    def _normalized_copy_source(self, source: torch.Tensor) -> torch.Tensor:
        source = source[..., -self.copy_window :].to(dtype=torch.long)
        if source.shape[-1] < self.copy_window:
            source = F.pad(source, (self.copy_window - source.shape[-1], 0))
        return source

    def _copy_keys(self, source_flat: torch.Tensor) -> torch.Tensor:
        source_embeddings = self.byte_embedding(source_flat)
        keys = self.copy_key(source_embeddings)
        if self.position_copy:
            previous = torch.cat(
                [torch.zeros_like(source_flat[:, :1]), source_flat[:, :-1]],
                dim=1,
            )
            keys = keys + self.copy_previous_key(self.byte_embedding(previous))
            positions = torch.arange(source_flat.shape[-1], device=source_flat.device)
            keys = keys + self.copy_position_key(positions)[None]
        if self.contextual_copy:
            following = torch.cat(
                [source_flat[:, 1:], torch.zeros_like(source_flat[:, :1])],
                dim=1,
            )
            following2 = torch.cat(
                [source_flat[:, 2:], torch.zeros_like(source_flat[:, :2])],
                dim=1,
            )
            keys = keys + self.copy_next_key(self.byte_embedding(following))
            keys = keys + self.copy_next2_key(self.byte_embedding(following2))
        if self.semantic_copy:
            keys = keys + self.copy_context_key(
                source_embeddings.transpose(1, 2)
            ).transpose(1, 2)
        return keys

    def _copy_bias(
        self,
        hidden: torch.Tensor,
        source: torch.Tensor | None,
        keys: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.copy_window <= 0 or source is None:
            return None
        source = self._normalized_copy_source(source)
        source_shape = source.shape
        source_flat = source.reshape(-1, source_shape[-1])
        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        if keys is None:
            keys = self._copy_keys(source_flat)
        query = self.copy_query(hidden_flat).unsqueeze(-1)
        scores = (keys @ query).squeeze(-1).float() / math.sqrt(max(self.copy_dim, 1))
        probs = torch.softmax(scores, dim=-1)
        copy_mass = torch.zeros(
            hidden_flat.shape[0],
            256,
            dtype=torch.float32,
            device=hidden_flat.device,
        )
        copy_ids = source_flat
        if self.lowercase_copy:
            copy_ids = torch.where(
                (copy_ids >= ord("A")) & (copy_ids <= ord("Z")),
                copy_ids + (ord("a") - ord("A")),
                copy_ids,
            )
        copy_mass.scatter_add_(1, copy_ids, probs)
        copy_logits = torch.log(copy_mass.clamp_min(1e-6))
        copy_logits = copy_logits - copy_logits.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.copy_gate(hidden_flat).float())
        return (copy_logits * (gate * self.copy_scale)).to(dtype=hidden.dtype).reshape(
            *hidden.shape[:-1],
            256,
        )

    def _copy_bias_sequence(
        self,
        hidden: torch.Tensor,
        source: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Compute the static-source copy distribution for a hidden sequence."""
        if self.copy_window <= 0 or source is None:
            return None
        source = self._normalized_copy_source(source)
        steps = hidden.shape[-2]
        source_flat = source.reshape(-1, source.shape[-1])
        hidden_flat = hidden.reshape(-1, steps, hidden.shape[-1])
        keys = self._copy_keys(source_flat)
        queries = self.copy_query(hidden_flat)
        scores = torch.bmm(queries, keys.transpose(1, 2)).float()
        scores = scores / math.sqrt(max(self.copy_dim, 1))
        probs = torch.softmax(scores, dim=-1)
        copy_mass = torch.zeros(
            hidden_flat.shape[0],
            steps,
            256,
            dtype=torch.float32,
            device=hidden_flat.device,
        )
        copy_ids = source_flat
        if self.lowercase_copy:
            copy_ids = torch.where(
                (copy_ids >= ord("A")) & (copy_ids <= ord("Z")),
                copy_ids + (ord("a") - ord("A")),
                copy_ids,
            )
        copy_mass.scatter_add_(
            2,
            copy_ids[:, None, :].expand(-1, steps, -1),
            probs,
        )
        copy_logits = torch.log(copy_mass.clamp_min(1e-6))
        copy_logits = copy_logits - copy_logits.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.copy_gate(hidden_flat).float())
        return (copy_logits * (gate * self.copy_scale)).to(
            dtype=hidden.dtype
        ).reshape(*hidden.shape[:-1], 256)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        prefix: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Teacher-force target bytes; logits at offset i use target[:i]."""
        hidden = torch.tanh(self.initial_state(context))
        decoder_input = self.bos.expand(*context.shape[:-1], -1)
        if prefix is not None:
            for offset in range(prefix.shape[-1]):
                decoder_input = self.byte_embedding(prefix[..., offset])
                hidden = self.cell(
                    decoder_input.reshape(-1, decoder_input.shape[-1]),
                    hidden.reshape(-1, hidden.shape[-1]),
                ).reshape_as(hidden)
        teacher_inputs = torch.cat(
            [
                decoder_input.unsqueeze(-2),
                self.byte_embedding(target[..., :-1]),
            ],
            dim=-2,
        )
        flat_inputs = teacher_inputs.reshape(
            -1,
            self.patch_size,
            teacher_inputs.shape[-1],
        ).transpose(0, 1)
        flat_hidden = hidden.reshape(-1, hidden.shape[-1]).unsqueeze(0)
        hidden_sequence, _ = torch._VF.gru(
            flat_inputs,
            flat_hidden,
            [
                self.cell.weight_ih,
                self.cell.weight_hh,
                self.cell.bias_ih,
                self.cell.bias_hh,
            ],
            True,
            1,
            0.0,
            self.training,
            False,
            False,
        )
        hidden_sequence = hidden_sequence.transpose(0, 1).reshape(
            *context.shape[:-1],
            self.patch_size,
            hidden.shape[-1],
        )
        logits = self.output(hidden_sequence)
        copy_bias = self._copy_bias_sequence(hidden_sequence, source)
        self.last_copy_logits = copy_bias
        if copy_bias is not None:
            logits = logits + copy_bias
        return list(logits.unbind(dim=-2))

    @torch.no_grad()
    def greedy(
        self,
        context: torch.Tensor,
        prefix: torch.Tensor | None = None,
        forced_first: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = torch.tanh(self.initial_state(context))
        decoder_input = self.bos.expand(*context.shape[:-1], -1)
        copy_source = source
        copy_keys = None
        if self.copy_window > 0 and copy_source is not None:
            copy_source = self._normalized_copy_source(copy_source)
            copy_keys = self._copy_keys(
                copy_source.reshape(-1, copy_source.shape[-1])
            )
        if prefix is not None:
            for offset in range(prefix.shape[-1]):
                decoder_input = self.byte_embedding(prefix[..., offset])
                hidden = self.cell(
                    decoder_input.reshape(-1, decoder_input.shape[-1]),
                    hidden.reshape(-1, hidden.shape[-1]),
                ).reshape_as(hidden)
        generated = []
        for offset in range(self.patch_size):
            hidden = self.cell(
                decoder_input.reshape(-1, decoder_input.shape[-1]),
                hidden.reshape(-1, hidden.shape[-1]),
            ).reshape_as(hidden)
            logits = self.output(hidden)
            copy_bias = self._copy_bias(hidden, copy_source, keys=copy_keys)
            if copy_bias is not None:
                logits = logits + copy_bias
            next_byte = logits.argmax(dim=-1)
            if offset == 0 and forced_first is not None:
                next_byte = forced_first
            generated.append(next_byte)
            decoder_input = self.byte_embedding(next_byte)
        return torch.stack(generated, dim=-1)


class CausalByteLM(nn.Module):
    def __init__(self, d_model=128, d_abi=64, layers=3, heads=4, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(256, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, batch_first=True, norm_first=True
        )
        self.core = nn.TransformerEncoder(block, layers)
        self.to_abi = nn.Sequential(nn.Linear(d_model, d_abi), nn.LayerNorm(d_abi))
        self.from_abi = nn.Linear(d_abi, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)
        self.register_buffer("canonical_head", canonical_brick_head(d_abi))

    def forward(self, x: torch.Tensor, brick=None):
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.emb(x) + self.pos(positions)[None]
        h = self.core(h, mask=causal_mask(x.shape[1], x.device))
        abi = self.to_abi(h)
        logits = self.head(self.norm(h + self.from_abi(abi)))
        if brick is not None:
            delta = brick(abi) - abi
            logits = logits + delta @ self.canonical_head
        return logits, abi

    def boundary_abi(self, abi: torch.Tensor, patch_size: int) -> torch.Tensor:
        usable = abi.shape[1] // patch_size * patch_size
        return abi[:, patch_size - 1 : usable : patch_size]


class CausalBytePatchLM(nn.Module):
    """Global transformer over completed patches plus a causal local GRU decoder."""

    def __init__(
        self, patch_size=4, d_byte=48, d_model=128, d_abi=64, layers=3,
        heads=4, max_patches=64, continuous_local=False,
        direct_global_context=False, ngram_buckets=0,
        local_decoder="gru", conv_layers=4, mtp_depth=0,
        transition_logits: torch.Tensor | None = None,
        patch_unit_buckets=0,
        local_layers=2,
        patch_prediction=False,
        patch_prediction_stride=1,
        patch_prediction_mode="factorized",
        patch_generation_width=96,
        patch_generation_bytes=0,
        patch_prediction_rollout_training=False,
        patch_prediction_rollout_mix=1.0,
        patch_generation_context=0,
        patch_generation_copy_window=0,
        patch_generation_copy_dim=32,
        patch_generation_copy_scale=4.0,
        patch_generation_position_copy=False,
        patch_generation_contextual_copy=False,
        patch_generation_lowercase_copy=False,
        patch_generation_semantic_copy=False,
        patch_prediction_detach_context=False,
        patch_prediction_context="global",
        tie_byte_embeddings=False,
        context_buckets=0,
        context_order=3,
        context_logits: torch.Tensor | None = None,
        transition_logit_scale=1.0,
        context_logit_scale=1.0,
        trainable_prior_gates=False,
        trainable_transition_head=True,
        trainable_context_head=True,
        prior_dropout=0.0,
        dynamic_prior_gates=False,
        repeat_suppression_window=0,
        repeat_suppression_scale=0.0,
        trainable_repeat_suppression=False,
        local_position_embeddings=False,
        modern_blocks=False,
        fused_attention=False,
        local_window=16,
        coarse_patch_size=0,
        coarse_layers=0,
        global_conv_layers=0,
        global_gru_layers=0,
        local_width=0,
        dropout=0.0,
        qk_norm=False,
        patch_encoder_layers=0,
        patch_encoder_window=16,
        mod_layers=0,
        mod_capacity=0.5,
        mod_group_size=8,
        mod_share_weights=False,
        global_block="attention",
        routed_cake_experts=0,
        shared_cake_layers=0,
        default_cake_route=None,
        sparse_state_local_window=32,
        sparse_state_dilated_offsets=(32, 48, 64, 96),
        sparse_state_chunk_size=16,
        abi_patch_cell_static_generation=False,
        abi_patch_cell_global_update_interval=1,
        abi_patch_cell_fast_global_decode=False,
        abi_patch_cell_fast_local_runtime=False,
        abi_patch_cell_lightweight_context_update=False,
        abi_patch_cell_lightweight_context_blend=0.15,
        generation_min_word_chars=0,
        generation_repeat_suppression_window=0,
        generation_repeat_suppression_scale=0.0,
        domain_cache_order=0,
        domain_cache_logit_scale=0.0,
        domain_cache_override=False,
        copy_attention=False,
        copy_attention_dim=32,
        copy_attention_scale=4.0,
        copy_attention_window=128,
        copy_transducer=False,
        copy_transducer_dim=32,
        copy_transducer_scale=4.0,
        copy_transducer_window=128,
        copy_transducer_logit_mode="prob",
        copy_transducer_projection="soft",
        span_width=4,
        span_verifier=False,
        span_prefix_conditioning=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = int(d_model)
        self.continuous_local = continuous_local
        self.direct_global_context = direct_global_context
        self.ngram_buckets = ngram_buckets
        self.local_decoder = local_decoder
        self.mtp_depth = mtp_depth
        self.patch_unit_buckets = patch_unit_buckets
        self.patch_prediction = patch_prediction
        self.patch_prediction_stride = patch_prediction_stride
        self.patch_prediction_mode = patch_prediction_mode
        self.patch_generation_width = patch_generation_width
        self.patch_generation_bytes = int(patch_generation_bytes or patch_size)
        if self.patch_generation_bytes <= 0:
            raise ValueError("patch_generation_bytes must be positive")
        if self.patch_generation_bytes % self.patch_size != 0:
            raise ValueError(
                "patch_generation_bytes must be a multiple of patch_size"
            )
        self.patch_prediction_rollout_training = bool(
            patch_prediction_rollout_training
        )
        self.patch_prediction_rollout_mix = float(patch_prediction_rollout_mix)
        if not 0.0 <= self.patch_prediction_rollout_mix <= 1.0:
            raise ValueError("patch_prediction_rollout_mix must be in [0, 1]")
        self.patch_generation_context = patch_generation_context
        self.patch_generation_copy_window = int(patch_generation_copy_window)
        self.patch_generation_copy_dim = int(patch_generation_copy_dim)
        self.patch_generation_copy_scale = float(patch_generation_copy_scale)
        self.patch_generation_position_copy = bool(
            patch_generation_position_copy
        )
        self.patch_generation_contextual_copy = bool(
            patch_generation_contextual_copy
        )
        self.patch_generation_lowercase_copy = bool(
            patch_generation_lowercase_copy
        )
        self.patch_generation_semantic_copy = bool(
            patch_generation_semantic_copy
        )
        self.patch_prediction_detach_context = (
            patch_prediction_detach_context
        )
        self.patch_prediction_context = patch_prediction_context
        if patch_prediction_context not in {"global", "local"}:
            raise ValueError(
                "patch_prediction_context must be global or local"
            )
        self.tie_byte_embeddings = tie_byte_embeddings
        self.context_buckets = context_buckets
        self.context_order = context_order
        self.domain_cache_order = int(domain_cache_order)
        self.domain_cache_logit_scale = float(domain_cache_logit_scale)
        self.domain_cache_override = bool(domain_cache_override)
        self.copy_attention = bool(copy_attention)
        self.copy_attention_dim = int(copy_attention_dim)
        self.copy_attention_scale = float(copy_attention_scale)
        self.copy_attention_window = int(copy_attention_window)
        self.copy_transducer = bool(copy_transducer)
        self.copy_transducer_dim = int(copy_transducer_dim)
        self.copy_transducer_scale = float(copy_transducer_scale)
        self.copy_transducer_window = int(copy_transducer_window)
        self.copy_transducer_logit_mode = str(copy_transducer_logit_mode)
        self.copy_transducer_projection = str(copy_transducer_projection)
        if self.copy_transducer_logit_mode not in {
            "prob",
            "centered_prob",
            "centered_log",
        }:
            raise ValueError(
                "copy_transducer_logit_mode must be 'prob', 'centered_prob', or 'centered_log'"
            )
        if self.copy_transducer_projection not in {"soft", "argmax"}:
            raise ValueError(
                "copy_transducer_projection must be 'soft' or 'argmax'"
            )
        self.span_width = int(span_width)
        self.span_verifier = bool(span_verifier)
        self.span_prefix_conditioning = bool(span_prefix_conditioning)
        self.trainable_prior_gates = trainable_prior_gates
        self.trainable_transition_head = trainable_transition_head
        self.trainable_context_head = trainable_context_head
        self.prior_dropout = float(prior_dropout)
        self.dynamic_prior_gates = dynamic_prior_gates
        self.repeat_suppression_window = int(repeat_suppression_window)
        self.local_position_embeddings = local_position_embeddings
        if trainable_repeat_suppression:
            init_scale = torch.tensor(
                max(float(repeat_suppression_scale), 1e-6)
            )
            self.repeat_suppression_log_scale = nn.Parameter(
                torch.log(torch.expm1(init_scale))
            )
        else:
            self.repeat_suppression_scale = float(repeat_suppression_scale)
        self.modern_blocks = modern_blocks
        self.fused_attention = fused_attention
        self.local_window = local_window
        self.coarse_patch_size = coarse_patch_size
        self.coarse_layers = coarse_layers
        self.global_conv_layers = global_conv_layers
        self.global_gru_layers = global_gru_layers
        self.local_width = local_width or d_model
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.patch_encoder_layers = patch_encoder_layers
        self.patch_encoder_window = patch_encoder_window
        self.mod_layers = mod_layers
        self.mod_capacity = mod_capacity
        self.mod_group_size = mod_group_size
        self.mod_share_weights = mod_share_weights
        self.global_block = global_block
        self.routed_cake_experts = int(routed_cake_experts)
        self.shared_cake_layers = int(shared_cake_layers)
        self.default_cake_route = (
            None if default_cake_route is None else int(default_cake_route)
        )
        if self.routed_cake_experts < 0 or self.routed_cake_experts == 1:
            raise ValueError(
                "routed_cake_experts must be zero or at least two"
            )
        if not 0 <= self.shared_cake_layers <= int(layers):
            raise ValueError("shared_cake_layers must be within the global depth")
        if self.shared_cake_layers and not self.routed_cake_experts:
            raise ValueError("shared_cake_layers requires routed_cake_experts")
        if self.default_cake_route is not None and not (
            self.routed_cake_experts
            and 0 <= self.default_cake_route < self.routed_cake_experts
        ):
            raise ValueError("default_cake_route requires a valid routed cake index")
        self.sparse_state_local_window = sparse_state_local_window
        self.sparse_state_dilated_offsets = tuple(
            sparse_state_dilated_offsets
        )
        self.sparse_state_chunk_size = sparse_state_chunk_size
        self.abi_patch_cell_static_generation = bool(
            abi_patch_cell_static_generation
        )
        self.abi_patch_cell_global_update_interval = int(
            abi_patch_cell_global_update_interval
        )
        self.abi_patch_cell_fast_global_decode = bool(
            abi_patch_cell_fast_global_decode
        )
        self.abi_patch_cell_fast_local_runtime = bool(
            abi_patch_cell_fast_local_runtime
        )
        self.abi_patch_cell_lightweight_context_update = bool(
            abi_patch_cell_lightweight_context_update
        )
        self.abi_patch_cell_lightweight_context_blend = float(
            abi_patch_cell_lightweight_context_blend
        )
        self.generation_min_word_chars = int(generation_min_word_chars)
        self.generation_repeat_suppression_window = int(
            generation_repeat_suppression_window
        )
        self.generation_repeat_suppression_scale = float(
            generation_repeat_suppression_scale
        )
        self.profile_timing = False
        self.last_profile = {}
        if global_block not in {
            "attention",
            "sparse_state_patch",
            "selective_state_patch",
        }:
            raise ValueError(
                "global_block must be attention, sparse_state_patch, or selective_state_patch"
            )
        if global_block in {"sparse_state_patch", "selective_state_patch"} and not modern_blocks:
            raise ValueError(f"{global_block} requires modern_blocks")
        if self.local_width % heads:
            raise ValueError("local_width must be divisible by heads")
        if (
            patch_prediction_context == "local"
            and self.local_width != d_model
        ):
            raise ValueError(
                "local patch-prediction context requires local_width=d_model"
            )
        if (
            global_conv_layers < 0
            or global_gru_layers < 0
            or global_conv_layers + global_gru_layers > layers
        ):
            raise ValueError(
                "global mixer replacement count must be within layers"
            )
        if coarse_patch_size:
            if coarse_patch_size <= patch_size:
                raise ValueError(
                    "coarse_patch_size must exceed patch_size"
                )
            if coarse_patch_size % patch_size:
                raise ValueError(
                    "coarse_patch_size must be divisible by patch_size"
                )
            if not coarse_layers:
                raise ValueError(
                    "coarse_layers must be positive with coarse patching"
                )
        self.transition_head = nn.Embedding(256, 256)
        if transition_logits is None:
            nn.init.zeros_(self.transition_head.weight)
        else:
            if transition_logits.shape != (256, 256):
                raise ValueError("transition logits must have shape [256, 256]")
            with torch.no_grad():
                self.transition_head.weight.copy_(transition_logits)
        self.transition_head.weight.requires_grad_(trainable_transition_head)
        if trainable_prior_gates:
            self.transition_logit_scale = nn.Parameter(
                torch.tensor(float(transition_logit_scale))
            )
            self.context_logit_scale = nn.Parameter(
                torch.tensor(float(context_logit_scale))
            )
        else:
            self.transition_logit_scale = float(transition_logit_scale)
            self.context_logit_scale = float(context_logit_scale)
        if context_buckets:
            self.context_head = nn.Embedding(context_buckets, 256)
            if context_logits is None:
                nn.init.zeros_(self.context_head.weight)
            else:
                if context_logits.shape != (context_buckets, 256):
                    raise ValueError(
                        "context logits must match [context_buckets, 256]"
                    )
                with torch.no_grad():
                    self.context_head.weight.copy_(context_logits)
            self.context_head.weight.requires_grad_(trainable_context_head)
        self.register_buffer(
            "domain_cache_keys",
            torch.empty(0, dtype=torch.long),
        )
        self.register_buffer(
            "domain_cache_logits",
            torch.empty(0, 256, dtype=torch.float32),
        )
        self.byte_emb = nn.Embedding(256, d_byte)
        if self.copy_attention:
            self.copy_query = nn.Linear(self.local_width, self.copy_attention_dim, bias=False)
            self.copy_key = nn.Linear(d_byte, self.copy_attention_dim, bias=False)
        if self.copy_transducer:
            self.copy_transducer_query = nn.Linear(self.local_width, self.copy_transducer_dim, bias=False)
            self.copy_transducer_key = nn.Linear(d_byte, self.copy_transducer_dim, bias=False)
            self.copy_transducer_gate = nn.Linear(self.local_width, 1)
        if ngram_buckets:
            self.bigram_emb = nn.Embedding(ngram_buckets, d_byte)
            self.trigram_emb = nn.Embedding(ngram_buckets, d_byte)
        if patch_unit_buckets:
            self.patch_unit_emb = nn.Embedding(patch_unit_buckets, d_byte)
        patch_input_width = (
            d_model
            if patch_encoder_layers
            else patch_size * d_byte
            + (d_byte if patch_unit_buckets else 0)
        )
        if patch_encoder_layers:
            self.patch_encoder_in = nn.Linear(d_byte, d_model)
            self.patch_encoder_core = nn.ModuleList(
                FusedModernCausalBlock(
                    d_model,
                    heads,
                    dropout,
                    qk_norm,
                )
                for _ in range(patch_encoder_layers)
            )
            self.patch_encoder_norm = nn.LayerNorm(d_model)
        self.patch_proj = nn.Linear(patch_input_width, d_model)
        self.patch_pos = nn.Embedding(max_patches, d_model)
        if coarse_patch_size:
            self.coarse_patch_proj = nn.Linear(
                coarse_patch_size * d_byte, d_model
            )
            self.coarse_patch_pos = nn.Embedding(
                max_patches * patch_size // coarse_patch_size,
                d_model,
            )
            self.coarse_to_fine = nn.Linear(
                d_model, d_model, bias=False
            )
            block_type = (
                FusedModernCausalBlock
                if fused_attention
                else ModernCausalBlock
            )
            self.coarse_core = nn.ModuleList(
                (
                    block_type(d_model, heads, dropout, qk_norm)
                    if fused_attention
                    else block_type(d_model, heads, dropout)
                )
                for _ in range(coarse_layers)
            )
        if self.routed_cake_experts:
            if not modern_blocks or not fused_attention:
                raise ValueError(
                    "routed cakes require modern_blocks and fused_attention"
                )
            if global_block != "attention":
                raise ValueError(
                    "routed cakes currently require the attention global block"
                )
            if global_conv_layers or global_gru_layers or mod_layers:
                raise ValueError(
                    "routed cakes do not combine with replacement/refinement layers"
                )
            self.core = nn.ModuleList(
                [
                    FusedModernCausalBlock(
                        d_model,
                        heads,
                        dropout,
                        qk_norm,
                    )
                    for _ in range(self.shared_cake_layers)
                ]
                + [
                    Top1RoutedCakeBlock(
                        d_model,
                        heads,
                        self.routed_cake_experts,
                        dropout,
                        qk_norm,
                    )
                    for _ in range(layers - self.shared_cake_layers)
                ]
            )
        elif modern_blocks:
            if global_block == "sparse_state_patch":
                block_type = SparseStatePatchBlock
            elif global_block == "selective_state_patch":
                block_type = SelectiveStatePatchBlock
            else:
                block_type = (
                    FusedModernCausalBlock
                    if fused_attention
                    else ModernCausalBlock
                )
            attention_layers = (
                layers - global_conv_layers - global_gru_layers
            )
            self.core = nn.ModuleList(
                [
                    GatedCausalConvBlock(
                        d_model, 2 ** (index % 4)
                    )
                    for index in range(global_conv_layers)
                ]
                + [
                    ResidualCausalGRUBlock(d_model)
                    for _ in range(global_gru_layers)
                ]
                + [
                    (
                        block_type(
                            d_model,
                            heads,
                            sparse_state_local_window,
                            self.sparse_state_dilated_offsets,
                            sparse_state_chunk_size,
                            dropout,
                            qk_norm,
                        )
                        if global_block == "sparse_state_patch"
                        else (
                            block_type(d_model, heads, dropout, qk_norm)
                            if global_block == "selective_state_patch"
                            else (
                                block_type(d_model, heads, dropout, qk_norm)
                                if fused_attention
                                else block_type(d_model, heads, dropout)
                            )
                        )
                    )
                    for _ in range(attention_layers)
                ]
            )
            if mod_layers:
                if not fused_attention:
                    raise ValueError(
                        "mixture-of-depth currently requires fused attention"
                    )
                self.mod_refinement = MixtureOfDepthRefinement(
                    d_model,
                    heads,
                    mod_layers,
                    mod_capacity,
                    mod_group_size,
                    mod_share_weights,
                    dropout,
                    qk_norm,
                )
        else:
            block = nn.TransformerEncoderLayer(
                d_model, heads, d_model * 4, batch_first=True, norm_first=True
            )
            self.core = nn.TransformerEncoder(block, layers)
        self.to_abi = nn.Sequential(nn.Linear(d_model, d_abi), nn.LayerNorm(d_abi))
        self.from_abi = nn.Linear(d_abi, d_model)
        self.bos_context = nn.Parameter(torch.zeros(1, 1, d_abi))
        local_input_width = (
            d_byte + d_model + (d_model if patch_encoder_layers else 0)
        )
        if local_decoder == "gru":
            self.local = nn.GRU(
                local_input_width, self.local_width, batch_first=True
            )
        elif local_decoder == "conv":
            self.local_in = nn.Linear(
                local_input_width, self.local_width
            )
            self.local_blocks = nn.ModuleList(
                CausalConvBlock(self.local_width, 2**index)
                for index in range(conv_layers)
            )
            self.local_norm = nn.LayerNorm(self.local_width)
        elif local_decoder in {"parallel_patch", "span_patch_decoder"}:
            self.local_in = nn.Linear(d_model, self.local_width)
            offset_count = self.span_width if local_decoder == "span_patch_decoder" else patch_size
            if offset_count <= 0:
                raise ValueError("span_width must be positive")
            self.local_offsets = nn.Embedding(offset_count, self.local_width)
            self.local_norm = nn.LayerNorm(self.local_width)
            if (
                local_decoder == "span_patch_decoder"
                and self.span_prefix_conditioning
            ):
                self.span_prefix_proj = nn.Linear(
                    d_byte,
                    self.local_width,
                    bias=False,
                )
            if local_decoder == "span_patch_decoder":
                self.span_refine = nn.Sequential(
                    nn.Linear(self.local_width, self.local_width),
                    nn.GELU(),
                    nn.Linear(self.local_width, self.local_width),
                )
            if local_decoder == "span_patch_decoder" and self.span_verifier:
                self.span_verifier_head = nn.Linear(self.local_width, 1)
        elif local_decoder == "routed_window_transformer":
            if not self.routed_cake_experts or self.routed_cake_experts < 5:
                raise ValueError(
                    "routed_window_transformer requires at least five routed cakes"
                )
            if self.shared_cake_layers != layers - 1:
                raise ValueError(
                    "routed_window_transformer requires one routed global tail layer"
                )
            self.local_in = nn.Linear(local_input_width, self.local_width)
            self.local_norm = nn.LayerNorm(self.local_width)
        elif local_decoder == "abi_patch_cell":
            if patch_size != 2:
                raise ValueError("abi_patch_cell requires patch_size=2")
            if not direct_global_context:
                raise ValueError("abi_patch_cell requires direct_global_context")
            self.local_in = nn.Linear(d_model, self.local_width)
            self.local_offsets = nn.Embedding(patch_size, self.local_width)
            self.abi_cell_byte0 = nn.Linear(d_byte, self.local_width, bias=False)
            self.abi_cell_gate = nn.Linear(d_model + d_byte, self.local_width)
            self.abi_cell_byte1 = nn.Linear(d_byte, self.local_width, bias=False)
            self.abi_cell_next_gate = nn.Linear(d_model + d_byte, self.local_width)
            self.abi_cell_refine = nn.Sequential(
                nn.LayerNorm(self.local_width),
                nn.Linear(self.local_width, self.local_width),
                nn.GELU(),
                nn.Linear(self.local_width, self.local_width),
            )
            self.local_norm = nn.LayerNorm(self.local_width)
        elif local_decoder in {
            "transformer",
            "patch_transformer",
            "window_transformer",
        }:
            self.local_in = nn.Linear(
                local_input_width, self.local_width
            )
            if local_position_embeddings:
                self.local_pos = nn.Embedding(
                    max_patches * patch_size, self.local_width
                )
            if self.routed_cake_experts:
                if not fused_attention:
                    raise ValueError(
                        "routed local cakes require fused_attention"
                    )
                self.local_core = nn.ModuleList(
                    Top1RoutedCakeBlock(
                        self.local_width,
                        heads,
                        self.routed_cake_experts,
                        dropout,
                        qk_norm,
                    )
                    for _ in range(local_layers)
                )
            elif modern_blocks:
                block_type = (
                    FusedModernCausalBlock
                    if fused_attention
                    else ModernCausalBlock
                )
                self.local_core = nn.ModuleList(
                    (
                        block_type(
                            self.local_width,
                            heads,
                            dropout,
                            qk_norm,
                        )
                        if fused_attention
                        else block_type(
                            self.local_width, heads, dropout
                        )
                    )
                    for _ in range(local_layers)
                )
            else:
                local_block = nn.TransformerEncoderLayer(
                    self.local_width,
                    heads,
                    self.local_width * 4,
                    batch_first=True,
                    norm_first=True,
                )
                self.local_core = nn.TransformerEncoder(
                    local_block, local_layers
                )
            self.local_norm = nn.LayerNorm(self.local_width)
        else:
            raise ValueError(f"unsupported local decoder: {local_decoder}")
        if dynamic_prior_gates:
            self.prior_gate = nn.Linear(self.local_width, 2)
            with torch.no_grad():
                transition_init = torch.tensor(
                    float(transition_logit_scale)
                ).clamp(1e-4, 1.0 - 1e-4)
                context_init = torch.tensor(
                    float(context_logit_scale)
                ).clamp(1e-4, 1.0 - 1e-4)
                self.prior_gate.weight.zero_()
                self.prior_gate.bias[0] = torch.logit(transition_init)
                self.prior_gate.bias[1] = torch.logit(context_init)
        if tie_byte_embeddings:
            self.output_to_byte = nn.Linear(
                self.local_width, d_byte, bias=False
            )
            self.output_bias = nn.Parameter(torch.zeros(256))
            self.output_logit_scale = nn.Parameter(torch.tensor(1.0))
            nn.init.normal_(
                self.output_to_byte.weight, std=d_model ** -0.5
            )
        else:
            self.head = nn.Linear(self.local_width, 256)
        self.aux_heads = nn.ModuleList(
            nn.Linear(self.local_width, 256)
            for _ in range(mtp_depth)
        )
        if patch_prediction:
            if patch_prediction_mode == "factorized":
                self.patch_prediction_heads = nn.ModuleList(
                    nn.Linear(d_model, 256) for _ in range(patch_size)
                )
            elif patch_prediction_mode == "autoregressive":
                self.patch_generator = AutoregressivePatchHead(
                    d_model,
                    self.byte_emb,
                    patch_generation_width,
                    self.patch_generation_bytes,
                    copy_window=self.patch_generation_copy_window,
                    copy_dim=self.patch_generation_copy_dim,
                    copy_scale=self.patch_generation_copy_scale,
                    position_copy=self.patch_generation_position_copy,
                    contextual_copy=self.patch_generation_contextual_copy,
                    lowercase_copy=self.patch_generation_lowercase_copy,
                    semantic_copy=self.patch_generation_semantic_copy,
                )
            else:
                raise ValueError(
                    "patch_prediction_mode must be factorized or autoregressive"
                )
        if self.default_cake_route is not None:
            for module in self.modules():
                if isinstance(module, Top1RoutedCakeBlock):
                    module.set_route(self.default_cake_route)
        self.register_buffer("canonical_head", canonical_brick_head(d_abi))

    def set_cake_route(self, route: int | None) -> None:
        """Pin all routed global/local cakes to one expert, or restore routing."""
        routed = [
            module
            for module in self.modules()
            if isinstance(module, Top1RoutedCakeBlock)
        ]
        if not routed and route is not None:
            raise RuntimeError("this model has no routed cake blocks")
        for module in routed:
            module.set_route(route)

    def sparse_cake_parameters(
        self,
        route: int,
        *,
        include_router: bool = False,
    ):
        """Yield shared state plus only the selected experts for AdamW."""
        routed = [
            module
            for module in self.modules()
            if isinstance(module, Top1RoutedCakeBlock)
        ]
        if not routed:
            raise RuntimeError("this model has no routed cake blocks")
        routed_parameter_ids = {
            id(parameter)
            for module in routed
            for parameter in module.parameters()
        }
        active_parameter_ids = {
            id(parameter)
            for module in routed
            for parameter in module.active_expert_parameters(route)
        }
        if include_router:
            active_parameter_ids.update(
                id(parameter)
                for module in routed
                for parameter in (
                    list(module.router_norm.parameters())
                    + list(module.router.parameters())
                )
            )
        for parameter in self.parameters():
            if parameter.requires_grad and (
                id(parameter) not in routed_parameter_ids
                or id(parameter) in active_parameter_ids
            ):
                yield parameter

    def _transition_prior(
        self, x: torch.Tensor, scale: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        if scale is None:
            scale = self.transition_logit_scale
        prior = self.transition_head(x) * scale
        return F.dropout(prior, p=self.prior_dropout, training=self.training)

    def _context_prior(
        self, context_ids: torch.Tensor, scale: torch.Tensor | float | None = None
    ) -> torch.Tensor:
        if scale is None:
            scale = self.context_logit_scale
        prior = self.context_head(context_ids) * scale
        return F.dropout(prior, p=self.prior_dropout, training=self.training)

    def _last_context_id(self, recent: torch.Tensor) -> torch.Tensor:
        context_ids = torch.zeros(
            recent.shape[0],
            dtype=recent.dtype,
            device=recent.device,
        )
        length = recent.shape[1]
        for lag in range(self.context_order):
            if lag < length:
                value = recent[:, length - 1 - lag]
            else:
                value = torch.zeros_like(context_ids)
            context_ids = (context_ids * 257 + value + 1) % self.context_buckets
        return context_ids

    def _copy_attention_prior(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if not self.copy_attention:
            return hidden.new_zeros(*hidden.shape[:2], 256)
        source = x
        if self.copy_attention_window > 0 and source.shape[1] > self.copy_attention_window:
            source = source[:, -self.copy_attention_window :]
        query = self.copy_query(hidden)
        keys = self.copy_key(self.byte_emb(source))
        scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(
            max(self.copy_attention_dim, 1)
        )
        source_len = source.shape[1]
        target_len = hidden.shape[1]
        if source_len == target_len:
            causal = torch.ones(
                target_len,
                source_len,
                dtype=torch.bool,
                device=x.device,
            ).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        prior = hidden.new_zeros(hidden.shape[0], target_len, 256)
        prior.scatter_add_(
            2,
            source[:, None, :].expand(-1, target_len, -1),
            probs.to(prior.dtype),
        )
        return prior * self.copy_attention_scale

    def _copy_attention_next_prior(
        self,
        recent: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if not self.copy_attention:
            return hidden.new_zeros(hidden.shape[0], 256)
        source = recent
        if self.copy_attention_window > 0 and source.shape[1] > self.copy_attention_window:
            source = source[:, -self.copy_attention_window :]
        query = self.copy_query(hidden)
        keys = self.copy_key(self.byte_emb(source))
        scores = torch.matmul(query[:, None, :], keys.transpose(1, 2)).squeeze(1)
        scores = scores / math.sqrt(max(self.copy_attention_dim, 1))
        probs = torch.softmax(scores, dim=-1)
        prior = hidden.new_zeros(hidden.shape[0], 256)
        prior.scatter_add_(1, source, probs.to(prior.dtype))
        return prior * self.copy_attention_scale

    def _copy_transducer_logits(
        self,
        source_bytes: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.copy_transducer:
            return hidden.new_zeros(*hidden.shape[:2], 256), hidden.new_zeros(
                *hidden.shape[:2], 1
            )
        source = source_bytes
        if self.copy_transducer_window > 0 and source.shape[1] > self.copy_transducer_window:
            source = source[:, -self.copy_transducer_window :]
        query = self.copy_transducer_query(hidden)
        keys = self.copy_transducer_key(self.byte_emb(source))
        copy_scores = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(
            max(self.copy_transducer_dim, 1)
        )
        source_len = source.shape[1]
        target_len = hidden.shape[1]
        if source_len == target_len:
            causal = torch.ones(
                target_len,
                source_len,
                dtype=torch.bool,
                device=source.device,
            ).tril()
            copy_scores = copy_scores.masked_fill(~causal, float("-inf"))
        copy_probs = torch.softmax(copy_scores, dim=-1)
        if self.copy_transducer_projection == "argmax":
            max_positions = copy_probs.argmax(dim=-1, keepdim=True)
            copy_probs = torch.zeros_like(copy_probs).scatter_(
                -1,
                max_positions,
                copy_probs.gather(-1, max_positions),
            )
        copy_logits = hidden.new_zeros(hidden.shape[0], target_len, 256)
        copy_logits.scatter_add_(
            2,
            source[:, None, :].expand(-1, target_len, -1),
            copy_probs.to(copy_logits.dtype),
        )
        if self.copy_transducer_logit_mode == "centered_prob":
            copy_logits = copy_logits - copy_logits.mean(dim=-1, keepdim=True)
        elif self.copy_transducer_logit_mode == "centered_log":
            copy_logits = copy_logits.clamp_min(1e-8).log()
            copy_logits = copy_logits - copy_logits.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.copy_transducer_gate(hidden))
        return copy_logits * (gate * self.copy_transducer_scale), copy_scores

    def _copy_transducer_next_logits(
        self,
        source_bytes: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, scores = self._copy_transducer_logits(source_bytes, hidden[:, None, :])
        return logits[:, 0], scores[:, 0]

    def _span_hidden_from_context(
        self,
        context_h: torch.Tensor,
        prefix_bytes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project patch states into fixed-width future-span decoder states."""
        if self.local_decoder != "span_patch_decoder":
            raise RuntimeError("span hidden requires span_patch_decoder")
        offset_positions = torch.arange(self.span_width, device=context_h.device)
        hidden = (
            self.local_in(context_h).unsqueeze(2)
            + self.local_offsets(offset_positions)[None, None]
        )
        if prefix_bytes is not None and self.span_prefix_conditioning:
            prefix_emb = self.byte_emb(prefix_bytes)
            shifted = F.pad(prefix_emb[:, :, :-1], (0, 0, 1, 0))
            prefix_sum = shifted.cumsum(dim=2)
            hidden = hidden + self.span_prefix_proj(prefix_sum)
        hidden = hidden + self.span_refine(hidden)
        return self.local_norm(hidden)

    def _span_logits_from_context(
        self,
        context_h: torch.Tensor,
        prefix_bytes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._span_hidden_from_context(context_h, prefix_bytes)
        logits = self._byte_logits(hidden)
        return logits, hidden

    def set_domain_cache(
        self,
        keys: torch.Tensor,
        logits: torch.Tensor,
        *,
        order: int | None = None,
        logit_scale: float | None = None,
    ) -> None:
        if keys.numel() == 0:
            self.domain_cache_keys = torch.empty(
                0, dtype=torch.long, device=self.domain_cache_keys.device
            )
            self.domain_cache_logits = torch.empty(
                0, 256, dtype=torch.float32, device=self.domain_cache_logits.device
            )
            return
        if logits.shape != (keys.numel(), 256):
            raise ValueError("domain cache logits must have shape [keys, 256]")
        order_idx = torch.argsort(keys.to(torch.long))
        self.domain_cache_keys = keys.to(
            device=self.domain_cache_keys.device, dtype=torch.long
        )[order_idx]
        self.domain_cache_logits = logits.to(
            device=self.domain_cache_logits.device, dtype=torch.float32
        )[order_idx]
        if order is not None:
            self.domain_cache_order = int(order)
        if logit_scale is not None:
            self.domain_cache_logit_scale = float(logit_scale)

    def _domain_cache_context_keys(self, x: torch.Tensor) -> torch.Tensor:
        keys = torch.zeros_like(x, dtype=torch.long)
        modulus = 2305843009213693951
        for lag in range(self.domain_cache_order):
            shifted = F.pad(x[:, : x.shape[1] - lag], (lag, 0)).to(torch.long)
            keys = torch.remainder(keys * 257 + shifted + 1, modulus)
        return keys

    def _last_domain_cache_key(self, recent: torch.Tensor) -> torch.Tensor:
        keys = torch.zeros(
            recent.shape[0],
            dtype=torch.long,
            device=recent.device,
        )
        modulus = 2305843009213693951
        length = recent.shape[1]
        for lag in range(self.domain_cache_order):
            if lag < length:
                value = recent[:, length - 1 - lag].to(torch.long)
            else:
                value = torch.zeros_like(keys)
            keys = torch.remainder(keys * 257 + value + 1, modulus)
        return keys

    def _domain_cache_prior_from_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if (
            self.domain_cache_order <= 0
            or self.domain_cache_logit_scale == 0.0
            or self.domain_cache_keys.numel() == 0
        ):
            return keys.new_zeros(*keys.shape, 256, dtype=torch.float32)
        flat = keys.reshape(-1).to(self.domain_cache_keys.device)
        indexes = torch.searchsorted(self.domain_cache_keys, flat)
        valid = indexes < self.domain_cache_keys.numel()
        safe_indexes = indexes.clamp(max=max(self.domain_cache_keys.numel() - 1, 0))
        valid = valid & (self.domain_cache_keys[safe_indexes] == flat)
        prior = self.domain_cache_logits.new_zeros(flat.shape[0], 256)
        if bool(valid.any()):
            prior[valid] = self.domain_cache_logits[safe_indexes[valid]]
        return (
            prior.reshape(*keys.shape, 256).to(keys.device)
            * self.domain_cache_logit_scale
        )

    def _last_domain_cache_prior(self, recent: torch.Tensor) -> torch.Tensor:
        keys = self._last_domain_cache_key(recent)
        return self._domain_cache_prior_from_keys(keys)

    def _domain_cache_active(self, prior: torch.Tensor) -> torch.Tensor:
        return prior.abs().sum(dim=-1, keepdim=True) > 0

    def _dynamic_prior_scales(
        self, local_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.dynamic_prior_gates:
            return self.transition_logit_scale, self.context_logit_scale
        gates = torch.sigmoid(self.prior_gate(local_hidden))
        return gates[..., 0:1], gates[..., 1:2]

    def _repeat_suppression_bias(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.repeat_suppression_window <= 0:
            return None
        if hasattr(self, "repeat_suppression_log_scale"):
            scale = F.softplus(self.repeat_suppression_log_scale)
        else:
            scale = x.new_tensor(self.repeat_suppression_scale, dtype=torch.float32)
        if float(scale.detach().cpu()) <= 0.0:
            return None
        batch, length = x.shape
        counts = x.new_zeros(batch, length, 256, dtype=torch.float32)
        for lag in range(self.repeat_suppression_window):
            shifted = torch.full_like(x, -1)
            if lag == 0:
                shifted = x
            elif lag < length:
                shifted[:, lag:] = x[:, : length - lag]
            valid = shifted >= 0
            safe = shifted.clamp_min(0)
            counts.scatter_add_(
                2,
                safe.unsqueeze(-1),
                valid.to(dtype=counts.dtype).unsqueeze(-1),
            )
        return -scale * counts

    def _generation_repeat_suppression_bias(
        self, recent: torch.Tensor
    ) -> torch.Tensor | None:
        window = self.generation_repeat_suppression_window
        scale = self.generation_repeat_suppression_scale
        if window <= 0 or scale <= 0.0 or recent.numel() == 0:
            return None
        tail = recent[:, -window:]
        counts = recent.new_zeros(recent.shape[0], 256, dtype=torch.float32)
        counts.scatter_add_(
            1,
            tail.clamp_min(0),
            torch.ones_like(tail, dtype=counts.dtype),
        )
        return -float(scale) * counts

    def _apply_generation_word_shape_constraints(
        self, logits: torch.Tensor, recent: torch.Tensor
    ) -> torch.Tensor:
        if self.generation_min_word_chars <= 1 or recent.shape[1] < 2:
            return logits
        previous = recent[:, -1]
        previous2 = recent[:, -2]
        previous_is_alpha = (
            ((previous >= ord("a")) & (previous <= ord("z")))
            | ((previous >= ord("A")) & (previous <= ord("Z")))
        )
        previous2_is_boundary = previous2 <= ord(" ")
        block_space = previous_is_alpha & previous2_is_boundary
        constrained = logits.clone()
        constrained[block_space, ord(" ")] = float("-inf")
        constrained[block_space, ord("\n")] = float("-inf")
        constrained[block_space, ord("\r")] = float("-inf")
        constrained[block_space, ord("\t")] = float("-inf")
        return constrained

    def _profile_start(self) -> float | None:
        if not self.profile_timing:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _profile_stop(self, name: str, started: float | None) -> None:
        if started is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_profile[name] = (
            self.last_profile.get(name, 0.0) + time.perf_counter() - started
        )

    def forward(
        self,
        x: torch.Tensor,
        brick=None,
        return_aux: bool = False,
        return_patch_prediction: bool = False,
        return_generated_patch: bool = False,
        patch_prediction_context_indices: torch.Tensor | None = None,
    ):
        if self.profile_timing:
            self.last_profile = {}
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        flat_byte_h = self.byte_emb(x)
        if self.ngram_buckets:
            previous = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
            previous2 = torch.cat(
                [torch.zeros_like(x[:, :2]), x[:, :-2]], dim=1
            )
            bigram_ids = (previous * 257 + x) % self.ngram_buckets
            trigram_ids = (
                previous2 * 65537 + previous * 257 + x
            ) % self.ngram_buckets
            flat_byte_h = (
                flat_byte_h
                + self.bigram_emb(bigram_ids)
                + self.trigram_emb(trigram_ids)
            )
        byte_h = flat_byte_h.reshape(
            batch, -1, self.patch_size, flat_byte_h.shape[-1]
        )
        if self.patch_encoder_layers:
            if usable % self.patch_encoder_window:
                raise ValueError(
                    "sequence length must divide patch_encoder_window"
                )
            encoded_bytes = self.patch_encoder_in(flat_byte_h).reshape(
                batch * (usable // self.patch_encoder_window),
                self.patch_encoder_window,
                -1,
            )
            encoded_bytes = run_modern_stack(
                self.patch_encoder_core,
                encoded_bytes,
                causal_mask(self.patch_encoder_window, x.device),
            ).reshape(batch, usable, -1)
            encoded_bytes = self.patch_encoder_norm(encoded_bytes)
            patch_features = encoded_bytes[
                :, self.patch_size - 1 :: self.patch_size
            ]
        else:
            patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets and not self.patch_encoder_layers:
            patch_ids = patches[:, :, 0].clone()
            for index in range(1, self.patch_size):
                patch_ids = (
                    patch_ids * 257 + patches[:, :, index]
                ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        if self.coarse_patch_size:
            if usable % self.coarse_patch_size:
                raise ValueError(
                    "sequence length must be divisible by coarse_patch_size"
                )
            coarse_features = flat_byte_h.reshape(
                batch,
                -1,
                self.coarse_patch_size * flat_byte_h.shape[-1],
            )
            coarse_h = self.coarse_patch_proj(coarse_features)
            coarse_positions = torch.arange(
                coarse_h.shape[1], device=x.device
            )
            coarse_h = (
                coarse_h + self.coarse_patch_pos(coarse_positions)[None]
            )
            coarse_mask = causal_mask(
                coarse_h.shape[1], coarse_h.device
            )
            coarse_h = run_modern_stack(
                self.coarse_core, coarse_h, coarse_mask
            )
            shifted_coarse = torch.cat(
                [
                    coarse_h.new_zeros(batch, 1, coarse_h.shape[-1]),
                    coarse_h[:, :-1],
                ],
                dim=1,
            )
            ratio = self.coarse_patch_size // self.patch_size
            coarse_context = shifted_coarse.repeat_interleave(
                ratio, dim=1
            )
            patch_h = patch_h + self.coarse_to_fine(coarse_context)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        global_mask = causal_mask(patch_h.shape[1], patch_h.device)
        profile_started = self._profile_start()
        if self.modern_blocks:
            global_h = run_modern_stack(self.core, patch_h, global_mask)
        else:
            global_h = self.core(patch_h, mask=global_mask)
        route_mask = None
        if self.mod_layers:
            global_h, route_mask = self.mod_refinement(
                global_h, global_mask
            )
        self._profile_stop("global_core_seconds", profile_started)
        completed_abi = self.to_abi(global_h)
        context_abi = torch.cat(
            [self.bos_context.expand(batch, 1, -1), completed_abi[:, :-1]], dim=1
        )
        if self.direct_global_context:
            bos_global = global_h.new_zeros(batch, 1, global_h.shape[-1])
            global_context = torch.cat([bos_global, global_h[:, :-1]], dim=1)
            context = global_context.unsqueeze(2).expand(
                -1, -1, self.patch_size, -1
            )
        else:
            context = self.from_abi(context_abi).unsqueeze(2).expand(
                -1, -1, self.patch_size, -1
            )
        local_parts = [byte_h, context]
        if self.patch_encoder_layers:
            local_parts.append(encoded_bytes.reshape(
                batch, -1, self.patch_size, encoded_bytes.shape[-1]
            ))
        local_in = torch.cat(local_parts, dim=-1)
        profile_started = self._profile_start()
        if self.local_decoder == "abi_patch_cell":
            previous_patch_context = context[:, :, 0, :]
            previous_context_hidden = self.local_in(previous_patch_context)
            if self.direct_global_context:
                current_context_hidden = self.local_in(global_h)
            else:
                current_context_hidden = self.local_in(self.from_abi(completed_abi))
            byte0_h = byte_h[:, :, 0, :]
            byte1_h = byte_h[:, :, 1, :]
            gated = torch.sigmoid(
                self.abi_cell_gate(torch.cat([previous_patch_context, byte0_h], dim=-1))
            )
            byte1_hidden = (
                previous_context_hidden
                + self.local_offsets.weight[1]
                + gated * self.abi_cell_byte0(byte0_h)
            )
            next_gated = torch.sigmoid(
                self.abi_cell_next_gate(torch.cat([global_h, byte1_h], dim=-1))
            )
            next_patch_byte0_hidden = (
                current_context_hidden
                + self.local_offsets.weight[0]
                + next_gated * self.abi_cell_byte1(byte1_h)
            )
            local_out = torch.stack([byte1_hidden, next_patch_byte0_hidden], dim=2).reshape(
                batch, usable, -1
            )
            local_out = local_out + self.abi_cell_refine(local_out)
            logits = self._byte_logits(self.local_norm(local_out))
        elif self.local_decoder in {"parallel_patch", "span_patch_decoder"}:
            patch_context = context[:, :, 0, :]
            if self.local_decoder == "span_patch_decoder":
                byte_context = patch_context.repeat_interleave(
                    self.patch_size,
                    dim=1,
                )
                offset_positions = torch.arange(usable, device=x.device) % self.span_width
                local_out = (
                    self.local_in(byte_context)
                    + self.local_offsets(offset_positions)[None]
                )
            else:
                offset_positions = torch.arange(self.patch_size, device=x.device)
                local_out = (
                    self.local_in(patch_context).unsqueeze(2)
                    + self.local_offsets(offset_positions)[None, None]
                ).reshape(batch, usable, -1)
            logits = self._byte_logits(self.local_norm(local_out))
        elif self.local_decoder in {
            "window_transformer",
            "routed_window_transformer",
        }:
            if usable % self.local_window:
                raise ValueError("sequence length must be divisible by local_window")
            windowed_in = local_in.reshape(
                batch * (usable // self.local_window),
                self.local_window,
                -1,
            )
            local_out = self.local_in(windowed_in)
            local_mask = causal_mask(self.local_window, x.device)
            if self.local_decoder == "routed_window_transformer":
                routed_tail = self.core[-1]
                if not isinstance(routed_tail, Top1RoutedCakeBlock):
                    raise RuntimeError("routed local decoder requires a routed tail")
                local_out = run_modern_stack(
                    routed_tail.experts[1:5],
                    local_out,
                    local_mask,
                )
            elif self.modern_blocks:
                local_out = run_modern_stack(
                    self.local_core, local_out, local_mask
                )
            else:
                local_out = self.local_core(local_out, mask=local_mask)
            local_out = local_out.reshape(batch, usable, -1)
            logits = self._byte_logits(self.local_norm(local_out))
        elif self.local_decoder == "patch_transformer":
            patch_local_in = local_in.reshape(
                batch * patches.shape[1], self.patch_size, -1
            )
            local_out = self.local_in(patch_local_in)
            if self.local_position_embeddings:
                local_positions = torch.arange(
                    self.patch_size, device=x.device
                )
                local_out = (
                    local_out
                    + self.local_pos(local_positions)[None]
                )
            local_mask = causal_mask(self.patch_size, x.device)
            if self.modern_blocks:
                local_out = run_modern_stack(
                    self.local_core, local_out, local_mask
                )
            else:
                local_out = self.local_core(
                    local_out, mask=local_mask
                )
            local_out = local_out.reshape(batch, usable, -1)
            normalized = self.local_norm(local_out)
            logits = self._byte_logits(normalized)
        elif self.local_decoder == "transformer":
            local_out = self.local_in(local_in.reshape(batch, usable, -1))
            if self.local_position_embeddings:
                local_positions = torch.arange(usable, device=x.device)
                local_out = (
                    local_out
                    + self.local_pos(local_positions)[None]
                )
            local_mask = causal_mask(usable, x.device)
            if self.modern_blocks:
                local_out = run_modern_stack(
                    self.local_core, local_out, local_mask
                )
            else:
                local_out = self.local_core(
                    local_out, mask=local_mask
                )
            logits = self._byte_logits(self.local_norm(local_out))
        elif self.local_decoder == "conv":
            local_out = self.local_in(local_in.reshape(batch, usable, -1))
            for block in self.local_blocks:
                local_out = block(local_out)
            logits = self._byte_logits(self.local_norm(local_out))
        elif self.continuous_local:
            local_in = local_in.reshape(batch, usable, -1)
            local_out, _ = self.local(local_in)
            logits = self._byte_logits(local_out)
        else:
            local_in = local_in.reshape(
                batch * patches.shape[1], self.patch_size, -1
            )
            local_out, _ = self.local(local_in)
            logits = self._byte_logits(local_out).reshape(batch, usable, 256)
        self._profile_stop("local_decoder_seconds", profile_started)
        transition_prior_scale = None
        context_prior_scale = None
        if self.dynamic_prior_gates:
            gate_source = (
                self.local_norm(local_out)
                if hasattr(self, "local_norm")
                else local_out
            )
            transition_prior_scale, context_prior_scale = (
                self._dynamic_prior_scales(gate_source)
            )
        logits = logits + self._transition_prior(x, transition_prior_scale)
        if self.context_buckets:
            context_ids = self._context_ids(x)
            logits = logits + self._context_prior(
                context_ids, context_prior_scale
            )
        if self.copy_attention:
            copy_hidden = (
                self.local_norm(local_out)
                if hasattr(self, "local_norm")
                else local_out
            )
            logits = logits + self._copy_attention_prior(x, copy_hidden)
        copy_scores = None
        if self.copy_transducer:
            copy_hidden = (
                self.local_norm(local_out)
                if hasattr(self, "local_norm")
                else local_out
            )
            copy_logits, copy_scores = self._copy_transducer_logits(x, copy_hidden)
            logits = logits + copy_logits
        if self.domain_cache_order > 0 and self.domain_cache_logit_scale != 0.0:
            domain_prior = self._domain_cache_prior_from_keys(
                self._domain_cache_context_keys(x)
            )
            if self.domain_cache_override:
                active = domain_prior.abs().sum(dim=-1, keepdim=True) > 0
                logits = torch.where(active, domain_prior.to(logits.dtype), logits)
            else:
                logits = logits + domain_prior
        repeat_bias = self._repeat_suppression_bias(x)
        if repeat_bias is not None:
            logits = logits + repeat_bias
        if brick is not None:
            delta = brick(context_abi) - context_abi
            patch_bias = delta @ self.canonical_head
            logits = logits + patch_bias.unsqueeze(2).expand(
                -1, -1, self.patch_size, -1
            ).reshape(batch, usable, 256)
        span_future_logits = None
        span_future_copy_scores = None
        if self.local_decoder == "span_patch_decoder":
            span_prefix_bytes = x.new_zeros(
                batch,
                global_h.shape[1],
                self.span_width,
            )
            starts = (
                (torch.arange(global_h.shape[1], device=x.device) + 1)
                * self.patch_size
            )
            for patch_index, start in enumerate(starts.tolist()):
                available = min(self.span_width, max(usable - start, 0))
                if available > 0:
                    span_prefix_bytes[:, patch_index, :available] = x[
                        :, start : start + available
                    ]
            span_future_logits, span_future_hidden = self._span_logits_from_context(
                global_h,
                span_prefix_bytes,
            )
            if self.copy_transducer:
                flat_span_hidden = span_future_hidden.reshape(
                    batch,
                    global_h.shape[1] * self.span_width,
                    -1,
                )
                _, flat_span_copy_scores = self._copy_transducer_logits(
                    x,
                    flat_span_hidden,
                )
                span_future_copy_scores = flat_span_copy_scores.reshape(
                    batch,
                    global_h.shape[1],
                    self.span_width,
                    -1,
                )
        if return_aux:
            auxiliary = [head(local_out) for head in self.aux_heads]
            if span_future_logits is not None:
                auxiliary.append(span_future_logits)
            if span_future_copy_scores is not None:
                auxiliary.append(span_future_copy_scores)
            if self.copy_transducer and copy_scores is not None:
                auxiliary.append(copy_scores)
            if return_patch_prediction and self.patch_prediction:
                profile_started = self._profile_start()
                if self.patch_prediction_context == "local":
                    prediction_source = self.local_norm(local_out)[
                        :, self.patch_size - 1 :: self.patch_size
                    ]
                else:
                    prediction_source = global_h
                prediction_context = prediction_source[
                    :, :: self.patch_prediction_stride
                ]
                if patch_prediction_context_indices is not None:
                    context_indices = patch_prediction_context_indices.to(
                        device=x.device,
                        dtype=torch.long,
                    ).clamp(0, prediction_context.shape[1] - 1)
                    prediction_context = prediction_context.gather(
                        1,
                        context_indices[:, None, None].expand(
                            -1,
                            1,
                            prediction_context.shape[-1],
                        ),
                    )
                if self.patch_prediction_detach_context:
                    prediction_context = prediction_context.detach()
                if self.patch_prediction_mode == "autoregressive":
                    next_patches = self.patch_prediction_targets(x)[
                        :, :: self.patch_prediction_stride
                    ]
                    generation_prefix = None
                    if self.patch_generation_context:
                        generation_prefix = self._patch_generation_prefixes(
                            x
                        )[:, :: self.patch_prediction_stride]
                    generation_copy_source = None
                    if self.patch_generation_copy_window:
                        generation_copy_source = self._patch_generation_copy_sources(
                            x
                        )[:, :: self.patch_prediction_stride]
                    if patch_prediction_context_indices is not None:
                        target_index = context_indices[:, None, None]
                        next_patches = next_patches.gather(
                            1,
                            target_index.expand(
                                -1,
                                1,
                                next_patches.shape[-1],
                            ),
                        )
                        if generation_prefix is not None:
                            generation_prefix = generation_prefix.gather(
                                1,
                                target_index.expand(
                                    -1,
                                    1,
                                    generation_prefix.shape[-1],
                                ),
                            )
                        if generation_copy_source is not None:
                            generation_copy_source = generation_copy_source.gather(
                                1,
                                target_index.expand(
                                    -1,
                                    1,
                                    generation_copy_source.shape[-1],
                                ),
                            )
                    conditioning_bytes = next_patches
                    if self.training and self.patch_prediction_rollout_training:
                        rollout_bytes = self.patch_generator.greedy(
                            prediction_context,
                            generation_prefix,
                            source=generation_copy_source,
                        )
                        if self.patch_prediction_rollout_mix >= 1.0:
                            conditioning_bytes = rollout_bytes
                        elif self.patch_prediction_rollout_mix > 0.0:
                            rollout_mask = torch.rand(
                                rollout_bytes.shape,
                                device=rollout_bytes.device,
                            ) < self.patch_prediction_rollout_mix
                            conditioning_bytes = torch.where(
                                rollout_mask,
                                rollout_bytes,
                                next_patches,
                            )
                    patch_predictions = self.patch_generator(
                        prediction_context,
                        conditioning_bytes,
                        generation_prefix,
                        source=generation_copy_source,
                    )
                    if self.patch_prediction_context == "local":
                        patch_predictions[0] = logits[
                            :, self.patch_size - 1 :: self.patch_size
                        ][:, :: self.patch_prediction_stride]
                else:
                    patch_predictions = [
                        head(prediction_context)
                        for head in self.patch_prediction_heads
                    ]
                if return_generated_patch:
                    if self.patch_prediction_mode == "autoregressive":
                        forced_first = None
                        if self.patch_prediction_context == "local":
                            forced_first = logits[:, -1].argmax(dim=-1)
                        generated_patch = self.patch_generator.greedy(
                            prediction_context[:, -1],
                            (
                                x[:, -self.patch_generation_context :]
                                if self.patch_generation_context
                                else None
                            ),
                            forced_first=forced_first,
                            source=(
                                x[:, -self.patch_generation_copy_window :]
                                if self.patch_generation_copy_window
                                else None
                            ),
                        )
                    else:
                        generated_patch = torch.stack(
                            [
                                prediction[:, -1].argmax(dim=-1)
                                for prediction in patch_predictions
                            ],
                            dim=-1,
                        )
                    self._profile_stop(
                        "patch_prediction_seconds", profile_started
                    )
                    return (
                        logits,
                        context_abi,
                        auxiliary,
                        patch_predictions,
                        generated_patch,
                    )
                self._profile_stop(
                    "patch_prediction_seconds", profile_started
                )
                return logits, context_abi, auxiliary, patch_predictions
            if self.mod_layers:
                return logits, context_abi, auxiliary, route_mask
            return logits, context_abi, auxiliary
        return logits, context_abi

    def _byte_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.tie_byte_embeddings:
            byte_space = F.normalize(
                self.output_to_byte(hidden), dim=-1
            )
            byte_weights = F.normalize(self.byte_emb.weight, dim=-1)
            return (
                self.output_logit_scale.exp()
                * (byte_space @ byte_weights.transpose(0, 1))
                + self.output_bias
            )
        return self.head(hidden)

    def _patch_generation_prefixes(self, x: torch.Tensor) -> torch.Tensor:
        """Recent-byte windows ending at each completed source patch."""
        if self.patch_generation_context < self.patch_size:
            raise ValueError(
                "patch_generation_context must be at least patch_size"
            )
        return self._patch_generation_byte_windows(
            x,
            self.patch_generation_context,
        )

    def _patch_generation_copy_sources(self, x: torch.Tensor) -> torch.Tensor:
        """Recent-byte copy windows ending at each completed source patch."""
        if self.patch_generation_copy_window < self.patch_size:
            raise ValueError(
                "patch_generation_copy_window must be at least patch_size"
            )
        return self._patch_generation_byte_windows(
            x,
            self.patch_generation_copy_window,
        )

    def _patch_generation_byte_windows(
        self,
        x: torch.Tensor,
        window: int,
    ) -> torch.Tensor:
        return F.pad(
            x,
            (
                int(window) - self.patch_size,
                0,
            ),
        ).unfold(
            1,
            int(window),
            self.patch_size,
        )

    def domain_cake_patch_predictions(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Teacher-forced generation loss without the legacy local decoder.

        This is the training path for a routed domain cake: the frozen shared
        foundation and frozen portable decoder remain in the forward graph,
        while only the selected tail cake needs gradients and optimizer state.
        """
        if not self.patch_prediction or self.patch_prediction_mode != "autoregressive":
            raise RuntimeError("domain cake training requires autoregressive patch prediction")
        if self.patch_prediction_context != "global":
            raise RuntimeError("domain cake training requires global patch context")
        if self.patch_encoder_layers or self.coarse_patch_size:
            raise RuntimeError("domain cake training does not support hierarchical patch encoders")
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        if usable == 0:
            raise ValueError("input must contain at least one complete patch")
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        flat_byte_h = self.byte_emb(x)
        byte_h = flat_byte_h.reshape(
            batch, -1, self.patch_size, flat_byte_h.shape[-1]
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = patches[:, :, 0].clone()
            for index in range(1, self.patch_size):
                patch_ids = (
                    patch_ids * 257 + patches[:, :, index]
                ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        mask = causal_mask(patch_h.shape[1], patch_h.device)
        global_h = (
            run_modern_stack(self.core, patch_h, mask)
            if self.modern_blocks
            else self.core(patch_h, mask=mask)
        )
        prediction_context = global_h[:, :: self.patch_prediction_stride]
        if self.patch_prediction_detach_context:
            prediction_context = prediction_context.detach()
        targets = self.patch_prediction_targets(x)[
            :, :: self.patch_prediction_stride
        ]
        prefix = (
            self._patch_generation_prefixes(x)[:, :: self.patch_prediction_stride]
            if self.patch_generation_context
            else None
        )
        source = (
            self._patch_generation_copy_sources(x)[:, :: self.patch_prediction_stride]
            if self.patch_generation_copy_window
            else None
        )
        if context_indices is not None:
            context_indices = context_indices.to(
                device=x.device,
                dtype=torch.long,
            ).clamp(0, prediction_context.shape[1] - 1)
            gather_index = context_indices[:, None, None]
            prediction_context = prediction_context.gather(
                1,
                gather_index.expand(-1, 1, prediction_context.shape[-1]),
            )
            targets = targets.gather(
                1,
                gather_index.expand(-1, 1, targets.shape[-1]),
            )
            if prefix is not None:
                prefix = prefix.gather(
                    1,
                    gather_index.expand(-1, 1, prefix.shape[-1]),
                )
            if source is not None:
                source = source.gather(
                    1,
                    gather_index.expand(-1, 1, source.shape[-1]),
                )
        predictions = self.patch_generator(
            prediction_context,
            targets,
            prefix,
            source=source,
        )
        return predictions, targets

    @torch.no_grad()
    def generate_next_patch(
        self, x: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor:
        """Generate from the global path without running the local LM decoder."""
        if self.local_decoder not in {"parallel_patch", "abi_patch_cell"} and not self.patch_prediction:
            raise RuntimeError("patch prediction is not enabled")
        if self.patch_prediction_context == "local":
            return self(
                x,
                return_aux=True,
                return_patch_prediction=True,
                return_generated_patch=True,
            )[4]
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        if usable == 0:
            raise ValueError("input must contain at least one complete patch")
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        flat_byte_h = self.byte_emb(x)
        if self.ngram_buckets:
            previous = torch.cat(
                [torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1
            )
            previous2 = torch.cat(
                [torch.zeros_like(x[:, :2]), x[:, :-2]], dim=1
            )
            bigram_ids = (previous * 257 + x) % self.ngram_buckets
            trigram_ids = (
                previous2 * 65537 + previous * 257 + x
            ) % self.ngram_buckets
            flat_byte_h = (
                flat_byte_h
                + self.bigram_emb(bigram_ids)
                + self.trigram_emb(trigram_ids)
            )
        byte_h = flat_byte_h.reshape(
            batch, -1, self.patch_size, flat_byte_h.shape[-1]
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = patches[:, :, 0].clone()
            for index in range(1, self.patch_size):
                patch_ids = (
                    patch_ids * 257 + patches[:, :, index]
                ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        mask = causal_mask(patch_h.shape[1], patch_h.device)
        if self.modern_blocks:
            global_h = run_modern_stack(self.core, patch_h, mask)
        else:
            global_h = self.core(patch_h, mask=mask)
        if self.patch_prediction_mode == "autoregressive":
            return self.patch_generator.greedy(
                global_h[:, -1],
                (
                    x[:, -self.patch_generation_context :]
                    if self.patch_generation_context
                    else None
                ),
                source=(
                    x[:, -self.patch_generation_copy_window :]
                    if self.patch_generation_copy_window
                    else None
                ),
            )
        if self.local_decoder == "abi_patch_cell":
            context_hidden = self.local_in(global_h[:, -1])
            last_byte_h = self.byte_emb(x[:, -1])
            next_gated = torch.sigmoid(
                self.abi_cell_next_gate(torch.cat([global_h[:, -1], last_byte_h], dim=-1))
            )
            first_hidden = (
                context_hidden
                + self.local_offsets.weight[0]
                + next_gated * self.abi_cell_byte1(last_byte_h)
            )
            first_hidden = first_hidden + self.abi_cell_refine(first_hidden)
            first_logits = self._byte_logits(self.local_norm(first_hidden))
            last_byte = x[:, -1]
            first_logits = first_logits + self._transition_prior(last_byte)
            if self.context_buckets:
                first_context_id = self._last_context_id(x)
                first_logits = first_logits + self._context_prior(first_context_id)
            first = first_logits.argmax(dim=-1)
            first_byte_h = self.byte_emb(first)
            gated = torch.sigmoid(
                self.abi_cell_gate(torch.cat([global_h[:, -1], first_byte_h], dim=-1))
            )
            second_hidden = (
                context_hidden
                + self.local_offsets.weight[1]
                + gated * self.abi_cell_byte0(first_byte_h)
            )
            second_hidden = second_hidden + self.abi_cell_refine(second_hidden)
            second_logits = self._byte_logits(self.local_norm(second_hidden))
            second_logits = second_logits + self._transition_prior(first)
            if self.context_buckets:
                second_context = torch.cat([x, first[:, None]], dim=1)
                second_context_id = self._last_context_id(second_context)
                second_logits = second_logits + self._context_prior(second_context_id)
            logits = torch.stack([first_logits, second_logits], dim=1)
            if return_logits:
                return logits
            return torch.stack([first, second_logits.argmax(dim=-1)], dim=-1)
        if self.local_decoder == "parallel_patch":
            offset_positions = torch.arange(self.patch_size, device=x.device)
            local_out = (
                self.local_in(global_h[:, -1]).unsqueeze(1)
                + self.local_offsets(offset_positions)[None]
            )
            logits = self._byte_logits(self.local_norm(local_out))
            if return_logits:
                return logits
            return logits.argmax(dim=-1)
        return torch.stack(
            [
                head(global_h[:, -1]).argmax(dim=-1)
                for head in self.patch_prediction_heads
            ],
            dim=-1,
        )

    @torch.no_grad()
    def prepare_patch_generator_cuda_graph(self, batch_size: int = 1) -> dict:
        """Capture fixed-shape local draft decoding for launch-efficient CUDA inference."""
        existing = getattr(self, "_patch_generator_cuda_graph_runtime", None)
        if existing is not None:
            return dict(existing["summary"])
        if not next(self.parameters()).is_cuda:
            raise RuntimeError("CUDA graph preparation requires a CUDA model")
        if self.patch_prediction_mode != "autoregressive":
            raise RuntimeError("CUDA graph preparation requires autoregressive patch prediction")
        if self.patch_generation_context:
            raise RuntimeError("CUDA graph preparation does not support generation prefixes")
        if self.patch_generation_copy_window <= 0:
            raise RuntimeError("CUDA graph preparation requires a fixed copy window")

        started = time.perf_counter()
        original_greedy = self.patch_generator.greedy

        class GreedyTraceWrapper(nn.Module):
            def __init__(self, head: nn.Module):
                super().__init__()
                self.head = head

            def forward(self, context: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
                return self.head.greedy(context, source=source)

        device = next(self.parameters()).device
        sample_context = torch.zeros(
            int(batch_size),
            self.d_model,
            device=device,
        )
        sample_source = torch.zeros(
            int(batch_size),
            self.patch_generation_copy_window,
            dtype=torch.long,
            device=device,
        )
        traced = torch.jit.freeze(
            torch.jit.trace(
                GreedyTraceWrapper(self.patch_generator).eval(),
                (sample_context, sample_source),
                check_trace=False,
            )
        )
        static_context = sample_context.clone()
        static_source = sample_source.clone()
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                static_output = traced(static_context, static_source)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = traced(static_context, static_source)

        def graph_greedy(
            context: torch.Tensor,
            prefix: torch.Tensor | None = None,
            forced_first: torch.Tensor | None = None,
            source: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if (
                prefix is not None
                or forced_first is not None
                or source is None
                or context.shape != static_context.shape
            ):
                return original_greedy(
                    context,
                    prefix,
                    forced_first=forced_first,
                    source=source,
                )
            normalized_source = self.patch_generator._normalized_copy_source(source)
            static_context.copy_(context)
            static_source.copy_(normalized_source)
            graph.replay()
            return static_output

        object.__setattr__(self.patch_generator, "greedy", graph_greedy)
        summary = {
            "enabled": True,
            "batch_size": int(batch_size),
            "generation_bytes": int(self.patch_generation_bytes),
            "setup_seconds": time.perf_counter() - started,
        }
        object.__setattr__(
            self,
            "_patch_generator_cuda_graph_runtime",
            {
                "graph": graph,
                "traced": traced,
                "static_context": static_context,
                "static_source": static_source,
                "static_output": static_output,
                "original_greedy": original_greedy,
                "summary": summary,
            },
        )
        return dict(summary)

    @torch.inference_mode()
    def begin_patch_prediction_cached_generation(self, x: torch.Tensor) -> dict:
        """Prefill global caches for exact autoregressive draft patch generation."""
        if (
            not self.patch_prediction
            or self.patch_prediction_context != "global"
            or self.patch_prediction_mode != "autoregressive"
            or not self.modern_blocks
        ):
            raise RuntimeError("cached patch prediction generation is unsupported")
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        if usable == 0:
            raise ValueError("input must contain at least one complete patch")
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        flat_byte_h = self.byte_emb(x)
        if self.ngram_buckets:
            previous = torch.cat(
                [torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1
            )
            previous2 = torch.cat(
                [torch.zeros_like(x[:, :2]), x[:, :-2]], dim=1
            )
            bigram_ids = (previous * 257 + x) % self.ngram_buckets
            trigram_ids = (
                previous2 * 65537 + previous * 257 + x
            ) % self.ngram_buckets
            flat_byte_h = (
                flat_byte_h
                + self.bigram_emb(bigram_ids)
                + self.trigram_emb(trigram_ids)
            )
        byte_h = flat_byte_h.reshape(
            batch, -1, self.patch_size, flat_byte_h.shape[-1]
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = patches[:, :, 0].clone()
            for index in range(1, self.patch_size):
                patch_ids = (
                    patch_ids * 257 + patches[:, :, index]
                ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        global_caches = []
        global_hidden = patch_h
        for block in self.core:
            global_hidden, cache = block.prefill_with_cache(global_hidden)
            if isinstance(cache, tuple):
                cache_key, cache_value = cache
                cache = {
                    "key": cache_key,
                    "value": cache_value,
                    "length": cache_key.shape[2],
                }
            global_caches.append(cache)
        recent_keep = max(
            64,
            int(self.patch_generation_context),
            int(self.patch_generation_copy_window),
            int(self.domain_cache_order),
        )
        return {
            "recent_bytes": x[:, -recent_keep:],
            "bytes": x[:, -recent_keep:],
            "recent_keep": recent_keep,
            "last_global": global_hidden[:, -1],
            "global_caches": global_caches,
            "patch_count": patches.shape[1],
        }

    def _patch_features_for_generated_patch(
        self,
        patch: torch.Tensor,
        recent: torch.Tensor,
    ) -> torch.Tensor:
        patch_byte_h = self.byte_emb(patch)
        if self.ngram_buckets:
            context = torch.cat([recent[:, -2:], patch], dim=1)
            patch_start = context.shape[1] - patch.shape[1]
            previous = context[:, patch_start - 1 : -1]
            previous2 = context[:, patch_start - 2 : -2]
            bigram_ids = (previous * 257 + patch) % self.ngram_buckets
            trigram_ids = (
                previous2 * 65537 + previous * 257 + patch
            ) % self.ngram_buckets
            patch_byte_h = (
                patch_byte_h
                + self.bigram_emb(bigram_ids)
                + self.trigram_emb(trigram_ids)
            )
        patch_features = patch_byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = patch[:, :, 0].clone() if patch.dim() == 3 else patch[:, 0].clone()
            for index in range(1, self.patch_size):
                patch_ids = (
                    patch_ids * 257 + patch[:, index]
                ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        return patch_features

    def patch_prediction_targets(self, x: torch.Tensor) -> torch.Tensor:
        """Return future byte spans aligned to every completed source patch."""
        usable = x.shape[1] // self.patch_size * self.patch_size
        patch_count = usable // self.patch_size
        if patch_count == 0:
            return x.new_zeros(x.shape[0], 0, self.patch_generation_bytes)
        starts = (
            torch.arange(patch_count, device=x.device, dtype=torch.long) + 1
        ) * self.patch_size
        offsets = torch.arange(
            self.patch_generation_bytes,
            device=x.device,
            dtype=torch.long,
        )
        padded = F.pad(x, (0, self.patch_generation_bytes), value=0)
        return padded[:, starts[:, None] + offsets[None, :]]

    def _advance_patch_prediction_cache(
        self,
        state: dict,
        patch: torch.Tensor,
    ) -> None:
        """Advance the global cache by one model-width patch."""
        recent = state["recent_bytes"]
        patch_features = self._patch_features_for_generated_patch(patch, recent)
        position = state["patch_count"]
        position_index = min(position, self.patch_pos.num_embeddings - 1)
        next_global = (
            self.patch_proj(patch_features) + self.patch_pos.weight[position_index]
        ).unsqueeze(1)
        for index, block in enumerate(self.core):
            next_global, cache = block.decode_with_cache(
                next_global, state["global_caches"][index]
            )
            state["global_caches"][index] = cache
        state["last_global"] = next_global[:, 0]
        new_recent = torch.cat([recent, patch], dim=1)
        recent_keep = int(state.get("recent_keep", 64))
        state["recent_bytes"] = (
            new_recent[:, -recent_keep:]
            if new_recent.shape[1] > recent_keep
            else new_recent
        )
        state["bytes"] = state["recent_bytes"]
        state["patch_count"] = position + 1
        state["position_overflow"] = bool(
            state.get("position_overflow", False)
            or position >= self.patch_pos.num_embeddings
        )

    @torch.inference_mode()
    def cached_patch_prediction_step(
        self,
        state: dict,
    ) -> torch.Tensor:
        """Emit one draft patch and advance exact global KV state."""
        prefix = (
            state["recent_bytes"][:, -self.patch_generation_context :]
            if self.patch_generation_context
            else None
        )
        patch = self.patch_generator.greedy(
            state["last_global"],
            prefix,
            source=(
                state["recent_bytes"][:, -self.patch_generation_copy_window :]
                if self.patch_generation_copy_window
                else None
            ),
        )
        for offset in range(0, patch.shape[1], self.patch_size):
            self._advance_patch_prediction_cache(
                state,
                patch[:, offset : offset + self.patch_size],
            )
        return patch

    @torch.inference_mode()
    def cached_patch_prediction_steps(self, state: dict, steps: int) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        patches = [self.cached_patch_prediction_step(state) for _ in range(int(steps))]
        return torch.cat(patches, dim=1)

    @torch.inference_mode()
    def begin_abi_patch_cell_cached_generation(self, x: torch.Tensor) -> dict:
        """Prefill global caches for fast ABI patch-cell generation."""
        if (
            self.local_decoder != "abi_patch_cell"
            or self.patch_size != 2
            or not self.direct_global_context
            or not self.modern_blocks
        ):
            raise RuntimeError("cached ABI patch-cell generation is unsupported")
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        if usable == 0:
            raise ValueError("input must contain at least one complete patch")
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        flat_byte_h = self.byte_emb(x)
        if self.ngram_buckets:
            previous = torch.cat(
                [torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1
            )
            previous2 = torch.cat(
                [torch.zeros_like(x[:, :2]), x[:, :-2]], dim=1
            )
            bigram_ids = (previous * 257 + x) % self.ngram_buckets
            trigram_ids = (
                previous2 * 65537 + previous * 257 + x
            ) % self.ngram_buckets
            flat_byte_h = (
                flat_byte_h
                + self.bigram_emb(bigram_ids)
                + self.trigram_emb(trigram_ids)
            )
        byte_h = flat_byte_h.reshape(
            batch, -1, self.patch_size, flat_byte_h.shape[-1]
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = patches[:, :, 0].clone()
            for index in range(1, self.patch_size):
                patch_ids = (
                    patch_ids * 257 + patches[:, :, index]
                ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        global_caches = []
        global_hidden = patch_h
        for block in self.core:
            global_hidden, cache = block.prefill_with_cache(global_hidden)
            if isinstance(cache, tuple):
                cache_key, cache_value = cache
                cache = {
                    "key": cache_key,
                    "value": cache_value,
                    "length": cache_key.shape[2],
                }
            global_caches.append(cache)
        return {
            "recent_bytes": x[:, -64:],
            "recent_byte_history": x[:, -64:].detach().cpu().tolist(),
            "bytes": x[:, -64:],
            "last_global": global_hidden[:, -1],
            "context_hidden": self.local_in(global_hidden[:, -1]),
            "global_caches": global_caches,
            "patch_count": patches.shape[1],
            "generated_patch_count": 0,
        }

    def _select_no_repeat_byte(
        self,
        logits: torch.Tensor,
        recent: torch.Tensor,
        no_repeat_ngram: int,
    ) -> torch.Tensor:
        if no_repeat_ngram <= 1 or recent.shape[1] < no_repeat_ngram - 1:
            return logits.argmax(dim=-1)
        selected = []
        for batch_index in range(logits.shape[0]):
            history = recent[batch_index].detach().cpu().tolist()
            prefix = tuple(history[-(no_repeat_ngram - 1) :])
            banned = {
                history[index + no_repeat_ngram - 1]
                for index in range(len(history) - no_repeat_ngram + 1)
                if tuple(history[index : index + no_repeat_ngram - 1]) == prefix
            }
            top = logits[batch_index].argmax()
            if int(top.detach().cpu()) not in banned:
                selected.append(top)
                continue
            scores = logits[batch_index].clone()
            scores[list(banned)] = float("-inf")
            selected.append(scores.argmax())
        return torch.stack(selected).to(logits.device)

    def _select_no_repeat_byte_with_history(
        self,
        logits: torch.Tensor,
        histories: list[list[int]],
        no_repeat_ngram: int,
    ) -> torch.Tensor:
        if no_repeat_ngram <= 1:
            return logits.argmax(dim=-1)
        selected = []
        for batch_index, history in enumerate(histories):
            if len(history) < no_repeat_ngram - 1:
                byte = logits[batch_index].argmax()
                selected.append(byte)
                continue
            prefix = tuple(history[-(no_repeat_ngram - 1) :])
            banned = {
                history[index + no_repeat_ngram - 1]
                for index in range(len(history) - no_repeat_ngram + 1)
                if tuple(history[index : index + no_repeat_ngram - 1]) == prefix
            }
            top = logits[batch_index].argmax()
            if int(top.detach().cpu()) not in banned:
                selected.append(top)
                continue
            scores = logits[batch_index].clone()
            scores[list(banned)] = float("-inf")
            selected.append(scores.argmax())
        return torch.stack(selected).to(logits.device)

    @torch.inference_mode()
    def cached_abi_patch_cell_step(
        self,
        state: dict,
        no_repeat_ngram: int = 0,
        return_logits: bool = False,
    ):
        """Emit one ABI patch-cell patch using cached global state."""
        context = state["last_global"]
        recent = state["recent_bytes"]
        recent_histories = state.get("recent_byte_history")
        first_domain_prior = self._last_domain_cache_prior(recent)
        first_cache_active = (
            self.domain_cache_override
            and bool(self._domain_cache_active(first_domain_prior).all().detach().cpu())
        )
        context_hidden = state.get("context_hidden")
        if first_cache_active:
            first_logits = first_domain_prior
        else:
            if context_hidden is None:
                context_hidden = self.local_in(context)
                state["context_hidden"] = context_hidden
            last_byte_h = self.byte_emb(recent[:, -1])
            next_gated = torch.sigmoid(
                self.abi_cell_next_gate(torch.cat([context, last_byte_h], dim=-1))
            )
            first_hidden = (
                context_hidden
                + self.local_offsets.weight[0]
                + next_gated * self.abi_cell_byte1(last_byte_h)
            )
            if not self.abi_patch_cell_fast_local_runtime:
                first_hidden = first_hidden + self.abi_cell_refine(first_hidden)
            first_logits = self._byte_logits(self.local_norm(first_hidden))
            last_byte = recent[:, -1]
            first_logits = first_logits + self._transition_prior(last_byte)
            if self.context_buckets:
                first_context_id = self._last_context_id(recent)
                first_logits = first_logits + self._context_prior(first_context_id)
            if self.domain_cache_override:
                active = self._domain_cache_active(first_domain_prior)
                first_logits = torch.where(
                    active,
                    first_domain_prior.to(first_logits.dtype),
                    first_logits,
                )
            else:
                first_logits = first_logits + first_domain_prior
        repeat_bias = self._generation_repeat_suppression_bias(recent)
        if repeat_bias is not None:
            first_logits = first_logits + repeat_bias[:, -1]
        first_logits = self._apply_generation_word_shape_constraints(
            first_logits, recent
        )
        if recent_histories is not None:
            first = self._select_no_repeat_byte_with_history(
                first_logits,
                recent_histories,
                no_repeat_ngram,
            )
        else:
            first = self._select_no_repeat_byte(first_logits, recent, no_repeat_ngram)
        recent_with_first = torch.cat([recent, first[:, None]], dim=1)
        second_domain_prior = self._last_domain_cache_prior(recent_with_first)
        second_cache_active = (
            self.domain_cache_override
            and bool(self._domain_cache_active(second_domain_prior).all().detach().cpu())
        )
        if second_cache_active:
            second_logits = second_domain_prior
        else:
            if context_hidden is None:
                context_hidden = self.local_in(context)
                state["context_hidden"] = context_hidden
            first_byte_h = self.byte_emb(first)
            gated = torch.sigmoid(
                self.abi_cell_gate(torch.cat([context, first_byte_h], dim=-1))
            )
            second_hidden = (
                context_hidden
                + self.local_offsets.weight[1]
                + gated * self.abi_cell_byte0(first_byte_h)
            )
            if not self.abi_patch_cell_fast_local_runtime:
                second_hidden = second_hidden + self.abi_cell_refine(second_hidden)
            second_logits = self._byte_logits(self.local_norm(second_hidden))
            second_logits = second_logits + self._transition_prior(first)
            if self.context_buckets:
                second_context_id = self._last_context_id(recent_with_first)
                second_logits = second_logits + self._context_prior(second_context_id)
            if self.domain_cache_override:
                active = self._domain_cache_active(second_domain_prior)
                second_logits = torch.where(
                    active,
                    second_domain_prior.to(second_logits.dtype),
                    second_logits,
                )
            else:
                second_logits = second_logits + second_domain_prior
        repeat_bias = self._generation_repeat_suppression_bias(recent_with_first)
        if repeat_bias is not None:
            second_logits = second_logits + repeat_bias[:, -1]
        second_logits = self._apply_generation_word_shape_constraints(
            second_logits, recent_with_first
        )
        if recent_histories is not None:
            first_list = first.detach().cpu().tolist()
            for history, byte in zip(recent_histories, first_list):
                history.append(int(byte))
                if len(history) > 64:
                    del history[:-64]
            second = self._select_no_repeat_byte_with_history(
                second_logits,
                recent_histories,
                no_repeat_ngram,
            )
        else:
            second = self._select_no_repeat_byte(
                second_logits, recent_with_first, no_repeat_ngram
            )
        patch = torch.stack([first, second], dim=-1)

        position = state["patch_count"]
        generated_patch_count = int(state.get("generated_patch_count", 0)) + 1
        update_interval = self.abi_patch_cell_global_update_interval
        should_update_global = (
            not self.abi_patch_cell_static_generation
            and update_interval > 0
            and generated_patch_count % update_interval == 0
        )
        if should_update_global or self.abi_patch_cell_lightweight_context_update:
            patch_byte_h = self.byte_emb(patch)
            if self.ngram_buckets:
                previous0 = recent[:, -1]
                previous1 = patch[:, 0]
                if recent.shape[1] >= 2:
                    previous20 = recent[:, -2]
                else:
                    previous20 = torch.zeros_like(previous0)
                previous21 = recent[:, -1]
                bigram_ids = torch.stack(
                    [
                        (previous0 * 257 + patch[:, 0]) % self.ngram_buckets,
                        (previous1 * 257 + patch[:, 1]) % self.ngram_buckets,
                    ],
                    dim=1,
                )
                trigram_ids = torch.stack(
                    [
                        (previous20 * 65537 + previous0 * 257 + patch[:, 0])
                        % self.ngram_buckets,
                        (previous21 * 65537 + previous1 * 257 + patch[:, 1])
                        % self.ngram_buckets,
                    ],
                    dim=1,
                )
                patch_byte_h = (
                    patch_byte_h
                    + self.bigram_emb(bigram_ids)
                    + self.trigram_emb(trigram_ids)
                )
            patch_features = patch_byte_h.flatten(-2)
            if self.patch_unit_buckets:
                patch_ids = (patch[:, 0] * 257 + patch[:, 1]) % self.patch_unit_buckets
                patch_features = torch.cat(
                    [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
                )
            projected_patch = self.patch_proj(patch_features)
            position_index = min(position, self.patch_pos.num_embeddings - 1)
            next_global = (
                projected_patch + self.patch_pos.weight[position_index]
            ).unsqueeze(1)
        if should_update_global:
            for index, block in enumerate(self.core):
                if (
                    self.abi_patch_cell_fast_global_decode
                    and hasattr(block, "decode_state_only_with_cache")
                ):
                    next_global, cache = block.decode_state_only_with_cache(
                        next_global, state["global_caches"][index]
                    )
                else:
                    next_global, cache = block.decode_with_cache(
                        next_global, state["global_caches"][index]
                    )
                state["global_caches"][index] = cache
            state["last_global"] = next_global[:, 0]
            state["context_hidden"] = self.local_in(state["last_global"])
        elif self.abi_patch_cell_lightweight_context_update:
            blend = self.abi_patch_cell_lightweight_context_blend
            state["last_global"] = (
                (1.0 - blend) * state["last_global"]
                + blend * projected_patch
            )
            state["context_hidden"] = self.local_in(state["last_global"])
        new_recent = torch.cat([recent, patch], dim=1)
        state["recent_bytes"] = new_recent[:, -64:] if new_recent.shape[1] > 64 else new_recent
        if recent_histories is not None:
            second_list = second.detach().cpu().tolist()
            for history, byte in zip(recent_histories, second_list):
                history.append(int(byte))
                if len(history) > 64:
                    del history[:-64]
        state["bytes"] = state["recent_bytes"]
        state["patch_count"] = position + 1
        state["generated_patch_count"] = generated_patch_count
        if return_logits:
            return patch, torch.stack([first_logits, second_logits], dim=1)
        return patch

    @torch.inference_mode()
    def cached_abi_patch_cell_steps(
        self,
        state: dict,
        steps: int,
        no_repeat_ngram: int = 0,
    ) -> torch.Tensor:
        """Emit multiple ABI patch-cell patches from one cached-generation call.

        This preserves the exact single-step semantics and ABI state while giving
        runtime benchmarks a lower-overhead entry point.  It is intentionally a
        semantic wrapper first; deeper fusion can replace the body later without
        changing callers or transfer behavior.
        """
        if steps <= 0:
            raise ValueError("steps must be positive")
        patches = [
            self.cached_abi_patch_cell_step(
                state,
                no_repeat_ngram=no_repeat_ngram,
            )
            for _ in range(int(steps))
        ]
        return torch.cat(patches, dim=1)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in ("domain_cache_keys", "domain_cache_logits"):
            key = prefix + name
            if key in state_dict:
                current = self._buffers.get(name)
                incoming = state_dict[key]
                if current is not None and current.shape != incoming.shape:
                    self._buffers[name] = torch.empty(
                        incoming.shape,
                        dtype=incoming.dtype,
                        device=current.device,
                    )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.no_grad()
    def generate_verified_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Draft two bytes globally, then verify with the high-quality local LM."""
        if self.patch_size != 2:
            raise RuntimeError("verified generation currently requires patch_size=2")
        if self.local_decoder != "window_transformer":
            raise RuntimeError(
                "verified generation currently requires window_transformer"
            )
        if not self.direct_global_context:
            raise RuntimeError(
                "verified generation currently requires direct global context"
            )
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        flat_byte_h = self.byte_emb(x)
        byte_h = flat_byte_h.reshape(
            batch, -1, self.patch_size, flat_byte_h.shape[-1]
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = (
                patches[:, :, 0] * 257 + patches[:, :, 1]
            ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        mask = causal_mask(patch_h.shape[1], patch_h.device)
        if self.modern_blocks:
            global_h = run_modern_stack(self.core, patch_h, mask)
        else:
            global_h = self.core(patch_h, mask=mask)
        prefix = (
            x[:, -self.patch_generation_context :]
            if self.patch_generation_context
            else None
        )
        if self.patch_prediction_mode == "autoregressive":
            draft = self.patch_generator.greedy(
                global_h[:, -1],
                prefix,
                source=(
                    x[:, -self.patch_generation_copy_window :]
                    if self.patch_generation_copy_window
                    else None
                ),
            )
        else:
            draft = torch.stack(
                [
                    head(global_h[:, -1]).argmax(dim=-1)
                    for head in self.patch_prediction_heads
                ],
                dim=-1,
            )

        verification_window = min(
            self.local_window, self.patch_size * 2
        )
        source_count = verification_window - self.patch_size
        source_bytes = x[:, -source_count:]
        source_patch_context = torch.cat(
            [
                global_h.new_zeros(batch, 1, global_h.shape[-1]),
                global_h[:, :-1],
            ],
            dim=1,
        ).unsqueeze(2).expand(-1, -1, self.patch_size, -1)
        source_context = source_patch_context.reshape(
            batch, usable, -1
        )[:, -source_count:]
        target_context = global_h[:, -1:].expand(
            -1, self.patch_size, -1
        )
        window_context = torch.cat(
            [source_context, target_context], dim=1
        )

        def verify(candidate: torch.Tensor) -> torch.Tensor:
            window_bytes = torch.cat([source_bytes, candidate], dim=1)
            local_input = torch.cat(
                [self.byte_emb(window_bytes), window_context], dim=-1
            )
            local_out = self.local_in(local_input)
            local_mask = causal_mask(verification_window, x.device)
            if self.modern_blocks:
                local_out = run_modern_stack(
                    self.local_core, local_out, local_mask
                )
            else:
                local_out = self.local_core(local_out, mask=local_mask)
            logits = self._byte_logits(self.local_norm(local_out))
            logits = logits + self._transition_prior(window_bytes)
            if self.context_buckets:
                logits = logits + self._context_prior(
                    self._context_ids(window_bytes)
                )
            return logits

        verified_logits = verify(draft)
        first = verified_logits[:, source_count - 1].argmax(dim=-1)
        second = verified_logits[:, source_count].argmax(dim=-1)
        rejected = first != draft[:, 0]
        if rejected.any():
            corrected = draft.clone()
            corrected[:, 0] = first
            corrected_logits = verify(corrected)
            corrected_second = corrected_logits[
                :, source_count
            ].argmax(dim=-1)
            second = torch.where(rejected, corrected_second, second)
        return torch.stack([first, second], dim=-1)

    @torch.no_grad()
    def generate_cached_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Generate two exact local-LM bytes using one prefill and one cache step."""
        if (
            self.patch_size != 2
            or self.local_decoder != "window_transformer"
            or not self.direct_global_context
            or not self.modern_blocks
            or not self.fused_attention
        ):
            raise RuntimeError(
                "cached generation requires fused modern patch_size=2 "
                "window_transformer with direct global context"
            )
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        flat_byte_h = self.byte_emb(x)
        byte_h = flat_byte_h.reshape(
            batch, -1, self.patch_size, flat_byte_h.shape[-1]
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = (
                patches[:, :, 0] * 257 + patches[:, :, 1]
            ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        mask = causal_mask(patch_h.shape[1], patch_h.device)
        global_h = run_modern_stack(self.core, patch_h, mask)

        source_bytes = x[:, -self.local_window :]
        source_patch_count = self.local_window // self.patch_size
        source_globals = global_h[:, -source_patch_count:]
        prior_global = torch.cat(
            [
                global_h.new_zeros(batch, 1, global_h.shape[-1]),
                global_h[:, :-1],
            ],
            dim=1,
        )[:, -source_patch_count:]
        source_context = prior_global.unsqueeze(2).expand(
            -1, -1, self.patch_size, -1
        ).reshape(batch, self.local_window, -1)
        local_input = torch.cat(
            [self.byte_emb(source_bytes), source_context], dim=-1
        )
        local_hidden = self.local_in(local_input)
        caches = []
        for block in self.local_core:
            local_hidden, cache = block.prefill_with_cache(local_hidden)
            caches.append(cache)
        normalized = self.local_norm(local_hidden)
        first_logits = self._byte_logits(normalized[:, -1])
        first_logits = first_logits + self._transition_prior(
            source_bytes[:, -1]
        )
        first = first_logits.argmax(dim=-1)

        target_context = source_globals[:, -1]
        next_input = self.local_in(
            torch.cat(
                [self.byte_emb(first), target_context], dim=-1
            )
        ).unsqueeze(1)
        for index, block in enumerate(self.local_core):
            next_input, _ = block.decode_with_cache(
                next_input, caches[index]
            )
        second_logits = self._byte_logits(
            self.local_norm(next_input)[:, 0]
        )
        second_logits = second_logits + self._transition_prior(first)
        second = second_logits.argmax(dim=-1)
        return torch.stack([first, second], dim=-1)

    @torch.inference_mode()
    def begin_cached_generation(
        self,
        x: torch.Tensor,
        profile: bool = False,
        keep_full_history: bool = False,
        fast_prefill_if_aligned: bool = False,
    ) -> dict:
        """Prefill global patch caches for stateful fixed-size patch generation."""
        profile_times: dict[str, float] = {}
        profile_start = time.perf_counter() if profile else 0.0

        def mark(name: str) -> None:
            nonlocal profile_start
            if not profile:
                return
            now = time.perf_counter()
            profile_times[name] = now - profile_start
            profile_start = now

        if (
            self.local_decoder != "window_transformer"
            or not self.direct_global_context
            or not self.modern_blocks
            or not self.fused_attention
        ):
            raise RuntimeError("unsupported model for cached generation")
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        if patches.shape[1] >= self.patch_pos.num_embeddings:
            raise ValueError(
                "prompt must leave at least one patch position for generation"
            )
        mark("validate_prompt")
        byte_h = self.byte_emb(x).reshape(
            batch, -1, self.patch_size, self.byte_emb.embedding_dim
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = (
                patches[:, :, 0] * 257 + patches[:, :, 1]
            ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        mark("byte_and_patch_features")
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        mark("patch_projection")
        global_caches = []
        global_hidden = patch_h
        for block in self.core:
            global_hidden, cache = block.prefill_with_cache(global_hidden)
            if isinstance(cache, tuple):
                cache_key, cache_value = cache
                cache = {
                    "key": cache_key,
                    "value": cache_value,
                    "length": cache_key.shape[2],
                }
            global_caches.append(cache)
        mark("global_core_cache_prefill")
        source_bytes = x[:, -self.local_window :]
        source_patch_count = self.local_window // self.patch_size
        all_prior = torch.nn.functional.pad(
            global_hidden[:, :-1], (0, 0, 1, 0), mode="constant", value=0.0
        )
        byte_width = self.byte_emb.embedding_dim
        local_position = usable % self.local_window
        fast_prefill_active = bool(fast_prefill_if_aligned and local_position == 0)
        if fast_prefill_active:
            source_byte_h = self.byte_emb(x[:, -1:]).reshape(
                batch,
                1,
                self.byte_emb.embedding_dim,
            )
            patch_index = max(patches.shape[1] - 1, 0)
            byte_offset = (usable - 1) % self.patch_size
            prior_patch_index = patch_index - 1 if byte_offset == 0 else patch_index
            if prior_patch_index < 0:
                source_context = global_hidden.new_zeros(batch, 1, global_hidden.shape[-1])
            else:
                source_context = global_hidden[:, prior_patch_index : prior_patch_index + 1]
        else:
            source_byte_h = byte_h[:, -source_patch_count:].reshape(
                batch, self.local_window, self.byte_emb.embedding_dim
            )
            prior_global = all_prior[:, -source_patch_count:]
            source_context = prior_global[:, :, None, :].expand(
                batch, source_patch_count, self.patch_size, -1
            ).reshape(batch, self.local_window, -1)
        local_hidden = F.linear(
            source_byte_h,
            self.local_in.weight[:, :byte_width],
            self.local_in.bias,
        ) + F.linear(
            source_context,
            self.local_in.weight[:, byte_width:],
        )
        mark("local_inputs")
        local_caches = []
        for block in self.local_core:
            local_hidden, cache = block.prefill_with_cache(local_hidden)
            if isinstance(cache, tuple):
                cache_key, cache_value = cache
                cache = {
                    "key": cache_key,
                    "value": cache_value,
                    "length": cache_key.shape[2],
                }
            local_caches.append(cache)
        mark("local_cache_prefill")
        next_hidden = self.local_norm(local_hidden)[:, -1]
        next_logits = self._byte_logits(next_hidden)
        next_logits = next_logits + self._transition_prior(
            source_bytes[:, -1]
        )
        if self.context_buckets:
            next_logits = next_logits + self._context_prior(
                self._last_context_id(x)
            )
        if self.copy_attention:
            next_logits = next_logits + self._copy_attention_next_prior(
                x,
                next_hidden,
            )
        if self.domain_cache_order > 0 and self.domain_cache_logit_scale != 0.0:
            domain_prior = self._last_domain_cache_prior(x)
            if self.domain_cache_override:
                active = self._domain_cache_active(domain_prior)
                next_logits = torch.where(active, domain_prior, next_logits)
            else:
                next_logits = next_logits + domain_prior
        mark("next_logits")
        recent_keep = max(64, int(self.domain_cache_order))
        state = {
            "bytes_history": x if keep_full_history else None,
            "recent_bytes": x[:, -recent_keep:],
            "bytes": x[:, -recent_keep:],
            "recent_keep": recent_keep,
            "last_global": global_hidden[:, -1],
            "global_caches": global_caches,
            "local_caches": local_caches,
            "local_position": local_position,
            "next_logits": next_logits,
            "patch_count": patches.shape[1],
            "keep_full_history": keep_full_history,
            "fast_prefill_active": fast_prefill_active,
        }
        if profile:
            state["profile_seconds"] = profile_times
        return state

    def _local_decode_blocks(
        self,
        token_hidden: torch.Tensor,
        local_caches: list,
        reset: bool,
    ) -> tuple[torch.Tensor, list]:
        """Fused local block decode with minimal overhead."""
        next_caches = []
        for index, block in enumerate(self.local_core):
            if reset:
                token_hidden, cache = block.prefill_with_cache(token_hidden)
            else:
                token_hidden, cache = block.decode_with_cache(
                    token_hidden, local_caches[index]
                )
            next_caches.append(cache)
        return token_hidden, next_caches


    @torch.inference_mode()
    def cached_generation_step(
        self,
        state: dict,
        forced_patch: torch.Tensor | None = None,
        return_logits: bool = False,
        no_repeat_ngram: int = 0,
    ):
        """Emit one exact local-LM patch and append it to global KV state."""
        full_history = state.get("bytes_history")
        recent_history = state.get("recent_bytes")
        next_logits = state["next_logits"]
        use_repeat_blocking = no_repeat_ngram > 1
        generation_recent = recent_history

        def select_byte(logits: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
            if no_repeat_ngram <= 1 or prefix.shape[1] < no_repeat_ngram - 1:
                return logits.argmax(dim=-1)
            if prefix.shape[1] < no_repeat_ngram:
                return logits.argmax(dim=-1)
            suffix = prefix[:, -(no_repeat_ngram - 1) :]
            windows = prefix.unfold(1, no_repeat_ngram, 1)
            matching_prefix = (
                windows[:, :, : no_repeat_ngram - 1] == suffix[:, None, :]
            ).all(dim=-1)
            banned_tokens = windows[:, :, -1]
            banned_counts = torch.zeros(
                logits.shape,
                device=logits.device,
                dtype=torch.int16,
            )
            banned_counts.scatter_reduce_(
                1,
                banned_tokens,
                matching_prefix.to(dtype=torch.int16),
                reduce="amax",
                include_self=False,
            )
            banned = banned_counts.to(dtype=torch.bool)
            masked = logits.masked_fill(banned, float("-inf"))
            all_banned = torch.isneginf(masked).all(dim=-1)
            selected = masked.argmax(dim=-1)
            greedy = logits.argmax(dim=-1)
            return torch.where(all_banned, greedy, selected)

        prefix = (
            full_history if full_history is not None else recent_history
        ) if use_repeat_blocking else None
        target_context = state["last_global"]
        byte_width = self.byte_emb.embedding_dim
        target_context_hidden = F.linear(
            target_context,
            self.local_in.weight[:, byte_width:],
            self.local_in.bias,
        )
        def process_local_byte(byte: torch.Tensor, recent: torch.Tensor) -> torch.Tensor:
            byte_emb_out = self.byte_emb(byte)
            byte_proj = F.linear(byte_emb_out, self.local_in.weight[:, :byte_width])
            token_hidden = (byte_proj + target_context_hidden).unsqueeze(1)
            reset = state["local_position"] == 0
            next_caches = []
            for index, block in enumerate(self.local_core):
                if reset:
                    token_hidden, cache = block.prefill_with_cache(token_hidden)
                else:
                    token_hidden, cache = block.decode_with_cache(
                        token_hidden, state["local_caches"][index]
                    )
                next_caches.append(cache)
            state["local_caches"] = next_caches
            state["local_position"] = (state["local_position"] + 1) % self.local_window
            next_hidden = self.local_norm(token_hidden)[:, 0]
            logits = self._byte_logits(next_hidden)
            logits = logits + self._transition_prior(byte)
            if self.context_buckets:
                logits = logits + self._context_prior(
                    self._last_context_id(recent)
                )
            if self.copy_attention:
                logits = logits + self._copy_attention_next_prior(
                    recent,
                    next_hidden,
                )
            return logits

        emitted = []
        emitted_logits = []
        for offset in range(self.patch_size):
            if (
                generation_recent is not None
                and self.domain_cache_order > 0
                and self.domain_cache_logit_scale != 0.0
            ):
                domain_prior = self._last_domain_cache_prior(generation_recent)
                if self.domain_cache_override:
                    active = self._domain_cache_active(domain_prior)
                    next_logits = torch.where(active, domain_prior, next_logits)
                else:
                    next_logits = next_logits + domain_prior
            emitted_logits.append(next_logits)
            next_byte = (
                forced_patch[:, offset]
                if forced_patch is not None
                else select_byte(next_logits, prefix)
            )
            emitted.append(next_byte)
            if generation_recent is None:
                generation_recent = next_byte[:, None]
            else:
                generation_recent = torch.cat([generation_recent, next_byte[:, None]], dim=1)
                recent_keep = int(state.get("recent_keep", 64))
                if generation_recent.shape[1] > recent_keep:
                    generation_recent = generation_recent[:, -recent_keep:]
            if use_repeat_blocking:
                prefix = (
                    torch.cat([prefix, next_byte[:, None]], dim=1)
                    if prefix is not None
                    else next_byte[:, None]
                )
            next_logits = process_local_byte(next_byte, generation_recent)
        patch = torch.stack(emitted, dim=-1)
        state["next_logits"] = next_logits

        patch_features = self.byte_emb(patch).flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = (
                patch[:, 0] * 257 + patch[:, 1]
            ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        position = state["patch_count"]
        next_global = self.patch_proj(patch_features)
        position_index = min(position, self.patch_pos.num_embeddings - 1)
        next_global = (
            next_global + self.patch_pos.weight[position_index]
        ).unsqueeze(1)
        for index, block in enumerate(self.core):
            next_global, cache = block.decode_with_cache(
                next_global, state["global_caches"][index]
            )
            state["global_caches"][index] = cache
        state["last_global"] = next_global[:, 0]
        if full_history is not None:
            state["bytes_history"] = torch.cat([full_history, patch], dim=1)
            recent_keep = int(state.get("recent_keep", 64))
            state["recent_bytes"] = state["bytes_history"][:, -recent_keep:]
        elif recent_history is None:
            state["recent_bytes"] = patch
        else:
            new_recent = torch.cat([recent_history, patch], dim=1)
            recent_keep = int(state.get("recent_keep", 64))
            state["recent_bytes"] = (
                new_recent[:, -recent_keep:]
                if new_recent.shape[1] > recent_keep
                else new_recent
            )
        state["bytes"] = state["recent_bytes"]
        state["patch_count"] = position + 1
        state["position_overflow"] = position >= self.patch_pos.num_embeddings
        if return_logits:
            return patch, torch.stack(emitted_logits, dim=1)
        return patch

    @torch.no_grad()
    def generate_next_span(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Generate a fixed-width neural byte span from the global patch state."""
        if self.local_decoder != "span_patch_decoder":
            raise RuntimeError("span generation requires span_patch_decoder")
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        if usable == 0:
            raise ValueError("input must contain at least one complete patch")
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        byte_h = self.byte_emb(x).reshape(
            batch, -1, self.patch_size, self.byte_emb.embedding_dim
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = patches[:, :, 0].clone()
            for index in range(1, self.patch_size):
                patch_ids = (
                    patch_ids * 257 + patches[:, :, index]
                ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        mask = causal_mask(patch_h.shape[1], patch_h.device)
        if self.modern_blocks:
            global_h = run_modern_stack(self.core, patch_h, mask)
        else:
            global_h = self.core(patch_h, mask=mask)
        base_hidden = self.local_in(global_h[:, -1])
        recent = x
        emitted = []
        emitted_logits = []
        prefix_sum = base_hidden.new_zeros(base_hidden.shape)
        for offset in range(self.span_width):
            next_hidden = self.local_norm(
                (
                    base_hidden
                    + self.local_offsets.weight[offset]
                    + prefix_sum
                )
                + self.span_refine(
                    base_hidden
                    + self.local_offsets.weight[offset]
                    + prefix_sum
                )
            )
            logits = self._byte_logits(next_hidden)
            logits = logits + self._transition_prior(recent[:, -1])
            if self.context_buckets:
                logits = logits + self._context_prior(
                    self._last_context_id(recent)
                )
            if self.copy_attention:
                logits = logits + self._copy_attention_next_prior(
                    recent,
                    next_hidden,
                )
            if self.copy_transducer:
                copy_logits, _ = self._copy_transducer_next_logits(
                    recent,
                    next_hidden,
                )
                logits = logits + copy_logits
            emitted_logits.append(logits)
            byte = logits.argmax(dim=-1)
            emitted.append(byte)
            if self.span_prefix_conditioning:
                prefix_sum = prefix_sum + self.span_prefix_proj(self.byte_emb(byte))
            recent = torch.cat([recent, byte[:, None]], dim=1)
        span = torch.stack(emitted, dim=-1)
        if return_logits:
            return span, torch.stack(emitted_logits, dim=1)
        return span

    @torch.no_grad()
    def generate_next_span_parallel(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Generate a fixed-width span with one parallel decoder head call."""
        if self.local_decoder != "span_patch_decoder":
            raise RuntimeError("span generation requires span_patch_decoder")
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        if usable == 0:
            raise ValueError("input must contain at least one complete patch")
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        byte_h = self.byte_emb(x).reshape(
            batch, -1, self.patch_size, self.byte_emb.embedding_dim
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = patches[:, :, 0].clone()
            for index in range(1, self.patch_size):
                patch_ids = (
                    patch_ids * 257 + patches[:, :, index]
                ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        mask = causal_mask(patch_h.shape[1], patch_h.device)
        if self.modern_blocks:
            global_h = run_modern_stack(self.core, patch_h, mask)
        else:
            global_h = self.core(patch_h, mask=mask)
        span_logits, span_hidden = self._span_logits_from_context(global_h[:, -1:])
        span_logits = span_logits[:, 0]
        span_hidden = span_hidden[:, 0]
        if self.copy_attention:
            span_logits = span_logits + self._copy_attention_prior(x, span_hidden)
        if self.copy_transducer:
            copy_logits, _ = self._copy_transducer_logits(x, span_hidden)
            span_logits = span_logits + copy_logits
        span = span_logits.argmax(dim=-1)
        if return_logits:
            return span, span_logits
        return span

    @torch.inference_mode()
    def begin_span_cached_generation(self, x: torch.Tensor) -> dict:
        """Prefill global caches for fixed-width span generation."""
        if (
            self.local_decoder != "span_patch_decoder"
            or not self.direct_global_context
            or not self.modern_blocks
        ):
            raise RuntimeError("unsupported model for span cached generation")
        if self.span_width % self.patch_size:
            raise RuntimeError("span_width must be divisible by patch_size")
        batch, length = x.shape
        usable = length // self.patch_size * self.patch_size
        if usable == 0:
            raise ValueError("input must contain at least one complete patch")
        x = x[:, :usable]
        patches = x.reshape(batch, -1, self.patch_size)
        byte_h = self.byte_emb(x).reshape(
            batch, -1, self.patch_size, self.byte_emb.embedding_dim
        )
        patch_features = byte_h.flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = patches[:, :, 0].clone()
            for index in range(1, self.patch_size):
                patch_ids = (
                    patch_ids * 257 + patches[:, :, index]
                ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        global_caches = []
        global_hidden = patch_h
        for block in self.core:
            global_hidden, cache = block.prefill_with_cache(global_hidden)
            if isinstance(cache, tuple):
                cache_key, cache_value = cache
                cache = {
                    "key": cache_key,
                    "value": cache_value,
                    "length": cache_key.shape[2],
                }
            global_caches.append(cache)
        recent_keep = max(128, int(self.copy_transducer_window), int(self.copy_attention_window))
        return {
            "recent_bytes": x[:, -recent_keep:],
            "bytes": x[:, -recent_keep:],
            "recent_keep": recent_keep,
            "last_global": global_hidden[:, -1],
            "global_caches": global_caches,
            "patch_count": patches.shape[1],
        }

    @torch.inference_mode()
    def cached_span_generation_step(
        self,
        state: dict,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Emit one neural span and advance global patch state."""
        if self.local_decoder != "span_patch_decoder":
            raise RuntimeError("span generation requires span_patch_decoder")
        base_hidden = self.local_in(state["last_global"])
        recent = state["recent_bytes"]
        emitted = []
        emitted_logits = []
        prefix_sum = base_hidden.new_zeros(base_hidden.shape)
        for offset in range(self.span_width):
            next_hidden = self.local_norm(
                (
                    base_hidden
                    + self.local_offsets.weight[offset]
                    + prefix_sum
                )
                + self.span_refine(
                    base_hidden
                    + self.local_offsets.weight[offset]
                    + prefix_sum
                )
            )
            logits = self._byte_logits(next_hidden)
            logits = logits + self._transition_prior(recent[:, -1])
            if self.context_buckets:
                logits = logits + self._context_prior(
                    self._last_context_id(recent)
                )
            if self.copy_attention:
                logits = logits + self._copy_attention_next_prior(
                    recent,
                    next_hidden,
                )
            if self.copy_transducer:
                copy_logits, _ = self._copy_transducer_next_logits(
                    recent,
                    next_hidden,
                )
                logits = logits + copy_logits
            emitted_logits.append(logits)
            byte = logits.argmax(dim=-1)
            emitted.append(byte)
            if self.span_prefix_conditioning:
                prefix_sum = prefix_sum + self.span_prefix_proj(self.byte_emb(byte))
            recent = torch.cat([recent, byte[:, None]], dim=1)
            if recent.shape[1] > int(state.get("recent_keep", 128)):
                recent = recent[:, -int(state.get("recent_keep", 128)) :]
        span = torch.stack(emitted, dim=-1)
        state["recent_bytes"] = recent
        state["bytes"] = recent
        for start in range(0, self.span_width, self.patch_size):
            patch = span[:, start : start + self.patch_size]
            patch_features = self.byte_emb(patch).flatten(-2)
            if self.patch_unit_buckets:
                patch_ids = patch[:, 0].clone()
                for index in range(1, self.patch_size):
                    patch_ids = (
                        patch_ids * 257 + patch[:, index]
                    ) % self.patch_unit_buckets
                patch_features = torch.cat(
                    [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
                )
            position = state["patch_count"]
            position_index = min(position, self.patch_pos.num_embeddings - 1)
            next_global = (
                self.patch_proj(patch_features)
                + self.patch_pos.weight[position_index]
            ).unsqueeze(1)
            for index, block in enumerate(self.core):
                next_global, cache = block.decode_with_cache(
                    next_global,
                    state["global_caches"][index],
                )
                state["global_caches"][index] = cache
            state["last_global"] = next_global[:, 0]
            state["patch_count"] = position + 1
        if return_logits:
            return span, torch.stack(emitted_logits, dim=1)
        return span

    def _context_ids(self, x: torch.Tensor) -> torch.Tensor:
        context_ids = torch.zeros_like(x)
        for lag in range(self.context_order):
            shifted = F.pad(x[:, : x.shape[1] - lag], (lag, 0))
            context_ids = (
                context_ids * 257 + shifted + 1
            ) % self.context_buckets
        return context_ids


class CausalVariableBytePatchLM(nn.Module):
    """Tokenizer-free variable patches ending at whitespace or a max length."""

    def __init__(
        self,
        max_patch_size=8,
        d_byte=48,
        d_model=384,
        d_abi=128,
        layers=8,
        heads=8,
        max_patches=256,
        continuous_local=True,
        transition_boundary_table: torch.Tensor | None = None,
        ordered_patch_encoder=True,
        reset_local_decoder=True,
    ):
        super().__init__()
        self.max_patch_size = max_patch_size
        self.continuous_local = continuous_local
        self.ordered_patch_encoder = ordered_patch_encoder
        self.reset_local_decoder = reset_local_decoder
        if transition_boundary_table is None:
            transition_boundary_table = torch.zeros(65536, dtype=torch.bool)
        if transition_boundary_table.shape != (65536,):
            raise ValueError("transition boundary table must have shape [65536]")
        self.register_buffer(
            "transition_boundary_table",
            transition_boundary_table.to(dtype=torch.bool),
        )
        self.byte_emb = nn.Embedding(256, d_byte)
        self.length_emb = nn.Embedding(max_patch_size + 1, d_model)
        self.patch_proj = nn.Linear(d_byte * 2, d_model)
        if ordered_patch_encoder:
            self.patch_encoder = nn.GRU(d_byte, d_model, batch_first=True)
        self.patch_pos = nn.Embedding(max_patches, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, batch_first=True, norm_first=True
        )
        self.core = nn.TransformerEncoder(block, layers)
        self.to_abi = nn.Sequential(nn.Linear(d_model, d_abi), nn.LayerNorm(d_abi))
        self.bos_context = nn.Parameter(torch.zeros(1, 1, d_abi))
        self.local = nn.GRU(d_byte + d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, 256)
        self.register_buffer("canonical_head", canonical_brick_head(d_abi))

    def _layout(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return byte->patch ids/offsets, patch lengths, and valid mask."""
        batch, length = x.shape
        patch_ids = torch.zeros(batch, length, dtype=torch.long, device=x.device)
        patch_offsets = torch.zeros(
            batch, length, dtype=torch.long, device=x.device
        )
        all_lengths: list[list[int]] = []
        max_count = 0
        whitespace = {9, 10, 13, 32}
        for row_index, row in enumerate(x.tolist()):
            lengths = []
            start = 0
            patch_index = 0
            for byte_index, value in enumerate(row):
                current_length = byte_index - start + 1
                patch_ids[row_index, byte_index] = patch_index
                patch_offsets[row_index, byte_index] = current_length - 1
                transition_break = False
                if byte_index > start:
                    transition_id = row[byte_index - 1] * 256 + value
                    transition_break = bool(
                        self.transition_boundary_table[transition_id].item()
                    )
                if (
                    value in whitespace
                    or transition_break
                    or current_length >= self.max_patch_size
                ):
                    lengths.append(current_length)
                    patch_index += 1
                    start = byte_index + 1
            if start < length:
                lengths.append(length - start)
            all_lengths.append(lengths)
            max_count = max(max_count, len(lengths))
        lengths_tensor = torch.zeros(
            batch, max_count, dtype=torch.long, device=x.device
        )
        valid = torch.zeros(batch, max_count, dtype=torch.bool, device=x.device)
        for row_index, lengths in enumerate(all_lengths):
            lengths_tensor[row_index, : len(lengths)] = torch.tensor(
                lengths, device=x.device
            )
            valid[row_index, : len(lengths)] = True
        return patch_ids, patch_offsets, lengths_tensor, valid

    def forward(self, x: torch.Tensor, brick=None):
        batch, length = x.shape
        byte_h = self.byte_emb(x)
        patch_ids, patch_offsets, patch_lengths, valid = self._layout(x)
        patch_count = patch_lengths.shape[1]
        packed_bytes = byte_h.new_zeros(
            batch, patch_count, self.max_patch_size, byte_h.shape[-1]
        )
        batch_indices = torch.arange(batch, device=x.device).unsqueeze(1)
        packed_bytes[batch_indices, patch_ids, patch_offsets] = byte_h
        if self.ordered_patch_encoder:
            encoded, _ = self.patch_encoder(
                packed_bytes.reshape(
                    batch * patch_count,
                    self.max_patch_size,
                    byte_h.shape[-1],
                )
            )
            flat_lengths = patch_lengths.reshape(-1).clamp_min(1)
            flat_indices = torch.arange(
                batch * patch_count, device=x.device
            )
            patch_h = encoded[
                flat_indices, flat_lengths - 1
            ].reshape(batch, patch_count, -1)
        else:
            sums = packed_bytes.sum(dim=2)
            means = sums / patch_lengths.clamp_min(1).unsqueeze(-1)
            ends = packed_bytes.gather(
                2,
                (patch_lengths.clamp_min(1) - 1)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, 1, byte_h.shape[-1]),
            ).squeeze(2)
            patch_h = self.patch_proj(torch.cat([means, ends], dim=-1))
        patch_h = patch_h + self.length_emb(patch_lengths.clamp_max(
            self.max_patch_size
        ))
        positions = torch.arange(patch_count, device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        global_h = self.core(
            patch_h,
            mask=causal_mask(patch_count, x.device),
            src_key_padding_mask=~valid,
        )
        completed_abi = self.to_abi(global_h)
        context_abi = torch.cat(
            [self.bos_context.expand(batch, 1, -1), completed_abi[:, :-1]],
            dim=1,
        )
        bos_global = global_h.new_zeros(batch, 1, global_h.shape[-1])
        patch_context = torch.cat([bos_global, global_h[:, :-1]], dim=1)
        if self.reset_local_decoder:
            packed_context = patch_context.unsqueeze(2).expand(
                -1, -1, self.max_patch_size, -1
            )
            packed_local_in = torch.cat(
                [packed_bytes, packed_context], dim=-1
            ).reshape(batch * patch_count, self.max_patch_size, -1)
            packed_local_out, _ = self.local(packed_local_in)
            packed_logits = self.head(packed_local_out).reshape(
                batch, patch_count, self.max_patch_size, 256
            )
            logits = packed_logits[
                batch_indices, patch_ids, patch_offsets
            ]
        else:
            byte_context = patch_context.gather(
                1,
                patch_ids.unsqueeze(-1).expand(
                    -1, -1, patch_context.shape[-1]
                ),
            )
            local_in = torch.cat([byte_h, byte_context], dim=-1)
            local_out, _ = self.local(local_in)
            logits = self.head(local_out)
        if brick is not None:
            delta = brick(context_abi) - context_abi
            correction = delta @ self.canonical_head
            byte_correction = correction.gather(
                1,
                patch_ids.unsqueeze(-1).expand(-1, -1, correction.shape[-1]),
            )
            logits = logits + byte_correction
        metadata = {
            "patch_ids": patch_ids,
            "patch_offsets": patch_offsets,
            "patch_lengths": patch_lengths,
            "valid_patches": valid,
        }
        return logits, context_abi, metadata


class CausalAdaptiveBytePatchLM(nn.Module):
    """Causal 2/4-byte patches with the validated fused local decoder."""

    def __init__(
        self,
        d_byte=48,
        d_model=384,
        d_abi=128,
        layers=4,
        local_layers=4,
        heads=8,
        max_patches=128,
        local_window=16,
        transition_boundary_table: torch.Tensor | None = None,
    ):
        super().__init__()
        self.patch_size = 2
        self.max_patch_size = 4
        self.local_window = local_window
        if transition_boundary_table is None:
            transition_boundary_table = torch.zeros(
                65536, dtype=torch.bool
            )
        if transition_boundary_table.shape != (65536,):
            raise ValueError("transition boundary table must have shape [65536]")
        self.register_buffer(
            "transition_boundary_table",
            transition_boundary_table.to(dtype=torch.bool),
        )
        self.byte_emb = nn.Embedding(256, d_byte)
        self.patch_proj = nn.Linear(4 * d_byte, d_model)
        self.patch_length = nn.Embedding(5, d_model)
        self.patch_pos = nn.Embedding(max_patches, d_model)
        self.core = nn.ModuleList(
            FusedModernCausalBlock(d_model, heads)
            for _ in range(layers)
        )
        self.to_abi = nn.Sequential(
            nn.Linear(d_model, d_abi), nn.LayerNorm(d_abi)
        )
        self.bos_context = nn.Parameter(torch.zeros(1, 1, d_abi))
        self.local_in = nn.Linear(d_byte + d_model, d_model)
        self.local_core = nn.ModuleList(
            FusedModernCausalBlock(d_model, heads)
            for _ in range(local_layers)
        )
        self.local_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)
        self.transition_head = nn.Embedding(256, 256)
        nn.init.zeros_(self.transition_head.weight)
        self.register_buffer("canonical_head", canonical_brick_head(d_abi))

    def _layout(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Group causal two-byte units into patches of length two or four."""
        batch, length = x.shape
        if length % 4:
            raise ValueError("adaptive patch input length must be divisible by 4")
        blocks = x.reshape(batch, -1, 4)
        transition_ids = blocks[:, :, 0] * 256 + blocks[:, :, 1]
        boundary = self.transition_boundary_table[transition_ids]
        boundary = boundary | (
            (blocks[:, :, 1] == 9)
            | (blocks[:, :, 1] == 10)
            | (blocks[:, :, 1] == 13)
            | (blocks[:, :, 1] == 32)
        )
        merge = ~boundary
        candidate_valid = torch.stack(
            [torch.ones_like(merge), boundary], dim=-1
        ).reshape(batch, -1)
        candidate_ids = candidate_valid.long().cumsum(dim=1) - 1
        max_count = int(candidate_valid.sum(dim=1).max().item())
        valid = torch.arange(
            max_count, device=x.device
        ).unsqueeze(0) < candidate_valid.sum(dim=1, keepdim=True)
        lengths = torch.zeros(
            batch, max_count, dtype=torch.long, device=x.device
        )
        candidate_lengths = torch.stack(
            [
                torch.where(
                    merge,
                    torch.full_like(blocks[:, :, 0], 4),
                    torch.full_like(blocks[:, :, 0], 2),
                ),
                torch.full_like(blocks[:, :, 0], 2),
            ],
            dim=-1,
        ).reshape(batch, -1)
        lengths.scatter_reduce_(
            1,
            candidate_ids.clamp_min(0),
            candidate_lengths * candidate_valid,
            reduce="amax",
            include_self=True,
        )
        block_first_ids = candidate_ids[:, 0::2]
        block_second_ids = candidate_ids[:, 1::2]
        byte_patch_ids = torch.stack(
            [
                block_first_ids,
                block_first_ids,
                torch.where(merge, block_first_ids, block_second_ids),
                torch.where(merge, block_first_ids, block_second_ids),
            ],
            dim=-1,
        )
        patch_ids = byte_patch_ids.reshape(batch, length)
        offsets = torch.tensor(
            [0, 1, 2, 3], device=x.device
        ).view(1, 1, 4).expand_as(byte_patch_ids).clone()
        offsets[:, :, 2:] = torch.where(
            merge.unsqueeze(-1),
            offsets[:, :, 2:],
            offsets[:, :, 2:] - 2,
        )
        return patch_ids, offsets.reshape(batch, length), lengths, valid

    def forward(self, x: torch.Tensor, brick=None):
        batch, length = x.shape
        byte_h = self.byte_emb(x)
        patch_ids, patch_offsets, patch_lengths, valid = self._layout(x)
        patch_count = patch_lengths.shape[1]
        packed = byte_h.new_zeros(
            batch, patch_count, self.max_patch_size, byte_h.shape[-1]
        )
        batch_ids = torch.arange(batch, device=x.device).unsqueeze(1)
        packed[batch_ids, patch_ids, patch_offsets] = byte_h
        patch_h = self.patch_proj(packed.flatten(-2))
        patch_h = patch_h + self.patch_length(patch_lengths)
        positions = torch.arange(patch_count, device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        global_h = run_modern_stack(
            self.core, patch_h, causal_mask(patch_count, x.device)
        )
        completed_abi = self.to_abi(global_h)
        context_abi = torch.cat(
            [
                self.bos_context.expand(batch, 1, -1),
                completed_abi[:, :-1],
            ],
            dim=1,
        )
        prior_global = torch.cat(
            [
                global_h.new_zeros(batch, 1, global_h.shape[-1]),
                global_h[:, :-1],
            ],
            dim=1,
        )
        byte_context = prior_global.gather(
            1,
            patch_ids.unsqueeze(-1).expand(-1, -1, global_h.shape[-1]),
        )
        local_hidden = self.local_in(
            torch.cat([byte_h, byte_context], dim=-1)
        )
        if length % self.local_window:
            raise ValueError(
                "sequence length must be divisible by local_window"
            )
        local_hidden = local_hidden.reshape(
            batch * (length // self.local_window),
            self.local_window,
            -1,
        )
        local_hidden = run_modern_stack(
            self.local_core,
            local_hidden,
            causal_mask(self.local_window, x.device),
        ).reshape(batch, length, -1)
        logits = self.head(self.local_norm(local_hidden))
        logits = logits + self.transition_head(x)
        if brick is not None:
            delta = brick(context_abi) - context_abi
            patch_bias = delta @ self.canonical_head
            logits = logits + patch_bias.gather(
                1,
                patch_ids.unsqueeze(-1).expand(-1, -1, 256),
            )
        return logits, context_abi, {
            "patch_ids": patch_ids,
            "patch_offsets": patch_offsets,
            "patch_lengths": patch_lengths,
            "valid_patches": valid,
        }
