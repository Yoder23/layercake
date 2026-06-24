"""Strictly causal byte and byte-patch models for measured experiments."""

from __future__ import annotations

import time

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
        h = h + self.dropout(
            self.down(
                F.silu(self.gate(normalized)) * self.up(normalized)
            )
        )
        return h, (all_key, all_value)


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
        return h, (all_key, all_value)


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
    ):
        super().__init__()
        self.byte_embedding = byte_embedding
        self.patch_size = patch_size
        self.initial_state = nn.Linear(context_width, hidden_width)
        self.bos = nn.Parameter(torch.zeros(byte_embedding.embedding_dim))
        self.cell = nn.GRUCell(byte_embedding.embedding_dim, hidden_width)
        self.output = nn.Linear(hidden_width, 256)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        prefix: torch.Tensor | None = None,
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
        predictions = []
        for offset in range(self.patch_size):
            hidden = self.cell(
                decoder_input.reshape(-1, decoder_input.shape[-1]),
                hidden.reshape(-1, hidden.shape[-1]),
            ).reshape_as(hidden)
            predictions.append(self.output(hidden))
            decoder_input = self.byte_embedding(target[..., offset])
        return predictions

    @torch.no_grad()
    def greedy(
        self,
        context: torch.Tensor,
        prefix: torch.Tensor | None = None,
        forced_first: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = torch.tanh(self.initial_state(context))
        decoder_input = self.bos.expand(*context.shape[:-1], -1)
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
            next_byte = self.output(hidden).argmax(dim=-1)
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
        patch_generation_context=0,
        patch_prediction_detach_context=False,
        patch_prediction_context="global",
        tie_byte_embeddings=False,
        context_buckets=0,
        context_order=3,
        context_logits: torch.Tensor | None = None,
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
        sparse_state_local_window=32,
        sparse_state_dilated_offsets=(32, 48, 64, 96),
        sparse_state_chunk_size=16,
    ):
        super().__init__()
        self.patch_size = patch_size
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
        self.patch_generation_context = patch_generation_context
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
        self.local_position_embeddings = local_position_embeddings
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
        self.sparse_state_local_window = sparse_state_local_window
        self.sparse_state_dilated_offsets = tuple(
            sparse_state_dilated_offsets
        )
        self.sparse_state_chunk_size = sparse_state_chunk_size
        self.profile_timing = False
        self.last_profile = {}
        if global_block not in {"attention", "sparse_state_patch"}:
            raise ValueError(
                "global_block must be attention or sparse_state_patch"
            )
        if global_block == "sparse_state_patch" and not modern_blocks:
            raise ValueError("sparse_state_patch requires modern_blocks")
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
        self.byte_emb = nn.Embedding(256, d_byte)
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
        if modern_blocks:
            block_type = (
                SparseStatePatchBlock
                if global_block == "sparse_state_patch"
                else (
                    FusedModernCausalBlock
                    if fused_attention
                    else ModernCausalBlock
                )
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
                            if fused_attention
                            else block_type(d_model, heads, dropout)
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
            if modern_blocks:
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
                    patch_size,
                )
            else:
                raise ValueError(
                    "patch_prediction_mode must be factorized or autoregressive"
                )
        self.register_buffer("canonical_head", canonical_brick_head(d_abi))

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
        if self.local_decoder == "window_transformer":
            if usable % self.local_window:
                raise ValueError("sequence length must be divisible by local_window")
            windowed_in = local_in.reshape(
                batch * (usable // self.local_window),
                self.local_window,
                -1,
            )
            local_out = self.local_in(windowed_in)
            local_mask = causal_mask(self.local_window, x.device)
            if self.modern_blocks:
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
        logits = logits + self.transition_head(x)
        if self.context_buckets:
            context_ids = self._context_ids(x)
            logits = logits + self.context_head(context_ids)
        if brick is not None:
            delta = brick(context_abi) - context_abi
            patch_bias = delta @ self.canonical_head
            logits = logits + patch_bias.unsqueeze(2).expand(
                -1, -1, self.patch_size, -1
            ).reshape(batch, usable, 256)
        if return_aux:
            auxiliary = [head(local_out) for head in self.aux_heads]
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
                if self.patch_prediction_detach_context:
                    prediction_context = prediction_context.detach()
                if self.patch_prediction_mode == "autoregressive":
                    next_patches = torch.cat(
                        [patches[:, 1:], torch.zeros_like(patches[:, :1])],
                        dim=1,
                    )[:, :: self.patch_prediction_stride]
                    generation_prefix = None
                    if self.patch_generation_context:
                        generation_prefix = self._patch_generation_prefixes(
                            x
                        )[:, :: self.patch_prediction_stride]
                    patch_predictions = self.patch_generator(
                        prediction_context,
                        next_patches,
                        generation_prefix,
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
        return F.pad(
            x,
            (
                self.patch_generation_context - self.patch_size,
                0,
            ),
        ).unfold(
            1,
            self.patch_generation_context,
            self.patch_size,
        )

    @torch.no_grad()
    def generate_next_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Generate from the global path without running the local LM decoder."""
        if not self.patch_prediction:
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
            )
        return torch.stack(
            [
                head(global_h[:, -1]).argmax(dim=-1)
                for head in self.patch_prediction_heads
            ],
            dim=-1,
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
            draft = self.patch_generator.greedy(global_h[:, -1], prefix)
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
            logits = logits + self.transition_head(window_bytes)
            if self.context_buckets:
                logits = logits + self.context_head(
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
        first_logits = first_logits + self.transition_head(
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
        second_logits = second_logits + self.transition_head(first)
        second = second_logits.argmax(dim=-1)
        return torch.stack([first, second], dim=-1)

    @torch.no_grad()
    def begin_cached_generation(self, x: torch.Tensor) -> dict:
        """Prefill global patch caches for stateful two-byte generation."""
        if (
            self.patch_size != 2
            or self.local_decoder != "window_transformer"
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
        patch_h = self.patch_proj(patch_features)
        positions = torch.arange(patch_h.shape[1], device=x.device)
        patch_h = patch_h + self.patch_pos(positions)[None]
        global_caches = []
        global_hidden = patch_h
        for block in self.core:
            global_hidden, cache = block.prefill_with_cache(global_hidden)
            global_caches.append(cache)
        source_bytes = x[:, -self.local_window :]
        source_patch_count = self.local_window // self.patch_size
        prior_global = torch.cat(
            [
                global_hidden.new_zeros(
                    batch, 1, global_hidden.shape[-1]
                ),
                global_hidden[:, :-1],
            ],
            dim=1,
        )[:, -source_patch_count:]
        source_context = prior_global.unsqueeze(2).expand(
            -1, -1, self.patch_size, -1
        ).reshape(batch, self.local_window, -1)
        local_hidden = self.local_in(
            torch.cat(
                [self.byte_emb(source_bytes), source_context], dim=-1
            )
        )
        local_caches = []
        for block in self.local_core:
            local_hidden, cache = block.prefill_with_cache(local_hidden)
            local_caches.append(cache)
        next_logits = self._byte_logits(
            self.local_norm(local_hidden)[:, -1]
        )
        next_logits = next_logits + self.transition_head(
            source_bytes[:, -1]
        )
        return {
            "bytes": x,
            "global_hidden": global_hidden,
            "global_caches": global_caches,
            "local_caches": local_caches,
            "local_position": usable % self.local_window,
            "next_logits": next_logits,
            "patch_count": patches.shape[1],
        }

    @torch.no_grad()
    def cached_generation_step(
        self,
        state: dict,
        forced_patch: torch.Tensor | None = None,
        return_logits: bool = False,
        no_repeat_ngram: int = 0,
    ):
        """Emit one exact local-LM patch and append it to global KV state."""
        x = state["bytes"]
        global_hidden = state["global_hidden"]
        first_logits = state["next_logits"]

        def select_byte(logits: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
            if no_repeat_ngram <= 1 or prefix.shape[1] < no_repeat_ngram - 1:
                return logits.argmax(dim=-1)
            selected = []
            for row, row_prefix in zip(logits, prefix):
                prefix_list = row_prefix.tolist()
                existing = {
                    tuple(prefix_list[index : index + no_repeat_ngram])
                    for index in range(
                        0, len(prefix_list) - no_repeat_ngram + 1
                    )
                }
                choice = row.argmax().item()
                for candidate in torch.argsort(row, descending=True).tolist():
                    trial = tuple(
                        prefix_list[-(no_repeat_ngram - 1) :]
                        + [int(candidate)]
                    )
                    if trial not in existing:
                        choice = int(candidate)
                        break
                selected.append(choice)
            return torch.tensor(selected, device=logits.device, dtype=torch.long)

        first = (
            forced_patch[:, 0]
            if forced_patch is not None
            else select_byte(first_logits, x)
        )

        target_context = global_hidden[:, -1]
        def process_local_byte(byte: torch.Tensor) -> torch.Tensor:
            token_hidden = self.local_in(
                torch.cat(
                    [self.byte_emb(byte), target_context], dim=-1
                )
            ).unsqueeze(1)
            reset = state["local_position"] == 0
            next_caches = []
            for index, block in enumerate(self.local_core):
                if reset:
                    token_hidden, cache = block.prefill_with_cache(
                        token_hidden
                    )
                else:
                    token_hidden, cache = block.decode_with_cache(
                        token_hidden, state["local_caches"][index]
                    )
                next_caches.append(cache)
            state["local_caches"] = next_caches
            state["local_position"] = (
                state["local_position"] + 1
            ) % self.local_window
            logits = self._byte_logits(
                self.local_norm(token_hidden)[:, 0]
            )
            return logits + self.transition_head(byte)

        second_logits = process_local_byte(first)
        second = (
            forced_patch[:, 1]
            if forced_patch is not None
            else select_byte(second_logits, torch.cat([x, first[:, None]], dim=1))
        )
        patch = torch.stack([first, second], dim=-1)
        state["next_logits"] = process_local_byte(second)

        patch_features = self.byte_emb(patch).flatten(-2)
        if self.patch_unit_buckets:
            patch_ids = (
                patch[:, 0] * 257 + patch[:, 1]
            ) % self.patch_unit_buckets
            patch_features = torch.cat(
                [patch_features, self.patch_unit_emb(patch_ids)], dim=-1
            )
        position = state["patch_count"]
        if position >= self.patch_pos.num_embeddings:
            raise ValueError("generation exceeded trained patch positions")
        next_global = self.patch_proj(patch_features)
        next_global = (
            next_global + self.patch_pos.weight[position]
        ).unsqueeze(1)
        for index, block in enumerate(self.core):
            next_global, cache = block.decode_with_cache(
                next_global, state["global_caches"][index]
            )
            state["global_caches"][index] = cache
        state["global_hidden"] = torch.cat(
            [global_hidden, next_global], dim=1
        )
        state["bytes"] = torch.cat([x, patch], dim=1)
        state["patch_count"] = position + 1
        if return_logits:
            return patch, torch.stack([first_logits, second_logits], dim=1)
        return patch

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
