"""Source-aligned progressive core with separate attention and MLP cakes."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

from .source_aligned_progressive_replacement_core import SourceAlignedProgressiveReplacementCore
from .structural_causal_core import RMSNorm, _rotary


class DualPathProgressiveLayer(nn.Module):
    def __init__(self, full_width: int, bottleneck_width: int, heads: int, intermediate_size: int, *, rms_epsilon: float, rope_theta: float) -> None:
        super().__init__()
        if bottleneck_width % heads:
            raise ValueError("dual-path bottleneck width must divide attention heads")
        self.full_width = int(full_width); self.bottleneck_width = int(bottleneck_width); self.heads = int(heads)
        self.head_dim = self.bottleneck_width // self.heads
        if self.head_dim % 2: raise ValueError("dual-path rotary head dimension must be even")
        self.intermediate_size = int(intermediate_size); self.rope_theta = float(rope_theta)
        self.input_norm = RMSNorm(self.full_width, rms_epsilon)
        self.attention_input_projection = nn.Linear(self.full_width, self.bottleneck_width, bias=False)
        self.attention_norm = RMSNorm(self.bottleneck_width, rms_epsilon)
        self.qkv_proj = nn.Linear(self.bottleneck_width, 3 * self.bottleneck_width, bias=False)
        self.o_proj = nn.Linear(self.bottleneck_width, self.bottleneck_width, bias=False)
        self.attention_output_projection = nn.Linear(self.bottleneck_width, self.full_width, bias=False)
        self.post_attention_norm = RMSNorm(self.full_width, rms_epsilon)
        self.mlp_input_projection = nn.Linear(self.full_width, self.bottleneck_width, bias=False)
        self.mlp_norm = RMSNorm(self.bottleneck_width, rms_epsilon)
        self.gate_up_proj = nn.Linear(self.bottleneck_width, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.bottleneck_width, bias=False)
        self.mlp_output_projection = nn.Linear(self.bottleneck_width, self.full_width, bias=False)

    def _qkv(self, latent: torch.Tensor, positions: torch.Tensor):
        batch, length, _ = latent.shape
        qkv = self.qkv_proj(self.attention_norm(latent)).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        return _rotary(query.transpose(1, 2), positions, self.rope_theta), _rotary(key.transpose(1, 2), positions, self.rope_theta), value.transpose(1, 2)

    @staticmethod
    def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, causal: bool) -> torch.Tensor:
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / sqrt(query.shape[-1])
        if causal:
            q, k = scores.shape[-2:]
            mask = torch.triu(torch.ones(q, k, dtype=torch.bool, device=scores.device), diagonal=1 + k - q)
            scores = scores.masked_fill(mask, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1).to(value.dtype), value)

    def _mlp_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        latent = self.mlp_input_projection(self.post_attention_norm(hidden))
        gate, up = self.gate_up_proj(self.mlp_norm(latent)).chunk(2, dim=-1)
        return self.mlp_output_projection(self.down_proj(F.silu(gate) * up))

    def forward_with_cache(self, hidden: torch.Tensor, positions: torch.Tensor):
        latent = self.attention_input_projection(self.input_norm(hidden))
        query, key, value = self._qkv(latent, positions)
        attended = self._attention(query, key, value, causal=True).transpose(1, 2).contiguous().view_as(latent)
        hidden = hidden + self.attention_output_projection(self.o_proj(attended))
        hidden = hidden + self._mlp_delta(hidden)
        return hidden, key, value

    def incremental(self, hidden: torch.Tensor, position: torch.Tensor, previous_key: torch.Tensor, previous_value: torch.Tensor):
        latent = self.attention_input_projection(self.input_norm(hidden))
        query, key, value = self._qkv(latent, position)
        all_keys = torch.cat((previous_key, key), dim=2); all_values = torch.cat((previous_value, value), dim=2)
        attended = self._attention(query, all_keys, all_values, causal=False).transpose(1, 2).contiguous().view_as(latent)
        hidden = hidden + self.attention_output_projection(self.o_proj(attended))
        hidden = hidden + self._mlp_delta(hidden)
        return hidden, all_keys, all_values


class DualPathProgressiveCore(SourceAlignedProgressiveReplacementCore):
    def __init__(self, *, fixed_vocab_size: int, full_width: int = 3072, bottleneck_width: int = 192, attention_heads: int = 2, replacement_layers: int = 32, intermediate_size: int = 768, rms_epsilon: float = 1e-5, rope_theta: float = 10000.0, maximum_source_actions: int = 192, maximum_target_actions: int = 320, maximum_sequence_actions: int = 512) -> None:
        requested_layers = int(replacement_layers)
        super().__init__(fixed_vocab_size=fixed_vocab_size, full_width=full_width, bottleneck_width=bottleneck_width, attention_heads=attention_heads, replacement_layers=0, intermediate_size=intermediate_size, rms_epsilon=rms_epsilon, rope_theta=rope_theta, maximum_source_actions=maximum_source_actions, maximum_target_actions=maximum_target_actions, maximum_sequence_actions=maximum_sequence_actions)
        self.replacement_layers = requested_layers
        self.layers = nn.ModuleList(DualPathProgressiveLayer(self.full_width, self.bottleneck_width, self.attention_heads, self.intermediate_size, rms_epsilon=self.rms_epsilon, rope_theta=self.rope_theta) for _ in range(self.replacement_layers))

    @staticmethod
    def parameter_count_for_config(*, fixed_vocab_size: int, full_width: int, bottleneck_width: int, replacement_layers: int, intermediate_size: int) -> int:
        copied = 2 * fixed_vocab_size * full_width + (2 * replacement_layers + 1) * full_width
        per_layer = 4 * full_width * bottleneck_width + 4 * bottleneck_width * bottleneck_width + 3 * bottleneck_width * intermediate_size + 2 * bottleneck_width
        return copied + replacement_layers * per_layer
