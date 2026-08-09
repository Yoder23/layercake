"""Compact Phi-compatible causal core with persistent rotary KV state."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .portable_token_plan import BOS_ID, EOS_ID


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden.float() * torch.rsqrt(variance + self.epsilon)
        return (normalized * self.weight.float()).to(hidden.dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _rotary(value: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    dimension = value.shape[-1]
    frequencies = 1.0 / (theta ** (torch.arange(0, dimension, 2, device=value.device, dtype=torch.float32) / dimension))
    phases = torch.outer(positions.float(), frequencies)
    embedding = torch.cat((phases, phases), dim=-1).to(value.dtype)
    cosine = embedding.cos()[None, None]
    sine = embedding.sin()[None, None]
    return value * cosine + _rotate_half(value) * sine


@dataclass
class StructuralCausalCoreState:
    source_ids: torch.Tensor
    source_lexemes: list[bytes]
    generated_actions: list[int]
    layer_keys: tuple[torch.Tensor, ...]
    layer_values: tuple[torch.Tensor, ...]
    next_logits: torch.Tensor
    sequence_length: int
    complete: bool = False


class StructuralCausalLayer(nn.Module):
    def __init__(self, width: int, heads: int, intermediate_size: int, *, rms_epsilon: float, rope_theta: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("structural causal width must divide attention heads")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        if self.head_dim % 2:
            raise ValueError("rotary head dimension must be even")
        self.intermediate_size = int(intermediate_size)
        self.rope_theta = float(rope_theta)
        self.input_norm = RMSNorm(self.width, rms_epsilon)
        self.qkv_proj = nn.Linear(self.width, 3 * self.width, bias=False)
        self.o_proj = nn.Linear(self.width, self.width, bias=False)
        self.post_attention_norm = RMSNorm(self.width, rms_epsilon)
        self.gate_up_proj = nn.Linear(self.width, 2 * self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.width, bias=False)

    def _qkv(self, hidden: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden.shape
        qkv = self.qkv_proj(hidden).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = _rotary(query.transpose(1, 2), positions, self.rope_theta)
        key = _rotary(key.transpose(1, 2), positions, self.rope_theta)
        return query, key, value.transpose(1, 2)

    @staticmethod
    def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, causal: bool) -> torch.Tensor:
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / sqrt(query.shape[-1])
        if causal:
            query_length, key_length = scores.shape[-2:]
            mask = torch.triu(torch.ones(query_length, key_length, dtype=torch.bool, device=scores.device), diagonal=1 + key_length - query_length)
            scores = scores.masked_fill(mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
        return torch.matmul(probabilities, value)

    def _mlp(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(self.post_attention_norm(hidden)).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    def forward_with_cache(self, hidden: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self._qkv(self.input_norm(hidden), positions)
        attended = self._attention(query, key, value, causal=True).transpose(1, 2).contiguous().view_as(hidden)
        hidden = hidden + self.o_proj(attended)
        hidden = hidden + self._mlp(hidden)
        return hidden, key, value

    def incremental(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        previous_key: torch.Tensor,
        previous_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self._qkv(self.input_norm(hidden), position)
        all_keys = torch.cat((previous_key, key), dim=2)
        all_values = torch.cat((previous_value, value), dim=2)
        attended = self._attention(query, all_keys, all_values, causal=False).transpose(1, 2).contiguous().view_as(hidden)
        hidden = hidden + self.o_proj(attended)
        hidden = hidden + self._mlp(hidden)
        return hidden, all_keys, all_values


class StructuralCausalCore(nn.Module):
    """Untied Phi-compatible decoder exposed through the generic LayerCake boundary."""

    def __init__(
        self,
        *,
        fixed_vocab_size: int,
        model_width: int = 192,
        attention_heads: int = 2,
        decoder_layers: int = 4,
        intermediate_size: int = 768,
        rms_epsilon: float = 1e-5,
        rope_theta: float = 10000.0,
        maximum_source_actions: int = 192,
        maximum_target_actions: int = 320,
        maximum_sequence_actions: int = 512,
    ) -> None:
        super().__init__()
        if maximum_source_actions + maximum_target_actions > maximum_sequence_actions:
            raise ValueError("declared source and target bounds exceed causal context")
        self.fixed_vocab_size = int(fixed_vocab_size)
        self.model_width = int(model_width)
        self.attention_heads = int(attention_heads)
        self.decoder_layers = int(decoder_layers)
        self.intermediate_size = int(intermediate_size)
        self.rms_epsilon = float(rms_epsilon)
        self.rope_theta = float(rope_theta)
        self.maximum_source_actions = int(maximum_source_actions)
        self.maximum_target_actions = int(maximum_target_actions)
        self.maximum_sequence_actions = int(maximum_sequence_actions)
        self.token_embedding = nn.Embedding(self.fixed_vocab_size, self.model_width)
        self.layers = nn.ModuleList(
            StructuralCausalLayer(
                self.model_width,
                self.attention_heads,
                self.intermediate_size,
                rms_epsilon=self.rms_epsilon,
                rope_theta=self.rope_theta,
            )
            for _ in range(self.decoder_layers)
        )
        self.final_norm = RMSNorm(self.model_width, self.rms_epsilon)
        self.lm_head = nn.Linear(self.model_width, self.fixed_vocab_size, bias=False)
        self.tokenizer: Any | None = None

    def canonical_config(self) -> dict[str, Any]:
        return {
            "fixed_vocab_size": self.fixed_vocab_size,
            "model_width": self.model_width,
            "attention_heads": self.attention_heads,
            "decoder_layers": self.decoder_layers,
            "intermediate_size": self.intermediate_size,
            "rms_epsilon": self.rms_epsilon,
            "rope_theta": self.rope_theta,
            "maximum_source_actions": self.maximum_source_actions,
            "maximum_target_actions": self.maximum_target_actions,
            "maximum_sequence_actions": self.maximum_sequence_actions,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def bind_tokenizer(self, tokenizer: Any) -> "StructuralCausalCore":
        if int(tokenizer.vocab_size) != self.fixed_vocab_size:
            raise ValueError("model and tokenizer vocabulary sizes differ")
        self.tokenizer = tokenizer
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or not input_ids.shape[1] or input_ids.shape[1] > self.maximum_sequence_actions:
            raise ValueError("structural causal input has invalid shape")
        hidden = self.token_embedding(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        for layer in self.layers:
            hidden, _, _ = layer.forward_with_cache(hidden, positions)
        return self.lm_head(self.final_norm(hidden))

    def prefill_ids(self, source_ids: list[int], source_lexemes: list[bytes]) -> StructuralCausalCoreState:
        if not source_ids or len(source_ids) > self.maximum_source_actions:
            raise ValueError("structural causal source is empty or exceeds bound")
        device = self.token_embedding.weight.device
        values = torch.tensor([source_ids + [BOS_ID]], dtype=torch.long, device=device)
        positions = torch.arange(values.shape[1], device=device)
        hidden = self.token_embedding(values)
        keys: list[torch.Tensor] = []
        cached_values: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, key, value = layer.forward_with_cache(hidden, positions)
            keys.append(key)
            cached_values.append(value)
        logits = self.lm_head(self.final_norm(hidden[:, -1]))
        return StructuralCausalCoreState(values[:, :-1], source_lexemes, [], tuple(keys), tuple(cached_values), logits, values.shape[1])

    def prefill_bytes(self, prompt: bytes | str) -> StructuralCausalCoreState:
        if self.tokenizer is None:
            raise ValueError("structural causal tokenizer is not bound")
        source_ids, source_lexemes = self.tokenizer.encode_source(prompt)
        return self.prefill_ids(source_ids, source_lexemes)

    @torch.inference_mode()
    def decode_step(self, state: StructuralCausalCoreState) -> tuple[int, StructuralCausalCoreState]:
        if state.complete:
            raise ValueError("structural causal request is already complete")
        action = int(state.next_logits.argmax(dim=-1).item())
        state.generated_actions.append(action)
        state.complete = action == EOS_ID or len(state.generated_actions) >= self.maximum_target_actions
        if state.complete or state.sequence_length >= self.maximum_sequence_actions:
            state.complete = True
            return action, state
        device = self.token_embedding.weight.device
        token = torch.tensor([[action]], dtype=torch.long, device=device)
        position = torch.tensor([state.sequence_length], dtype=torch.long, device=device)
        hidden = self.token_embedding(token)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for layer, previous_key, previous_value in zip(self.layers, state.layer_keys, state.layer_values):
            hidden, key, value = layer.incremental(hidden, position, previous_key, previous_value)
            keys.append(key)
            values.append(value)
        state.layer_keys = tuple(keys)
        state.layer_values = tuple(values)
        state.next_logits = self.lm_head(self.final_norm(hidden[:, -1]))
        state.sequence_length += 1
        return action, state

    @torch.inference_mode()
    def generate_bytes(self, prompt: bytes | str, *, maximum_actions: int | None = None) -> bytes:
        if self.tokenizer is None:
            raise ValueError("structural causal tokenizer is not bound")
        state = self.prefill_bytes(prompt)
        limit = self.maximum_target_actions if maximum_actions is None else min(int(maximum_actions), self.maximum_target_actions)
        while not state.complete and len(state.generated_actions) < limit:
            self.decode_step(state)
        return self.tokenizer.decode_actions(state.generated_actions, state.source_lexemes)
