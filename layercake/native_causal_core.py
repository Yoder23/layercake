"""Compact tied-vocabulary causal core with persistent incremental state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .portable_token_plan import BOS_ID, EOS_ID, PAD_ID


@dataclass
class NativeCausalCoreState:
    source_ids: torch.Tensor
    source_lexemes: list[bytes]
    generated_actions: list[int]
    layer_inputs: tuple[torch.Tensor, ...]
    next_logits: torch.Tensor
    sequence_length: int
    complete: bool = False


class NativeCausalLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, *, causal_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        hidden = hidden + self.dropout(attended)
        normalized = self.norm2(hidden)
        return hidden + self.dropout(self.linear2(self.dropout(F.gelu(self.linear1(normalized)))))

    def incremental(self, hidden: torch.Tensor, previous_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm1(hidden)
        memory = torch.cat((previous_inputs, normalized), dim=1)
        attended = self.attention(normalized, memory, memory, need_weights=False)[0]
        hidden = hidden + self.dropout(attended)
        normalized = self.norm2(hidden)
        hidden = hidden + self.dropout(self.linear2(self.dropout(F.gelu(self.linear1(normalized)))))
        return hidden, memory


class TiedNativeCausalCore(nn.Module):
    """Decoder-only UTF-8-facing neural core with a tied external vocabulary."""

    def __init__(
        self,
        *,
        fixed_vocab_size: int,
        model_width: int = 256,
        attention_heads: int = 8,
        decoder_layers: int = 6,
        feedforward_width: int = 1024,
        dropout: float = 0.1,
        maximum_source_actions: int = 192,
        maximum_target_actions: int = 320,
        maximum_sequence_actions: int = 512,
    ) -> None:
        super().__init__()
        if model_width % attention_heads:
            raise ValueError("model width must divide attention heads")
        if maximum_source_actions + maximum_target_actions > maximum_sequence_actions:
            raise ValueError("declared source and target bounds exceed causal context")
        self.fixed_vocab_size = int(fixed_vocab_size)
        self.model_width = int(model_width)
        self.attention_heads = int(attention_heads)
        self.decoder_layers = int(decoder_layers)
        self.feedforward_width = int(feedforward_width)
        self.dropout = float(dropout)
        self.maximum_source_actions = int(maximum_source_actions)
        self.maximum_target_actions = int(maximum_target_actions)
        self.maximum_sequence_actions = int(maximum_sequence_actions)
        self.token_embedding = nn.Embedding(self.fixed_vocab_size, self.model_width, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(self.maximum_sequence_actions, self.model_width)
        self.layers = nn.ModuleList(
            NativeCausalLayer(self.model_width, self.attention_heads, self.feedforward_width, self.dropout)
            for _ in range(self.decoder_layers)
        )
        self.final_norm = nn.LayerNorm(self.model_width)
        self.output_bias = nn.Parameter(torch.zeros(self.fixed_vocab_size))
        self.tokenizer: Any | None = None

    def canonical_config(self) -> dict[str, Any]:
        return {
            "fixed_vocab_size": self.fixed_vocab_size,
            "model_width": self.model_width,
            "attention_heads": self.attention_heads,
            "decoder_layers": self.decoder_layers,
            "feedforward_width": self.feedforward_width,
            "dropout": self.dropout,
            "maximum_source_actions": self.maximum_source_actions,
            "maximum_target_actions": self.maximum_target_actions,
            "maximum_sequence_actions": self.maximum_sequence_actions,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def bind_tokenizer(self, tokenizer: Any) -> "TiedNativeCausalCore":
        if int(tokenizer.vocab_size) != self.fixed_vocab_size:
            raise ValueError("model and tokenizer vocabulary sizes differ")
        self.tokenizer = tokenizer
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.maximum_sequence_actions:
            raise ValueError("causal input has invalid shape")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        causal = torch.triu(torch.ones(input_ids.shape[1], input_ids.shape[1], dtype=torch.bool, device=input_ids.device), diagonal=1)
        padding = input_ids.eq(PAD_ID)
        for layer in self.layers:
            hidden = layer(hidden, causal_mask=causal, padding_mask=padding)
        hidden = self.final_norm(hidden)
        return F.linear(hidden, self.token_embedding.weight, self.output_bias)

    def prefill_ids(self, source_ids: list[int], source_lexemes: list[bytes]) -> NativeCausalCoreState:
        if not source_ids or len(source_ids) > self.maximum_source_actions:
            raise ValueError("causal source is empty or exceeds bound")
        device = self.token_embedding.weight.device
        values = torch.tensor([source_ids + [BOS_ID]], dtype=torch.long, device=device)
        positions = torch.arange(values.shape[1], device=device)
        hidden = self.token_embedding(values) + self.position_embedding(positions)[None]
        causal = torch.triu(torch.ones(values.shape[1], values.shape[1], dtype=torch.bool, device=device), diagonal=1)
        padding = values.eq(PAD_ID)
        caches: list[torch.Tensor] = []
        for layer in self.layers:
            normalized = layer.norm1(hidden)
            caches.append(normalized)
            attended = layer.attention(normalized, normalized, normalized, attn_mask=causal, key_padding_mask=padding, need_weights=False)[0]
            hidden = hidden + layer.dropout(attended)
            normalized2 = layer.norm2(hidden)
            hidden = hidden + layer.dropout(layer.linear2(layer.dropout(F.gelu(layer.linear1(normalized2)))))
        next_logits = F.linear(self.final_norm(hidden[:, -1]), self.token_embedding.weight, self.output_bias)
        return NativeCausalCoreState(values[:, :-1], source_lexemes, [], tuple(caches), next_logits, values.shape[1])

    def prefill_bytes(self, prompt: bytes | str) -> NativeCausalCoreState:
        if self.tokenizer is None:
            raise ValueError("causal tokenizer is not bound")
        source_ids, source_lexemes = self.tokenizer.encode_source(prompt)
        return self.prefill_ids(source_ids, source_lexemes)

    @torch.inference_mode()
    def decode_step(self, state: NativeCausalCoreState) -> tuple[int, NativeCausalCoreState]:
        if state.complete:
            raise ValueError("causal request is already complete")
        action = int(state.next_logits.argmax(dim=-1).item())
        state.generated_actions.append(action)
        state.complete = action == EOS_ID or len(state.generated_actions) >= self.maximum_target_actions
        if state.complete:
            return action, state
        if state.sequence_length >= self.maximum_sequence_actions:
            state.complete = True
            return action, state
        device = self.token_embedding.weight.device
        token = torch.tensor([[action]], dtype=torch.long, device=device)
        position = torch.tensor([state.sequence_length], dtype=torch.long, device=device)
        hidden = self.token_embedding(token) + self.position_embedding(position)[None]
        caches: list[torch.Tensor] = []
        for layer, previous in zip(self.layers, state.layer_inputs):
            hidden, memory = layer.incremental(hidden, previous)
            caches.append(memory)
        state.layer_inputs = tuple(caches)
        state.next_logits = F.linear(self.final_norm(hidden[:, -1]), self.token_embedding.weight, self.output_bias)
        state.sequence_length += 1
        return action, state

    @torch.inference_mode()
    def generate_bytes(self, prompt: bytes | str, *, maximum_actions: int | None = None) -> bytes:
        if self.tokenizer is None:
            raise ValueError("causal tokenizer is not bound")
        state = self.prefill_bytes(prompt)
        limit = self.maximum_target_actions if maximum_actions is None else min(int(maximum_actions), self.maximum_target_actions)
        while not state.complete and len(state.generated_actions) < limit:
            self.decode_step(state)
        return self.tokenizer.decode_actions(state.generated_actions, state.source_lexemes)
