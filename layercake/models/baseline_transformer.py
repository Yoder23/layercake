"""Modern tokenizer transformer used as a non-sabotaged same-scale baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import hashlib
import heapq
import json

import torch
from torch import nn
import torch.nn.functional as F


class BytePairTokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = list(merges or [])
        self._rebuild()

    def _rebuild(self) -> None:
        self.pieces: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for index, (left, right) in enumerate(self.merges, start=256):
            if left not in self.pieces or right not in self.pieces:
                raise ValueError("BPE merge references an undefined piece")
            self.pieces[index] = self.pieces[left] + self.pieces[right]
        self.merge_ids = {pair: index for index, pair in enumerate(self.merges, start=256)}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, corpus: bytes, merge_count: int = 64) -> "BytePairTokenizer":
        if not corpus:
            raise ValueError("BPE training corpus must be non-empty")
        sequence = list(corpus)
        merges: list[tuple[int, int]] = []
        for new_id in range(256, 256 + merge_count):
            counts = Counter(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = min(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
            if frequency < 2:
                break
            merges.append(pair)
            replaced: list[int] = []
            index = 0
            while index < len(sequence):
                if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                    replaced.append(new_id)
                    index += 2
                else:
                    replaced.append(sequence[index])
                    index += 1
            sequence = replaced
        return cls(merges)

    def encode(self, value: bytes | str) -> list[int]:
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not value:
            return []
        tokens = list(value)
        previous = [index - 1 for index in range(len(tokens))]
        following = [index + 1 for index in range(len(tokens))]
        following[-1] = -1
        alive = [True] * len(tokens)
        ranks = {
            pair: (new_id - 256, new_id)
            for pair, new_id in self.merge_ids.items()
        }
        queue: list[tuple[int, int, int, int]] = []

        def schedule(left: int) -> None:
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right < 0 or not alive[right]:
                return
            ranked = ranks.get((tokens[left], tokens[right]))
            if ranked is not None:
                rank, new_id = ranked
                heapq.heappush(queue, (rank, left, right, new_id))

        for index in range(len(tokens) - 1):
            schedule(index)
        while queue:
            rank, left, right, new_id = heapq.heappop(queue)
            if (
                not alive[left] or not alive[right] or following[left] != right
                or ranks.get((tokens[left], tokens[right])) != (rank, new_id)
            ):
                continue
            tokens[left] = new_id
            alive[right] = False
            successor = following[right]
            following[left] = successor
            if successor >= 0:
                previous[successor] = left
            schedule(previous[left])
            schedule(left)
        encoded = []
        index = 0
        while index >= 0:
            if alive[index]:
                encoded.append(tokens[index])
            index = following[index]
        return encoded

    def decode(self, ids: list[int]) -> bytes:
        try:
            return b"".join(self.pieces[index] for index in ids)
        except KeyError as exc:
            raise ValueError(f"unknown BPE token: {exc.args[0]}") from exc

    def canonical_dict(self) -> dict:
        return {"format": "layercake-byte-bpe/1", "merges": [list(pair) for pair in self.merges]}

    def hash(self) -> str:
        raw = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    width: int
    layers: int
    heads: int
    max_tokens: int
    expansion: int = 4
    architecture_version: str = "modern-bpe-transformer/1"

    def canonical_dict(self) -> dict:
        return asdict(self)


class CausalSwiGLUBlock(nn.Module):
    def __init__(self, width: int, heads: int, expansion: int):
        super().__init__()
        if width % heads:
            raise ValueError("transformer width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attn_out = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.gate_up = nn.Linear(width, 2 * width * expansion, bias=False)
        self.down = nn.Linear(width * expansion, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(self.attn_norm(hidden)).reshape(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = [item.transpose(1, 2) for item in qkv.unbind(dim=2)]
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        hidden = hidden + self.attn_out(attended.transpose(1, 2).reshape(batch, length, width))
        gate, value = self.gate_up(self.ffn_norm(hidden)).chunk(2, dim=-1)
        return hidden + self.down(F.silu(gate) * value)

    def forward_cached(
        self,
        hidden: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, length, width = hidden.shape
        qkv = self.qkv(self.attn_norm(hidden)).reshape(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = [item.transpose(1, 2) for item in qkv.unbind(dim=2)]
        if past is not None:
            key = torch.cat([past[0], key], dim=2)
            value = torch.cat([past[1], value], dim=2)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=past is None and length > 1
        )
        hidden = hidden + self.attn_out(attended.transpose(1, 2).reshape(batch, length, width))
        gate, ffn_value = self.gate_up(self.ffn_norm(hidden)).chunk(2, dim=-1)
        hidden = hidden + self.down(F.silu(gate) * ffn_value)
        return hidden, (key.detach(), value.detach())


@dataclass
class TransformerGenerationState:
    keys_values: list[tuple[torch.Tensor, torch.Tensor]]
    next_logits: torch.Tensor
    token_ids: torch.Tensor
    generated_ids: torch.Tensor


class ModernBPETransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.max_tokens, config.width)
        self.blocks = nn.ModuleList(
            CausalSwiGLUBlock(config.width, config.heads, config.expansion)
            for _ in range(config.layers)
        )
        self.norm = nn.LayerNorm(config.width)
        # Tied output embeddings are a credible modern default.
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] > self.config.max_tokens:
            raise ValueError("token ids exceed the configured transformer context")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.embedding(token_ids) + self.position(positions)[None]
        for block in self.blocks:
            hidden = block(hidden)
        return F.linear(self.norm(hidden), self.embedding.weight)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.inference_mode()
    def prefill(self, token_ids: torch.Tensor) -> TransformerGenerationState:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("prefill requires non-empty [batch, sequence] token ids")
        if token_ids.shape[1] > self.config.max_tokens:
            raise ValueError("token ids exceed the configured transformer context")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.embedding(token_ids) + self.position(positions)[None]
        keys_values = []
        for block in self.blocks:
            hidden, cache = block.forward_cached(hidden)
            keys_values.append(cache)
        logits = F.linear(self.norm(hidden[:, -1]), self.embedding.weight)
        return TransformerGenerationState(
            keys_values=keys_values,
            next_logits=logits,
            token_ids=token_ids,
            generated_ids=token_ids[:, :0],
        )

    @torch.inference_mode()
    def decode_step(
        self,
        state: TransformerGenerationState,
        next_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TransformerGenerationState]:
        logits = state.next_logits
        selected = logits.argmax(-1) if next_token is None else next_token.to(logits.device).long().flatten()
        position = state.token_ids.shape[1] + state.generated_ids.shape[1]
        if position >= self.config.max_tokens:
            raise ValueError("transformer KV cache reached max_tokens")
        hidden = self.embedding(selected[:, None]) + self.position.weight[position][None, None]
        new_cache = []
        for block, past in zip(self.blocks, state.keys_values):
            hidden, cache = block.forward_cached(hidden, past)
            new_cache.append(cache)
        state.keys_values = new_cache
        state.next_logits = F.linear(self.norm(hidden[:, 0]), self.embedding.weight)
        state.generated_ids = torch.cat([state.generated_ids, selected[:, None]], dim=1)
        return logits, state

    @torch.inference_mode()
    def decode_many(
        self, state: TransformerGenerationState, count: int
    ) -> tuple[torch.Tensor, TransformerGenerationState]:
        rows = []
        for _ in range(count):
            logits, state = self.decode_step(state)
            rows.append(logits[:, None])
        empty = state.next_logits.new_zeros(state.next_logits.shape[0], 0, self.config.vocab_size)
        return (torch.cat(rows, dim=1) if rows else empty), state


def matched_transformer_config(
    target_parameters: int,
    *,
    vocab_size: int,
    max_tokens: int,
    tolerance: float = 0.05,
) -> TransformerConfig:
    best: tuple[float, TransformerConfig, int] | None = None
    for width in range(48, 257, 16):
        heads = 4 if width % 4 == 0 else 2
        for layers in range(1, 25):
            config = TransformerConfig(vocab_size, width, layers, heads, max_tokens)
            count = ModernBPETransformer(config).parameter_count()
            difference = abs(count - target_parameters) / target_parameters
            if best is None or difference < best[0]:
                best = difference, config, count
            if difference <= tolerance:
                return config
    assert best is not None
    raise ValueError(
        f"could not parameter-match transformer within {tolerance:.1%}; best was {best[0]:.1%}"
    )
