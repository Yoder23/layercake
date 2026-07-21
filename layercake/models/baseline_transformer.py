"""Modern tokenizer transformer used as a non-sabotaged same-scale baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import hashlib
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
        sequence = list(value)
        for pair, new_id in self.merge_ids.items():
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
        return sequence

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
