"""Tokenizer-free byte and byte-patch front ends."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class PatchMetadata:
    boundaries: tuple[tuple[int, int], ...]
    original_length: int
    padded_length: int
    patching_mode: str

    @property
    def patch_count(self) -> int:
        return len(self.boundaries)

    @property
    def compression_ratio(self) -> float:
        return self.original_length / max(self.patch_count, 1)


class ByteCodec:
    @staticmethod
    def encode_text(text: str, encoding: str = "utf-8") -> list[int]:
        return list(text.encode(encoding))

    @staticmethod
    def decode_bytes(
        byte_ids: Sequence[int], encoding: str = "utf-8", errors: str = "strict"
    ) -> str:
        return bytes(int(x) for x in byte_ids).decode(encoding, errors=errors)


class ByteEncoder(nn.Module):
    def __init__(self, d_byte: int):
        super().__init__()
        self.embedding = nn.Embedding(256, d_byte)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        if byte_ids.dtype != torch.long:
            byte_ids = byte_ids.long()
        if byte_ids.numel() and (byte_ids.min() < 0 or byte_ids.max() > 255):
            raise ValueError("byte ids must be in [0, 255]")
        return self.embedding(byte_ids)


class ByteDecoder(nn.Module):
    def __init__(self, d_byte: int):
        super().__init__()
        self.projection = nn.Linear(d_byte, 256)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.projection(states)


class FixedBytePatcher:
    def __init__(self, patch_size: int = 4):
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.mode = f"fixed:{patch_size}"

    def boundaries(self, byte_ids: Sequence[int]) -> PatchMetadata:
        n = len(byte_ids)
        bounds = tuple((i, min(i + self.patch_size, n)) for i in range(0, n, self.patch_size))
        return PatchMetadata(bounds, n, n, self.mode)


class WhitespaceBytePatcher:
    """Ends patches at ASCII whitespace or at ``max_patch_size``."""

    def __init__(self, max_patch_size: int = 16):
        if max_patch_size <= 0:
            raise ValueError("max_patch_size must be positive")
        self.max_patch_size = max_patch_size
        self.mode = f"whitespace:{max_patch_size}"

    def boundaries(self, byte_ids: Sequence[int]) -> PatchMetadata:
        bounds: list[tuple[int, int]] = []
        start = 0
        for index, value in enumerate(byte_ids, start=1):
            if value in (9, 10, 13, 32) or index - start >= self.max_patch_size:
                bounds.append((start, index))
                start = index
        if start < len(byte_ids):
            bounds.append((start, len(byte_ids)))
        return PatchMetadata(tuple(bounds), len(byte_ids), len(byte_ids), self.mode)


class DifficultyPatcher:
    """Interface stub for learned entropy/difficulty boundaries."""

    mode = "difficulty:untrained"

    def boundaries(self, byte_ids: Sequence[int]) -> PatchMetadata:
        raise NotImplementedError(
            "DifficultyPatcher requires a trained next-byte difficulty estimator"
        )


class BytePatchEncoder(nn.Module):
    """Encode bytes, pool variable patches, and project to the model dimension."""

    def __init__(self, d_model: int, d_byte: int = 64, patcher=None):
        super().__init__()
        self.byte_encoder = ByteEncoder(d_byte)
        self.patcher = patcher or FixedBytePatcher(4)
        self.projection = nn.Linear(d_byte, d_model)

    def forward(
        self, byte_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[PatchMetadata]]:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, bytes]")
        embedded = self.byte_encoder(byte_ids)
        batch_patches: list[torch.Tensor] = []
        metadata: list[PatchMetadata] = []
        max_patches = 0
        for row, row_emb in zip(byte_ids, embedded):
            ids = row.tolist()
            meta = self.patcher.boundaries(ids)
            metadata.append(meta)
            patches = [
                row_emb[start:end].mean(dim=0) for start, end in meta.boundaries
            ]
            if not patches:
                patches = [row_emb.new_zeros(row_emb.shape[-1])]
            stacked = torch.stack(patches)
            batch_patches.append(stacked)
            max_patches = max(max_patches, stacked.shape[0])
        padded = embedded.new_zeros(len(batch_patches), max_patches, embedded.shape[-1])
        for index, patches in enumerate(batch_patches):
            padded[index, : patches.shape[0]] = patches
        return self.projection(padded), metadata


class BytePatchDecoder(nn.Module):
    """Expand patch states over original byte boundaries then predict bytes."""

    def __init__(self, d_model: int):
        super().__init__()
        self.byte_decoder = ByteDecoder(d_model)

    def forward(
        self, patch_states: torch.Tensor, metadata: Sequence[PatchMetadata]
    ) -> torch.Tensor:
        max_bytes = max((m.original_length for m in metadata), default=0)
        expanded = patch_states.new_zeros(
            patch_states.shape[0], max_bytes, patch_states.shape[-1]
        )
        for batch_index, meta in enumerate(metadata):
            for patch_index, (start, end) in enumerate(meta.boundaries):
                expanded[batch_index, start:end] = patch_states[batch_index, patch_index]
        return self.byte_decoder(expanded)
