#!/usr/bin/env python3
"""
Data utilities for LayerCake LM training.

All training scripts here assume you’ve pre-tokenized your corpora into 1D numpy
arrays of integer token IDs and saved them as .npy files.

The dataset builds sliding windows over that 1D stream.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LM1DDataset(Dataset):
    """
    Language modeling dataset sliced from a 1D token_id array.

    Given a 1D LongTensor tokens: [N],
    we construct samples (x, y) where:
      x = tokens[i : i+seq_len]
      y = tokens[i+1 : i+1+seq_len]
    """

    def __init__(self, token_ids_1d: torch.Tensor, seq_len: int):
        super().__init__()
        assert token_ids_1d.dim() == 1, "token_ids_1d must be 1D LongTensor"
        self.tokens = token_ids_1d
        self.seq_len = seq_len
        # last index such that i + 1 + seq_len <= len(tokens)
        self._max_start = self.tokens.size(0) - (self.seq_len + 1)
        self._max_start = max(self._max_start, 0)

    def __len__(self) -> int:
        return self._max_start + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx > self._max_start:
            raise IndexError(idx)
        start = idx
        end_x = start + self.seq_len
        end_y = start + self.seq_len + 1
        x = self.tokens[start:end_x]       # [seq_len]
        y = self.tokens[start + 1:end_y]   # [seq_len]
        return x, y


def load_tokens(path: str) -> torch.Tensor:
    """
    Load 1D tokens from a .npy file and return as LongTensor.
    """
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D token array at {path}, got shape={arr.shape}")
    return torch.from_numpy(arr.astype("int64"))


def make_lm_dataloader(
    token_ids_1d: torch.Tensor,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    ds = LM1DDataset(token_ids_1d, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
