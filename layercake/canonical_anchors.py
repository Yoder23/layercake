"""Deterministic causal anchors shared across interfaces, sizes, and seeds."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def canonical_byte_table(d_abi: int, device=None, dtype=torch.float32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(8675309)
    table = torch.randn(256, d_abi, generator=generator)
    table = F.normalize(table, dim=-1)
    return table.to(device=device, dtype=dtype)


def causal_byte_anchors(
    byte_ids: torch.Tensor, d_abi: int, decay: float = 0.875
) -> torch.Tensor:
    """Return a deterministic normalized prefix state after every observed byte."""
    table = canonical_byte_table(d_abi, byte_ids.device)
    embedded = table[byte_ids]
    state = embedded.new_zeros(byte_ids.shape[0], d_abi)
    anchors = []
    for index in range(byte_ids.shape[1]):
        state = decay * state + embedded[:, index]
        anchors.append(F.layer_norm(state, (d_abi,)))
    return torch.stack(anchors, dim=1)


def patch_context_anchors(
    byte_ids: torch.Tensor, d_abi: int, patch_size: int
) -> torch.Tensor:
    """BOS plus prefix anchors after each completed patch except the current one."""
    anchors = causal_byte_anchors(byte_ids, d_abi)
    usable = byte_ids.shape[1] // patch_size * patch_size
    completed = anchors[:, patch_size - 1 : usable : patch_size]
    bos = anchors.new_zeros(byte_ids.shape[0], 1, d_abi)
    return torch.cat([bos, completed[:, :-1]], dim=1)
