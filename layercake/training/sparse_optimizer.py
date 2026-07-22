"""Measured sparse optimizer construction and resident-state accounting."""

from __future__ import annotations

import torch

from layercake.models.foundation_v2 import LayerCakeFoundationV2


def sparse_adamw(
    model: LayerCakeFoundationV2,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    # AdamW allocates moment tensors lazily. Inactive experts have no gradient,
    # so they consume neither updates nor optimizer tensors until selected.
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def optimizer_state_report(optimizer: torch.optim.Optimizer) -> dict:
    tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    return {
        "resident_tensor_elements": sum(tensor.numel() for tensor in tensors),
        "resident_tensor_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors),
        "parameters_with_state": sum(bool(state) for state in optimizer.state.values()),
    }

