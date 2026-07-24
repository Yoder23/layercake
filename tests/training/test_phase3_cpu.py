from __future__ import annotations

import torch

from layercake.training.phase3_cpu import (
    TrainingOnlyHorizonHeads,
    _estimated_training_operations,
    _sampled_multihorizon_loss,
)


def test_sampled_objective_keeps_vocabulary_gradient_sparse() -> None:
    torch.manual_seed(3)
    weight = torch.nn.Parameter(torch.randn(32, 8))
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    targets = {
        1: torch.tensor([[1, 2, 3, 4]]),
        2: torch.tensor([[2, 3, 4, -100]]),
    }
    heads = TrainingOnlyHorizonHeads(8, (1, 2))
    loss, metrics = _sampled_multihorizon_loss(
        hidden,
        targets,
        weight,
        heads,
        negatives=8,
        generator=torch.Generator().manual_seed(5),
    )
    loss.backward()
    assert weight.grad is not None
    assert weight.grad.is_sparse
    assert metrics["supervised_target_units"] == 7


def test_three_block_operation_count_is_below_six_block_control() -> None:
    common = {
        "batch": 1,
        "sequence": 64,
        "width": 768,
        "candidates": 2200,
        "horizons": (1, 2, 4),
    }
    layercake = _estimated_training_operations(
        layers=3, active_cake=True, **common
    )
    transformer = _estimated_training_operations(
        layers=6, active_cake=False, **common
    )
    assert layercake < transformer
    assert layercake / transformer < 0.75
