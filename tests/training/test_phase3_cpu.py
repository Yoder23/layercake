from __future__ import annotations

import torch

from layercake.training.phase3_cpu import (
    TrainingOnlyHorizonHeads,
    _chunked_exact_next_token_loss,
    _estimated_training_operations,
    _sampled_multihorizon_loss,
    _sparse_sgd_step,
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


def test_chunked_exact_softmax_matches_materialized_cross_entropy() -> None:
    torch.manual_seed(7)
    weight = torch.nn.Parameter(torch.randn(19, 8))
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    targets = torch.tensor(
        [[1, 2, 3, 4, 5], [6, 7, -100, 8, 9]],
        dtype=torch.long,
    )
    loss, metrics = _chunked_exact_next_token_loss(
        hidden,
        targets,
        weight,
        maximum_positions=32,
        vocabulary_chunk_rows=4,
    )
    valid = targets >= 0
    expected = torch.nn.functional.cross_entropy(
        hidden[valid] @ weight.transpose(0, 1),
        targets[valid],
    )
    torch.testing.assert_close(loss, expected)
    assert metrics["exact_softmax_positions"] == 9
    assert metrics["exact_softmax_chunks"] == 5


def test_exact_softmax_uses_stateless_dense_vocabulary_update() -> None:
    torch.manual_seed(11)
    weight = torch.nn.Parameter(torch.randn(13, 6))
    hidden = torch.randn(1, 4, 6, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4]])
    loss, _ = _chunked_exact_next_token_loss(
        hidden,
        targets,
        weight,
        maximum_positions=4,
        vocabulary_chunk_rows=5,
    )
    loss.backward()
    assert weight.grad is not None
    assert not weight.grad.is_sparse
    before = weight.detach().clone()
    rows = _sparse_sgd_step(weight, 0.01)
    assert rows == weight.shape[0]
    assert weight.grad is None
    assert not torch.equal(before, weight)
