from __future__ import annotations

import torch

from layercake.domain_runtime import AttentiveHostResidualCake
from layercake.training.phase4_python_cake import (
    _configure_copy_path_only,
    _tensor_subset_sha256,
)


def test_attentive_residual_prefill_and_step_preserve_abi_shape():
    cake = AttentiveHostResidualCake(
        d_abi=16, hidden_width=24, layers=1, heads=4, expansion=2
    )
    semantic = torch.randn(2, 5, 16)
    adapted, caches = cake(semantic)
    assert adapted.shape == semantic.shape
    assert len(caches) == 1
    assert caches[0].shape == (2, 5, 24)
    step, next_caches = cake.step(torch.randn(2, 16), caches)
    assert step.shape == (2, 16)
    assert next_caches[0].shape == (2, 6, 24)


def test_attentive_residual_starts_as_identity():
    cake = AttentiveHostResidualCake(
        d_abi=8, hidden_width=12, layers=1, heads=3, expansion=2
    )
    semantic = torch.randn(1, 4, 8)
    adapted, _ = cake(semantic)
    torch.testing.assert_close(adapted, semantic)


def test_attentive_copy_path_retains_raw_abi_history_incrementally():
    cake = AttentiveHostResidualCake(
        d_abi=8,
        hidden_width=12,
        layers=1,
        heads=3,
        expansion=2,
        copy_width=4,
    )
    semantic = torch.randn(1, 4, 8)
    adapted, caches = cake(semantic)
    torch.testing.assert_close(adapted, semantic)
    assert len(caches) == 2
    assert caches[-1].shape == semantic.shape
    step, next_caches = cake.step(torch.randn(1, 8), caches)
    assert step.shape == (1, 8)
    assert next_caches[-1].shape == (1, 5, 8)
    scores = cake.copy_scores(semantic)
    assert scores.shape == (1, 4, 4)
    assert scores[0, 0, 1] < -1.0e20


def test_copy_value_projection_is_identity_initialized():
    cake = AttentiveHostResidualCake(
        d_abi=8,
        hidden_width=12,
        layers=1,
        heads=3,
        expansion=2,
        copy_width=4,
        copy_value_projection=True,
    )
    expected = torch.eye(8)
    torch.testing.assert_close(cake.copy_value.weight, expected)


def test_selective_copy_starts_closed_and_uses_hard_value_read_in_eval():
    cake = AttentiveHostResidualCake(
        d_abi=8,
        hidden_width=12,
        layers=1,
        heads=3,
        expansion=2,
        copy_width=4,
        copy_value_projection=True,
        selective_copy=True,
    ).eval()
    assert cake.copy_gate.bias.item() == -4.0
    assert torch.count_nonzero(cake.copy_gate.weight) == 0
    semantic = torch.randn(1, 4, 8)
    context = cake._copy_context(semantic)
    assert context is not None
    for position in range(context.shape[1]):
        assert any(
            torch.equal(context[0, position], semantic[0, source])
            for source in range(position + 1)
        )


def test_transition_copy_starts_as_current_state_projection():
    cake = AttentiveHostResidualCake(
        d_abi=8,
        hidden_width=12,
        layers=1,
        heads=3,
        expansion=2,
        copy_width=4,
        copy_value_projection=True,
        selective_copy=True,
        transition_copy=True,
    ).eval()
    semantic = torch.randn(1, 4, 8)
    projected = cake.project_copy_positions(
        semantic,
        torch.tensor([0, 0]),
        torch.tensor([1, 3]),
    )
    torch.testing.assert_close(
        projected, semantic[0, torch.tensor([1, 3])]
    )


def test_copy_path_only_freezes_semantic_parent():
    cake = AttentiveHostResidualCake(
        d_abi=768,
        hidden_width=384,
        layers=1,
        heads=6,
        copy_width=64,
        copy_value_projection=True,
        selective_copy=True,
        transition_copy=True,
    )
    trainable = set(_configure_copy_path_only(cake))
    assert trainable == {
        "copy_query.weight",
        "copy_key.weight",
        "copy_gate.weight",
        "copy_gate.bias",
        "copy_transition_value.weight",
    }
    assert all(
        parameter.requires_grad == (name in trainable)
        for name, parameter in cake.named_parameters()
    )


def test_frozen_subset_hash_supports_scalar_parameters():
    state = {"alpha": torch.tensor(0.0), "weight": torch.eye(2)}
    first = _tensor_subset_sha256(state, set(state))
    second = _tensor_subset_sha256(state, set(state))
    assert first == second
