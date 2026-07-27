from __future__ import annotations

import torch

from layercake.domain_runtime import AttentiveHostResidualCake


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
