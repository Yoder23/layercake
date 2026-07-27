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
