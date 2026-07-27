from __future__ import annotations

import torch

from layercake.domain_runtime import RecurrentHostResidualCake


def test_recurrent_host_residual_preserves_abi_shape_and_state():
    cake = RecurrentHostResidualCake(d_abi=16, hidden_width=24)
    semantic = torch.randn(2, 5, 16)
    adapted, state = cake(semantic)
    assert adapted.shape == semantic.shape
    assert state.shape == (1, 2, 24)
    step, next_state = cake(torch.randn(2, 16), state)
    assert step.shape == (2, 16)
    assert next_state.shape == state.shape


def test_recurrent_host_residual_starts_as_identity():
    cake = RecurrentHostResidualCake(d_abi=8, hidden_width=12)
    semantic = torch.randn(1, 3, 8)
    adapted, _ = cake(semantic)
    torch.testing.assert_close(adapted, semantic)
