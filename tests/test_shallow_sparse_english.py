from __future__ import annotations

import torch

from layercake.models.shallow_sparse_english import (
    ShallowSparseEnglishConfig,
    ShallowSparseEnglishCore,
)


def tiny_config() -> ShallowSparseEnglishConfig:
    return ShallowSparseEnglishConfig(
        vocab_size=128,
        width=48,
        layers=3,
        heads=4,
        max_tokens=64,
        task_cakes=10,
        task_cake_rank=64,
    )


def test_exactly_one_task_cake_is_called_for_batch_one():
    model = ShallowSparseEnglishCore(tiny_config()).eval()
    tokens = torch.randint(0, 128, (1, 8))
    result = model(tokens, task_routes=torch.tensor([4]))
    assert result["logits"].shape == (1, 8, 128)
    assert model.last_cake_calls == (4,)


def test_incremental_state_keeps_route_and_grows_cache():
    model = ShallowSparseEnglishCore(tiny_config()).eval()
    tokens = torch.randint(0, 128, (1, 8))
    state = model.prefill(tokens)
    route = state["task_routes"].clone()
    _, state = model.decode_step(state)
    assert torch.equal(state["task_routes"], route)
    assert state["generated_ids"].shape == (1, 1)
    assert model.last_cake_calls == (int(route.item()),)


def test_active_parameter_count_excludes_nine_inactive_cakes():
    model = ShallowSparseEnglishCore(tiny_config())
    one = sum(
        parameter.numel() for parameter in model.task_cakes[0].parameters()
    )
    assert model.active_parameter_count() == model.parameter_count() - 9 * one
