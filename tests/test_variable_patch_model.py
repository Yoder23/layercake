import torch

from layercake.causal_byte_models import (
    CausalAdaptiveBytePatchLM,
    CausalVariableBytePatchLM,
)


def test_variable_patch_model_shapes_and_compression():
    model = CausalVariableBytePatchLM(
        max_patch_size=8,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
    )
    x = torch.tensor([list(b"hello world\n123")], dtype=torch.long)
    logits, abi, metadata = model(x)
    assert logits.shape == (1, 15, 256)
    assert abi.shape[1] == metadata["valid_patches"].sum().item()
    assert metadata["patch_lengths"].sum().item() == 15


def test_variable_patch_model_is_causal_before_changed_patch():
    model = CausalVariableBytePatchLM(
        max_patch_size=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
    )
    model.eval()
    x = torch.tensor([list(b"abc def ghi jkl")], dtype=torch.long)
    changed = x.clone()
    changed[:, 8:] = torch.tensor([list(b"XYZ 123")], dtype=torch.long)
    logits, _, _ = model(x)
    changed_logits, _, _ = model(changed)
    assert torch.equal(logits[:, :8], changed_logits[:, :8])


def test_transition_difficulty_table_adds_boundaries():
    table = torch.zeros(65536, dtype=torch.bool)
    table[ord("a") * 256 + ord("b")] = True
    model = CausalVariableBytePatchLM(
        max_patch_size=8,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
        transition_boundary_table=table,
    )
    x = torch.tensor([list(b"zabcz")], dtype=torch.long)
    _, _, metadata = model(x)
    assert metadata["valid_patches"].sum().item() == 2


def test_variable_patch_batched_layout_is_row_independent():
    model = CausalVariableBytePatchLM(
        max_patch_size=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
    )
    rows = torch.tensor([list(b"ab cd123"), list(b"abcdefgh")], dtype=torch.long)
    patch_ids, offsets, lengths, valid = model._layout(rows)
    assert patch_ids[0].tolist() == [0, 0, 0, 1, 1, 1, 1, 2]
    assert offsets[0].tolist() == [0, 1, 2, 0, 1, 2, 3, 0]
    assert lengths[0, valid[0]].tolist() == [3, 4, 1]
    assert patch_ids[1].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert lengths[1, valid[1]].tolist() == [4, 4]


def test_adaptive_two_four_layout_is_causal_and_compressed():
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=8,
        local_window=4,
    )
    model.eval()
    x = torch.tensor([list(b"abcd efghijklmno")], dtype=torch.long)
    logits, abi, metadata = model(x)
    assert logits.shape == (1, 16, 256)
    assert metadata["patch_lengths"].sum().item() == 16
    lengths = metadata["patch_lengths"][metadata["valid_patches"]]
    assert set(lengths.tolist()) <= {2, 4}
    assert 16 / metadata["valid_patches"].sum().item() > 2
    changed = x.clone()
    changed[:, 12:] = torch.tensor([list(b"WXYZ")])
    changed_logits, changed_abi, _ = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    assert torch.equal(abi[:, :3], changed_abi[:, :3])


def test_adaptive_hashed_transition_context_is_causal_and_trainable():
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=8,
        local_window=4,
        transition_context_buckets=64,
    )
    x = torch.tensor([list(b"abcdefghijklmnop")], dtype=torch.long)
    changed = x.clone()
    changed[:, 12:] = torch.tensor([list(b"WXYZ")])
    logits, _, _ = model(x)
    changed_logits, _, _ = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    logits.sum().backward()
    assert model.transition_context_head.weight.grad is not None


def test_adaptive_incremental_state_matches_full_forward_and_persists() -> None:
    torch.manual_seed(17)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=2,
        local_layers=2,
        heads=4,
        max_patches=16,
        local_window=4,
        transition_context_buckets=64,
        routed_experts=4,
        expert_expansion=1,
        routing_mode="learned_top1",
    ).eval()
    x = torch.tensor([list(b"ab cd123efghijkl")], dtype=torch.long)
    full_logits, _, metadata = model(x)
    state = model.prefill_incremental(x)
    assert torch.allclose(state["next_logits"], full_logits[:, -1], atol=2e-6, rtol=0.0)
    assert state["byte_count"] == x.shape[1]
    assert state["patch_count"] == int(metadata["valid_patches"].sum())
    previous_cache_length = state["global_caches"][0][0].shape[2]
    model.incremental_step(state, state["next_logits"].argmax(dim=-1))
    assert state["byte_count"] == x.shape[1] + 1
    assert state["global_caches"][0][0].shape[2] >= previous_cache_length


def test_adaptive_incremental_router_physically_executes_only_selected_experts() -> None:
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=8,
        local_window=4,
        routed_experts=4,
        expert_expansion=1,
        routing_mode="learned_top1",
    ).eval()
    model.routed.set_route(2)
    calls = [0, 0, 0, 0]
    handles = [
        expert.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(index, calls[index] + 1)
        )
        for index, expert in enumerate(model.routed.experts)
    ]
    try:
        state = model.prefill_incremental(
            torch.tensor([list(b"abcdefghijklmnop")], dtype=torch.long)
        )
    finally:
        for handle in handles:
            handle.remove()
    assert sum(calls) == state["patch_count"]
    assert sum(value > 0 for value in calls) < len(calls)
    assert len(state["routed_expert_trace"]) == state["patch_count"]


def test_adaptive_gru_local_mixer_incremental_parity() -> None:
    torch.manual_seed(23)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=2,
        heads=4,
        max_patches=16,
        local_window=4,
        routed_experts=4,
        expert_expansion=1,
        routing_mode="learned_top1",
        local_mixer="gru",
    ).eval()
    x = torch.tensor([list(b"ab cd123efghijkl")], dtype=torch.long)
    full_logits, _, _ = model(x)
    state = model.prefill_incremental(x)
    assert torch.allclose(state["next_logits"], full_logits[:, -1], atol=2e-6, rtol=0.0)
