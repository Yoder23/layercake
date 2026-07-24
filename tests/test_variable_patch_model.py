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


def test_adaptive_completed_patch_context_is_causal_and_incremental() -> None:
    torch.manual_seed(19)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=2,
        local_layers=2,
        heads=4,
        max_patches=16,
        local_window=4,
        routed_experts=4,
        expert_expansion=1,
        routing_mode="learned_top1",
        use_completed_patch_context=True,
    ).eval()
    x = torch.tensor([list(b"ab cd123efghij  ")], dtype=torch.long)
    logits, _, metadata = model(x)
    changed = x.clone()
    changed[:, 12:] = torch.tensor([list(b"WXYZ")])
    changed_logits, _, _ = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    state = model.prefill_incremental(x)
    assert torch.allclose(
        state["next_logits"], logits[:, -1], atol=3e-6, rtol=0.0
    )
    assert state["patch_count"] == int(metadata["valid_patches"].sum())


def test_adaptive_whitespace_any_boundaries_close_causally() -> None:
    torch.manual_seed(21)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=2,
        heads=4,
        max_patches=16,
        local_window=4,
        local_context_mode="sliding",
        patch_boundary_mode="whitespace_any",
        use_completed_patch_context=True,
        word_context_buckets=128,
        word_context_dim=8,
        word_context_order=2,
        word_context_whitespace_only=True,
        word_recurrent_buckets=128,
        word_recurrent_embedding_dim=8,
        word_recurrent_hidden_dim=16,
    ).eval()
    x = torch.tensor([list(b"a bc def ghi jkl")], dtype=torch.long)
    logits, _, metadata = model(x)
    lengths = metadata["patch_lengths"][metadata["valid_patches"]]
    assert set(lengths.tolist()) <= {1, 2, 3, 4}
    whitespace = (x == 32) | (x == 9) | (x == 10) | (x == 13)
    current_lengths = metadata["patch_lengths"].gather(
        1, metadata["patch_ids"]
    )
    assert torch.all(
        metadata["patch_offsets"][whitespace]
        == current_lengths[whitespace] - 1
    )
    state = model.prefill_incremental(x)
    assert torch.allclose(
        state["next_logits"], logits[:, -1], atol=3e-6, rtol=0.0
    )
    model.train()
    model(x)[0].sum().backward()
    assert model.word_context_emb.weight.grad is not None
    assert model.word_context_proj.weight.grad is not None
    assert model.word_recurrent_emb.weight.grad is not None
    assert model.word_recurrent_cell.weight_ih.grad is not None
    assert model.word_recurrent_proj.weight.grad is not None


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


def test_adaptive_low_rank_higher_context_is_causal_trainable_and_incremental() -> None:
    torch.manual_seed(29)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=16,
        local_window=4,
        routed_experts=4,
        expert_expansion=1,
        routing_mode="learned_top1",
        higher_context_buckets=128,
        higher_context_dim=8,
        higher_context_order=4,
    ).eval()
    x = torch.tensor([list(b"ab cd123efghijkl")], dtype=torch.long)
    changed = x.clone()
    changed[:, 12:] = torch.tensor([list(b"WXYZ")])
    full_logits, _, _ = model(x)
    changed_logits, _, _ = model(changed)
    assert torch.allclose(full_logits[:, :12], changed_logits[:, :12], atol=1e-6, rtol=0.0)
    state = model.prefill_incremental(x)
    assert torch.allclose(state["next_logits"], full_logits[:, -1], atol=2e-6, rtol=0.0)
    model.train()
    model(x)[0].sum().backward()
    assert model.higher_context_emb.weight.grad is not None
    assert model.higher_context_proj.weight.grad is not None


def test_adaptive_future_byte_heads_are_training_only_and_receive_gradients() -> None:
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=8,
        local_window=4,
        future_byte_horizons=(2, 4),
    )
    x = torch.tensor([list(b"abcdefghijklmnop")], dtype=torch.long)
    _, _, metadata = model(x)
    assert set(metadata["future_byte_logits"]) == {2, 4}
    sum(value.sum() for value in metadata["future_byte_logits"].values()).backward()
    assert all(head.weight.grad is not None for head in model.future_heads.values())
    model.eval()
    state = model.prefill_incremental(x)
    assert state["next_logits"].shape == (1, 256)
    assert set(state["next_future_logits"]) == {2, 4}


def test_adaptive_two_byte_state_update_matches_two_single_updates() -> None:
    torch.manual_seed(31)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=2,
        local_layers=2,
        heads=4,
        max_patches=16,
        local_window=4,
        routed_experts=4,
        expert_expansion=1,
        routing_mode="learned_top1",
        future_byte_horizons=(2,),
        conditional_future_transition=True,
    ).eval()
    prompt = torch.tensor([list(b"abcdefghijklmnop")], dtype=torch.long)
    sequential = model.prefill_incremental(prompt)
    blocked = model.prefill_incremental(prompt)
    values = torch.tensor([ord("q"), ord("r")], dtype=torch.long)
    model.incremental_step(sequential, values[0:1])
    model.incremental_step(sequential, values[1:2])
    model.incremental_step_many(blocked, values)
    assert sequential["byte_count"] == blocked["byte_count"]
    assert sequential["patch_count"] == blocked["patch_count"]
    assert len(sequential["pending_bytes"]) == len(blocked["pending_bytes"])
    assert sequential["routed_expert_trace"] == blocked["routed_expert_trace"]
    assert torch.allclose(
        sequential["next_logits"], blocked["next_logits"], atol=3e-6, rtol=0.0
    )
    assert torch.allclose(
        sequential["next_future_logits"][2],
        blocked["next_future_logits"][2],
        atol=3e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        sequential["last_global"], blocked["last_global"], atol=3e-6, rtol=0.0
    )
    for sequential_cache, blocked_cache in zip(
        sequential["local_caches"], blocked["local_caches"]
    ):
        assert torch.allclose(
            sequential_cache[0], blocked_cache[0], atol=3e-6, rtol=0.0
        )
        assert torch.allclose(
            sequential_cache[1], blocked_cache[1], atol=3e-6, rtol=0.0
        )
    first = torch.tensor([ord("s")], dtype=torch.long)
    assert torch.allclose(
        model.proposed_future_logits(blocked, 2, first),
        blocked["next_future_logits"][2] + model.transition_head(first),
        atol=0.0,
        rtol=0.0,
    )


def test_adaptive_incremental_patch_positions_wrap_after_trained_table() -> None:
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=2,
        local_window=4,
        local_context_mode="sliding",
        patch_boundary_mode="whitespace_any",
        use_completed_patch_context=True,
    ).eval()
    state = model.init_incremental_state()
    for value in b"a b c d e f ":
        model.incremental_step(
            state, torch.tensor([value], dtype=torch.long)
        )
    assert state["patch_count"] > model.patch_pos.num_embeddings
    assert state["next_logits"].shape == (1, 256)


def test_adaptive_full_forward_patch_positions_wrap_after_trained_table() -> None:
    torch.manual_seed(37)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=2,
        local_window=4,
        local_context_mode="sliding",
        patch_boundary_mode="whitespace_any",
        use_completed_patch_context=True,
    ).eval()
    x = torch.tensor([list(b"a b c d ")], dtype=torch.long)
    full_logits, _, metadata = model(x)
    assert int(metadata["valid_patches"].sum()) > model.patch_pos.num_embeddings
    state = model.prefill_incremental(x)
    assert torch.allclose(
        state["next_logits"], full_logits[:, -1], atol=3e-6, rtol=0.0
    )


def test_adaptive_prompt_memory_boundary_is_incremental_and_trainable() -> None:
    torch.manual_seed(41)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=16,
        local_window=4,
        local_context_mode="sliding",
        patch_boundary_mode="whitespace_any",
        use_completed_patch_context=True,
        prompt_memory_adapter_dim=16,
    ).eval()
    prompt = torch.tensor([list(b"raw prompt..")], dtype=torch.long)
    boundary = torch.tensor([prompt.shape[1] - 1])
    full_logits, _, metadata = model(
        prompt, prompt_boundary_indexes=boundary
    )
    state = model.prefill_incremental(prompt)
    assert state["prompt_memory_active"]
    assert not state["pending_bytes"]
    assert metadata["patch_ids"][0, -1] == metadata["valid_patches"].sum() - 1
    assert torch.allclose(
        state["next_logits"], full_logits[:, -1], atol=3e-6, rtol=0.0
    )
    model.train()
    model.zero_grad(set_to_none=True)
    model(prompt, prompt_boundary_indexes=boundary)[0].sum().backward()
    assert model.prompt_memory_adapter[-1].weight.grad is not None


def test_adaptive_prompt_cross_attention_is_incremental_and_trainable() -> None:
    torch.manual_seed(43)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=16,
        local_window=4,
        local_context_mode="sliding",
        patch_boundary_mode="whitespace_any",
        use_completed_patch_context=True,
        prompt_cross_attention_dim=8,
    ).eval()
    torch.nn.init.normal_(model.prompt_cross_out.weight, std=0.01)
    prompt = torch.tensor([list(b"raw prompt..")], dtype=torch.long)
    boundary = torch.tensor([prompt.shape[1] - 1])
    full_logits, _, _ = model(
        prompt, prompt_boundary_indexes=boundary
    )
    state = model.prefill_incremental(prompt)
    assert state["prompt_memory_active"]
    assert state["prompt_cross_keys"].shape == (1, prompt.shape[1], 8)
    assert torch.allclose(
        state["next_logits"], full_logits[:, -1], atol=4e-6, rtol=0.0
    )
    model.train()
    model.zero_grad(set_to_none=True)
    model(prompt, prompt_boundary_indexes=boundary)[0].sum().backward()
    assert model.prompt_cross_out.weight.grad is not None
    assert model.prompt_cross_query.weight.grad is not None


def test_adaptive_prompt_pointer_is_incremental_and_trainable() -> None:
    torch.manual_seed(47)
    model = CausalAdaptiveBytePatchLM(
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        local_layers=1,
        heads=4,
        max_patches=16,
        local_window=4,
        local_context_mode="sliding",
        patch_boundary_mode="whitespace_any",
        use_completed_patch_context=True,
        prompt_pointer_dim=8,
    ).eval()
    prompt = torch.tensor([list(b"raw prompt..")], dtype=torch.long)
    boundary = torch.tensor([prompt.shape[1] - 1])
    full_logits, _, metadata = model(
        prompt, prompt_boundary_indexes=boundary
    )
    state = model.prefill_incremental(prompt)
    assert state["prompt_pointer_keys"].shape == (1, prompt.shape[1], 8)
    assert torch.allclose(
        state["next_logits"], full_logits[:, -1], atol=4e-6, rtol=0.0
    )
    assert torch.allclose(
        state["next_logits"].float().exp().sum(dim=-1),
        torch.ones(1),
        atol=2e-6,
        rtol=0.0,
    )
    assert metadata["prompt_pointer_mean_gate"] is not None
    model.train()
    model.zero_grad(set_to_none=True)
    model(prompt, prompt_boundary_indexes=boundary)[0].sum().backward()
    assert model.prompt_pointer_gate.weight.grad is not None
    assert model.prompt_pointer_query.weight.grad is not None
