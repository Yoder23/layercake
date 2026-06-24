import torch
import pytest

from layercake.causal_byte_models import (
    CausalByteLM,
    CausalBytePatchLM,
    FusedModernCausalBlock,
    MixtureOfDepthRefinement,
    SparseStatePatchBlock,
)
from layercake.canonical_anchors import causal_byte_anchors, patch_context_anchors


def test_causal_models_shapes_and_patch_context_shift():
    x = torch.randint(0, 256, (2, 16))
    byte = CausalByteLM(d_model=32, d_abi=16, layers=1, heads=4, max_len=16)
    patch = CausalBytePatchLM(patch_size=4, d_byte=8, d_model=32, d_abi=16, layers=1, heads=4, max_patches=4)
    byte_logits, byte_abi = byte(x)
    patch_logits, patch_abi = patch(x)
    assert byte_logits.shape == (2, 16, 256)
    assert byte.boundary_abi(byte_abi, 4).shape == (2, 4, 16)
    assert patch_logits.shape == (2, 16, 256)
    assert patch_abi.shape == (2, 4, 16)
    x_changed = x.clone()
    x_changed[:, 4:] = torch.randint(0, 256, x_changed[:, 4:].shape)
    _, changed_abi = patch(x_changed)
    assert torch.equal(patch_abi[:, 0], changed_abi[:, 0])


def test_canonical_heads_and_anchors_are_seed_independent():
    torch.manual_seed(1)
    a = CausalByteLM(d_model=32, d_abi=16, layers=1, heads=4, max_len=16)
    torch.manual_seed(2)
    b = CausalByteLM(d_model=32, d_abi=16, layers=1, heads=4, max_len=16)
    assert torch.equal(a.canonical_head, b.canonical_head)
    x = torch.randint(0, 256, (2, 16))
    anchors = causal_byte_anchors(x, 16)
    patch_anchors = patch_context_anchors(x, 16, 4)
    assert anchors.shape == (2, 16, 16)
    assert patch_anchors.shape == (2, 4, 16)
    assert torch.equal(patch_anchors[:, 0], torch.zeros_like(patch_anchors[:, 0]))


def test_continuous_local_decoder_preserves_shapes():
    x = torch.randint(0, 256, (2, 16))
    model = CausalBytePatchLM(
        patch_size=4, d_byte=8, d_model=32, d_abi=16, layers=1,
        heads=4, max_patches=4, continuous_local=True
    )
    logits, abi = model(x)
    assert logits.shape == (2, 16, 256)
    assert abi.shape == (2, 4, 16)


def test_direct_global_context_preserves_abi_contract():
    model = CausalBytePatchLM(
        patch_size=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=4,
        continuous_local=True,
        direct_global_context=True,
    )
    x = torch.randint(0, 256, (2, 16))
    model.eval()
    logits, abi = model(x)
    assert logits.shape == (2, 16, 256)
    assert abi.shape == (2, 4, 16)


def test_causal_ngram_features_preserve_shapes():
    model = CausalBytePatchLM(
        patch_size=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=4,
        continuous_local=True,
        direct_global_context=True,
        ngram_buckets=64,
    )
    x = torch.randint(0, 256, (2, 16))
    model.eval()
    logits, abi = model(x)
    assert logits.shape == (2, 16, 256)
    assert abi.shape == (2, 4, 16)


def test_parallel_conv_decoder_is_strictly_causal():
    model = CausalBytePatchLM(
        patch_size=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=4,
        continuous_local=True,
        direct_global_context=True,
        local_decoder="conv",
        conv_layers=3,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, _ = model(x)
    changed_logits, _ = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])


def test_multi_byte_prediction_heads_are_training_only_outputs():
    model = CausalBytePatchLM(
        patch_size=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=4,
        continuous_local=True,
        mtp_depth=2,
    )
    model.eval()
    x = torch.randint(0, 256, (2, 16))
    logits, abi = model(x)
    logits_aux, abi_aux, auxiliary = model(x, return_aux=True)
    assert torch.equal(logits, logits_aux)
    assert torch.equal(abi, abi_aux)
    assert len(auxiliary) == 2
    assert auxiliary[0].shape == logits.shape


def test_transition_head_changes_next_byte_logits():
    transition = torch.zeros(256, 256)
    transition[ord("a"), ord("b")] = 5
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=4,
        continuous_local=True,
        transition_logits=transition,
    )
    assert model.transition_head.weight[ord("a"), ord("b")] == 5


def test_exact_byte_pair_units_preserve_tokenizer_free_shapes():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        continuous_local=True,
        patch_unit_buckets=65536,
    )
    x = torch.randint(0, 256, (2, 16))
    logits, abi = model(x)
    assert logits.shape == (2, 16, 256)
    assert abi.shape == (2, 8, 16)


def test_hierarchical_local_transformer_is_causal():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        continuous_local=True,
        direct_global_context=True,
        local_decoder="transformer",
        local_layers=1,
        local_position_embeddings=True,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, _ = model(x)
    changed_logits, _ = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    assert model.local_pos.num_embeddings == 16


def test_patch_local_transformer_preserves_shapes_and_causality():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="patch_transformer",
        local_layers=1,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, abi = model(x)
    changed_logits, _ = model(changed)
    assert logits.shape == (1, 16, 256)
    assert abi.shape == (1, 8, 16)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])


def test_factorized_patch_prediction_heads():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="transformer",
        local_layers=1,
        patch_prediction=True,
    )
    x = torch.randint(0, 256, (2, 16))
    output = model(
        x, return_aux=True, return_patch_prediction=True
    )
    logits, abi, auxiliary, patch_predictions = output
    assert logits.shape == (2, 16, 256)
    assert abi.shape == (2, 8, 16)
    assert auxiliary == []
    assert len(patch_predictions) == 2
    assert patch_predictions[0].shape == (2, 8, 256)


def test_autoregressive_patch_prediction_is_teacher_forced_causally():
    model = CausalBytePatchLM(
        patch_size=3,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="transformer",
        local_layers=1,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=12,
    )
    model.eval()
    x = torch.randint(0, 256, (2, 24))
    changed = x.clone()
    changed[:, 4] = (changed[:, 4] + 1) % 256
    original = model(
        x, return_aux=True, return_patch_prediction=True
    )[3]
    modified = model(
        changed, return_aux=True, return_patch_prediction=True
    )[3]
    # Source patch 0 predicts target patch 1. Changing target offset 1 cannot
    # affect logits for offsets 0 or 1, but does affect offset 2.
    assert torch.equal(original[0][:, 0], modified[0][:, 0])
    assert torch.equal(original[1][:, 0], modified[1][:, 0])
    assert not torch.equal(original[2][:, 0], modified[2][:, 0])


def test_autoregressive_patch_generation_shape_and_range():
    model = CausalBytePatchLM(
        patch_size=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=4,
        direct_global_context=True,
        local_decoder="transformer",
        local_layers=1,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=12,
    )
    model.eval()
    x = torch.randint(0, 256, (2, 16))
    output = model(
        x,
        return_aux=True,
        return_patch_prediction=True,
        return_generated_patch=True,
    )
    generated = output[4]
    assert generated.shape == (2, 4)
    assert generated.dtype == torch.long
    assert generated.min() >= 0
    assert generated.max() < 256
    fast_generated = model.generate_next_patch(x)
    assert torch.equal(generated, fast_generated)


def test_autoregressive_patch_context_has_one_window_per_source_patch():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="transformer",
        local_layers=1,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=12,
        patch_generation_context=4,
    )
    model.eval()
    x = torch.randint(0, 256, (2, 16))
    predictions = model(
        x, return_aux=True, return_patch_prediction=True
    )[3]
    prefixes = model._patch_generation_prefixes(x)
    assert len(predictions) == 2
    assert predictions[0].shape == (2, 8, 256)
    assert prefixes.shape == (2, 8, 4)
    assert torch.equal(prefixes[:, 0, -2:], x[:, :2])
    assert torch.equal(prefixes[:, 1], x[:, :4])
    assert torch.equal(prefixes[:, -1], x[:, -4:])
    assert model.generate_next_patch(x).shape == (2, 2)


def test_verified_patch_generation_uses_local_lm():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=12,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    generated = model.generate_verified_patch(x)
    assert generated.shape == (1, 2)
    assert generated.dtype == torch.long


def test_patch_generator_can_use_local_boundary_context():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_prediction_context="local",
        patch_generation_width=12,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    predictions = model(
        x, return_aux=True, return_patch_prediction=True
    )[3]
    assert predictions[0].shape == (1, 8, 256)
    assert model.generate_next_patch(x).shape == (1, 2)


def test_fused_block_cached_step_matches_full_forward_last_token():
    block = FusedModernCausalBlock(32, 4)
    block.eval()
    prefix = torch.randn(2, 5, 32)
    token = torch.randn(2, 1, 32)
    full = block(torch.cat([prefix, token], dim=1))
    _, cache = block.prefill_with_cache(prefix)
    decoded, _ = block.decode_with_cache(token, cache)
    assert torch.allclose(full[:, -1:], decoded, atol=1e-5, rtol=1e-5)


def test_cached_patch_generation_shape():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
        patch_prediction=True,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    generated = model.generate_cached_patch(x)
    assert generated.shape == (1, 2)
    state = model.begin_cached_generation(x[:, :8])
    first = model.cached_generation_step(state)
    forced = torch.randint(0, 256, (1, 2))
    second, logits = model.cached_generation_step(
        state, forced_patch=forced, return_logits=True
    )
    constrained = model.cached_generation_step(state, no_repeat_ngram=4)
    assert first.shape == (1, 2)
    assert second.shape == (1, 2)
    assert constrained.shape == (1, 2)
    assert torch.equal(second, forced)
    assert logits.shape == (1, 2, 256)
    assert state["bytes"].shape == (1, 14)


def test_sparse_state_patch_block_shape_grad_and_determinism():
    block = SparseStatePatchBlock(
        width=32,
        heads=4,
        local_window=4,
        dilated_offsets=(4, 6),
        chunk_size=4,
    )
    block.eval()
    h = torch.randn(2, 9, 32, requires_grad=True)
    first = block(h)
    second = block(h)
    assert first.shape == h.shape
    assert torch.allclose(first, second)
    first.sum().backward()
    assert h.grad is not None
    assert h.grad.abs().sum() > 0


def test_sparse_state_patch_block_is_causal():
    block = SparseStatePatchBlock(
        width=32,
        heads=4,
        local_window=4,
        dilated_offsets=(4, 6),
        chunk_size=4,
    )
    block.eval()
    h = torch.randn(1, 10, 32)
    changed = h.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:])
    original = block(h)
    modified = block(changed)
    assert torch.allclose(original[:, :7], modified[:, :7], atol=1e-5)


def test_sparse_state_patch_block_cache_matches_full_forward():
    block = SparseStatePatchBlock(
        width=32,
        heads=4,
        local_window=4,
        dilated_offsets=(4, 6),
        chunk_size=4,
    )
    block.eval()
    prefix = torch.randn(2, 7, 32)
    token = torch.randn(2, 1, 32)
    full = block(torch.cat([prefix, token], dim=1))
    _, cache = block.prefill_with_cache(prefix)
    decoded, _ = block.decode_with_cache(token, cache)
    assert torch.allclose(full[:, -1:], decoded, atol=1e-5, rtol=1e-5)


def test_sparse_state_patch_model_cached_generation_shape():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
        patch_prediction=True,
        global_block="sparse_state_patch",
        sparse_state_local_window=4,
        sparse_state_dilated_offsets=(4, 6),
        sparse_state_chunk_size=4,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 8))
    state = model.begin_cached_generation(x)
    generated = model.cached_generation_step(state, no_repeat_ngram=4)
    assert generated.shape == (1, 2)
    assert state["bytes"].shape == (1, 10)


def test_multiscale_coarse_context_is_causal():
    model = CausalBytePatchLM(
        patch_size=2,
        coarse_patch_size=4,
        coarse_layers=1,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, abi = model(x)
    changed_logits, changed_abi = model(changed)
    assert logits.shape == (1, 16, 256)
    assert abi.shape == (1, 8, 16)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    assert torch.equal(abi[:, :6], changed_abi[:, :6])


def test_multiscale_patch_configuration_validation():
    with pytest.raises(ValueError):
        CausalBytePatchLM(
            patch_size=2,
            coarse_patch_size=3,
            coarse_layers=1,
        )


def test_hybrid_global_convolution_is_causal():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=3,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
        global_conv_layers=2,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, abi = model(x)
    changed_logits, changed_abi = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    assert torch.equal(abi[:, :6], changed_abi[:, :6])


def test_global_convolution_count_is_validated():
    with pytest.raises(ValueError):
        CausalBytePatchLM(layers=2, global_conv_layers=3)


def test_hybrid_global_gru_is_causal():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=3,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
        global_gru_layers=1,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, abi = model(x)
    changed_logits, changed_abi = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    assert torch.equal(abi[:, :6], changed_abi[:, :6])


def test_wider_local_decoder_preserves_shapes_and_causality():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        local_width=48,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, abi = model(x)
    changed_logits, changed_abi = model(changed)
    assert logits.shape == (1, 16, 256)
    assert abi.shape == (1, 8, 16)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    assert torch.equal(abi[:, :6], changed_abi[:, :6])


def test_causal_patch_encoder_preserves_future_independence():
    model = CausalBytePatchLM(
        patch_size=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=4,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
        patch_encoder_layers=1,
        patch_encoder_window=4,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, abi = model(x)
    changed_logits, changed_abi = model(changed)
    assert logits.shape == (1, 16, 256)
    assert abi.shape == (1, 4, 16)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    assert torch.equal(abi[:, :3], changed_abi[:, :3])
    assert model.local_in.in_features == 8 + 32 + 32


def test_mixture_of_depth_has_fixed_capacity_and_is_deterministic():
    refinement = MixtureOfDepthRefinement(
        width=32,
        heads=4,
        layers=1,
        capacity_ratio=0.25,
        group_size=4,
    )
    refinement.eval()
    h = torch.randn(2, 12, 32)
    first = refinement.route_mask(h)
    second = refinement.route_mask(h)
    assert torch.equal(first, second)
    assert torch.equal(first.sum(dim=1), torch.tensor([3, 3]))


def test_mixture_of_depth_routing_cannot_see_future_groups():
    refinement = MixtureOfDepthRefinement(
        width=32,
        heads=4,
        layers=1,
        capacity_ratio=0.5,
        group_size=4,
    )
    refinement.eval()
    h = torch.randn(1, 12, 32)
    changed = h.clone()
    changed[:, 8:] = torch.randn_like(changed[:, 8:])
    original = refinement.route_mask(h)
    modified = refinement.route_mask(changed)
    assert torch.equal(original[:, :8], modified[:, :8])


def test_mixture_of_depth_can_share_refinement_weights():
    refinement = MixtureOfDepthRefinement(
        width=32,
        heads=4,
        layers=3,
        capacity_ratio=0.5,
        group_size=4,
        share_weights=True,
    )
    assert refinement.layers == 3
    assert len(refinement.blocks) == 1


def test_mixture_of_depth_model_is_causal():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
        mod_layers=1,
        mod_capacity=0.5,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, abi = model(x)
    changed_logits, changed_abi = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])
    assert torch.equal(abi[:, :6], changed_abi[:, :6])


def test_tied_byte_input_output_embeddings():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="transformer",
        local_layers=1,
        tie_byte_embeddings=True,
    )
    x = torch.randint(0, 256, (2, 16))
    logits, _ = model(x)
    assert logits.shape == (2, 16, 256)
    assert not hasattr(model, "head")


def test_hashed_byte_context_head_is_causal():
    context_logits = torch.zeros(32, 256)
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="transformer",
        local_layers=1,
        context_buckets=32,
        context_order=3,
        context_logits=context_logits,
    )
    x = torch.randint(0, 256, (1, 16))
    ids = model._context_ids(x)
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    changed_ids = model._context_ids(changed)
    assert torch.equal(ids[:, :12], changed_ids[:, :12])


def test_modern_swiglu_blocks_preserve_shapes():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="transformer",
        local_layers=1,
        modern_blocks=True,
    )
    x = torch.randint(0, 256, (2, 16))
    logits, abi = model(x)
    assert logits.shape == (2, 16, 256)
    assert abi.shape == (2, 8, 16)


def test_windowed_local_attention_is_causal():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, _ = model(x)
    changed_logits, _ = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])


def test_fused_modern_attention_is_causal():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    changed = x.clone()
    changed[:, 12:] = torch.randint(0, 256, (1, 4))
    logits, _ = model(x)
    changed_logits, _ = model(changed)
    assert torch.equal(logits[:, :12], changed_logits[:, :12])


def test_cached_generation_no_repeat_uses_tensor_mask():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=12,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
    )
    model.eval()
    x = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3]], dtype=torch.long)
    state = model.begin_cached_generation(x)
    logits = torch.full((1, 256), -100.0)
    logits[0, 4] = 100.0
    logits[0, 5] = 99.0
    state["next_logits"] = logits
    patch = model.cached_generation_step(state, no_repeat_ngram=4)
    assert patch.shape == (1, 2)
    assert patch[0, 0].item() == 5
