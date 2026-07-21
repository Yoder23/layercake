import torch
import pytest

from layercake.causal_byte_models import (
    AutoregressivePatchHead,
    CausalByteLM,
    CausalBytePatchLM,
    FusedModernCausalBlock,
    MixtureOfDepthRefinement,
    SelectiveStatePatchBlock,
    SparseStatePatchBlock,
    Top1RoutedCakeBlock,
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


def test_trainable_prior_gates_receive_gradients():
    transition_logits = torch.randn(256, 256) * 0.01
    context_logits = torch.randn(64, 256) * 0.01
    model = CausalBytePatchLM(
        patch_size=1,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        local_window=8,
        context_buckets=64,
        context_order=2,
        transition_logits=transition_logits,
        context_logits=context_logits,
        transition_logit_scale=0.35,
        context_logit_scale=0.65,
        trainable_prior_gates=True,
        dynamic_prior_gates=True,
        prior_dropout=0.1,
    )
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss.backward()
    assert model.prior_gate.weight.grad is not None
    assert model.prior_gate.bias.grad is not None


def test_repeat_suppression_bias_is_causal_and_trainable():
    model = CausalBytePatchLM(
        patch_size=1,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=1,
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        local_window=8,
        repeat_suppression_window=4,
        repeat_suppression_scale=0.1,
        trainable_repeat_suppression=True,
    )
    x = torch.tensor([[1, 2, 1, 3, 1, 4, 5, 6]], dtype=torch.long)
    bias = model._repeat_suppression_bias(x)
    assert bias is not None
    assert bias.shape == (1, 8, 256)
    assert bias[0, 4, 1] < bias[0, 4, 2]
    logits, _ = model(x)
    loss = logits.square().mean()
    loss.backward()
    assert model.repeat_suppression_log_scale.grad is not None
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


def test_cached_patch_prediction_matches_full_recompute_loop():
    torch.manual_seed(123)
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
        continuous_local=True,
        direct_global_context=True,
        modern_blocks=True,
        fused_attention=True,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=24,
        patch_prediction_context="global",
    )
    model.eval()
    prompt = torch.randint(0, 256, (1, 12))

    full_context = prompt.clone()
    full_patches = []
    for _ in range(4):
        patch = model.generate_next_patch(full_context)
        full_patches.append(patch)
        full_context = torch.cat([full_context, patch], dim=1)

    state = model.begin_patch_prediction_cached_generation(prompt)
    cached = model.cached_patch_prediction_steps(state, 4)

    assert torch.equal(cached, torch.cat(full_patches, dim=1))


def test_autoregressive_span_prediction_and_cached_generation():
    torch.manual_seed(124)
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
        continuous_local=True,
        direct_global_context=True,
        modern_blocks=True,
        fused_attention=True,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=24,
        patch_generation_bytes=6,
        patch_prediction_context="global",
    )
    model.eval()
    prompt = torch.randint(0, 256, (1, 12))
    targets = model.patch_prediction_targets(prompt)
    predictions = model(
        prompt,
        return_aux=True,
        return_patch_prediction=True,
    )[3]

    assert targets.shape == (1, 6, 6)
    assert torch.equal(targets[:, 0], prompt[:, 2:8])
    assert len(predictions) == 6

    first = model.generate_next_patch(prompt)
    full_second = model.generate_next_patch(torch.cat([prompt, first], dim=1))
    state = model.begin_patch_prediction_cached_generation(prompt)
    cached_first = model.cached_patch_prediction_step(state)
    cached_second = model.cached_patch_prediction_step(state)

    assert first.shape == (1, 6)
    assert torch.equal(cached_first, first)
    assert torch.equal(cached_second, full_second)
    assert state["patch_count"] == prompt.shape[1] // 2 + 6


def test_patch_generation_span_requires_complete_model_patches():
    with pytest.raises(ValueError, match="multiple of patch_size"):
        CausalBytePatchLM(patch_size=4, patch_generation_bytes=6)


def test_fused_patch_teacher_forcing_matches_gru_cell_recurrence():
    torch.manual_seed(125)
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=12,
        patch_generation_bytes=6,
    )
    head = model.patch_generator
    context = torch.randn(2, 3, 32)
    target = torch.randint(0, 256, (2, 3, 6))

    fused = torch.stack(head(context, target), dim=-2)
    hidden = torch.tanh(head.initial_state(context))
    decoder_input = head.bos.expand(*context.shape[:-1], -1)
    recurrent = []
    for offset in range(6):
        hidden = head.cell(
            decoder_input.reshape(-1, decoder_input.shape[-1]),
            hidden.reshape(-1, hidden.shape[-1]),
        ).reshape_as(hidden)
        recurrent.append(head.output(hidden))
        decoder_input = head.byte_embedding(target[..., offset])

    assert torch.allclose(fused, torch.stack(recurrent, dim=-2))


def test_patch_prediction_can_gather_one_context_per_batch_row():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=12,
        patch_generation_bytes=6,
    )
    x = torch.randint(0, 256, (2, 16))
    predictions = model(
        x,
        return_aux=True,
        return_patch_prediction=True,
        patch_prediction_context_indices=torch.tensor([2, 5]),
    )[3]

    assert len(predictions) == 6
    assert predictions[0].shape == (2, 1, 256)


def test_position_aware_copy_normalizes_short_sources():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=16,
        patch_generation_bytes=6,
        patch_generation_copy_window=8,
        patch_generation_copy_dim=8,
        patch_generation_position_copy=True,
    )
    source = torch.randint(0, 256, (2, 5))
    normalized = model.patch_generator._normalized_copy_source(source)

    assert normalized.shape == (2, 8)
    assert torch.equal(normalized[:, -5:], source)
    assert model.generate_next_patch(torch.randint(0, 256, (2, 8))).shape == (2, 6)


def test_contextual_copy_keys_receive_gradients():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=16,
        patch_generation_bytes=6,
        patch_generation_copy_window=8,
        patch_generation_copy_dim=8,
        patch_generation_position_copy=True,
        patch_generation_contextual_copy=True,
    )
    x = torch.randint(0, 256, (2, 16))
    predictions = model(
        x,
        return_aux=True,
        return_patch_prediction=True,
    )[3]
    sum(prediction.mean() for prediction in predictions).backward()

    assert model.patch_generator.copy_next_key.weight.grad is not None
    assert model.patch_generator.copy_next2_key.weight.grad is not None


def test_lowercase_copy_maps_uppercase_source_mass():
    embedding = torch.nn.Embedding(256, 8)
    head = AutoregressivePatchHead(
        context_width=16,
        byte_embedding=embedding,
        hidden_width=12,
        patch_size=2,
        copy_window=4,
        copy_dim=8,
        lowercase_copy=True,
    )
    with torch.no_grad():
        head.copy_query.weight.zero_()
        head.copy_key.weight.zero_()
        head.copy_gate.weight.zero_()
        head.copy_gate.bias.fill_(10.0)
    hidden = torch.zeros(1, 12)
    source = torch.tensor([[ord("A"), ord("B"), ord("C"), ord("D")]])
    bias = head._copy_bias(hidden, source)

    assert bias[0, ord("a")] > bias[0, ord("A")]
    assert bias[0, ord("d")] > bias[0, ord("D")]


def test_semantic_copy_context_convolution_receives_gradients():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_generation_width=16,
        patch_generation_bytes=6,
        patch_generation_copy_window=40,
        patch_generation_copy_dim=8,
        patch_generation_position_copy=True,
        patch_generation_semantic_copy=True,
    )
    x = torch.randint(0, 256, (2, 16))
    predictions = model(
        x,
        return_aux=True,
        return_patch_prediction=True,
    )[3]
    sum(prediction.mean() for prediction in predictions).backward()

    assert model.patch_generator.copy_context_key.weight.grad is not None


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


def test_autoregressive_patch_generation_copy_window_shapes():
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
        patch_generation_copy_window=6,
        patch_generation_copy_dim=8,
        patch_generation_copy_scale=2.0,
        modern_blocks=True,
        fused_attention=True,
    )
    model.eval()
    x = torch.randint(0, 256, (2, 16))
    predictions = model(
        x,
        return_aux=True,
        return_patch_prediction=True,
    )[3]
    copy_sources = model._patch_generation_copy_sources(x)
    generated = model.generate_next_patch(x)
    state = model.begin_patch_prediction_cached_generation(x)
    cached = model.cached_patch_prediction_step(state)

    assert len(predictions) == 2
    assert predictions[0].shape == (2, 8, 256)
    assert copy_sources.shape == (2, 8, 6)
    assert generated.shape == (2, 2)
    assert cached.shape == (2, 2)


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


def test_cached_patch_generation_marks_position_overflow_without_raising():
    model = CausalBytePatchLM(
        patch_size=2,
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
        patch_prediction=True,
    )
    model.eval()
    state = model.begin_cached_generation(torch.randint(0, 256, (1, 6)))
    first = model.cached_generation_step(state)
    second = model.cached_generation_step(state)
    assert first.shape == (1, 2)
    assert second.shape == (1, 2)
    assert state["position_overflow"] is True


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
    for patch_size in (2, 4):
        model = CausalBytePatchLM(
            patch_size=patch_size,
            d_byte=8,
            d_model=32,
            d_abi=16,
            layers=1,
            heads=4,
            max_patches=8,
            direct_global_context=True,
            local_decoder="window_transformer",
            local_layers=1,
            local_window=8,
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
        assert generated.shape == (1, patch_size)
        assert state["recent_bytes"].shape == (1, 8 + patch_size)


def test_selective_state_patch_block_shape_grad_determinism_and_cache():
    block = SelectiveStatePatchBlock(width=32, heads=4, dropout=0.0)
    block.eval()
    h = torch.randn(2, 9, 32, requires_grad=True)
    first = block(h)
    second = block(h)
    assert first.shape == h.shape
    assert torch.allclose(first, second)
    first.sum().backward()
    assert h.grad is not None
    assert h.grad.abs().sum() > 0

    prefix = torch.randn(2, 7, 32)
    token = torch.randn(2, 1, 32)
    full = block(torch.cat([prefix, token], dim=1))
    _, cache = block.prefill_with_cache(prefix)
    decoded, _ = block.decode_with_cache(token, cache)
    assert torch.allclose(full[:, -1:], decoded, atol=1e-5, rtol=1e-5)


def test_selective_state_patch_block_is_causal():
    block = SelectiveStatePatchBlock(width=32, heads=4, dropout=0.0)
    block.eval()
    h = torch.randn(1, 10, 32)
    changed = h.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:])
    original = block(h)
    modified = block(changed)
    assert torch.allclose(original[:, :7], modified[:, :7], atol=1e-5, rtol=1e-5)


def test_selective_state_abipatchcell_generation_and_causality():
    torch.manual_seed(456)
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="abi_patch_cell",
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        dropout=0.0,
        qk_norm=False,
        global_block="selective_state_patch",
    )
    model.eval()
    x = torch.randint(0, 256, (2, 16))
    logits_a, abi_a = model(x)
    logits_b, abi_b = model(x)
    assert logits_a.shape == (2, 16, 256)
    assert abi_a.shape == (2, 8, 16)
    assert torch.allclose(logits_a, logits_b)
    assert torch.allclose(abi_a, abi_b)
    assert model.generate_next_patch(x).shape == (2, 2)

    changed_future = x.clone()
    changed_future[:, 12:] = torch.randint(0, 256, (2, 4))
    changed_logits, _ = model(changed_future)
    assert torch.allclose(logits_a[:, :8], changed_logits[:, :8], atol=1e-5, rtol=1e-5)


def test_parallel_patch_decoder_shape_generation_and_causality():
    torch.manual_seed(123)
    model = CausalBytePatchLM(
        patch_size=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="parallel_patch",
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        local_window=8,
        dropout=0.0,
        qk_norm=False,
        global_block="sparse_state_patch",
        sparse_state_local_window=4,
        sparse_state_dilated_offsets=(4, 6),
        sparse_state_chunk_size=4,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    logits, _ = model(x)
    assert logits.shape == (1, 16, 256)
    generated = model.generate_next_patch(x)
    assert generated.shape == (1, 4)

    changed_future = x.clone()
    changed_future[:, 12:] = torch.randint(0, 256, (1, 4))
    changed_logits, _ = model(changed_future)
    assert torch.allclose(logits[:, :8], changed_logits[:, :8], atol=1e-5, rtol=1e-5)


def test_abi_patch_cell_shape_generation_causality_and_gradients():
    torch.manual_seed(321)
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="abi_patch_cell",
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        dropout=0.0,
        qk_norm=False,
        global_block="sparse_state_patch",
        sparse_state_local_window=4,
        sparse_state_dilated_offsets=(4, 6),
        sparse_state_chunk_size=4,
    )
    model.eval()
    x = torch.randint(0, 256, (2, 16))
    logits_a, abi_a = model(x)
    logits_b, abi_b = model(x)
    assert logits_a.shape == (2, 16, 256)
    assert abi_a.shape[-1] == 16
    assert torch.allclose(logits_a, logits_b)
    assert torch.allclose(abi_a, abi_b)
    generated = model.generate_next_patch(x)
    assert generated.shape == (2, 2)

    changed_future = x.clone()
    changed_future[:, 12:] = torch.randint(0, 256, (2, 4))
    changed_logits, _ = model(changed_future)
    assert torch.allclose(logits_a[:, :8], changed_logits[:, :8], atol=1e-5, rtol=1e-5)

    model.train()
    y = torch.randint(0, 256, (2, 16))
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
    loss.backward()
    assert model.abi_cell_byte0.weight.grad is not None
    assert model.abi_cell_gate.weight.grad is not None
    assert model.abi_cell_byte1.weight.grad is not None
    assert model.abi_cell_next_gate.weight.grad is not None
    assert model.abi_cell_refine[1].weight.grad is not None


def test_abi_patch_cell_cached_multistep_matches_single_step_loop():
    torch.manual_seed(4321)
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="abi_patch_cell",
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        dropout=0.0,
        qk_norm=False,
        global_block="selective_state_patch",
        abi_patch_cell_global_update_interval=2,
        abi_patch_cell_fast_global_decode=True,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    single_state = model.begin_abi_patch_cell_cached_generation(x)
    multi_state = model.begin_abi_patch_cell_cached_generation(x)

    single = torch.cat(
        [
            model.cached_abi_patch_cell_step(
                single_state,
                no_repeat_ngram=3,
            )
            for _ in range(4)
        ],
        dim=1,
    )
    multi = model.cached_abi_patch_cell_steps(
        multi_state,
        4,
        no_repeat_ngram=3,
    )

    assert torch.equal(single, multi)
    assert torch.equal(single_state["recent_bytes"], multi_state["recent_bytes"])
    assert torch.equal(single_state["last_global"], multi_state["last_global"])
    assert single_state["patch_count"] == multi_state["patch_count"]
    assert single_state["generated_patch_count"] == multi_state["generated_patch_count"]


def test_span_patch_decoder_shapes_and_cached_step():
    torch.manual_seed(2468)
    model = CausalBytePatchLM(
        patch_size=2,
        span_width=4,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
        direct_global_context=True,
        local_decoder="span_patch_decoder",
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        dropout=0.0,
        qk_norm=False,
        copy_transducer=True,
        copy_transducer_dim=8,
        copy_transducer_window=16,
    )
    model.eval()
    x = torch.randint(0, 256, (2, 16))
    logits, abi = model(x)
    assert logits.shape == (2, 16, 256)
    assert abi.shape == (2, 8, 16)
    logits_aux, _, auxiliary = model(x, return_aux=True)
    assert logits_aux.shape == logits.shape
    span_aux = [
        item for item in auxiliary
        if item.ndim == 4 and item.shape[2:] == (4, 256)
    ]
    assert len(span_aux) == 1
    assert span_aux[0].shape == (2, 8, 4, 256)
    span, span_logits = model.generate_next_span(x, return_logits=True)
    assert span.shape == (2, 4)
    assert span_logits.shape == (2, 4, 256)
    state = model.begin_span_cached_generation(x)
    cached, cached_logits = model.cached_span_generation_step(
        state,
        return_logits=True,
    )
    assert cached.shape == (2, 4)
    assert cached_logits.shape == (2, 4, 256)
    assert state["patch_count"] == 10


def test_span_patch_decoder_width8_shape():
    model = CausalBytePatchLM(
        patch_size=2,
        span_width=8,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=16,
        direct_global_context=True,
        local_decoder="span_patch_decoder",
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        dropout=0.0,
        qk_norm=False,
    )
    model.eval()
    x = torch.randint(0, 256, (1, 16))
    span = model.generate_next_span(x)
    assert span.shape == (1, 8)


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


def test_top1_routed_cake_backpropagates_only_selected_expert():
    block = Top1RoutedCakeBlock(
        width=32,
        heads=4,
        experts=3,
        dropout=0.0,
    )
    block.set_route(1)
    hidden = torch.randn(2, 8, 32, requires_grad=True)

    block(hidden).square().mean().backward()

    assert hidden.grad is not None
    assert all(parameter.grad is None for parameter in block.router.parameters())
    for index, expert in enumerate(block.experts):
        gradients = [parameter.grad for parameter in expert.parameters()]
        if index == 1:
            assert all(gradient is not None for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)


def test_fused_swiglu_is_math_equivalent_after_weight_migration():
    torch.manual_seed(812)
    dense = FusedModernCausalBlock(32, 4, fused_swiglu=False).eval()
    fused = FusedModernCausalBlock(32, 4, fused_swiglu=True).eval()
    dense_state = dense.state_dict()
    fused_state = {
        key: value
        for key, value in dense_state.items()
        if key not in {"gate.weight", "up.weight"}
    }
    fused_state["gate_up.weight"] = torch.cat(
        [dense_state["gate.weight"], dense_state["up.weight"]], dim=0
    )
    fused.load_state_dict(fused_state, strict=True)
    hidden = torch.randn(2, 8, 32)

    assert torch.equal(dense(hidden), fused(hidden))


def test_sparse_cake_parameters_exclude_inactive_experts():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="parallel_patch",
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        routed_cake_experts=3,
    )
    model.set_cake_route(2)
    optimized_ids = {
        id(parameter) for parameter in model.sparse_cake_parameters(2)
    }
    routed = model.core[0]

    assert all(id(parameter) in optimized_ids for parameter in routed.experts[2].parameters())
    assert all(id(parameter) not in optimized_ids for parameter in routed.experts[0].parameters())
    assert all(id(parameter) not in optimized_ids for parameter in routed.experts[1].parameters())
    assert all(id(parameter) not in optimized_ids for parameter in routed.router.parameters())
    assert id(model.byte_emb.weight) in optimized_ids
    assert len(optimized_ids) < sum(1 for _ in model.parameters())


def test_default_dense_cake_state_dict_and_outputs_remain_exact():
    kwargs = {
        "patch_size": 2,
        "d_byte": 8,
        "d_model": 32,
        "d_abi": 16,
        "layers": 1,
        "heads": 4,
        "max_patches": 8,
        "direct_global_context": True,
        "local_decoder": "window_transformer",
        "local_layers": 1,
        "local_width": 32,
        "modern_blocks": True,
        "fused_attention": True,
    }
    torch.manual_seed(731)
    source = CausalBytePatchLM(**kwargs).eval()
    receiver = CausalBytePatchLM(**kwargs).eval()
    receiver.load_state_dict(source.state_dict(), strict=True)
    inputs = torch.randint(0, 256, (2, 16))

    source_logits, source_abi = source(inputs)
    receiver_logits, receiver_abi = receiver(inputs)

    assert torch.equal(source_logits, receiver_logits)
    assert torch.equal(source_abi, receiver_abi)


def test_default_cake_route_is_pinned_on_construction():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="parallel_patch",
        local_width=32,
        modern_blocks=True,
        fused_attention=True,
        routed_cake_experts=3,
        default_cake_route=2,
    )

    assert model.core[0].route_override == 2


def test_domain_cake_prediction_can_train_one_selected_context():
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=8,
        d_model=32,
        d_abi=16,
        layers=2,
        shared_cake_layers=1,
        heads=4,
        max_patches=8,
        direct_global_context=True,
        local_decoder="routed_window_transformer",
        local_layers=0,
        local_width=32,
        local_window=4,
        modern_blocks=True,
        fused_attention=True,
        routed_cake_experts=5,
        default_cake_route=4,
        patch_prediction=True,
        patch_prediction_mode="autoregressive",
        patch_prediction_context="global",
        patch_generation_width=16,
        patch_generation_bytes=4,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.core[1].active_expert_parameters(4):
        parameter.requires_grad_(True)
    inputs = torch.randint(0, 256, (2, 16))
    predictions, targets = model.domain_cake_patch_predictions(
        inputs,
        context_indices=torch.tensor([1, 3]),
    )
    prediction_tensor = torch.stack(predictions, dim=2)

    assert prediction_tensor.shape == (2, 1, 4, 256)
    assert targets.shape == (2, 1, 4)
    torch.nn.functional.cross_entropy(
        prediction_tensor.reshape(-1, 256), targets.reshape(-1)
    ).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.core[1].experts[4].parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.core[1].experts[0].parameters()
    )
