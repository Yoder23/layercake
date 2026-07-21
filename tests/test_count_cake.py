import numpy as np
import torch

from layercake.count_cake import (
    CausalCompositeByteCache,
    CausalOnlineByteCache,
    HierarchicalCountCakeLM,
    PrunedBackoffByteCake,
    apply_causal_online_cache_to_observed,
    assert_parameter_budget,
    load_count_cake_bundle,
    save_count_cake_bundle,
    _parallel_affine_scan,
)
from layercake.count_cake_cpu import CountCakeCPUDecoder
from layercake.count_cake_speculative import CountCakeSpeculativeDecoder


def _corpus() -> torch.Tensor:
    payload = (b"the quick brown fox jumps over the lazy dog\n" * 40) + bytes(
        range(64)
    )
    return torch.tensor(list(payload), dtype=torch.long)


def test_parallel_affine_scan_matches_long_sequential_recurrence_and_gradients():
    torch.manual_seed(17)
    raw_decay = torch.randn(2, 513, 7, requires_grad=True)
    injection = torch.randn(2, 513, 7, requires_grad=True)
    decay = 0.05 + 0.945 * torch.sigmoid(raw_decay)
    parallel = _parallel_affine_scan(decay, injection)
    state = torch.zeros_like(injection[:, 0])
    sequential = []
    for offset in range(decay.shape[1]):
        state = decay[:, offset] * state + injection[:, offset]
        sequential.append(state)
    sequential = torch.stack(sequential, dim=1)
    assert torch.allclose(parallel, sequential, atol=2e-5, rtol=2e-5)
    parallel.square().mean().backward()
    assert torch.isfinite(raw_decay.grad).all()
    assert torch.isfinite(injection.grad).all()


def test_count_draft_speculation_is_exact_for_direct_confidence_model():
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus, state_budget=900, max_order=4
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=1,
        prediction_start=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
        local_decoder="position",
        byte_head="direct",
        confidence_gate=True,
    ).eval()
    prompt = corpus[:32].reshape(1, -1)
    reference_state = model.begin_cached_generation(prompt)
    expected = model.generate_cached(reference_state, patches=41)
    candidate_state = model.begin_cached_generation(prompt)
    decoder = CountCakeSpeculativeDecoder(
        model, CountCakeCPUDecoder(model), block_size=8
    )
    actual = decoder.generate_cached(candidate_state, patches=41)
    assert torch.equal(expected, actual)
    assert torch.allclose(
        reference_state["recurrent_state"],
        candidate_state["recurrent_state"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert candidate_state["speculative_emitted_bytes"] == 41
    assert candidate_state["speculative_rounds"] >= 1


def test_neural_count_order_router_is_causal_trainable_and_portable(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus, state_budget=900, max_order=4
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=1,
        prediction_start=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
        local_decoder="position",
        byte_head="direct",
        confidence_gate=True,
        count_order_routing=True,
    )
    rows = corpus[:96].reshape(2, 48)
    _, features, stages = cake.target_log_probs(
        rows, start=8, return_features=True, return_stages=True
    )
    assert features.shape == (2, 40, 4)
    assert stages.shape == (2, 40, cake.max_order + 1)
    assert torch.allclose(
        stages[..., -1], cake.target_log_probs(rows, start=8), atol=1e-6
    )
    loss = model.loss(rows, neural_auxiliary_weight=1.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.count_order_router.weight.grad is not None
    changed = rows.clone()
    changed[:, 32:] = torch.flip(changed[:, 32:], dims=(1,))
    with torch.no_grad():
        original = model.target_log_probs(rows)
        modified = model.target_log_probs(changed)
    assert torch.allclose(original[:, :24], modified[:, :24], atol=1e-6)
    bundle = tmp_path / "order-routed.npz"
    save_count_cake_bundle(model, bundle)
    restored, manifest = load_count_cake_bundle(bundle)
    assert manifest["model"]["count_order_routing"]
    with torch.no_grad():
        assert torch.allclose(
            original, restored.target_log_probs(rows), atol=1e-6
        )


def test_count_cake_is_budgeted_normalized_and_generation_aligned():
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus,
        state_budget=900,
        max_order=4,
    )
    assert cake.state_entries <= 900
    assert cake.state_entries > 256

    rows = corpus[:96].reshape(1, -1)
    observed = cake.target_log_probs(rows, start=8)
    for position in range(8, rows.shape[1]):
        distribution = cake.next_probabilities(rows[0, :position])
        assert torch.allclose(distribution.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(
            observed[0, position - 8],
            distribution[rows[0, position]].log(),
            atol=1e-6,
            rtol=1e-6,
        )


def test_hierarchical_count_cake_trains_all_neural_state_and_generates():
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus,
        state_budget=900,
        max_order=4,
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
    )
    rows = corpus[:96].reshape(2, 48)
    loss = model.loss(rows)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None for parameter in model.parameters()
    )
    assert model.logical_total_parameters == cake.state_entries + sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert_parameter_budget(
        model,
        target=model.logical_total_parameters,
        relative_tolerance=0.0,
    )
    generated = model.eval().generate_next_patch(rows[:1, :16])
    assert generated.shape == (1, 8)
    assert int(generated.min()) >= 0
    assert int(generated.max()) <= 255
    state = model.begin_cached_generation(rows[:1, :16])
    assert torch.equal(generated, model.generate_cached(state))
    reference_state = model.begin_cached_generation(rows[:1, :16])
    optimized_state = {
        name: value.clone() for name, value in reference_state.items()
    }
    reference = model.generate_cached(reference_state, patches=2)
    optimized = CountCakeCPUDecoder(model).generate_cached(
        optimized_state, patches=2
    )
    assert torch.equal(reference, optimized)
    assert torch.equal(
        reference_state["recurrent_state"],
        optimized_state["recurrent_state"],
    )


def test_delimiter_chunk_host_is_causal_trainable_and_portable(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus,
        state_budget=900,
        max_order=4,
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        chunking_mode="delimiter",
        d_byte=4,
        d_model=16,
        d_abi=8,
        dynamic_hash_buckets=64,
        dynamic_hash_width=8,
        local_width=12,
        local_recurrent=True,
        local_continuous=True,
        byte_head="direct",
        confidence_gate=True,
    )
    rows = corpus[:96].reshape(2, 48)
    loss = model.loss(rows)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.dynamic_byte_projection.weight.grad is not None
    assert model.dynamic_hash_embedding.weight.grad is not None
    assert model.patch_core.weight_ih_l0.grad is not None

    changed = rows.clone()
    changed[:, 32:] = torch.flip(changed[:, 32:], dims=(1,))
    with torch.no_grad():
        original = model.target_log_probs(rows)
        modified = model.target_log_probs(changed)
    assert torch.allclose(original[:, :24], modified[:, :24], atol=1e-6)

    bundle = tmp_path / "delimiter_count_cake.npz"
    save_count_cake_bundle(model, bundle)
    loaded, manifest = load_count_cake_bundle(bundle)
    assert manifest["model"]["chunking_mode"] == "delimiter"
    assert manifest["model"]["dynamic_hash_buckets"] == 64
    with torch.no_grad():
        restored = loaded.target_log_probs(rows)
    assert torch.allclose(original, restored, atol=1e-6)


def test_compositional_sparse_hash_is_trainable_and_portable(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus, state_budget=900, max_order=4
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        chunking_mode="delimiter",
        d_byte=4,
        d_model=16,
        d_abi=8,
        dynamic_hash_buckets=64,
        dynamic_hash_width=2,
        dynamic_hash_tables=4,
        dynamic_hash_sparse=True,
        local_width=12,
        local_recurrent=True,
        local_continuous=True,
        byte_head="direct",
    )
    rows = corpus[:96].reshape(2, 48)
    loss = model.loss(rows)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        embedding.weight.grad is not None
        and embedding.weight.grad.is_sparse
        for embedding in model.dynamic_hash_embeddings
    )
    bundle = tmp_path / "multi-hash.npz"
    save_count_cake_bundle(model, bundle)
    restored, manifest = load_count_cake_bundle(bundle)
    assert manifest["model"]["dynamic_hash_tables"] == 4
    assert restored.dynamic_hash_sparse
    with torch.no_grad():
        assert torch.allclose(
            model.target_log_probs(rows),
            restored.target_log_probs(rows),
            atol=1e-6,
        )


def test_sparse_causal_context_state_is_trainable_causal_and_portable(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus, state_budget=900, max_order=4
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=1,
        prediction_start=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        neural_context_buckets=64,
        neural_context_order=4,
        neural_context_sparse=True,
        local_width=12,
        byte_head="direct",
    )
    rows = corpus[:96].reshape(2, 48)
    original = model.target_log_probs(rows)
    loss = model.neural_loss(rows)
    loss.backward()
    gradient = model.neural_context_embedding.weight.grad
    assert torch.isfinite(loss)
    assert gradient is not None and gradient.is_sparse
    changed = rows.clone()
    changed[:, 32:] = torch.flip(changed[:, 32:], dims=(1,))
    with torch.no_grad():
        modified = model.target_log_probs(changed)
    assert torch.allclose(original[:, :24], modified[:, :24], atol=1e-6)
    bundle = tmp_path / "sparse-context.npz"
    save_count_cake_bundle(model, bundle)
    restored, manifest = load_count_cake_bundle(bundle)
    assert manifest["model"]["neural_context_sparse"]
    assert restored.neural_context_sparse
    with torch.no_grad():
        assert torch.allclose(
            original,
            restored.target_log_probs(rows),
            atol=1e-6,
        )


def test_scratchpad_context_is_causal_trainable_and_portable(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus, state_budget=900, max_order=4
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=32,
        scratchpad_stride=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
        local_recurrent=True,
        local_continuous=True,
        byte_head="direct",
    )
    rows = corpus[:192].reshape(2, 96)
    original = model.target_log_probs(rows)
    changed = rows.clone()
    changed[:, 40] = (changed[:, 40] + 17).remainder(256)
    modified = model.target_log_probs(changed)
    assert torch.allclose(original[:, :8], modified[:, :8], atol=1e-6)
    assert not torch.allclose(original[:, 9:24], modified[:, 9:24])
    loss = model.loss(rows)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.patch_core.weight_ih_l0.grad is not None
    bundle = tmp_path / "scratchpad.npz"
    save_count_cake_bundle(model, bundle)
    restored, manifest = load_count_cake_bundle(bundle)
    assert manifest["model"]["scratchpad_stride"] == 8
    with torch.no_grad():
        assert torch.allclose(
            original,
            restored.target_log_probs(rows),
            atol=1e-6,
        )


def test_parallel_selective_patch_core_is_causal_trainable_and_portable(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus,
        state_budget=900,
        max_order=4,
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        patch_layers=2,
        patch_core_type="selective_scan",
        local_width=12,
        local_recurrent=True,
        local_continuous=True,
        byte_head="direct",
    )
    rows = corpus[:96].reshape(2, 48)
    loss = model.loss(rows)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.patch_core.blocks[0].selective.weight.grad is not None
    assert model.patch_core.blocks[1].ffn_out.weight.grad is not None

    changed = rows.clone()
    changed[:, 32:] = torch.flip(changed[:, 32:], dims=(1,))
    with torch.no_grad():
        original = model.target_log_probs(rows)
        modified = model.target_log_probs(changed)
    assert torch.allclose(original[:, :24], modified[:, :24], atol=1e-6)

    bundle = tmp_path / "selective-count-cake.npz"
    save_count_cake_bundle(model, bundle)
    loaded, manifest = load_count_cake_bundle(bundle)
    assert manifest["model"]["patch_core_type"] == "selective_scan"
    with torch.no_grad():
        restored = loaded.target_log_probs(rows)
    assert torch.allclose(original, restored, atol=1e-6)


def test_continuous_dilated_decoder_uses_full_causal_receptive_field(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus, state_budget=900, max_order=4
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
        local_decoder="dilated_conv",
        local_continuous=True,
        local_layers=6,
        local_rank=8,
        byte_head="direct",
    )
    assert model.local_dilations == (1, 2, 4, 8, 16, 32)
    rows = corpus[:96].reshape(2, 48)
    original = model.target_log_probs(rows)
    changed = rows.clone()
    changed[:, 32:] = torch.flip(changed[:, 32:], dims=(1,))
    modified = model.target_log_probs(changed)
    assert torch.allclose(original[:, :24], modified[:, :24], atol=1e-6)
    loss = model.loss(rows)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.local_depthwise[-1].weight.grad is not None

    bundle = tmp_path / "continuous-dilated.npz"
    save_count_cake_bundle(model, bundle)
    restored, manifest = load_count_cake_bundle(bundle)
    assert manifest["model"]["local_layers"] == 6
    assert restored.local_dilations == model.local_dilations
    with torch.no_grad():
        restored_probability = restored.target_log_probs(rows)
    assert torch.allclose(original, restored_probability, atol=1e-6)


def test_continuous_dilated_decoder_persists_explicit_growth_schedule(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus, state_budget=900, max_order=4
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
        local_decoder="dilated_conv",
        local_continuous=True,
        local_layers=4,
        local_dilation_growth=4,
        local_rank=8,
        byte_head="direct",
    )
    assert model.local_dilations == (1, 4, 16, 64)
    bundle = tmp_path / "growth-four.npz"
    save_count_cake_bundle(model, bundle)
    restored, manifest = load_count_cake_bundle(bundle)
    assert manifest["model"]["local_dilation_growth"] == 4
    assert restored.local_dilations == model.local_dilations


def test_count_cake_bundle_transfer_is_bit_exact(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus,
        state_budget=700,
        max_order=3,
    )
    sender = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
    ).eval()
    path = tmp_path / "domain-cake.npz"
    saved = save_count_cake_bundle(sender, path, metadata={"domain": "test"})
    receiver, loaded = load_count_cake_bundle(path)
    receiver.eval()

    assert saved["format"] == loaded["format"]
    assert loaded["metadata"] == {"domain": "test"}
    assert receiver.logical_total_parameters == sender.logical_total_parameters
    for name, value in sender.state_dict().items():
        assert torch.equal(value, receiver.state_dict()[name]), name
    rows = corpus[:48].reshape(1, -1)
    assert torch.equal(
        sender.generate_next_patch(rows[:, :16]),
        receiver.generate_next_patch(rows[:, :16]),
    )
    assert torch.equal(
        sender.loss(rows, neural_auxiliary_weight=0.0),
        receiver.loss(rows, neural_auxiliary_weight=0.0),
    )


def test_hashed_high_order_count_cake_is_normalized_and_portable(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus,
        state_budget=10_000,
        max_order=8,
    )
    assert cake.max_order == 8
    assert cake.order_encodings[-2:] == ("hashed_index", "hashed_index")
    rows = corpus[:96].reshape(1, -1)
    observed = cake.target_log_probs(rows, start=16)
    for position in range(16, rows.shape[1]):
        distribution = cake.next_probabilities(rows[0, :position])
        assert torch.allclose(distribution.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(
            observed[0, position - 16],
            distribution[rows[0, position]].log(),
            atol=1e-6,
            rtol=1e-6,
        )

    model = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
    ).eval()
    state = model.begin_cached_generation(rows[:, :16])
    reference = model.generate_cached(state)
    cpu_state = model.begin_cached_generation(rows[:, :16])
    assert torch.equal(
        reference,
        CountCakeCPUDecoder(model).generate_cached(cpu_state),
    )
    path = tmp_path / "high-order.npz"
    save_count_cake_bundle(model, path)
    receiver, _ = load_count_cake_bundle(path)
    for name, value in model.state_dict().items():
        assert torch.equal(value, receiver.state_dict()[name]), name


def test_recurrent_local_cake_is_generation_aligned_and_portable(tmp_path):
    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus, state_budget=900, max_order=4
    )
    model = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
        local_recurrent=True,
        online_cache_specs=((3, 8.0),),
    ).eval()
    rows = corpus[:48].reshape(1, -1)
    targets = rows[:, 8:].reshape(1, -1, 8)
    context = model._patch_context(rows)
    observed, _ = model._neural_log_probs(context, targets)
    changed = targets.clone()
    changed[..., 4:] = torch.randint(0, 256, changed[..., 4:].shape)
    changed_observed, _ = model._neural_log_probs(context, changed)
    assert torch.equal(observed[..., :4], changed_observed[..., :4])
    direct = model.generate_next_patch(rows[:, :16])
    state = model.begin_cached_generation(rows[:, :16])
    assert torch.equal(direct, model.generate_cached(state))
    cpu_state = model.begin_cached_generation(rows[:, :16])
    assert torch.equal(
        direct,
        CountCakeCPUDecoder(model).generate_cached(cpu_state),
    )

    path = tmp_path / "recurrent.npz"
    save_count_cake_bundle(model, path)
    receiver, manifest = load_count_cake_bundle(path)
    assert manifest["model"]["local_recurrent"] is True
    assert manifest["model"]["online_cache_specs"] == [[3, 8.0]]
    assert torch.equal(direct, receiver.generate_next_patch(rows[:, :16]))


def test_online_cache_is_normalized_row_local_and_strictly_causal():
    cache = CausalOnlineByteCache(((2, 2.0), (1, 4.0)))
    history = bytearray(b"abab")
    prefix = bytearray()
    for target in history:
        cache.update(prefix, target)
        prefix.append(target)
    base = torch.full((256,), 1.0 / 256.0)
    probability = cache.probabilities(base, history)
    assert torch.allclose(probability.sum(), torch.tensor(1.0), atol=1e-7)
    assert torch.allclose(
        probability[ord("a")],
        torch.tensor(cache.observed_probability(1.0 / 256.0, history, ord("a"))),
    )

    rows = np.array(
        [list(b"abababab"), list(b"ababaxxx")], dtype=np.uint8
    )
    base_observed = np.full((2, 4), 1.0 / 256.0)
    observed = apply_causal_online_cache_to_observed(
        base_observed,
        rows,
        start=4,
        specs=((2, 2.0), (1, 4.0)),
    )
    # Future changes cannot alter either row's probability at the first target.
    assert observed[0, 0] == observed[1, 0]
    # Identical rows receive identical results, proving no cross-row state leak.
    duplicated = np.repeat(rows[:1], 2, axis=0)
    duplicate_result = apply_causal_online_cache_to_observed(
        base_observed,
        duplicated,
        start=4,
        specs=((2, 2.0), (1, 4.0)),
    )
    assert np.array_equal(duplicate_result[0], duplicate_result[1])


def test_composite_cache_matches_observed_cpu_and_torch_and_transfers(tmp_path):
    exact = ((4, 5.25), (2, 52.0))
    recent = ((6, 1.5), (4, 3.0))
    normalized = ((3, 18.0),)
    payload = b"Ab12 Ab12 Ab12 Ab12\n"
    cache = CausalCompositeByteCache(
        exact_specs=exact,
        recent_specs=recent,
        normalized_specs=normalized,
        window=16,
        normalization="classes",
    )
    history = cache.prefill(payload[:-1])
    base = torch.linspace(1.0, 256.0, 256)
    base /= base.sum()
    torch_probability = cache.probabilities(base, history)
    numpy_probability = cache.probabilities_numpy(base.numpy(), history)
    target = payload[-1]
    assert torch.allclose(torch_probability.sum(), torch.tensor(1.0), atol=1e-7)
    assert np.isclose(numpy_probability.sum(), 1.0)
    assert np.isclose(
        float(torch_probability[target]),
        cache.observed_probability(float(base[target]), history, target),
        rtol=1e-6,
    )
    assert np.allclose(torch_probability.numpy(), numpy_probability, rtol=1e-6)

    corpus = _corpus()
    cake = PrunedBackoffByteCake.train_from_bytes(
        corpus, state_budget=900, max_order=4
    )
    sender = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
        local_recurrent=True,
        online_cache_specs=exact,
        online_cache_window=16,
        recent_cache_specs=recent,
        normalized_cache_specs=normalized,
        cache_normalization="classes",
    ).eval()
    path = tmp_path / "composite-cache.npz"
    save_count_cake_bundle(sender, path)
    receiver, manifest = load_count_cake_bundle(path)
    assert manifest["model"]["online_cache_window"] == 16
    assert manifest["model"]["recent_cache_specs"] == [list(x) for x in recent]
    assert manifest["model"]["normalized_cache_specs"] == [
        list(x) for x in normalized
    ]
    prompt = corpus[:16].reshape(1, -1)
    sender_state = sender.begin_cached_generation(prompt)
    receiver_state = receiver.begin_cached_generation(prompt)
    sender_bytes = sender.generate_cached(sender_state, patches=2)
    assert torch.equal(sender_bytes, receiver.generate_cached(receiver_state, patches=2))
    cpu_state = receiver.begin_cached_generation(prompt)
    assert torch.equal(
        sender_bytes,
        CountCakeCPUDecoder(receiver).generate_cached(cpu_state, patches=2),
    )


def test_streaming_high_order_hash_abi_is_reachable_and_portable(tmp_path):
    corpus = _corpus().to(torch.uint8)
    cake = PrunedBackoffByteCake.train_streaming_from_bytes(
        corpus,
        device=torch.device("cpu"),
        state_budget=8_000,
        max_order=8,
        chunk_bytes=256,
        budget_mode="balanced",
    )
    assert cake.context_hash_bits[-2:] == (55, 55)
    history = corpus[:8].to(torch.long)
    context = 0
    for byte in history:
        context = (context * 257 + int(byte) + 1) & ((1 << 55) - 1)
    contexts = cake.context_keys_8
    location = torch.searchsorted(contexts, torch.tensor(context))
    assert location < contexts.numel()
    assert int(contexts[location]) == context

    model = HierarchicalCountCakeLM(
        cake,
        patch_size=8,
        d_byte=4,
        d_model=16,
        d_abi=8,
        local_width=12,
    ).eval()
    path = tmp_path / "streaming-hash.npz"
    save_count_cake_bundle(model, path)
    receiver, manifest = load_count_cake_bundle(path)
    assert manifest["count_cake"]["context_hash_bits"][-2:] == [55, 55]
    assert receiver.count_cake.context_hash_bits == cake.context_hash_bits


def test_hybrid2_preserves_two_dense_orders_then_balances():
    payload = torch.arange(256, dtype=torch.long).repeat(8)
    budget = 1_200
    model = PrunedBackoffByteCake.train_from_bytes(
        payload,
        state_budget=budget,
        max_order=5,
        budget_mode="hybrid2",
    )

    assert model.state_entries == budget
    assert model.order_entries[0] == 256
    assert model.order_entries[1] == 256
    assert max(model.order_entries[2:]) - min(model.order_entries[2:]) <= 1
