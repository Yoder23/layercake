from __future__ import annotations

import json

import pytest
import torch

from layercake.models.baseline_transformer import BytePairTokenizer
from layercake.models.representation_tokenizer import (
    HYBRID_CONTRACT_VERSION,
    HybridTokenByteTokenizer,
    WORD_BYTE_HYBRID_CONTRACT_VERSION,
    WordByteHybridTokenizer,
    tokenizer_from_document,
)
from layercake.models.sparse_bpe_layercake import (
    LayerCakeSparseBPECore,
    SparseBPELayerCakeConfig,
)


def test_hybrid_tokenizer_exact_roundtrip_and_raw_fallback():
    base = BytePairTokenizer.train(
        b" ordinary ordinary English English identifiers", merge_count=24
    )
    tokenizer = HybridTokenByteTokenizer(base)
    values = [
        b"ordinary English",
        "naïve café — 雪".encode("utf-8"),
        b"def parse_http2_id(value_7): return value_7[0]",
        b"C:\\models\\weights.bin",
        bytes([0xFF, 0xFE, 0x80]),
    ]
    for value in values:
        assert tokenizer.decode(tokenizer.encode(value)) == value
    identifier = b"parse_http2_id"
    assert tokenizer.encode(identifier) == list(identifier)
    unicode_value = "雪".encode("utf-8")
    assert tokenizer.encode(unicode_value) == list(unicode_value)


def test_hybrid_tokenizer_canonical_reload_is_identical():
    base = BytePairTokenizer.train(b" layercake layercake model model", merge_count=16)
    tokenizer = HybridTokenByteTokenizer(base)
    document = tokenizer.canonical_dict()
    assert document["hybrid_contract"]["version"] == HYBRID_CONTRACT_VERSION
    reloaded = tokenizer_from_document(json.loads(json.dumps(document)))
    value = b"value_7 = " + "雪".encode("utf-8")
    assert reloaded.encode(value) == tokenizer.encode(value)
    assert reloaded.decode(reloaded.encode(value)) == value


def test_word_byte_hybrid_uses_lexical_units_with_exact_byte_fallback():
    tokenizer = WordByteHybridTokenizer([
        b"LayerCake",
        b" prompt",
        b" grounding",
    ])
    assert tokenizer.encode(b"LayerCake prompt grounding") == [256, 257, 258]
    unknown = b" Nova_ID-7 " + "雪".encode("utf-8")
    assert tokenizer.encode(unknown) == list(unknown)
    values = [
        b"LayerCake prompt grounding",
        b"Nova_ID-7",
        "naïve café — 雪".encode("utf-8"),
        bytes([0xFF, 0xFE, 0x80]),
    ]
    document = tokenizer.canonical_dict()
    assert (
        document["hybrid_contract"]["version"]
        == WORD_BYTE_HYBRID_CONTRACT_VERSION
    )
    reloaded = tokenizer_from_document(json.loads(json.dumps(document)))
    for value in values:
        assert reloaded.decode(reloaded.encode(value)) == value


def test_cached_prompt_attention_incremental_state_is_prompt_conditioned():
    torch.manual_seed(3)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=320,
        width=32,
        layers=2,
        heads=4,
        max_tokens=64,
        expansion=1,
        routed_experts=3,
        expert_expansion=1,
        route_after_layers=1,
        prompt_conditioning=True,
        prompt_attention_pooling=True,
    )).eval()
    prompt = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
    state = model.prefill(prompt)
    assert state.prompt_context.shape == (1, 32)
    assert state.prompt_copy_bias.shape == (1, 320)
    assert torch.isfinite(state.next_logits).all()
    _, state = model.decode_step(state)
    assert state.generated_ids.shape == (1, 1)
    assert torch.isfinite(state.next_logits).all()


def test_multislot_prompt_state_is_prefill_only_and_prompt_conditioned():
    torch.manual_seed(5)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=320,
        width=32,
        layers=2,
        heads=4,
        max_tokens=64,
        expansion=1,
        routed_experts=3,
        expert_expansion=1,
        route_after_layers=1,
        prompt_conditioning=True,
        prompt_state_slots=4,
    )).eval()
    first = model.prefill(torch.tensor([[65, 66, 67, 68]], dtype=torch.long))
    second = model.prefill(torch.tensor([[65, 66, 90, 91]], dtype=torch.long))
    assert first.prompt_context.shape == (1, 32)
    assert first.prompt_copy_bias.shape == (1, 320)
    assert not torch.equal(first.prompt_context, second.prompt_context)
    cached_lengths = [cache[0].shape[-2] for cache in first.keys_values]
    _, first = model.decode_step(first)
    assert [
        cache[0].shape[-2] for cache in first.keys_values
    ] == [length + 1 for length in cached_lengths]
    assert first.prompt_context.shape == (1, 32)
    assert torch.isfinite(first.next_logits).all()


def test_recurrent_prompt_memory_is_bounded_and_changes_neural_logits():
    torch.manual_seed(7)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=320,
        width=32,
        layers=2,
        heads=4,
        max_tokens=64,
        expansion=1,
        routed_experts=3,
        expert_expansion=1,
        route_after_layers=1,
        prompt_conditioning=True,
        prompt_state_slots=4,
        recurrent_prompt_memory=True,
        prompt_memory_key_width=8,
    )).eval()
    state = model.prefill(
        torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long)
    )
    assert state.prompt_memory_slots.shape == (1, 4, 32)
    assert state.prompt_memory_copy_distribution.shape == (1, 4, 320)
    memory_elements = (
        state.prompt_memory_slots.numel()
        + state.prompt_memory_copy_distribution.numel()
    )
    recurrent_logits = state.next_logits.clone()
    state.prompt_memory_slots = None
    state.prompt_memory_copy_distribution = None
    _, state = model.decode_step(
        state, next_token=torch.tensor([70], dtype=torch.long)
    )
    assert (
        state.prompt_memory_slots is None
        and state.prompt_memory_copy_distribution is None
    )
    assert torch.isfinite(state.next_logits).all()
    fresh = model.prefill(
        torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long)
    )
    _, fresh = model.decode_step(
        fresh, next_token=torch.tensor([70], dtype=torch.long)
    )
    assert (
        fresh.prompt_memory_slots.numel()
        + fresh.prompt_memory_copy_distribution.numel()
    ) == memory_elements
    assert not torch.equal(recurrent_logits, state.next_logits)
    assert not torch.equal(fresh.next_logits, state.next_logits)


def test_hierarchical_prompt_memory_has_fixed_abi_and_sparse_pointer_bias():
    torch.manual_seed(11)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=320,
        width=32,
        layers=2,
        heads=4,
        max_tokens=96,
        expansion=1,
        routed_experts=3,
        expert_expansion=1,
        route_after_layers=1,
        prompt_conditioning=True,
        prompt_state_slots=4,
        hierarchical_prompt_memory=True,
        prompt_memory_key_width=8,
        prompt_memory_capacity=16,
        prompt_memory_chunk_size=4,
    )).eval()
    prompt = torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long)
    state = model.prefill(prompt)
    memory = state.hierarchical_prompt_memory
    assert [tuple(value.shape) for value in memory] == [
        (1, 16),
        (1, 16, 32),
        (1, 16),
        (1, 4, 32),
        (1, 4),
    ]
    assert torch.isfinite(state.next_logits).all()
    shapes = [tuple(value.shape) for value in memory]
    _, state = model.decode_step(
        state, next_token=torch.tensor([70], dtype=torch.long)
    )
    assert [
        tuple(value.shape) for value in state.hierarchical_prompt_memory
    ] == shapes
    assert state.generated_ids.shape == (1, 1)
    assert torch.isfinite(state.next_logits).all()
    assert model.last_prompt_memory_aux["pointer_mass"].item() == pytest.approx(
        1.0, abs=1e-6
    )


def test_structured_prompt_memory_is_encode_once_bounded_and_addressable():
    torch.manual_seed(13)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=320,
        width=32,
        layers=2,
        heads=4,
        max_tokens=96,
        expansion=1,
        routed_experts=3,
        expert_expansion=1,
        route_after_layers=1,
        prompt_conditioning=True,
        structured_prompt_memory=True,
        structured_prompt_roles=6,
        prompt_memory_key_width=8,
        prompt_memory_capacity=16,
    )).eval()
    state = model.prefill(
        torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long)
    )
    memory = state.structured_prompt_memory
    assert [tuple(value.shape) for value in memory] == [
        (1, 16),
        (1, 16, 32),
        (1, 16),
        (1, 6, 32),
    ]
    frozen = [value.clone() for value in memory]
    _, state = model.decode_step(
        state, next_token=torch.tensor([70], dtype=torch.long)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(frozen, state.structured_prompt_memory)
    )
    assert model.last_structured_pointer_weights.shape == (1, 1, 16)
    assert model.last_structured_pointer_weights.sum().item() == pytest.approx(
        1.0, abs=1e-6
    )
    assert torch.isfinite(state.next_logits).all()


def test_semantic_prompt_encoder_is_contextual_bounded_and_encode_once():
    torch.manual_seed(17)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=320,
        width=32,
        layers=2,
        heads=4,
        max_tokens=96,
        expansion=1,
        routed_experts=3,
        expert_expansion=1,
        route_after_layers=1,
        prompt_conditioning=True,
        semantic_prompt_encoder=True,
        semantic_prompt_slots=6,
        prompt_memory_key_width=8,
        prompt_memory_capacity=16,
    )).eval()
    prompt = torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long)
    state = model.prefill(prompt)
    memory = state.semantic_prompt_memory
    assert [tuple(value.shape) for value in memory] == [
        (1, 6, 32),
        (1, 6, 320),
    ]
    assert torch.allclose(memory[1].sum(dim=-1), torch.ones(1, 6))
    frozen = [value.clone() for value in memory]
    _, state = model.decode_step(
        state, next_token=torch.tensor([70], dtype=torch.long)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(frozen, state.semantic_prompt_memory)
    )
    assert model.last_semantic_slot_weights.shape == (1, 1, 6)
    assert model.last_semantic_slot_weights.sum().item() == pytest.approx(
        1.0, abs=1e-6
    )
    assert model.last_semantic_pointer_distribution.sum().item() == (
        pytest.approx(1.0, abs=1e-6)
    )
    alternate = model.prefill(
        torch.tensor([[65, 66, 90, 91, 92]], dtype=torch.long)
    )
    assert not torch.equal(state.prompt_context, alternate.prompt_context)
    assert torch.isfinite(state.next_logits).all()


def test_contextual_token_memory_is_direct_bounded_and_encode_once():
    torch.manual_seed(19)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=320,
        width=32,
        layers=2,
        heads=4,
        max_tokens=96,
        expansion=1,
        routed_experts=3,
        expert_expansion=1,
        route_after_layers=1,
        prompt_conditioning=True,
        contextual_token_memory=True,
        prompt_memory_key_width=8,
        prompt_memory_capacity=16,
    )).eval()
    prompt = torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long)
    state = model.prefill(prompt)
    memory = state.contextual_token_memory
    assert [tuple(value.shape) for value in memory] == [
        (1, 32),
        (1, 16),
        (1, 16, 32),
        (1, 16),
    ]
    assert memory[1][0, :5].tolist() == prompt[0].tolist()
    assert memory[3].sum().item() == 5
    frozen = [value.clone() for value in memory]
    _, state = model.decode_step(
        state, next_token=torch.tensor([70], dtype=torch.long)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(frozen, state.contextual_token_memory)
    )
    assert model.last_contextual_pointer_weights.shape == (1, 1, 16)
    assert model.last_contextual_pointer_weights.sum().item() == pytest.approx(
        1.0, abs=1e-6
    )
    assert torch.equal(
        model.last_contextual_pointer_ids,
        state.contextual_token_memory[1],
    )
    alternate = model.prefill(
        torch.tensor([[65, 66, 90, 91, 92]], dtype=torch.long)
    )
    assert not torch.equal(state.prompt_context, alternate.prompt_context)
    assert torch.isfinite(state.next_logits).all()


def test_contextual_token_memory_rejects_semantic_slot_combination():
    with pytest.raises(ValueError, match="exclusive"):
        SparseBPELayerCakeConfig(
            vocab_size=320,
            width=32,
            layers=2,
            heads=4,
            route_after_layers=1,
            prompt_conditioning=True,
            semantic_prompt_encoder=True,
            contextual_token_memory=True,
        )


def test_factorized_prompt_control_has_three_distinct_bounded_records():
    torch.manual_seed(29)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=320,
        width=32,
        layers=2,
        heads=4,
        max_tokens=96,
        expansion=1,
        routed_experts=3,
        expert_expansion=1,
        route_after_layers=1,
        prompt_conditioning=True,
        factorized_prompt_control=True,
        factorized_task_count=10,
        prompt_memory_key_width=8,
        prompt_memory_capacity=16,
    )).eval()
    prompt = torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long)
    state = model.prefill(prompt)
    control = state.factorized_prompt_control
    assert [tuple(value.shape) for value in control] == [
        (1, 32),
        (1, 16),
        (1, 16),
        (1, 3, 32),
        (1, 10),
        (1, 16),
    ]
    assert control[2].sum().item() == 5
    assert control[5].sum().item() == pytest.approx(1.0, abs=1e-6)
    assert not torch.equal(control[3][:, 0], control[3][:, 1])
    assert not torch.equal(control[3][:, 1], control[3][:, 2])
    frozen = [value.clone() for value in control]
    _, state = model.decode_step(
        state, next_token=torch.tensor([70], dtype=torch.long)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            frozen, state.factorized_prompt_control
        )
    )
    assert model.last_factor_pointer_weights.shape == (1, 1, 16)
    assert model.last_factor_pointer_weights.sum().item() == pytest.approx(
        1.0, abs=1e-6
    )
    assert torch.isfinite(state.next_logits).all()


def test_factorized_prompt_control_rejects_direct_memory_combination():
    with pytest.raises(ValueError, match="exclusive"):
        SparseBPELayerCakeConfig(
            vocab_size=320,
            width=32,
            layers=2,
            heads=4,
            route_after_layers=1,
            prompt_conditioning=True,
            contextual_token_memory=True,
            factorized_prompt_control=True,
        )
