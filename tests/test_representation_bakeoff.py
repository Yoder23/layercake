from __future__ import annotations

import json

import torch

from layercake.models.baseline_transformer import BytePairTokenizer
from layercake.models.representation_tokenizer import (
    HYBRID_CONTRACT_VERSION,
    HybridTokenByteTokenizer,
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
