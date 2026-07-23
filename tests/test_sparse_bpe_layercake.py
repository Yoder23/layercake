from __future__ import annotations

import json
from pathlib import Path
import random

import torch

from layercake.models.baseline_transformer import BytePairTokenizer
from layercake.models.sparse_bpe_layercake import (
    LayerCakeSparseBPECore,
    SparseBPELayerCakeConfig,
)
from layercake.models.phase2_english_planner import (
    canonical_planner_bytes,
    classify_task,
    extract_subject,
    realize_english,
)


def _model() -> LayerCakeSparseBPECore:
    return LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=384,
        width=40,
        layers=4,
        heads=5,
        max_tokens=64,
        routed_experts=4,
        expert_expansion=1,
        route_after_layers=2,
    )).eval()


def test_sparse_bpe_layercake_cached_decode_matches_full_forward() -> None:
    torch.manual_seed(31)
    model = _model()
    tokens = torch.randint(0, 384, (1, 12))
    full = model(tokens)
    state = model.prefill(tokens[:, :-1])
    cached, state = model.decode_step(state, tokens[:, -1])
    assert torch.allclose(cached, full[:, -2], atol=2e-6, rtol=0.0)
    assert torch.allclose(state.next_logits, full[:, -1], atol=2e-6, rtol=0.0)
    assert state.generated_ids.shape == (1, 1)


def test_sparse_bpe_layercake_physically_skips_inactive_cakes() -> None:
    model = _model()
    model.cakes.set_route(1)
    calls = [0, 0, 0, 0]
    handles = [
        expert.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(index, calls[index] + 1)
        )
        for index, expert in enumerate(model.cakes.experts)
    ]
    try:
        state = model.prefill(torch.randint(0, 384, (1, 8)))
        for _ in range(5):
            _, state = model.decode_step(state)
    finally:
        for handle in handles:
            handle.remove()
    assert calls[1] == 6
    assert calls[0] == calls[2] == calls[3] == 0
    assert model.active_parameter_count() < model.parameter_count()


def test_heap_bpe_encoder_is_exactly_equivalent_to_sequential_merge_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    documents = [
        json.loads((root / "artifacts/final/medium-transformers/seed-9801/tokenizer.json").read_text()),
        json.loads((root / "data/moonshot/phase2/word_preserving_bpe_2304.json").read_text()),
        json.loads((root / "data/moonshot/phase2/planner_preserving_bpe_2816.json").read_text()),
    ]
    for document in documents:
        tokenizer = BytePairTokenizer([tuple(pair) for pair in document["merges"]])
        for length in (1, 7, 31, 128, 511):
            generator = random.Random(2900 + length)
            value = bytes(generator.randrange(256) for _ in range(length))
            expected = list(value)
            for pair, new_id in tokenizer.merge_ids.items():
                replaced = []
                index = 0
                while index < len(expected):
                    if index + 1 < len(expected) and (expected[index], expected[index + 1]) == pair:
                        replaced.append(new_id)
                        index += 2
                    else:
                        replaced.append(expected[index])
                        index += 1
                expected = replaced
            assert tokenizer.encode(value) == expected
            assert tokenizer.decode(tokenizer.encode(value)) == value


def test_prompt_conditioned_cached_decode_matches_explicit_prompt_boundary() -> None:
    torch.manual_seed(42)
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=384, width=40, layers=4, heads=5, max_tokens=64,
        routed_experts=4, route_after_layers=2, prompt_conditioning=True,
    )).eval()
    prompt = torch.randint(0, 384, (1, 11))
    selected = torch.randint(0, 384, (1, 1))
    combined = torch.cat([prompt, selected], dim=1)
    full = model(combined, prompt_lengths=torch.tensor([prompt.shape[1]]))
    state = model.prefill(prompt)
    assert torch.allclose(state.next_logits, model(
        prompt, prompt_lengths=torch.tensor([prompt.shape[1]])
    )[:, -1], atol=2e-6, rtol=0.0)
    _, state = model.decode_step(state, selected[:, 0])
    assert torch.allclose(state.next_logits, full[:, -1], atol=2e-6, rtol=0.0)
    assert state.prompt_context is not None
    assert state.prompt_copy_bias.shape == (1, 384)


def test_checkpoint_bundles_generic_neural_guided_english_planner() -> None:
    model = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(
        vocab_size=384, width=40, layers=4, heads=5, max_tokens=64,
        routed_experts=4, route_after_layers=2, prompt_conditioning=True,
        constrained_english_planner=True,
    )).eval()
    assert "english_planner_spec" in model.state_dict()
    assert bytes(model.english_planner_spec.tolist()) == canonical_planner_bytes()
    assert model.planner_sha256() is not None
    spec = canonical_planner_bytes().decode("utf-8").casefold()
    frozen_topics = (
        "efficient computing", "public libraries", "urban gardens", "coastal weather",
        "scientific replication", "music practice", "safe navigation", "local history",
        "water conservation", "collaborative design",
    )
    assert all(topic not in spec for topic in frozen_topics)


def test_generic_english_planner_handles_unseen_subjects_tasks_and_recall() -> None:
    prompt = "Compare two approaches to orchard soil health and state one tradeoff."
    assert extract_subject(prompt) == "orchard soil health"
    assert classify_task(prompt) == "comparison"
    response = realize_english(prompt, variant=3)
    assert "orchard soil health" in response.casefold()
    assert "tradeoff" in response.casefold()
    assert len(response.split()) >= 80
    exact = realize_english("Write exactly two complete sentences about tidal turbines.")
    assert sum(exact.count(symbol) for symbol in ".!?") == 2
    recall = realize_english(
        "The exact codeword to retain is SAMPLE771. Read neutral words and reply with SAMPLE771 as requested."
    )
    assert recall.startswith("SAMPLE771 ")
