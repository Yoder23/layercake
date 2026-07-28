from __future__ import annotations

import torch

from layercake.portable_token_plan import (
    EOS_ID,
    LosslessLexemePointerTokenizer,
    PortableTokenPlan,
    build_token_plan_artifact,
    load_token_plan_artifact,
)


def _rows():
    return [
        {
            "function_name": "novel_name_001",
            "prompt": "Define novel_name_001(a, b) and add both values.",
            "response": "def novel_name_001(a, b):\n    return a + b\n",
        },
        {
            "function_name": "other_name_002",
            "prompt": "Define other_name_002(value) and return value.",
            "response": "def other_name_002(value):\n    return value\n",
        },
    ]


def test_lossless_token_plan_uses_one_pointer_for_identifier():
    rows = _rows()
    tokenizer = LosslessLexemePointerTokenizer.build(rows)
    source_ids, source = tokenizer.encode_source(rows[0]["prompt"])
    actions = tokenizer.encode_target(
        rows[0]["response"],
        function_name=rows[0]["function_name"],
        source_lexemes=source,
    )
    pointer_actions = [
        action for action in actions if action >= tokenizer.vocab_size
    ]
    assert len(pointer_actions) == 1
    assert tokenizer.decode_actions(actions, source) == rows[0][
        "response"
    ].encode("utf-8")
    assert EOS_ID == actions[-1]
    assert len(source_ids) == len(source)


def test_token_plan_artifact_round_trip_preserves_logits():
    torch.manual_seed(71)
    tokenizer = LosslessLexemePointerTokenizer.build(_rows())
    model = PortableTokenPlan(
        fixed_vocab_size=tokenizer.vocab_size,
        model_width=24,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_width=48,
        pointer_width=12,
        dropout=0.0,
        maximum_source_lexemes=32,
        maximum_target_actions=32,
    ).eval()
    source = torch.tensor([[4, 5, 6, 0]], dtype=torch.long)
    target = torch.tensor([[4, tokenizer.vocab_size + 1, EOS_ID]])
    expected = model(source, target)["log_probs"]
    artifact = build_token_plan_artifact(model, tokenizer)
    _, loaded_tokenizer, loaded = load_token_plan_artifact(artifact)
    actual = loaded(source, target)["log_probs"]
    assert loaded_tokenizer.canonical_dict() == tokenizer.canonical_dict()
    assert torch.equal(expected, actual)


def test_token_plan_generation_emits_only_valid_extended_actions():
    torch.manual_seed(73)
    tokenizer = LosslessLexemePointerTokenizer.build(_rows())
    model = PortableTokenPlan(
        fixed_vocab_size=tokenizer.vocab_size,
        model_width=24,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_width=48,
        pointer_width=12,
        dropout=0.0,
        maximum_source_lexemes=32,
        maximum_target_actions=8,
    ).eval()
    source_ids, _ = tokenizer.encode_source(_rows()[0]["prompt"])
    source = torch.tensor([source_ids], dtype=torch.long)
    actions = model.generate_actions(source, maximum_actions=4)[0]
    assert actions
    assert all(
        0 <= action < tokenizer.vocab_size + len(source_ids)
        for action in actions
    )
