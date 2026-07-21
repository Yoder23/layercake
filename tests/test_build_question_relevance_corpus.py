from __future__ import annotations

import json

from scripts.build_question_relevance_corpus import (
    EVAL_PROMPTS,
    build_eval_questions,
    build_rows,
)


def test_question_relevance_corpus_excludes_exact_guardrail_prompts():
    rows = build_rows(2)
    payloads = [json.loads(row)["text"] for row in rows]

    for prompt in EVAL_PROMPTS:
        assert all(not text.startswith(prompt) for text in payloads)
    assert any("top right" in text and "Save" in text for text in payloads)
    assert any("<item id=" in text for text in payloads)
    assert all('"item":{' not in text for text in payloads)
    assert any('"attrs":{"id":"42"}' in text for text in payloads)
    assert any('"target":"button#save"' in text for text in payloads)
    assert any("Given XML element" in text for text in payloads)
    assert any("Return the JSON edit action" in text for text in payloads)
    assert any("A user requests moving" in text for text in payloads)


def test_question_relevance_eval_is_heldout_and_uses_canonical_json():
    training_payloads = [json.loads(row)["text"] for row in build_rows(2)]
    evaluation = build_eval_questions()

    assert evaluation["seen"] == []
    assert len(evaluation["heldout"]) == 20
    assert all(
        not any(text.startswith(row["prompt"]) for text in training_payloads)
        for row in evaluation["heldout"]
    )
    assert all(row["expected"].startswith("{") for row in evaluation["heldout"])
