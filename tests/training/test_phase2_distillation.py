import hashlib
import json
from pathlib import Path

import torch

from layercake.models.baseline_transformer import BytePairTokenizer
from layercake.phase1_campaign import _headline_prompts
from layercake.training.phase2_distillation import (
    TOPICS,
    _instruction_batch,
    _instruction_focus_mask,
    _negative_prompt_rows,
    _prompt_rows,
)


def test_distillation_prompts_exclude_frozen_phase1_topics_and_exact_prompts() -> None:
    frozen = _headline_prompts()
    frozen_hashes = {row["sha256"] for row in frozen}
    frozen_topics = {
        "efficient computing", "public libraries", "urban gardens", "coastal weather",
        "scientific replication", "music practice", "safe navigation", "local history",
        "water conservation", "collaborative design",
    }
    rows = _prompt_rows()
    assert len(rows) == 360
    assert frozen_topics.isdisjoint(TOPICS)
    assert not frozen_hashes.intersection(row["prompt_sha256"] for row in rows)


def test_instruction_batch_masks_prompt_and_padding_tokens() -> None:
    tokenizer = BytePairTokenizer()
    tokens, labels, response_tokens, prompt_lengths = _instruction_batch(
        tokenizer,
        [
            {"prompt": "Question", "response": "Answer"},
            {"prompt": "Longer question", "response": "Reply"},
        ],
        device="cpu",
        max_tokens=64,
    )
    assert tokens.shape[0] == labels.shape[0] == 2
    assert tokens.shape[1] == labels.shape[1] + 1
    assert response_tokens == len("Answer") + len("Reply")
    assert (labels == -100).any()
    assert prompt_lengths.tolist() == [len("Question\n"), len("Longer question\n")]


def test_focus_mask_finds_topic_only_in_response_targets() -> None:
    tokenizer = BytePairTokenizer()
    rows = [{
        "prompt": "Teach me about river ecology.",
        "response": "river ecology connects water and life.",
        "topic": "river ecology",
        "task": "teach",
    }]
    _, labels, _, _ = _instruction_batch(
        tokenizer, rows, device="cpu", max_tokens=128
    )
    mask = _instruction_focus_mask(
        tokenizer, rows, labels, device=torch.device("cpu")
    )
    assert int(mask.sum()) == len(tokenizer.encode("river ecology"))
    assert bool((labels[mask] >= 0).all())


def test_negative_prompt_pair_preserves_task_and_changes_topic() -> None:
    rows = [
        {"prompt": "A", "response": "one", "topic": "river", "task": "teach"},
        {"prompt": "B", "response": "two", "topic": "forest", "task": "teach"},
    ]
    negative = _negative_prompt_rows(rows, rows)
    assert [row["task"] for row in negative] == ["teach", "teach"]
    assert [row["topic"] for row in negative] == ["forest", "river"]


def test_curated_distillation_artifact_preserves_frozen_suite_separation() -> None:
    root = Path(__file__).resolve().parents[2]
    corpus = root / "data/moonshot/phase2/instruction_distillation_curated.jsonl"
    manifest = json.loads(corpus.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    payload = corpus.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    exact_two = [row for row in rows if str(row["task"]) == "4"]
    recall = [row for row in rows if row["task"] == "long_context_recall"]
    assert manifest["status"] == "PASS"
    assert hashlib.sha256(payload).hexdigest() == manifest["corpus_sha256"]
    assert len(rows) == 480
    assert len(exact_two) == 60
    assert all(sum(row["response"].count(mark) for mark in ".!?") == 2 for row in exact_two)
    assert len(recall) == 120
    assert manifest["exact_phase1_prompt_overlap"] == 0
    assert manifest["frozen_long_context_codeword_overlap"] == []


def test_clean_curriculum_and_planner_tokenizer_remain_disjoint_and_exact() -> None:
    root = Path(__file__).resolve().parents[2]
    curriculum = root / "data/moonshot/phase2/instruction_curriculum_clean.jsonl"
    curriculum_manifest = json.loads(
        curriculum.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    tokenizer_path = root / "data/moonshot/phase2/planner_preserving_bpe_2816.json"
    tokenizer_manifest = json.loads(
        tokenizer_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    base = json.loads(
        (root / "data/moonshot/phase2/word_preserving_bpe_2304.json").read_text(encoding="utf-8")
    )
    extended = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert curriculum_manifest["status"] == "PASS"
    assert curriculum_manifest["exact_phase1_prompt_overlap"] == 0
    assert curriculum_manifest["frozen_answers_used"] is False
    assert curriculum_manifest["frozen_long_context_codewords_used"] is False
    assert tokenizer_manifest["status"] == "PASS"
    assert tokenizer_manifest["frozen_evaluation_content"] is False
    assert tokenizer_manifest["base_merges_preserved_as_exact_prefix"] is True
    assert extended["merges"][:len(base["merges"])] == base["merges"]
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == tokenizer_manifest["tokenizer_sha256"]
