import torch

from layercake.phase2_recertification import (
    _instruction_examples,
    _instruction_focus_masks,
    _instruction_sequence_coverage,
    _negative_prompt_indexes,
    _semantic_adherence_metrics,
)


def _row(prompt: str, response: str, task: str) -> dict[str, str]:
    return {
        "prompt": prompt,
        "response": response,
        "task": task,
    }


def test_instruction_sequence_coverage_exposes_truncated_responses() -> None:
    rows = [
        _row("short", "response", "a"),
        _row("x" * 32, "answer", "a"),
    ]
    coverage = _instruction_sequence_coverage(rows, 16)
    assert coverage["examples"] == 2
    assert coverage["fully_covered_response_examples"] == 1
    assert coverage["zero_response_target_examples"] == 1


def test_negative_prompts_stay_within_task_and_change_identity() -> None:
    rows = [
        _row("a0", "r0", "a"),
        _row("a1", "r1", "a"),
        _row("b0", "r2", "b"),
        _row("b1", "r3", "b"),
    ]
    negatives = _negative_prompt_indexes(rows)
    for index, negative in enumerate(negatives):
        assert negative != index
        assert rows[negative]["task"] == rows[index]["task"]


def test_mismatched_instruction_example_keeps_response_targets() -> None:
    rows = [
        _row("topic alpha", "answer alpha", "same"),
        _row("topic beta", "answer beta", "same"),
    ]
    correct, correct_mask = _instruction_examples(
        rows, [0], 32, torch.device("cpu")
    )
    wrong, wrong_mask = _instruction_examples(
        rows,
        [0],
        32,
        torch.device("cpu"),
        prompt_indexes=[1],
    )
    assert not torch.equal(correct, wrong)
    correct_targets = correct[:, 1:][correct_mask]
    wrong_targets = wrong[:, 1:][wrong_mask]
    assert bytes(correct_targets.tolist()) == b"answer alpha"
    assert bytes(wrong_targets.tolist()) == b"answer alpha"


def test_instruction_focus_mask_tracks_topic_after_prompt() -> None:
    rows = [{
        "prompt": "Explain alpha beta.",
        "response": "Alpha beta matters.",
        "task": "explain",
        "topic": "alpha beta",
    }]
    batch, response_mask = _instruction_examples(
        rows,
        [0],
        48,
        torch.device("cpu"),
        append_newline=False,
    )
    focus = _instruction_focus_masks(
        rows,
        [0],
        48,
        torch.device("cpu"),
        append_newline=False,
    )
    assert torch.all(focus <= response_mask)
    assert bytes(batch[:, 1:][focus].tolist()).lower() == b"alpha beta"


def test_semantic_adherence_requires_topic_length_and_structure() -> None:
    prompt = (
        "Write exactly two complete sentences about efficient computing. "
        "Your response must contain at least 80 words."
    )
    response = (
        ("Efficient computing reduces energy use while preserving useful "
         "performance " * 8).strip()
        + ". "
        + ("Efficient computing also lets modest hardware complete more "
           "valuable work " * 8).strip()
        + "."
    ).encode()
    metrics = _semantic_adherence_metrics(
        prompt, "instruction_following", response
    )
    assert metrics["minimum_80_words_pass"]
    assert metrics["topic_phrase_present"]
    assert metrics["sentence_count"] == 2
    assert metrics["core_adherence_pass"]
