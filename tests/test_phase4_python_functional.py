from __future__ import annotations

import json
from pathlib import Path

import torch

from layercake.training.phase4_python_cake import (
    _causal_copy_labels,
    _execute_tests,
    _extract_function,
    _on_policy_recovery_sequence,
    _prompt_post_token_pairs,
    generate_diverse_training_dataset,
    generate_dataset,
    generate_identifier_generalization_dataset,
    generate_unique_identifier_dataset,
)


def test_on_policy_recovery_uses_generated_inputs_and_gold_targets():
    sequence, targets, mask = _on_policy_recovery_sequence(
        [10, 11, 12],
        [20, 21, 22, 23],
        [90, 91],
    )
    assert sequence == [10, 11, 12, 90, 91, 22, 23]
    assert targets.tolist() == [11, 12, 20, 21, 22, 23]
    assert mask.tolist() == [False, False, True, True, True, True]


def test_copy_labels_select_first_state_after_observed_prompt_token():
    # hidden[j] predicts targets[j].  A copied token at target position 1
    # is therefore represented by hidden[2], not hidden[1].
    targets = torch.tensor([[10, 42, 11, 42, 12]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    response = torch.tensor([[False, False, False, True, True]])
    labels = _causal_copy_labels(targets, valid, response)
    assert labels.tolist() == [[-100, -100, -100, 2, -100]]


def test_prompt_value_pairs_use_post_token_states():
    targets = torch.tensor([[10, 11, 12, 13, 14]])
    valid = torch.ones_like(targets, dtype=torch.bool)
    response = torch.tensor([[False, False, False, True, True]])
    batches, sources, observed = _prompt_post_token_pairs(
        targets, valid, response
    )
    assert batches.tolist() == [0, 0, 0]
    assert sources.tolist() == [1, 2, 3]
    assert observed.tolist() == [10, 11, 12]


def test_generated_dataset_has_disjoint_promoted_depth(tmp_path: Path):
    manifest = generate_dataset(tmp_path / "python.jsonl")
    assert manifest["counts"] == {
        "train": 960,
        "validation": 64,
        "test": 128,
    }
    assert manifest["training_test_prompt_overlap"] == 0
    assert not manifest["syntax_only_counts_as_success"]


def test_diverse_repair_changes_only_training_rows(tmp_path: Path):
    source = tmp_path / "v1" / "python.jsonl"
    generate_dataset(source)
    manifest = generate_diverse_training_dataset(
        source, tmp_path / "v2" / "python.jsonl"
    )
    assert manifest["validation_rows_unchanged"]
    assert manifest["test_rows_unchanged"]
    assert (
        manifest["split_hashes"]["train"]
        != manifest["source_split_hashes"]["train"]
    )


def test_identifier_repair_keeps_heldout_rows_and_removes_family_shortcut(
    tmp_path: Path,
):
    source = tmp_path / "v1" / "python.jsonl"
    generate_dataset(source)
    diverse = tmp_path / "v2" / "python.jsonl"
    generate_diverse_training_dataset(source, diverse)
    output = tmp_path / "v3" / "python.jsonl"
    manifest = generate_identifier_generalization_dataset(diverse, output)
    assert manifest["validation_rows_unchanged"]
    assert manifest["test_rows_unchanged"]
    first = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert first["family"] not in first["function_name"]


def test_unique_identifier_repair_has_no_repeated_training_name(tmp_path: Path):
    source = tmp_path / "v1" / "python.jsonl"
    generate_dataset(source)
    output = tmp_path / "v4" / "python.jsonl"
    manifest = generate_unique_identifier_dataset(source, output)
    assert manifest["validation_rows_unchanged"]
    assert manifest["test_rows_unchanged"]
    assert (
        manifest["unique_training_identifiers"]
        == manifest["training_identifier_count"]
        == 960
    )


def test_safe_function_must_pass_real_unit_tests():
    source = "def add_test(a, b):\n    return a + b\n"
    passed, records = _execute_tests(
        source,
        "add_test",
        [
            {"args": [2, 3], "expected": 5},
            {"args": [-4, 1], "expected": -3},
        ],
    )
    assert passed
    assert all(record["status"] == "PASS" for record in records)


def test_unsafe_or_wrong_generated_code_is_not_functional_success():
    unsafe = "def add_test(a, b):\n    return __import__('os').getcwd()\n"
    passed, _ = _execute_tests(
        unsafe, "add_test", [{"args": [1, 2], "expected": 3}]
    )
    assert not passed
    wrong = "def add_test(a, b):\n    return a - b\n"
    passed, records = _execute_tests(
        wrong, "add_test", [{"args": [1, 2], "expected": 3}]
    )
    assert not passed
    assert records[0]["status"] == "WRONG_RESULT"


def test_extracts_only_the_expected_function():
    source, status = _extract_function(
        "some prose\ndef wanted(value):\n    return value * 2\n",
        "wanted",
    )
    assert status == "PARSED"
    assert source == "def wanted(value):\n    return value * 2\n"


def test_extracts_complete_function_before_invalid_trailing_generation():
    source, status = _extract_function(
        "def wanted(value):\n    return value * 2\nthis is invalid prose (\n",
        "wanted",
    )
    assert status == "PARSED"
    assert source == "def wanted(value):\n    return value * 2\n"
