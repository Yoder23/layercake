"""Phase 4 frozen-core training and functional evaluation for a Python cake.

This module deliberately keeps the Phase 2 model immutable.  It trains only a
``HostResidualCake`` over the canonical 768-wide semantic state and evaluates
autonomous code with held-out prompts and executable unit tests.  Syntax,
perplexity, and teacher-forced token accuracy are diagnostics, never functional
successes.
"""

from __future__ import annotations

import argparse
import ast
import builtins
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Sequence

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from layercake.models.routed_cakes import HostResidualCake
from layercake.domain_runtime import (
    AttentiveHostResidualCake,
    RecurrentHostResidualCake,
)
from layercake.training.data import sha256_file
from layercake.training.phase2_shallow_sparse import load_student


ROOT = Path(__file__).resolve().parents[2]
ABI_PATH = ROOT / "moonshot" / "phase2_canonical_semantic_abi_r3.json"
DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "moonshot"
    / "phase2_shallow_sparse_pretrained"
    / "student2400-seed-9824"
)
DEFAULT_DATASET = (
    ROOT / "data" / "moonshot" / "phase4" / "python_functional_v1.jsonl"
)
_COPY_PATH_PARAMETER_NAMES = frozenset(
    {
        "copy_query.weight",
        "copy_key.weight",
        "copy_gate.weight",
        "copy_gate.bias",
        "copy_transition_value.weight",
    }
)


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _tensor_subset_sha256(
    state: dict[str, torch.Tensor], names: set[str]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _configure_copy_path_only(
    cake: AttentiveHostResidualCake,
) -> tuple[str, ...]:
    available = {name for name, _ in cake.named_parameters()}
    missing = _COPY_PATH_PARAMETER_NAMES - available
    if missing:
        raise ValueError(
            "copy-path-only training requires selective transition copy; "
            f"missing {sorted(missing)}"
        )
    for name, parameter in cake.named_parameters():
        parameter.requires_grad_(name in _COPY_PATH_PARAMETER_NAMES)
    return tuple(sorted(_COPY_PATH_PARAMETER_NAMES))


@dataclass(frozen=True)
class Family:
    family_id: str
    parameters: tuple[str, ...]
    descriptions: tuple[str, ...]
    body: tuple[str, ...]
    cases: Callable[[random.Random], list[list[Any]]]
    oracle: Callable[..., Any]


def _families() -> tuple[Family, ...]:
    pair = lambda rng: [
        [rng.randint(-50, 50), rng.randint(-50, 50)] for _ in range(4)
    ]
    numbers = lambda rng: [
        [[rng.randint(-20, 20) for _ in range(size)]]
        for size in (0, 1, 5, 9)
    ]
    words = ("Layer Cake", "  quiet bridge  ", "Red BLUE red", "a  b   c")
    return (
        Family(
            "add",
            ("a", "b"),
            ("returns the sum of the two values", "adds both numbers"),
            ("return a + b",),
            pair,
            lambda a, b: a + b,
        ),
        Family(
            "subtract",
            ("a", "b"),
            ("subtracts the second value from the first", "returns a minus b"),
            ("return a - b",),
            pair,
            lambda a, b: a - b,
        ),
        Family(
            "multiply",
            ("a", "b"),
            ("returns the product of the two values", "multiplies both numbers"),
            ("return a * b",),
            pair,
            lambda a, b: a * b,
        ),
        Family(
            "absolute_difference",
            ("a", "b"),
            ("returns the absolute difference between two numbers",),
            ("return abs(a - b)",),
            pair,
            lambda a, b: abs(a - b),
        ),
        Family(
            "clamp",
            ("value", "low", "high"),
            ("clamps value to the inclusive low and high bounds",),
            ("return max(low, min(value, high))",),
            lambda rng: [[-4, 0, 10], [4, 0, 10], [14, 0, 10], [3, 3, 3]],
            lambda value, low, high: max(low, min(value, high)),
        ),
        Family(
            "is_even",
            ("number",),
            ("returns True exactly when number is even",),
            ("return number % 2 == 0",),
            lambda rng: [[value] for value in (-11, 0, 8, 17)],
            lambda number: number % 2 == 0,
        ),
        Family(
            "palindrome",
            ("text",),
            ("ignores letter case and returns whether text is a palindrome",),
            (
                "normalized = text.lower()",
                "return normalized == normalized[::-1]",
            ),
            lambda rng: [[value] for value in ("Level", "Python", "", "Rotor")],
            lambda text: text.lower() == text.lower()[::-1],
        ),
        Family(
            "reverse_text",
            ("text",),
            ("returns the characters of text in reverse order",),
            ("return text[::-1]",),
            lambda rng: [[value] for value in ("cake", "", "abc def", "x")],
            lambda text: text[::-1],
        ),
        Family(
            "count_vowels",
            ("text",),
            ("counts vowels in text without regard to case",),
            ("return sum(1 for char in text.lower() if char in \"aeiou\")",),
            lambda rng: [[value] for value in ("LayerCake", "", "rhythm", "AEIOU")],
            lambda text: sum(1 for char in text.lower() if char in "aeiou"),
        ),
        Family(
            "factorial",
            ("number",),
            ("returns the factorial of a nonnegative integer",),
            (
                "result = 1",
                "for value in range(2, number + 1):",
                "    result *= value",
                "return result",
            ),
            lambda rng: [[value] for value in (0, 1, 5, 7)],
            math.factorial,
        ),
        Family(
            "fibonacci",
            ("count",),
            ("returns a list containing the first count Fibonacci numbers",),
            (
                "values = []",
                "a, b = 0, 1",
                "for _ in range(count):",
                "    values.append(a)",
                "    a, b = b, a + b",
                "return values",
            ),
            lambda rng: [[value] for value in (0, 1, 5, 8)],
            lambda count: (
                lambda values: values
            )(_fibonacci(count)),
        ),
        Family(
            "is_prime",
            ("number",),
            ("returns whether number is prime",),
            (
                "if number < 2:",
                "    return False",
                "for divisor in range(2, int(number ** 0.5) + 1):",
                "    if number % divisor == 0:",
                "        return False",
                "return True",
            ),
            lambda rng: [[value] for value in (1, 2, 17, 21)],
            _is_prime,
        ),
        Family(
            "deduplicate",
            ("values",),
            ("removes duplicate values while preserving first-seen order",),
            (
                "result = []",
                "for value in values:",
                "    if value not in result:",
                "        result.append(value)",
                "return result",
            ),
            lambda rng: [
                [[1, 2, 1, 3, 2]],
                [[]],
                [["a", "a", "b"]],
                [[4, 4, 4]],
            ],
            lambda values: list(dict.fromkeys(values)),
        ),
        Family(
            "running_totals",
            ("values",),
            ("returns the running totals of a numeric list",),
            (
                "result = []",
                "total = 0",
                "for value in values:",
                "    total += value",
                "    result.append(total)",
                "return result",
            ),
            numbers,
            lambda values: _running_totals(values),
        ),
        Family(
            "flatten_once",
            ("groups",),
            ("flattens one level of nested lists",),
            (
                "result = []",
                "for group in groups:",
                "    result.extend(group)",
                "return result",
            ),
            lambda rng: [
                [[[1, 2], [3], []]],
                [[["a"], ["b", "c"]]],
                [[]],
                [[[0], [1, 2, 3]]],
            ],
            lambda groups: [item for group in groups for item in group],
        ),
        Family(
            "maximum",
            ("values",),
            ("returns the largest item in a nonempty list",),
            ("return max(values)",),
            lambda rng: [[[1]], [[-2, -9, -1]], [[3, 7, 2]], [[0, 0]]],
            max,
        ),
        Family(
            "minimum",
            ("values",),
            ("returns the smallest item in a nonempty list",),
            ("return min(values)",),
            lambda rng: [[[1]], [[-2, -9, -1]], [[3, 7, 2]], [[0, 0]]],
            min,
        ),
        Family(
            "average",
            ("values",),
            ("returns the arithmetic mean of a nonempty numeric list",),
            ("return sum(values) / len(values)",),
            lambda rng: [[[2]], [[1, 2, 3]], [[-2, 2]], [[1.5, 2.5]]],
            lambda values: sum(values) / len(values),
        ),
        Family(
            "word_count",
            ("text",),
            ("counts whitespace-separated words in text",),
            ("return len(text.split())",),
            lambda rng: [[value] for value in words],
            lambda text: len(text.split()),
        ),
        Family(
            "slugify",
            ("text",),
            ("strips text, lowercases it, and replaces spaces with hyphens",),
            ("return \"-\".join(text.strip().lower().split())",),
            lambda rng: [[value] for value in words],
            lambda text: "-".join(text.strip().lower().split()),
        ),
        Family(
            "merge_mappings",
            ("left", "right"),
            ("returns a new dictionary where right-hand values override left",),
            ("return {**left, **right}",),
            lambda rng: [
                [{}, {}],
                [{"a": 1}, {"b": 2}],
                [{"a": 1}, {"a": 2}],
                [{"x": 0, "y": 1}, {"y": 4}],
            ],
            lambda left, right: {**left, **right},
        ),
        Family(
            "invert_mapping",
            ("mapping",),
            ("swaps the keys and values of a dictionary",),
            ("return {value: key for key, value in mapping.items()}",),
            lambda rng: [
                [{}],
                [{"a": 1}],
                [{"a": 1, "b": 2}],
                [{"x": "red", "y": "blue"}],
            ],
            lambda mapping: {value: key for key, value in mapping.items()},
        ),
        Family(
            "character_frequencies",
            ("text",),
            ("returns a dictionary counting every character in text",),
            (
                "counts = {}",
                "for char in text:",
                "    counts[char] = counts.get(char, 0) + 1",
                "return counts",
            ),
            lambda rng: [[value] for value in ("", "aba", "cake", "a a")],
            lambda text: _frequencies(text),
        ),
        Family(
            "safe_divide",
            ("numerator", "denominator"),
            ("returns None for a zero denominator and the quotient otherwise",),
            (
                "if denominator == 0:",
                "    return None",
                "return numerator / denominator",
            ),
            lambda rng: [[4, 2], [1, 0], [-9, 3], [0, 5]],
            lambda numerator, denominator: (
                None if denominator == 0 else numerator / denominator
            ),
        ),
        Family(
            "chunks",
            ("values", "size"),
            ("splits values into consecutive lists of at most size items",),
            (
                "return [values[index:index + size] for index in range(0, len(values), size)]",
            ),
            lambda rng: [
                [[1, 2, 3, 4, 5], 2],
                [[], 3],
                [[1, 2], 5],
                [["a", "b", "c"], 1],
            ],
            lambda values, size: [
                values[index:index + size]
                for index in range(0, len(values), size)
            ],
        ),
        Family(
            "rotate_left",
            ("values", "places"),
            ("rotates a nonempty list left by places positions",),
            (
                "offset = places % len(values)",
                "return values[offset:] + values[:offset]",
            ),
            lambda rng: [
                [[1, 2, 3], 1],
                [[1, 2, 3], 4],
                [["a"], 9],
                [[0, 1, 2, 3], 2],
            ],
            lambda values, places: (
                values[places % len(values):] + values[:places % len(values)]
            ),
        ),
        Family(
            "find_index",
            ("values", "target"),
            ("returns the first index of target or minus one when absent",),
            (
                "for index, value in enumerate(values):",
                "    if value == target:",
                "        return index",
                "return -1",
            ),
            lambda rng: [
                [[1, 2, 3], 2],
                [[1, 2, 1], 1],
                [[], 4],
                [["a", "b"], "x"],
            ],
            lambda values, target: (
                values.index(target) if target in values else -1
            ),
        ),
        Family(
            "all_positive",
            ("values",),
            ("returns whether every numeric value is strictly positive",),
            ("return all(value > 0 for value in values)",),
            lambda rng: [[[1, 2]], [[1, 0]], [[-1, 2]], [[]]],
            lambda values: all(value > 0 for value in values),
        ),
        Family(
            "balanced_parentheses",
            ("text",),
            ("returns whether parentheses in text are balanced",),
            (
                "depth = 0",
                "for char in text:",
                "    if char == \"(\":",
                "        depth += 1",
                "    elif char == \")\":",
                "        depth -= 1",
                "        if depth < 0:",
                "            return False",
                "return depth == 0",
            ),
            lambda rng: [[value] for value in ("()", "(())", "(()", ")(")],
            _balanced_parentheses,
        ),
        Family(
            "title_words",
            ("text",),
            ("returns text converted to title case",),
            ("return text.title()",),
            lambda rng: [[value] for value in words],
            lambda text: text.title(),
        ),
        Family(
            "remove_none",
            ("values",),
            ("returns a list with every None item removed",),
            ("return [value for value in values if value is not None]",),
            lambda rng: [
                [[1, None, 2]],
                [[None]],
                [[]],
                [["a", None, "b"]],
            ],
            lambda values: [value for value in values if value is not None],
        ),
        Family(
            "sort_values",
            ("values",),
            ("returns the values in ascending order without changing the input",),
            ("return sorted(values)",),
            numbers,
            sorted,
        ),
    )


def _fibonacci(count: int) -> list[int]:
    values: list[int] = []
    a, b = 0, 1
    for _ in range(count):
        values.append(a)
        a, b = b, a + b
    return values


def _is_prime(number: int) -> bool:
    if number < 2:
        return False
    for divisor in range(2, int(number ** 0.5) + 1):
        if number % divisor == 0:
            return False
    return True


def _running_totals(values: list[float]) -> list[float]:
    result = []
    total = 0
    for value in values:
        total += value
        result.append(total)
    return result


def _frequencies(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for char in text:
        result[char] = result.get(char, 0) + 1
    return result


def _balanced_parentheses(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _render_response(name: str, family: Family) -> str:
    lines = [f"def {name}({', '.join(family.parameters)}):"]
    lines.extend(f"    {line}" if line else "" for line in family.body)
    return "\n".join(lines) + "\n"


def generate_dataset(output: Path, *, seed: int = 9404) -> dict[str, Any]:
    """Create disjoint train/validation/test prompts before model training."""

    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise RuntimeError(f"dataset artifact is immutable: {output}")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for family in _families():
        for index in range(30):
            name = f"{family.family_id}_train_{index:02d}"
            description = family.descriptions[index % len(family.descriptions)]
            prompt = (
                f"Write only valid Python code. Define {name}"
                f"({', '.join(family.parameters)}) that {description}."
            )
            rows.append(
                {
                    "id": f"train-{family.family_id}-{index:02d}",
                    "split": "train",
                    "family": family.family_id,
                    "prompt": prompt,
                    "response": _render_response(name, family),
                    "function_name": name,
                    "tests": [],
                }
            )
        for split, count, start in (("validation", 2, 30), ("test", 4, 32)):
            for offset in range(count):
                index = start + offset
                name = f"{family.family_id}_{split}_{offset:02d}"
                description = family.descriptions[
                    (index + 1) % len(family.descriptions)
                ]
                prompt = (
                    f"Return code only. Create a Python function named {name}"
                    f" with parameters {', '.join(family.parameters)}. It {description}."
                )
                cases = []
                for arguments in family.cases(rng):
                    cases.append(
                        {
                            "args": _jsonable(arguments),
                            "expected": _jsonable(family.oracle(*arguments)),
                        }
                    )
                rows.append(
                    {
                        "id": f"{split}-{family.family_id}-{offset:02d}",
                        "split": split,
                        "family": family.family_id,
                        "prompt": prompt,
                        "response": _render_response(name, family),
                        "function_name": name,
                        "tests": cases,
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )
    output.write_text(raw, encoding="utf-8")
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    }
    manifest = {
        "format": "layercake-phase4-python-functional-dataset/1",
        "status": "PREREGISTERED",
        "seed": seed,
        "dataset": (
            output.relative_to(ROOT).as_posix()
            if output.is_relative_to(ROOT)
            else str(output)
        ),
        "dataset_sha256": sha256_file(output),
        "counts": counts,
        "families": len(_families()),
        "test_definition": (
            "autonomous generated function passes every held-out unit test"
        ),
        "syntax_only_counts_as_success": False,
        "teacher_forced_metrics_count_as_success": False,
        "training_test_prompt_overlap": 0,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_diverse_training_dataset(
    source: Path,
    output: Path,
    *,
    seed: int = 9414,
) -> dict[str, Any]:
    """Repair only the training curriculum; preserve held-out rows byte-for-byte."""

    source = source if source.is_absolute() else ROOT / source
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise RuntimeError(f"dataset artifact is immutable: {output}")
    source_rows = _load_rows(source)
    families = {family.family_id: family for family in _families()}
    adjectives = (
        "amber",
        "calm",
        "delta",
        "exact",
        "green",
        "lunar",
        "rapid",
        "silver",
    )
    templates = (
        "Write only valid Python code. Define {name}({parameters}) that {description}.",
        "Return only Python source. Implement {name}({parameters}); it {description}.",
        "No prose. Create a Python function called {name} with parameters {parameters} that {description}.",
        "Produce valid code for {name}({parameters}). The function {description}.",
        "Define the Python callable {name}({parameters}) so it {description}. Return code only.",
        "Generate only a function named {name} taking {parameters}; it {description}.",
    )
    rows = []
    for row in source_rows:
        if row["split"] != "train":
            # Object reuse plus canonical serialization below guarantees the
            # semantic held-out records are identical, which the manifest hashes.
            rows.append(row)
            continue
        family = families[row["family"]]
        index = int(row["id"].rsplit("-", 1)[1])
        pattern = index % 4
        if pattern == 0:
            name = f"{family.family_id}_{adjectives[index % len(adjectives)]}_{index:02d}"
        elif pattern == 1:
            name = f"compute_{family.family_id}_{100 + index}"
        elif pattern == 2:
            name = f"lc_{adjectives[index % len(adjectives)]}_{family.family_id}"
        else:
            name = f"{family.family_id}_implementation_{index:02d}"
        description = family.descriptions[index % len(family.descriptions)]
        prompt = templates[index % len(templates)].format(
            name=name,
            parameters=", ".join(family.parameters),
            description=description,
        )
        rows.append(
            {
                **row,
                "prompt": prompt,
                "response": _render_response(name, family),
                "function_name": name,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    def split_sha(items: list[dict[str, Any]], split: str) -> str:
        return _canonical_sha([row for row in items if row["split"] == split])
    manifest = {
        "format": "layercake-phase4-python-functional-dataset/2",
        "status": "PREREGISTERED_TRAINING_ONLY_REPAIR",
        "seed": seed,
        "source_dataset": (
            source.relative_to(ROOT).as_posix()
            if source.is_relative_to(ROOT)
            else str(source)
        ),
        "source_dataset_sha256": sha256_file(source),
        "dataset": (
            output.relative_to(ROOT).as_posix()
            if output.is_relative_to(ROOT)
            else str(output)
        ),
        "dataset_sha256": sha256_file(output),
        "counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "split_hashes": {
            split: split_sha(rows, split)
            for split in ("train", "validation", "test")
        },
        "source_split_hashes": {
            split: split_sha(source_rows, split)
            for split in ("train", "validation", "test")
        },
        "validation_rows_unchanged": split_sha(rows, "validation")
        == split_sha(source_rows, "validation"),
        "test_rows_unchanged": split_sha(rows, "test")
        == split_sha(source_rows, "test"),
        "training_change": (
            "six instruction phrasings and four identifier patterns replace "
            "the leaked _train_## marker"
        ),
        "test_definition_unchanged": True,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    output.with_name("manifest_v2.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_identifier_generalization_dataset(
    source: Path,
    output: Path,
    *,
    seed: int = 9424,
) -> dict[str, Any]:
    """Force behavior-from-description and exact copying of varied identifiers."""

    source = source if source.is_absolute() else ROOT / source
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise RuntimeError(f"dataset artifact is immutable: {output}")
    source_rows = _load_rows(source)
    families = {family.family_id: family for family in _families()}
    prefixes = (
        "amber",
        "brisk",
        "careful",
        "delta",
        "exact",
        "lunar",
        "quiet",
        "silver",
    )
    suffixes = (
        "calculation",
        "configuration",
        "integration",
        "operation",
        "processor",
        "transformation",
        "validation",
        "workflow",
    )
    templates = (
        "Write only valid Python code. Define {name}({parameters}) that {description}.",
        "Return code only. Create a Python function named {name} with parameters {parameters}. It {description}.",
        "No prose: implement {name}({parameters}); this callable {description}.",
        "Produce valid Python source for a function called {name}, taking {parameters}, that {description}.",
        "Define {name}({parameters}) in Python so it {description}. Output only code.",
        "Generate only the Python function {name}({parameters}). It {description}.",
    )
    rows = []
    for row in source_rows:
        if row["split"] != "train":
            rows.append(row)
            continue
        family = families[row["family"]]
        index = int(row["id"].rsplit("-", 1)[1])
        family_index = list(families).index(family.family_id)
        prefix = prefixes[(index + family_index) % len(prefixes)]
        suffix = suffixes[(3 * index + family_index) % len(suffixes)]
        if index % 3 == 0:
            name = f"{prefix}_{suffix}_{100 + index}"
        elif index % 3 == 1:
            name = f"apply_{prefix}_{suffix}"
        else:
            name = f"{suffix}_{prefix}_{200 + index}"
        description = family.descriptions[index % len(family.descriptions)]
        prompt = templates[index % len(templates)].format(
            name=name,
            parameters=", ".join(family.parameters),
            description=description,
        )
        rows.append(
            {
                **row,
                "prompt": prompt,
                "response": _render_response(name, family),
                "function_name": name,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    def split_sha(items: list[dict[str, Any]], split: str) -> str:
        return _canonical_sha([row for row in items if row["split"] == split])
    manifest = {
        "format": "layercake-phase4-python-functional-dataset/3",
        "status": "PREREGISTERED_IDENTIFIER_GENERALIZATION_REPAIR",
        "seed": seed,
        "source_dataset": (
            source.relative_to(ROOT).as_posix()
            if source.is_relative_to(ROOT)
            else str(source)
        ),
        "source_dataset_sha256": sha256_file(source),
        "dataset": (
            output.relative_to(ROOT).as_posix()
            if output.is_relative_to(ROOT)
            else str(output)
        ),
        "dataset_sha256": sha256_file(output),
        "counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "split_hashes": {
            split: split_sha(rows, split)
            for split in ("train", "validation", "test")
        },
        "source_split_hashes": {
            split: split_sha(source_rows, split)
            for split in ("train", "validation", "test")
        },
        "validation_rows_unchanged": split_sha(rows, "validation")
        == split_sha(source_rows, "validation"),
        "test_rows_unchanged": split_sha(rows, "test")
        == split_sha(source_rows, "test"),
        "training_change": (
            "task-independent multi-token identifiers plus six prompt forms; "
            "behavior must be inferred from the description"
        ),
        "test_definition_unchanged": True,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    output.with_name("manifest_v3.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_unique_identifier_dataset(
    source: Path,
    output: Path,
    *,
    seed: int = 9434,
) -> dict[str, Any]:
    """Make every training identifier unique while preserving held-out rows."""

    source = source if source.is_absolute() else ROOT / source
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise RuntimeError(f"dataset artifact is immutable: {output}")
    source_rows = _load_rows(source)
    words = (
        "amber",
        "brisk",
        "careful",
        "delta",
        "exact",
        "lunar",
        "quiet",
        "silver",
        "vector",
        "willow",
    )
    rows = []
    train_index = 0
    for row in source_rows:
        if row["split"] != "train":
            rows.append(row)
            continue
        left = words[train_index % len(words)]
        right = words[(train_index * 7 + 3) % len(words)]
        nonce = (train_index * 7919 + seed) % 100_000
        name = f"{left}_{right}_{train_index:04d}_{nonce:05d}"
        old_name = row["function_name"]
        rows.append(
            {
                **row,
                "prompt": row["prompt"].replace(old_name, name),
                "response": row["response"].replace(old_name, name, 1),
                "function_name": name,
            }
        )
        train_index += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    def split_sha(items: list[dict[str, Any]], split: str) -> str:
        return _canonical_sha([row for row in items if row["split"] == split])
    training_names = [
        row["function_name"] for row in rows if row["split"] == "train"
    ]
    manifest = {
        "format": "layercake-phase4-python-functional-dataset/4",
        "status": "PREREGISTERED_UNIQUE_IDENTIFIER_REPAIR",
        "seed": seed,
        "source_dataset": (
            source.relative_to(ROOT).as_posix()
            if source.is_relative_to(ROOT)
            else str(source)
        ),
        "source_dataset_sha256": sha256_file(source),
        "dataset": (
            output.relative_to(ROOT).as_posix()
            if output.is_relative_to(ROOT)
            else str(output)
        ),
        "dataset_sha256": sha256_file(output),
        "counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "unique_training_identifiers": len(set(training_names)),
        "training_identifier_count": len(training_names),
        "split_hashes": {
            split: split_sha(rows, split)
            for split in ("train", "validation", "test")
        },
        "source_split_hashes": {
            split: split_sha(source_rows, split)
            for split in ("train", "validation", "test")
        },
        "validation_rows_unchanged": split_sha(rows, "validation")
        == split_sha(source_rows, "validation"),
        "test_rows_unchanged": split_sha(rows, "test")
        == split_sha(source_rows, "test"),
        "training_change": "every training function identifier is unique",
        "test_definition_unchanged": True,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    output.with_name("manifest_v4.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_rows(dataset: Path) -> list[dict[str, Any]]:
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    return [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _subsequence_start(values: list[int], pattern: list[int]) -> int | None:
    for start in range(len(values) - len(pattern) + 1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def _on_policy_recovery_sequence(
    prompt_ids: list[int],
    response_ids: list[int],
    generated: list[int],
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    if len(generated) > len(response_ids):
        raise ValueError("generated prefix is longer than the gold response")
    sequence = prompt_ids + generated + response_ids[len(generated):]
    targets = torch.tensor(sequence[1:], dtype=torch.long)
    response_start = len(prompt_ids) - 1
    response_stop = response_start + len(response_ids)
    targets[response_start:response_stop] = torch.tensor(
        response_ids, dtype=torch.long
    )
    mask = torch.zeros(len(targets), dtype=torch.bool)
    mask[response_start:response_stop] = True
    return sequence, targets, mask


@torch.inference_mode()
def cache_training_states(
    checkpoint: Path,
    dataset: Path,
    output: Path,
    *,
    max_response_tokens: int = 160,
) -> dict[str, Any]:
    """Cache only frozen-core semantic states used to fit the external cake."""

    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise RuntimeError(f"state cache artifact is immutable: {output}")
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    model, tokenizer, metadata = load_student(checkpoint)
    model.eval()
    states: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    prompt_routes: dict[str, int] = {}
    rows = [row for row in _load_rows(dataset) if row["split"] == "train"]
    raw_bytes = 0
    visible_tokens = 0
    for index, row in enumerate(rows):
        prompt_ids = tokenizer.encode(row["prompt"] + "\n")
        response_ids = tokenizer.encode(row["response"])[:max_response_tokens]
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        prompt_result = model(
            prompt_tensor,
            prompt_lengths=torch.tensor([len(prompt_ids)]),
            use_cache=False,
        )
        route = prompt_result["task_routes"]
        sequence = prompt_ids + response_ids
        ids = torch.tensor([sequence], dtype=torch.long)
        result = model(ids, task_routes=route)
        # Hidden at position t predicts target token t+1.  Only response tokens
        # contribute; the core and its output embedding remain frozen.
        start = len(prompt_ids) - 1
        stop = len(sequence) - 1
        states.append(result["hidden"][0, start:stop].half().cpu())
        targets.append(ids[0, start + 1:stop + 1].cpu())
        prompt_routes[str(int(route.item()))] = (
            prompt_routes.get(str(int(route.item())), 0) + 1
        )
        raw_bytes += len((row["prompt"] + "\n" + row["response"]).encode("utf-8"))
        visible_tokens += len(sequence)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if (index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "cached_rows": index + 1,
                        "semantic_training_units": sum(
                            item.shape[0] for item in states
                        ),
                        "wall_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    state_tensor = torch.cat(states)
    target_tensor = torch.cat(targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "semantic_states": state_tensor.contiguous(),
            "target_ids": target_tensor.contiguous(),
        },
        str(output),
    )
    evidence = {
        "format": "layercake-phase4-frozen-semantic-cache/1",
        "status": "COMPLETE",
        "checkpoint_sha256_before": metadata["checkpoint"]["sha256"],
        "checkpoint_sha256_after": sha256_file(checkpoint / "model.safetensors"),
        "core_parameters_changed": 0,
        "dataset_sha256": sha256_file(dataset),
        "rows": len(rows),
        "semantic_training_units": int(state_tensor.shape[0]),
        "state_width": int(state_tensor.shape[1]),
        "state_dtype": str(state_tensor.dtype),
        "raw_utf8_training_bytes_exposed": raw_bytes,
        "model_visible_nonpadding_units": visible_tokens,
        "prompt_routes": prompt_routes,
        "cache_path": output.relative_to(ROOT).as_posix(),
        "cache_sha256": sha256_file(output),
        "cache_bytes": output.stat().st_size,
        "cpu_wall_seconds": time.perf_counter() - started,
        "peak_process_resident_memory_bytes": peak_rss,
        "device": "cpu",
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path = output.with_suffix(".json")
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def train_cake(
    checkpoint: Path,
    cache: Path,
    output: Path,
    *,
    seed: int,
    rank: int = 768,
    steps: int = 2400,
    batch_size: int = 512,
    negative_count: int = 1536,
    learning_rate: float = 8.0e-4,
) -> dict[str, Any]:
    """Fit only the portable residual with sampled frozen-vocabulary contrast."""

    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    cache = cache if cache.is_absolute() else ROOT / cache
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise RuntimeError(f"cake checkpoint artifact is immutable: {output}")
    torch.manual_seed(seed)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    core, _, metadata = load_student(checkpoint)
    embedding = core.output_weight.detach().float().cpu()
    del core
    cached = load_file(str(cache), device="cpu")
    states = cached["semantic_states"]
    targets = cached["target_ids"].long()
    cake = HostResidualCake(d_abi=states.shape[1], rank=rank)
    cake.train()
    optimizer = torch.optim.AdamW(
        cake.parameters(), lr=learning_rate, weight_decay=0.01
    )
    curves: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(1, steps + 1):
        indexes = torch.randint(
            len(states), (batch_size,), generator=rng
        )
        hidden = states.index_select(0, indexes).float()
        target = targets.index_select(0, indexes)
        optimizer.zero_grad(set_to_none=True)
        adapted = cake(hidden)
        if negative_count > 0:
            random_ids = torch.randint(
                embedding.shape[0], (negative_count,), generator=rng
            )
            candidate_ids = torch.unique(torch.cat((target, random_ids)))
            candidate_embedding = embedding.index_select(0, candidate_ids)
            target_positions = torch.searchsorted(candidate_ids, target)
            logits = F.linear(adapted, candidate_embedding)
            loss = F.cross_entropy(logits, target_positions)
        else:
            # A zero sampled-negative count intentionally means authoritative
            # full-vocabulary cross-entropy, not a zero-negative approximation.
            logits = F.linear(adapted, embedding)
            loss = F.cross_entropy(logits, target)
        # Keep the residual bounded relative to its host state so portability
        # cannot be obtained by erasing the English representation.
        residual = adapted - hidden
        stability = residual.square().mean() / hidden.square().mean().clamp_min(1e-6)
        objective = loss + 0.01 * stability
        objective.backward()
        torch.nn.utils.clip_grad_norm_(cake.parameters(), 1.0)
        optimizer.step()
        loss_value = float(loss.detach())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in cake.state_dict().items()
            }
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "cross_entropy": loss_value,
                "stability_ratio": float(stability.detach()),
                "alpha": float(cake.alpha.detach()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(record)
            print(json.dumps(record), flush=True)
    assert best_state is not None
    cake.load_state_dict(best_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in cake.state_dict().items()
        },
        str(output),
    )
    wall = time.perf_counter() - started
    trainable = sum(parameter.numel() for parameter in cake.parameters())
    evidence = {
        "format": "layercake-phase4-python-host-residual-training/1",
        "status": "COMPLETE",
        "seed": seed,
        "device": "cpu",
        "core_checkpoint_sha256_before": metadata["checkpoint"]["sha256"],
        "core_checkpoint_sha256_after": sha256_file(
            checkpoint / "model.safetensors"
        ),
        "core_parameters_changed": 0,
        "cache_sha256": sha256_file(cache),
        "cake_checkpoint": output.relative_to(ROOT).as_posix(),
        "cake_checkpoint_sha256": sha256_file(output),
        "architecture": {"name": "host_residual", "d_abi": 768, "rank": rank},
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "negative_vocabulary_samples_per_step": negative_count,
        "training_objective": (
            "sampled_vocabulary_cross_entropy"
            if negative_count > 0
            else "full_vocabulary_cross_entropy"
        ),
        "trainable_parameters": trainable,
        "active_parameter_seconds_to_quality": trainable * wall,
        "end_to_end_cpu_wall_seconds": wall,
        "peak_process_resident_memory_bytes": peak_rss,
        "learning_curves": curves,
        "best_cross_entropy": best_loss,
        "energy_to_quality": {
            "status": "UNAVAILABLE",
            "reason": "no calibrated package energy meter is exposed",
        },
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output.with_suffix(".json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


@torch.inference_mode()
def cache_recurrent_training_states(
    checkpoint: Path,
    dataset: Path,
    output: Path,
    *,
    max_response_tokens: int = 160,
) -> dict[str, Any]:
    """Cache complete prompt/response semantic sequences and response masks."""

    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise RuntimeError(f"state cache artifact is immutable: {output}")
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    model, tokenizer, metadata = load_student(checkpoint)
    model.eval()
    all_states: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_lexical_copy_labels: list[torch.Tensor] = []
    offsets = [0]
    rows = [row for row in _load_rows(dataset) if row["split"] == "train"]
    raw_bytes = 0
    visible_tokens = 0
    prompt_routes: dict[str, int] = {}
    for index, row in enumerate(rows):
        prompt_ids = tokenizer.encode(row["prompt"] + "\n")
        response_ids = tokenizer.encode(row["response"])[:max_response_tokens]
        prompt_result = model(
            torch.tensor([prompt_ids], dtype=torch.long),
            prompt_lengths=torch.tensor([len(prompt_ids)]),
        )
        route = prompt_result["task_routes"]
        ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
        result = model(ids, task_routes=route)
        states = result["hidden"][0, :-1].half().cpu()
        targets = ids[0, 1:].cpu()
        mask = torch.zeros(len(targets), dtype=torch.bool)
        mask[max(0, len(prompt_ids) - 1):] = True
        lexical_copy_labels = torch.full(
            (len(targets),), -100, dtype=torch.int64
        )
        identifier_pattern = tokenizer.encode(" " + row["function_name"])
        prompt_identifier_start = _subsequence_start(
            prompt_ids, identifier_pattern
        )
        response_identifier_start = _subsequence_start(
            response_ids, identifier_pattern
        )
        if (
            prompt_identifier_start is None
            or response_identifier_start is None
        ):
            identifier_pattern = tokenizer.encode(row["function_name"])
            prompt_identifier_start = _subsequence_start(
                prompt_ids, identifier_pattern
            )
            response_identifier_start = _subsequence_start(
                response_ids, identifier_pattern
            )
        if (
            prompt_identifier_start is None
            or response_identifier_start is None
        ):
            raise RuntimeError(
                f"function identifier span not found for {row['id']}"
            )
        for identifier_offset in range(len(identifier_pattern)):
            response_target_position = (
                len(prompt_ids)
                - 1
                + response_identifier_start
                + identifier_offset
            )
            source_state_position = (
                prompt_identifier_start + identifier_offset
            )
            if response_target_position >= len(targets):
                break
            lexical_copy_labels[
                response_target_position
            ] = source_state_position
        all_states.append(states)
        all_targets.append(targets)
        all_masks.append(mask)
        all_lexical_copy_labels.append(lexical_copy_labels)
        offsets.append(offsets[-1] + len(targets))
        prompt_routes[str(int(route.item()))] = (
            prompt_routes.get(str(int(route.item())), 0) + 1
        )
        raw_bytes += len((row["prompt"] + "\n" + row["response"]).encode("utf-8"))
        visible_tokens += len(ids[0])
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if (index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "cached_rows": index + 1,
                        "causal_units": offsets[-1],
                        "wall_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "semantic_states": torch.cat(all_states).contiguous(),
            "target_ids": torch.cat(all_targets).contiguous(),
            "response_mask": torch.cat(all_masks).contiguous(),
            "lexical_copy_labels": torch.cat(
                all_lexical_copy_labels
            ).contiguous(),
            "row_offsets": torch.tensor(offsets, dtype=torch.int64),
        },
        str(output),
    )
    evidence = {
        "format": "layercake-phase4-recurrent-semantic-cache/1",
        "status": "COMPLETE",
        "checkpoint_sha256_before": metadata["checkpoint"]["sha256"],
        "checkpoint_sha256_after": sha256_file(checkpoint / "model.safetensors"),
        "core_parameters_changed": 0,
        "dataset_sha256": sha256_file(dataset),
        "rows": len(rows),
        "causal_units": offsets[-1],
        "response_training_units": int(torch.cat(all_masks).sum()),
        "lexical_copy_training_units": int(
            (torch.cat(all_lexical_copy_labels) >= 0).sum()
        ),
        "state_width": 768,
        "state_dtype": "torch.float16",
        "raw_utf8_training_bytes_exposed": raw_bytes,
        "model_visible_nonpadding_units": visible_tokens,
        "prompt_routes": prompt_routes,
        "cache_path": output.relative_to(ROOT).as_posix(),
        "cache_sha256": sha256_file(output),
        "cache_bytes": output.stat().st_size,
        "cpu_wall_seconds": time.perf_counter() - started,
        "peak_process_resident_memory_bytes": peak_rss,
        "device": "cpu",
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output.with_suffix(".json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


@torch.inference_mode()
def cache_on_policy_recovery_states(
    checkpoint: Path,
    dataset: Path,
    cake_checkpoint: Path,
    output: Path,
    *,
    max_response_tokens: int = 160,
    horizons: tuple[int, ...] = (8, 32, 64),
) -> dict[str, Any]:
    """Cache gold correction targets under real autonomous cake prefixes."""

    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    cake_checkpoint = (
        cake_checkpoint
        if cake_checkpoint.is_absolute()
        else ROOT / cake_checkpoint
    )
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise RuntimeError(f"state cache artifact is immutable: {output}")
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("on-policy recovery horizons must be positive")

    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    model, tokenizer, metadata = load_student(checkpoint)
    tensors = load_file(str(cake_checkpoint), device="cpu")
    if "blocks.0.attention.in_proj_weight" not in tensors:
        raise ValueError("on-policy recovery requires an attentive cake")
    hidden_width = int(tensors["input.weight"].shape[0])
    cake = AttentiveHostResidualCake(
        d_abi=768,
        hidden_width=hidden_width,
        layers=len(
            {
                name.split(".")[1]
                for name in tensors
                if name.startswith("blocks.")
            }
        ),
        heads=6,
        expansion=int(
            tensors["blocks.0.feedforward.0.weight"].shape[0]
            / hidden_width
        ),
        copy_width=(
            int(tensors["copy_query.weight"].shape[0])
            if "copy_query.weight" in tensors
            else 0
        ),
        copy_value_projection=(
            "copy_value.weight" in tensors
            or "copy_transition_value.weight" in tensors
        ),
        selective_copy="copy_gate.weight" in tensors,
        transition_copy="copy_transition_value.weight" in tensors,
    )
    cake.load_state_dict(tensors, strict=True)
    cake.eval()

    all_states: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    offsets = [0]
    rows = [row for row in _load_rows(dataset) if row["split"] == "train"]
    horizon_rows = {str(value): 0 for value in horizons}
    generated_units = 0
    mismatched_generated_units = 0
    early_eos_rows = 0
    for index, row in enumerate(rows):
        prompt_ids = tokenizer.encode(row["prompt"] + "\n")
        response_ids = tokenizer.encode(row["response"])[
            :max_response_tokens
        ]
        horizon = horizons[index % len(horizons)]
        requested = min(horizon, max(0, len(response_ids) - 1))
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        state = _domain_prefill(model, cake, prompt_tensor)
        generated = []
        for _ in range(requested):
            token = state["next_logits"].argmax(dim=-1)
            generated.append(int(token.item()))
            _domain_decode(model, cake, state, token)
            if token.item() == tokenizer.eos_token_id:
                early_eos_rows += 1
                break
        generated_count = len(generated)
        horizon_rows[str(horizon)] += 1
        generated_units += generated_count
        mismatched_generated_units += sum(
            generated[position] != response_ids[position]
            for position in range(generated_count)
        )
        (
            sequence,
            targets,
            mask,
        ) = _on_policy_recovery_sequence(
            prompt_ids, response_ids, generated
        )
        ids = torch.tensor([sequence], dtype=torch.long)
        result = model(
            ids,
            task_routes=state["task_routes"],
            use_cache=False,
        )
        states = result["hidden"][0, :-1].half().cpu()
        all_states.append(states)
        all_targets.append(targets)
        all_masks.append(mask)
        offsets.append(offsets[-1] + len(targets))
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if (index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "cached_rows": index + 1,
                        "generated_prefix_units": generated_units,
                        "mismatched_generated_prefix_units": (
                            mismatched_generated_units
                        ),
                        "wall_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "semantic_states": torch.cat(all_states).contiguous(),
            "target_ids": torch.cat(all_targets).contiguous(),
            "response_mask": torch.cat(all_masks).contiguous(),
            "row_offsets": torch.tensor(offsets, dtype=torch.int64),
        },
        str(output),
    )
    wall = time.perf_counter() - started
    evidence = {
        "format": "layercake-phase4-on-policy-recovery-cache/1",
        "status": "COMPLETE",
        "device": "cpu",
        "checkpoint_sha256_before": metadata["checkpoint"]["sha256"],
        "checkpoint_sha256_after": sha256_file(
            checkpoint / "model.safetensors"
        ),
        "core_parameters_changed": 0,
        "source_cake_checkpoint_sha256": sha256_file(cake_checkpoint),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(rows),
        "prefix_horizons": list(horizons),
        "horizon_rows": horizon_rows,
        "generation_policy": "greedy_argmax_integrated_core_plus_cake",
        "generated_prefix_units": generated_units,
        "mismatched_generated_prefix_units": mismatched_generated_units,
        "generated_prefix_mismatch_rate": (
            mismatched_generated_units / max(1, generated_units)
        ),
        "early_eos_rows": early_eos_rows,
        "causal_units": offsets[-1],
        "cache_path": output.relative_to(ROOT).as_posix(),
        "cache_sha256": sha256_file(output),
        "cache_bytes": output.stat().st_size,
        "cpu_wall_seconds": wall,
        "peak_process_resident_memory_bytes": peak_rss,
        "test_split_used_for_generation_or_targets": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output.with_suffix(".json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _causal_copy_labels(
    targets: torch.Tensor,
    valid: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Locate the post-token prompt state for response tokens copied from input.

    Cached hidden state j predicts targets[j], so it precedes that target.
    The state at j + 1 is the first causal ABI state that has observed it.
    """

    if targets.shape != valid.shape or targets.shape != response_mask.shape:
        raise ValueError("copy-label tensors must have identical shapes")
    labels = torch.full(targets.shape, -100, dtype=torch.long)
    for batch_index in range(targets.shape[0]):
        prompt_target_positions = torch.nonzero(
            valid[batch_index] & ~response_mask[batch_index],
            as_tuple=False,
        ).flatten()
        response_positions = torch.nonzero(
            valid[batch_index] & response_mask[batch_index],
            as_tuple=False,
        ).flatten()
        prompt_targets = targets[batch_index, prompt_target_positions]
        for position in response_positions.tolist():
            matches = prompt_target_positions[
                prompt_targets == targets[batch_index, position]
            ]
            if not matches.numel():
                continue
            post_token_position = int(matches[-1]) + 1
            if (
                post_token_position <= position
                and bool(valid[batch_index, post_token_position])
            ):
                labels[batch_index, position] = post_token_position
    return labels


def _prompt_post_token_pairs(
    targets: torch.Tensor,
    valid: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return batch indexes, post-token state indexes, and prompt token IDs."""

    if targets.shape != valid.shape or targets.shape != response_mask.shape:
        raise ValueError("prompt-value tensors must have identical shapes")
    batch_indexes, prompt_positions = torch.nonzero(
        valid & ~response_mask, as_tuple=True
    )
    source_positions = prompt_positions + 1
    in_range = source_positions < targets.shape[1]
    batch_indexes = batch_indexes[in_range]
    prompt_positions = prompt_positions[in_range]
    source_positions = source_positions[in_range]
    source_valid = valid[batch_indexes, source_positions]
    batch_indexes = batch_indexes[source_valid]
    prompt_positions = prompt_positions[source_valid]
    source_positions = source_positions[source_valid]
    return (
        batch_indexes,
        source_positions,
        targets[batch_indexes, prompt_positions],
    )


def train_recurrent_cake(
    checkpoint: Path,
    cache: Path,
    output: Path,
    *,
    seed: int,
    hidden_width: int = 1024,
    layers: int = 1,
    max_residual: float = 6.0,
    steps: int = 1600,
    batch_size: int = 8,
    learning_rate: float = 3.0e-4,
    initial_checkpoint: Path | None = None,
    prompt_retention_fraction: float = 0.0,
    prompt_retention_weight: float = 0.25,
    architecture: str = "recurrent",
    heads: int = 6,
    expansion: int = 4,
    copy_width: int = 0,
    pointer_supervision_weight: float = 0.0,
    copy_value_projection: bool = False,
    copy_value_supervision_weight: float = 0.0,
    selective_copy: bool = False,
    copy_gate_supervision_weight: float = 0.0,
    prompt_value_supervision_weight: float = 0.0,
    transition_copy: bool = False,
    copy_path_only: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    cache = cache if cache.is_absolute() else ROOT / cache
    output = output if output.is_absolute() else ROOT / output
    if output.exists():
        raise RuntimeError(f"cake checkpoint artifact is immutable: {output}")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    core, _, metadata = load_student(checkpoint)
    embedding = core.output_weight.detach().float().cpu()
    del core
    cached = load_file(str(cache), device="cpu")
    states = cached["semantic_states"]
    targets = cached["target_ids"].long()
    masks = cached["response_mask"].bool()
    lexical_copy_labels = (
        cached["lexical_copy_labels"].long()
        if "lexical_copy_labels" in cached
        else None
    )
    offsets = cached["row_offsets"].long().tolist()
    if architecture == "recurrent":
        cake = RecurrentHostResidualCake(
            d_abi=768,
            hidden_width=hidden_width,
            layers=layers,
            max_residual=max_residual,
        )
    elif architecture == "attentive":
        cake = AttentiveHostResidualCake(
            d_abi=768,
            hidden_width=hidden_width,
            layers=layers,
            heads=heads,
            expansion=expansion,
            max_residual=max_residual,
            copy_width=copy_width,
            copy_value_projection=copy_value_projection,
            selective_copy=selective_copy,
            transition_copy=transition_copy,
        )
    else:
        raise ValueError(f"unknown semantic cake architecture: {architecture}")
    initial_sha = None
    if initial_checkpoint is not None:
        initial_checkpoint = (
            initial_checkpoint
            if initial_checkpoint.is_absolute()
            else ROOT / initial_checkpoint
        )
        initial_tensors = load_file(str(initial_checkpoint), device="cpu")
        target_keys = set(cake.state_dict())
        source_keys = set(initial_tensors)
        expected_missing = target_keys - source_keys
        allowed_new_copy_tensors = {
            "copy_alpha",
            "copy_query.weight",
            "copy_key.weight",
            "copy_value.weight",
            "copy_transition_value.weight",
            "copy_gate.weight",
            "copy_gate.bias",
        }
        if not expected_missing <= allowed_new_copy_tensors:
            raise RuntimeError(
                "initial checkpoint is missing non-copy-path tensors"
            )
        unexpected_source = source_keys - target_keys
        if unexpected_source:
            raise RuntimeError(
                "initial checkpoint has incompatible source tensors"
            )
        incompatible = cake.load_state_dict(
            initial_tensors, strict=not expected_missing
        )
        if expected_missing:
            if set(incompatible.missing_keys) != expected_missing:
                raise RuntimeError(
                    "copy-path migration has unexpected missing tensors"
                )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "copy-path migration has unexpected source tensors"
                )
        if (
            "copy_value.weight" in expected_missing
            or "copy_transition_value.weight" in expected_missing
        ):
            with torch.no_grad():
                cake.copy_alpha.zero_()
        initial_sha = sha256_file(initial_checkpoint)
    trainable_parameter_names = tuple(
        name for name, _ in cake.named_parameters()
    )
    frozen_parameter_names: set[str] = set()
    frozen_parameters_sha256_before = None
    if copy_path_only:
        if not isinstance(cake, AttentiveHostResidualCake):
            raise ValueError(
                "copy-path-only training requires the attentive architecture"
            )
        trainable_parameter_names = _configure_copy_path_only(cake)
        frozen_parameter_names = (
            set(cake.state_dict()) - set(trainable_parameter_names)
        )
        frozen_parameters_sha256_before = _tensor_subset_sha256(
            cake.state_dict(), frozen_parameter_names
        )
    cake.train()
    trainable_parameters = [
        parameter for parameter in cake.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=learning_rate, weight_decay=0.01
    )
    curves: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_cross_entropy = float("inf")
    best_state = None
    row_count = len(offsets) - 1
    for step in range(1, steps + 1):
        selected = [rng.randrange(row_count) for _ in range(batch_size)]
        lengths = [offsets[row + 1] - offsets[row] for row in selected]
        length = max(lengths)
        batch_states = torch.zeros(batch_size, length, 768)
        batch_targets = torch.zeros(batch_size, length, dtype=torch.long)
        batch_masks = torch.zeros(batch_size, length, dtype=torch.bool)
        batch_valid = torch.zeros(batch_size, length, dtype=torch.bool)
        batch_lexical_copy_labels = torch.full(
            (batch_size, length), -100, dtype=torch.long
        )
        for batch_index, row in enumerate(selected):
            start, stop = offsets[row], offsets[row + 1]
            count = stop - start
            batch_states[batch_index, :count] = states[start:stop].float()
            batch_targets[batch_index, :count] = targets[start:stop]
            batch_masks[batch_index, :count] = masks[start:stop]
            batch_valid[batch_index, :count] = True
            if lexical_copy_labels is not None:
                batch_lexical_copy_labels[
                    batch_index, :count
                ] = lexical_copy_labels[start:stop]
        optimizer.zero_grad(set_to_none=True)
        adapted = (
            cake.training_forward(batch_states)
            if isinstance(cake, AttentiveHostResidualCake)
            else cake(batch_states)[0]
        )
        supervised = batch_masks.clone()
        prompt_selected = torch.zeros_like(batch_masks)
        if prompt_retention_fraction > 0:
            prompt_candidates = batch_valid & ~batch_masks
            prompt_selected = prompt_candidates & (
                torch.rand(prompt_candidates.shape)
                < prompt_retention_fraction
            )
            supervised |= prompt_selected
        selected_adapted = adapted[supervised]
        selected_targets = batch_targets[supervised]
        logits = F.linear(selected_adapted, embedding)
        losses = F.cross_entropy(logits, selected_targets, reduction="none")
        weights = torch.ones_like(losses)
        if prompt_selected.any():
            weights[prompt_selected[supervised]] = prompt_retention_weight
        loss = (losses * weights).sum() / weights.sum()
        residual = selected_adapted - batch_states[supervised]
        stability = residual.square().mean() / (
            batch_states[supervised].square().mean().clamp_min(1e-6)
        )
        objective = loss + 0.002 * stability
        pointer_loss = None
        copy_value_loss = None
        copy_gate_loss = None
        prompt_value_loss = None
        pointer_labels_count = 0
        prompt_value_units = 0
        if (
            isinstance(cake, AttentiveHostResidualCake)
            and cake.copy_width > 0
            and (
                pointer_supervision_weight > 0
                or copy_value_supervision_weight > 0
                or copy_gate_supervision_weight > 0
                or prompt_value_supervision_weight > 0
            )
        ):
            pointer_labels = (
                batch_lexical_copy_labels
                if lexical_copy_labels is not None
                else _causal_copy_labels(
                    batch_targets,
                    batch_valid,
                    batch_masks,
                )
            )
            pointer_valid = pointer_labels >= 0
            if pointer_valid.any():
                pointer_labels_count = int(pointer_valid.sum())
                if pointer_supervision_weight > 0:
                    pointer_scores = cake.copy_scores(batch_states)
                    pointer_loss = F.cross_entropy(
                        pointer_scores[pointer_valid],
                        pointer_labels[pointer_valid],
                    )
                    objective = (
                        objective
                        + pointer_supervision_weight * pointer_loss
                    )
                if (
                    copy_value_supervision_weight > 0
                    and cake.copy_value_projection
                ):
                    batch_indexes, _ = torch.nonzero(
                        pointer_valid, as_tuple=True
                    )
                    source_positions = pointer_labels[pointer_valid]
                    projected_values = cake.project_copy_positions(
                        batch_states,
                        batch_indexes,
                        source_positions,
                    )
                    copied_logits = F.linear(projected_values, embedding)
                    copy_value_loss = F.cross_entropy(
                        copied_logits,
                        batch_targets[pointer_valid],
                    )
                    objective = (
                        objective
                        + copy_value_supervision_weight * copy_value_loss
                    )
            if (
                copy_gate_supervision_weight > 0
                and cake.selective_copy
            ):
                gate_valid = batch_valid & batch_masks
                copy_gate_loss = F.binary_cross_entropy_with_logits(
                    cake.copy_gate_logits(batch_states)[gate_valid],
                    pointer_valid[gate_valid].float(),
                )
                objective = (
                    objective
                    + copy_gate_supervision_weight * copy_gate_loss
                )
            if (
                prompt_value_supervision_weight > 0
                and cake.copy_value_projection
            ):
                (
                    prompt_batch_indexes,
                    prompt_source_positions,
                    prompt_value_targets,
                ) = _prompt_post_token_pairs(
                    batch_targets,
                    batch_valid,
                    batch_masks,
                )
                prompt_value_logits = F.linear(
                    cake.project_copy_positions(
                        batch_states,
                        prompt_batch_indexes,
                        prompt_source_positions,
                    ),
                    embedding,
                )
                prompt_value_loss = F.cross_entropy(
                    prompt_value_logits, prompt_value_targets
                )
                prompt_value_units = int(prompt_value_targets.numel())
                objective = (
                    objective
                    + prompt_value_supervision_weight
                    * prompt_value_loss
                )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        loss_value = float(loss.detach())
        objective_value = float(objective.detach())
        selection_loss = (
            objective_value
            if (
                copy_value_supervision_weight > 0
                or copy_gate_supervision_weight > 0
                or prompt_value_supervision_weight > 0
            )
            else loss_value
        )
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_cross_entropy = loss_value
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in cake.state_dict().items()
            }
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "full_vocabulary_cross_entropy": loss_value,
                "stability_ratio": float(stability.detach()),
                "alpha": float(cake.alpha.detach()),
                "copy_alpha": (
                    float(cake.copy_alpha.detach())
                    if isinstance(cake, AttentiveHostResidualCake)
                    and cake.copy_width > 0
                    else None
                ),
                "pointer_cross_entropy": (
                    float(pointer_loss.detach())
                    if pointer_loss is not None
                    else None
                ),
                "pointer_supervised_units": pointer_labels_count,
                "copy_value_cross_entropy": (
                    float(copy_value_loss.detach())
                    if copy_value_loss is not None
                    else None
                ),
                "copy_gate_binary_cross_entropy": (
                    float(copy_gate_loss.detach())
                    if copy_gate_loss is not None
                    else None
                ),
                "prompt_value_cross_entropy": (
                    float(prompt_value_loss.detach())
                    if prompt_value_loss is not None
                    else None
                ),
                "prompt_value_supervised_units": prompt_value_units,
                "selection_objective": objective_value,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(record)
            print(json.dumps(record), flush=True)
    assert best_state is not None
    cake.load_state_dict(best_state)
    frozen_parameters_sha256_after = (
        _tensor_subset_sha256(cake.state_dict(), frozen_parameter_names)
        if copy_path_only
        else None
    )
    if (
        copy_path_only
        and frozen_parameters_sha256_after
        != frozen_parameters_sha256_before
    ):
        raise RuntimeError("copy-path-only training mutated a frozen tensor")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in cake.state_dict().items()
        },
        str(output),
    )
    wall = time.perf_counter() - started
    trainable = sum(
        parameter.numel()
        for parameter in cake.parameters()
        if parameter.requires_grad
    )
    evidence = {
        "format": "layercake-phase4-recurrent-host-residual-training/1",
        "status": "COMPLETE",
        "seed": seed,
        "device": "cpu",
        "core_checkpoint_sha256_before": metadata["checkpoint"]["sha256"],
        "core_checkpoint_sha256_after": sha256_file(
            checkpoint / "model.safetensors"
        ),
        "core_parameters_changed": 0,
        "cache_sha256": sha256_file(cache),
        "cake_checkpoint": output.relative_to(ROOT).as_posix(),
        "cake_checkpoint_sha256": sha256_file(output),
        "architecture": {
            "name": (
                "attentive_host_residual"
                if isinstance(cake, AttentiveHostResidualCake)
                else "recurrent_host_residual"
            ),
            "d_abi": 768,
            "hidden_width": hidden_width,
            "layers": layers,
            "max_residual": max_residual,
            **(
                {"heads": heads, "expansion": expansion}
                | {"copy_width": copy_width}
                | {"copy_value_projection": copy_value_projection}
                | {"selective_copy": selective_copy}
                | {"transition_copy": transition_copy}
                if isinstance(cake, AttentiveHostResidualCake)
                else {}
            ),
        },
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "training_objective": "full_vocabulary_cross_entropy",
        "initial_checkpoint_sha256": initial_sha,
        "prompt_retention_fraction": prompt_retention_fraction,
        "prompt_retention_weight": prompt_retention_weight,
        "pointer_supervision_weight": pointer_supervision_weight,
        "copy_value_supervision_weight": copy_value_supervision_weight,
        "copy_gate_supervision_weight": copy_gate_supervision_weight,
        "prompt_value_supervision_weight": prompt_value_supervision_weight,
        "copy_path_only": copy_path_only,
        "trainable_parameter_names": list(trainable_parameter_names),
        "frozen_parameters_sha256_before": (
            frozen_parameters_sha256_before
        ),
        "frozen_parameters_sha256_after": (
            frozen_parameters_sha256_after
        ),
        "trainable_parameters": trainable,
        "active_parameter_seconds_to_quality": trainable * wall,
        "end_to_end_cpu_wall_seconds": wall,
        "peak_process_resident_memory_bytes": peak_rss,
        "learning_curves": curves,
        "best_cross_entropy": best_cross_entropy,
        "best_selection_objective": best_loss,
        "energy_to_quality": {
            "status": "UNAVAILABLE",
            "reason": "no calibrated package energy meter is exposed",
        },
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output.with_suffix(".json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _domain_prefill(model, cake, input_ids: torch.Tensor) -> dict[str, Any]:
    result = model(
        input_ids,
        prompt_lengths=torch.full(
            (input_ids.shape[0],), input_ids.shape[1], dtype=torch.long
        ),
        use_cache=True,
    )
    cake_state = None
    if isinstance(cake, (RecurrentHostResidualCake, AttentiveHostResidualCake)):
        adapted_sequence, cake_state = cake(result["hidden"])
        adapted = adapted_sequence[:, -1]
    else:
        adapted = cake(result["hidden"][:, -1])
    return {
        "past_key_values": result["past_key_values"],
        "task_routes": result["task_routes"],
        "next_logits": F.linear(adapted, model.output_weight),
        "generated_ids": input_ids[:, :0],
        "cake_state": cake_state,
    }


def _domain_decode(model, cake, state: dict[str, Any], token: torch.Tensor) -> None:
    result = model(
        token[:, None],
        task_routes=state["task_routes"],
        past_key_values=state["past_key_values"],
        use_cache=True,
    )
    state["past_key_values"] = result["past_key_values"]
    if isinstance(cake, AttentiveHostResidualCake):
        adapted, cake_state = cake.step(
            result["hidden"][:, -1], state["cake_state"]
        )
        state["cake_state"] = cake_state
    elif isinstance(cake, RecurrentHostResidualCake):
        adapted, cake_state = cake(
            result["hidden"][:, -1], state["cake_state"]
        )
        state["cake_state"] = cake_state
    else:
        adapted = cake(result["hidden"][:, -1])
    state["next_logits"] = F.linear(adapted, model.output_weight)
    state["generated_ids"] = torch.cat(
        (state["generated_ids"], token[:, None]), dim=1
    )


@torch.inference_mode()
def _generate_code(
    model,
    tokenizer,
    prompt: str,
    *,
    cake: HostResidualCake | None,
    maximum_tokens: int = 192,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt + "\n")
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    started = time.perf_counter()
    if cake is None:
        state = model.prefill(ids)
    else:
        state = _domain_prefill(model, cake, ids)
    generated: list[int] = []
    first = None
    for _ in range(maximum_tokens):
        token = state["next_logits"].argmax(dim=-1)
        generated.append(int(token.item()))
        if first is None:
            first = time.perf_counter()
        text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if token.item() == tokenizer.eos_token_id:
            break
        if len(text.encode("utf-8")) >= 2048:
            break
        if cake is None:
            _, state = model.decode_step(state, next_token=token)
        else:
            _domain_decode(model, cake, state, token)
    completed = time.perf_counter()
    return {
        "text": tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "generated_tokens": len(generated),
        "time_to_first_output_seconds": (first or completed) - started,
        "total_latency_seconds": completed - started,
    }


_ALLOWED_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_ALLOWED_METHODS = {
    "append",
    "count",
    "extend",
    "get",
    "index",
    "items",
    "join",
    "lower",
    "replace",
    "reverse",
    "split",
    "strip",
    "title",
    "upper",
}
_FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


def _extract_function(text: str, expected_name: str) -> tuple[str | None, str]:
    cleaned = text.replace("```python", "").replace("```", "")
    starts = [
        index
        for index in range(len(cleaned))
        if cleaned.startswith("def ", index)
    ]
    for start in starts:
        candidate = cleaned[start:]
        lines = candidate.splitlines()
        # Autonomous decoders often continue after a complete answer.  Search
        # longest-first for a valid prefix so harmless trailing prose/junk
        # cannot turn an already complete function into a syntax failure.
        for end in range(len(lines), 1, -1):
            source = "\n".join(lines[:end]) + "\n"
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            matches = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == expected_name
            ]
            if matches:
                node = matches[0]
                function_source = (
                    "\n".join(lines[: node.end_lineno]) + "\n"
                )
                return function_source, "PARSED"
    return None, "NO_EXPECTED_FUNCTION"


def _validate_safe_function(source: str, expected_name: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, "SYNTAX_ERROR"
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if len(functions) != 1 or functions[0].name != expected_name:
        return False, "FUNCTION_CONTRACT"
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            return False, f"FORBIDDEN_{type(node).__name__}"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return False, "DUNDER_NAME"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr not in _ALLOWED_METHODS:
                return False, "ATTRIBUTE_NOT_ALLOWED"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_CALLS and node.func.id != expected_name:
                    return False, "CALL_NOT_ALLOWED"
            elif not isinstance(node.func, ast.Attribute):
                return False, "CALL_TARGET_NOT_ALLOWED"
    return True, "SAFE"


def _execute_tests(
    source: str, function_name: str, cases: list[dict[str, Any]]
) -> tuple[bool, list[dict[str, Any]]]:
    safe, reason = _validate_safe_function(source, function_name)
    if not safe:
        return False, [{"status": reason}]
    safe_builtins = {
        name: getattr(builtins, name)
        for name in _ALLOWED_CALLS
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    try:
        exec(compile(source, "<generated-python-cake>", "exec"), namespace)
        function = namespace[function_name]
    except Exception as exc:
        return False, [{"status": "LOAD_ERROR", "error": type(exc).__name__}]
    records = []
    for case in cases:
        try:
            actual = _jsonable(function(*case["args"]))
            passed = actual == case["expected"]
            records.append(
                {
                    "status": "PASS" if passed else "WRONG_RESULT",
                    "actual": actual,
                    "expected": case["expected"],
                }
            )
        except Exception as exc:
            records.append(
                {"status": "RUNTIME_ERROR", "error": type(exc).__name__}
            )
    return bool(records) and all(row["status"] == "PASS" for row in records), records


def evaluate_functional(
    checkpoint: Path,
    dataset: Path,
    output: Path,
    *,
    cake_checkpoint: Path | None,
    split: str = "validation",
    maximum_tokens: int = 192,
) -> dict[str, Any]:
    checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    dataset = dataset if dataset.is_absolute() else ROOT / dataset
    output = output if output.is_absolute() else ROOT / output
    model, tokenizer, metadata = load_student(checkpoint)
    cake = None
    cake_sha = None
    if cake_checkpoint is not None:
        cake_checkpoint = (
            cake_checkpoint
            if cake_checkpoint.is_absolute()
            else ROOT / cake_checkpoint
        )
        tensors = load_file(str(cake_checkpoint), device="cpu")
        if "blocks.0.attention.in_proj_weight" in tensors:
            hidden_width = int(tensors["input.weight"].shape[0])
            layers = len(
                {
                    name.split(".")[1]
                    for name in tensors
                    if name.startswith("blocks.")
                }
            )
            heads = 6
            expansion = int(
                tensors["blocks.0.feedforward.0.weight"].shape[0]
                / hidden_width
            )
            cake = AttentiveHostResidualCake(
                d_abi=768,
                hidden_width=hidden_width,
                layers=layers,
                heads=heads,
                expansion=expansion,
                copy_width=(
                    int(tensors["copy_query.weight"].shape[0])
                    if "copy_query.weight" in tensors
                    else 0
                ),
                copy_value_projection=(
                    "copy_value.weight" in tensors
                    or "copy_transition_value.weight" in tensors
                ),
                selective_copy="copy_gate.weight" in tensors,
                transition_copy="copy_transition_value.weight" in tensors,
            )
        elif "recurrent.weight_ih_l0" in tensors:
            hidden_width = int(tensors["recurrent.weight_hh_l0"].shape[1])
            layers = sum(
                name.startswith("recurrent.weight_ih_l")
                for name in tensors
            )
            cake = RecurrentHostResidualCake(
                d_abi=768,
                hidden_width=hidden_width,
                layers=layers,
            )
        else:
            rank = int(tensors["down.weight"].shape[0])
            cake = HostResidualCake(d_abi=768, rank=rank)
        cake.load_state_dict(tensors, strict=True)
        cake.eval()
        cake_sha = sha256_file(cake_checkpoint)
    rows = [row for row in _load_rows(dataset) if row["split"] == split]
    process = psutil.Process()
    records = []
    for index, row in enumerate(rows):
        generated = _generate_code(
            model,
            tokenizer,
            row["prompt"],
            cake=cake,
            maximum_tokens=maximum_tokens,
        )
        source, parse_status = _extract_function(
            generated["text"], row["function_name"]
        )
        passed = False
        tests = [{"status": parse_status}]
        if source is not None:
            passed, tests = _execute_tests(
                source, row["function_name"], row["tests"]
            )
        records.append(
            {
                "id": row["id"],
                "family": row["family"],
                "prompt": row["prompt"],
                "prompt_sha256": hashlib.sha256(
                    row["prompt"].encode("utf-8")
                ).hexdigest(),
                "expected_function": row["function_name"],
                "generated_text": generated["text"],
                "generated_text_sha256": hashlib.sha256(
                    generated["text"].encode("utf-8")
                ).hexdigest(),
                "extracted_source": source,
                "functional_success": passed,
                "tests": tests,
                "generated_tokens": generated["generated_tokens"],
                "time_to_first_output_seconds": generated[
                    "time_to_first_output_seconds"
                ],
                "total_latency_seconds": generated["total_latency_seconds"],
            }
        )
        print(
            json.dumps(
                {
                    "evaluated": index + 1,
                    "total": len(rows),
                    "successes": sum(
                        item["functional_success"] for item in records
                    ),
                }
            ),
            flush=True,
        )
    successes = sum(row["functional_success"] for row in records)
    document = {
        "format": "layercake-phase4-python-functional-evaluation/1",
        "status": "COMPLETE",
        "split": split,
        "system": "frozen_core_plus_cake" if cake else "frozen_core",
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "cake_checkpoint_sha256": cake_sha,
        "dataset_sha256": sha256_file(dataset),
        "distinct_prompts": len(rows),
        "functional_successes": successes,
        "functional_failures": len(rows) - successes,
        "functional_success_rate": successes / max(1, len(rows)),
        "functional_error_rate": (len(rows) - successes) / max(1, len(rows)),
        "median_time_to_first_output_seconds": statistics.median(
            row["time_to_first_output_seconds"] for row in records
        ),
        "median_total_latency_seconds": statistics.median(
            row["total_latency_seconds"] for row in records
        ),
        "resident_memory_bytes_after": int(process.memory_info().rss),
        "maximum_generation_tokens": maximum_tokens,
        "autonomous_generation": True,
        "teacher_at_inference": False,
        "syntax_only_counted_as_success": False,
        "every_task_requires_all_unit_tests": True,
        "records": records,
    }
    document["evidence_sha256"] = _canonical_sha(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        key: document[key]
        for key in (
            "status",
            "system",
            "distinct_prompts",
            "functional_successes",
            "functional_error_rate",
            "evidence_sha256",
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-dataset")
    generate.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    diversify = sub.add_parser("generate-diverse-dataset")
    diversify.add_argument("--source", type=Path, default=DEFAULT_DATASET)
    diversify.add_argument("--output", type=Path, required=True)
    generalize = sub.add_parser("generate-identifier-dataset")
    generalize.add_argument("--source", type=Path, required=True)
    generalize.add_argument("--output", type=Path, required=True)
    unique = sub.add_parser("generate-unique-identifier-dataset")
    unique.add_argument("--source", type=Path, required=True)
    unique.add_argument("--output", type=Path, required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    cache.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    cache.add_argument("--output", type=Path, required=True)
    recurrent_cache = sub.add_parser("cache-recurrent")
    recurrent_cache.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    recurrent_cache.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    recurrent_cache.add_argument("--output", type=Path, required=True)
    recovery_cache = sub.add_parser("cache-on-policy-recovery")
    recovery_cache.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    recovery_cache.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET
    )
    recovery_cache.add_argument("--cake-checkpoint", type=Path, required=True)
    recovery_cache.add_argument("--output", type=Path, required=True)
    recovery_cache.add_argument("--maximum-response-tokens", type=int, default=160)
    recovery_cache.add_argument("--horizons", default="8,32,64")
    train = sub.add_parser("train")
    train.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--rank", type=int, default=768)
    train.add_argument("--steps", type=int, default=2400)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--negative-count", type=int, default=1536)
    train.add_argument("--learning-rate", type=float, default=8.0e-4)
    recurrent = sub.add_parser("train-recurrent")
    recurrent.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    recurrent.add_argument("--cache", type=Path, required=True)
    recurrent.add_argument("--output", type=Path, required=True)
    recurrent.add_argument("--seed", type=int, required=True)
    recurrent.add_argument("--hidden-width", type=int, default=1024)
    recurrent.add_argument("--layers", type=int, default=1)
    recurrent.add_argument("--max-residual", type=float, default=6.0)
    recurrent.add_argument("--steps", type=int, default=1600)
    recurrent.add_argument("--batch-size", type=int, default=8)
    recurrent.add_argument("--learning-rate", type=float, default=3.0e-4)
    recurrent.add_argument("--initial-checkpoint", type=Path)
    recurrent.add_argument("--prompt-retention-fraction", type=float, default=0.0)
    recurrent.add_argument("--prompt-retention-weight", type=float, default=0.25)
    attentive = sub.add_parser("train-attentive")
    attentive.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    attentive.add_argument("--cache", type=Path, required=True)
    attentive.add_argument("--output", type=Path, required=True)
    attentive.add_argument("--seed", type=int, required=True)
    attentive.add_argument("--hidden-width", type=int, default=384)
    attentive.add_argument("--layers", type=int, default=1)
    attentive.add_argument("--heads", type=int, default=6)
    attentive.add_argument("--expansion", type=int, default=4)
    attentive.add_argument("--copy-width", type=int, default=0)
    attentive.add_argument("--initial-checkpoint", type=Path)
    attentive.add_argument("--pointer-supervision-weight", type=float, default=0.0)
    attentive.add_argument(
        "--copy-value-supervision-weight", type=float, default=0.0
    )
    attentive.add_argument(
        "--copy-gate-supervision-weight", type=float, default=0.0
    )
    attentive.add_argument(
        "--prompt-value-supervision-weight", type=float, default=0.0
    )
    attentive.add_argument(
        "--copy-value-projection", action="store_true"
    )
    attentive.add_argument("--selective-copy", action="store_true")
    attentive.add_argument("--transition-copy", action="store_true")
    attentive.add_argument("--copy-path-only", action="store_true")
    attentive.add_argument("--max-residual", type=float, default=6.0)
    attentive.add_argument("--steps", type=int, default=800)
    attentive.add_argument("--batch-size", type=int, default=8)
    attentive.add_argument("--learning-rate", type=float, default=3.0e-4)
    attentive.add_argument("--prompt-retention-fraction", type=float, default=0.25)
    attentive.add_argument("--prompt-retention-weight", type=float, default=0.25)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluate.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--cake-checkpoint", type=Path)
    evaluate.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    evaluate.add_argument("--maximum-tokens", type=int, default=192)
    args = parser.parse_args(argv)
    if args.command == "generate-dataset":
        result = generate_dataset(args.output)
    elif args.command == "generate-diverse-dataset":
        result = generate_diverse_training_dataset(
            args.source, args.output
        )
    elif args.command == "generate-identifier-dataset":
        result = generate_identifier_generalization_dataset(
            args.source, args.output
        )
    elif args.command == "generate-unique-identifier-dataset":
        result = generate_unique_identifier_dataset(
            args.source, args.output
        )
    elif args.command == "cache":
        result = cache_training_states(
            args.checkpoint, args.dataset, args.output
        )
    elif args.command == "cache-recurrent":
        result = cache_recurrent_training_states(
            args.checkpoint, args.dataset, args.output
        )
    elif args.command == "cache-on-policy-recovery":
        result = cache_on_policy_recovery_states(
            args.checkpoint,
            args.dataset,
            args.cake_checkpoint,
            args.output,
            max_response_tokens=args.maximum_response_tokens,
            horizons=tuple(
                int(value)
                for value in args.horizons.split(",")
                if value.strip()
            ),
        )
    elif args.command == "train":
        result = train_cake(
            args.checkpoint,
            args.cache,
            args.output,
            seed=args.seed,
            rank=args.rank,
            steps=args.steps,
            batch_size=args.batch_size,
            negative_count=args.negative_count,
            learning_rate=args.learning_rate,
        )
    elif args.command in {"train-recurrent", "train-attentive"}:
        result = train_recurrent_cake(
            args.checkpoint,
            args.cache,
            args.output,
            seed=args.seed,
            hidden_width=args.hidden_width,
            layers=args.layers,
            max_residual=args.max_residual,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            initial_checkpoint=(
                args.initial_checkpoint
            ),
            prompt_retention_fraction=args.prompt_retention_fraction,
            prompt_retention_weight=args.prompt_retention_weight,
            architecture=(
                "attentive"
                if args.command == "train-attentive"
                else "recurrent"
            ),
            heads=(args.heads if args.command == "train-attentive" else 6),
            expansion=(
                args.expansion if args.command == "train-attentive" else 4
            ),
            copy_width=(
                args.copy_width if args.command == "train-attentive" else 0
            ),
            pointer_supervision_weight=(
                args.pointer_supervision_weight
                if args.command == "train-attentive"
                else 0.0
            ),
            copy_value_projection=(
                args.copy_value_projection
                if args.command == "train-attentive"
                else False
            ),
            copy_value_supervision_weight=(
                args.copy_value_supervision_weight
                if args.command == "train-attentive"
                else 0.0
            ),
            selective_copy=(
                args.selective_copy
                if args.command == "train-attentive"
                else False
            ),
            copy_gate_supervision_weight=(
                args.copy_gate_supervision_weight
                if args.command == "train-attentive"
                else 0.0
            ),
            prompt_value_supervision_weight=(
                args.prompt_value_supervision_weight
                if args.command == "train-attentive"
                else 0.0
            ),
            transition_copy=(
                args.transition_copy
                if args.command == "train-attentive"
                else False
            ),
            copy_path_only=(
                args.copy_path_only
                if args.command == "train-attentive"
                else False
            ),
        )
    else:
        result = evaluate_functional(
            args.checkpoint,
            args.dataset,
            args.output,
            cake_checkpoint=args.cake_checkpoint,
            split=args.split,
            maximum_tokens=args.maximum_tokens,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
