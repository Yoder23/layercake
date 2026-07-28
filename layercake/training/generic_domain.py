"""Configuration-driven datasets and bounded functional checks for cakes.

The deployed cake contains only tensors and canonical metadata.  Evaluator
fixtures live in campaign evidence and are never packaged or available during
inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping


CONFIG_FORMAT = "layercake-generic-domain-authoring-config/1"
DATASET_FORMAT = "layercake-generic-domain-dataset/1"
EVALUATORS = frozenset({"sqlite_query", "regex_fullmatch"})
SPLITS = ("train", "validation", "test")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


class GenericDomainError(ValueError):
    """Raised when authoring data or a bounded evaluator is invalid."""


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(variables)
        except (KeyError, ValueError) as error:
            raise GenericDomainError(
                f"invalid or unknown authoring placeholder: {error}"
            ) from error
    if isinstance(value, list):
        return [_render(item, variables) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _render(item, variables)
            for key, item in value.items()
        }
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise GenericDomainError("authoring templates must contain JSON values")


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "format",
        "domain_id",
        "description",
        "data_seed",
        "rows_per_family",
        "families",
    }
    if set(config) != expected or config.get("format") != CONFIG_FORMAT:
        raise GenericDomainError(
            "generic authoring config is incomplete or ambiguous"
        )
    domain_id = config.get("domain_id")
    if not isinstance(domain_id, str) or not _IDENTIFIER.fullmatch(
        domain_id
    ):
        raise GenericDomainError("domain_id is not a canonical identifier")
    if not isinstance(config.get("description"), str) or not config[
        "description"
    ].strip():
        raise GenericDomainError("domain description is required")
    if (
        not isinstance(config.get("data_seed"), int)
        or isinstance(config.get("data_seed"), bool)
    ):
        raise GenericDomainError("data_seed must be an integer")
    counts = config.get("rows_per_family")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(SPLITS)
        or any(
            not isinstance(counts[split], int)
            or isinstance(counts[split], bool)
            or counts[split] <= 0
            for split in SPLITS
        )
    ):
        raise GenericDomainError(
            "rows_per_family must declare positive split counts"
        )
    families = config.get("families")
    if not isinstance(families, list) or len(families) < 1:
        raise GenericDomainError("at least one domain family is required")
    family_ids: set[str] = set()
    for family in families:
        allowed = {
            "id",
            "prompt_templates",
            "response_template",
            "copy_slots",
            "evaluation",
        }
        if not isinstance(family, dict) or set(family) != allowed:
            raise GenericDomainError(
                "domain family is incomplete or ambiguous"
            )
        identifier = family.get("id")
        if (
            not isinstance(identifier, str)
            or not _IDENTIFIER.fullmatch(identifier)
            or identifier in family_ids
        ):
            raise GenericDomainError(
                "family ids must be unique canonical identifiers"
            )
        family_ids.add(identifier)
        prompts = family.get("prompt_templates")
        if (
            not isinstance(prompts, list)
            or not prompts
            or any(
                not isinstance(prompt, str) or not prompt
                for prompt in prompts
            )
        ):
            raise GenericDomainError(
                "prompt_templates must contain non-empty strings"
            )
        if (
            not isinstance(family.get("response_template"), str)
            or not family["response_template"]
        ):
            raise GenericDomainError("response_template is required")
        copy_slots = family.get("copy_slots")
        if (
            not isinstance(copy_slots, list)
            or not copy_slots
            or len(copy_slots) != len(set(copy_slots))
            or any(
                not isinstance(slot, str)
                or not _IDENTIFIER.fullmatch(slot)
                for slot in copy_slots
            )
        ):
            raise GenericDomainError(
                "copy_slots must be unique canonical identifiers"
            )
        evaluation = family.get("evaluation")
        if (
            not isinstance(evaluation, dict)
            or evaluation.get("kind") not in EVALUATORS
        ):
            raise GenericDomainError(
                "family requires a supported functional evaluator"
            )


def render_dataset(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render deterministic, split-disjoint rows from one generic config."""

    _validate_config(config)
    domain_id = str(config["domain_id"])
    seed = int(config["data_seed"])
    counts = config["rows_per_family"]
    rows: list[dict[str, Any]] = []
    global_index = 0
    split_offsets = {"train": 0, "validation": 100_000, "test": 200_000}
    for split in SPLITS:
        for family_index, family in enumerate(config["families"]):
            for local_index in range(int(counts[split])):
                nonce = seed + split_offsets[split] + (
                    family_index * 10_000
                ) + local_index
                variables = {
                    "domain": domain_id,
                    "family": str(family["id"]),
                    "split": split,
                    "index": f"{local_index:04d}",
                    "nonce": str(nonce),
                }
                for slot_index, slot in enumerate(family["copy_slots"]):
                    variables[slot] = (
                        f"{slot}_{domain_id}_{split}_{nonce + slot_index}"
                    )
                prompt_templates = family["prompt_templates"]
                prompt = _render(
                    prompt_templates[local_index % len(prompt_templates)],
                    variables,
                )
                response = _render(
                    family["response_template"], variables
                )
                evaluation = _render(family["evaluation"], variables)
                copy_lexemes = [
                    variables[slot] for slot in family["copy_slots"]
                ]
                row = {
                    "format": DATASET_FORMAT,
                    "id": (
                        f"{domain_id}-{split}-{family['id']}-"
                        f"{local_index:04d}"
                    ),
                    "domain_id": domain_id,
                    "family": family["id"],
                    "split": split,
                    "prompt": prompt,
                    "response": response,
                    "copy_lexemes": copy_lexemes,
                    "evaluation": evaluation,
                    "generation_index": global_index,
                }
                validate_row(row)
                rows.append(row)
                global_index += 1
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise GenericDomainError("rendered row ids are not unique")
    for split in SPLITS:
        values = {
            value
            for row in rows
            if row["split"] == split
            for value in row["copy_lexemes"]
        }
        others = {
            value
            for row in rows
            if row["split"] != split
            for value in row["copy_lexemes"]
        }
        if values & others:
            raise GenericDomainError(
                "dynamic copy lexemes overlap across data splits"
            )
    return rows


def validate_row(row: Mapping[str, Any]) -> None:
    expected = {
        "format",
        "id",
        "domain_id",
        "family",
        "split",
        "prompt",
        "response",
        "copy_lexemes",
        "evaluation",
        "generation_index",
    }
    if set(row) != expected or row.get("format") != DATASET_FORMAT:
        raise GenericDomainError("generic domain row schema is invalid")
    for key in ("id", "domain_id", "family", "prompt", "response"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise GenericDomainError(f"row {key} must be a non-empty string")
    if row.get("split") not in SPLITS:
        raise GenericDomainError("row split is unsupported")
    values = row.get("copy_lexemes")
    if (
        not isinstance(values, list)
        or not values
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise GenericDomainError("row copy_lexemes are invalid")
    for value in values:
        if row["prompt"].count(value) != 1:
            raise GenericDomainError(
                "each copy lexeme must occur once in the prompt"
            )
        if value not in row["response"]:
            raise GenericDomainError(
                "each copy lexeme must occur in the response"
            )
    evaluation = row.get("evaluation")
    if (
        not isinstance(evaluation, dict)
        or evaluation.get("kind") not in EVALUATORS
    ):
        raise GenericDomainError("row evaluator is unsupported")
    if (
        not isinstance(row.get("generation_index"), int)
        or isinstance(row.get("generation_index"), bool)
        or row["generation_index"] < 0
    ):
        raise GenericDomainError("generation_index is invalid")


def write_dataset(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise GenericDomainError("generic domain dataset is immutable")
    rows = list(rows)
    for row in rows:
        validate_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GenericDomainError(f"cannot read dataset: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise GenericDomainError(
                f"invalid JSON on dataset line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise GenericDomainError("dataset rows must be objects")
        validate_row(row)
        rows.append(row)
    identifiers = [row["id"] for row in rows]
    if not rows or len(identifiers) != len(set(identifiers)):
        raise GenericDomainError(
            "dataset must contain rows with unique identifiers"
        )
    return rows


def _normalize_sql_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 9)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


def _evaluate_sql(
    generated: str,
    evaluation: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    allowed = {"kind", "setup", "expected_rows"}
    if set(evaluation) != allowed:
        raise GenericDomainError("sqlite evaluator schema is invalid")
    setup = evaluation.get("setup")
    expected = evaluation.get("expected_rows")
    if (
        not isinstance(setup, list)
        or not setup
        or any(not isinstance(statement, str) for statement in setup)
        or not isinstance(expected, list)
        or any(not isinstance(row, list) for row in expected)
    ):
        raise GenericDomainError("sqlite fixtures are invalid")
    query = generated.strip()
    if (
        len(query.encode("utf-8")) > 512
        or not re.match(r"^(?:SELECT|WITH)\b", query, re.IGNORECASE)
    ):
        return False, [{"status": "UNSAFE_OR_NON_QUERY_OUTPUT"}]
    connection = sqlite3.connect(":memory:")
    steps = 0

    def bounded() -> int:
        nonlocal steps
        steps += 1
        return int(steps > 10_000)

    connection.set_progress_handler(bounded, 100)
    try:
        for statement in setup:
            connection.execute(statement)
        cursor = connection.execute(query)
        observed = [
            [_normalize_sql_value(value) for value in row]
            for row in cursor.fetchall()
        ]
        normalized_expected = [
            [_normalize_sql_value(value) for value in row]
            for row in expected
        ]
        passed = observed == normalized_expected
        return passed, [
            {
                "status": "PASS" if passed else "RESULT_MISMATCH",
                "observed_rows": observed,
                "expected_rows": normalized_expected,
            }
        ]
    except sqlite3.Error as error:
        return False, [
            {
                "status": "SQL_ERROR",
                "error_class": type(error).__name__,
            }
        ]
    finally:
        connection.close()


def _evaluate_regex(
    generated: str,
    evaluation: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    allowed = {"kind", "must_match", "must_not_match"}
    if set(evaluation) != allowed:
        raise GenericDomainError("regex evaluator schema is invalid")
    positives = evaluation.get("must_match")
    negatives = evaluation.get("must_not_match")
    if (
        not isinstance(positives, list)
        or not positives
        or not isinstance(negatives, list)
        or not negatives
        or any(
            not isinstance(value, str)
            for value in positives + negatives
        )
    ):
        raise GenericDomainError("regex fixtures are invalid")
    pattern = generated.strip()
    if len(pattern.encode("utf-8")) > 256:
        return False, [{"status": "PATTERN_TOO_LARGE"}]
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        return False, [
            {
                "status": "REGEX_ERROR",
                "error_class": type(error).__name__,
            }
        ]
    checks = [
        {
            "text": value,
            "expected": True,
            "observed": compiled.fullmatch(value) is not None,
        }
        for value in positives
    ] + [
        {
            "text": value,
            "expected": False,
            "observed": compiled.fullmatch(value) is not None,
        }
        for value in negatives
    ]
    passed = all(row["expected"] == row["observed"] for row in checks)
    return passed, checks


def evaluate_generated(
    generated: bytes | str,
    row: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    """Run only the preregistered bounded functional evaluator."""

    validate_row(row)
    if isinstance(generated, bytes):
        try:
            text = generated.decode("utf-8")
        except UnicodeDecodeError:
            return False, [{"status": "INVALID_UTF8"}]
    else:
        text = generated
    evaluation = row["evaluation"]
    if evaluation["kind"] == "sqlite_query":
        return _evaluate_sql(text, evaluation)
    if evaluation["kind"] == "regex_fullmatch":
        return _evaluate_regex(text, evaluation)
    raise GenericDomainError("row evaluator is unsupported")
