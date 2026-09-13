"""Deterministic execution of compact, signed canonical factual cakes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable

import torch
from torch import nn


CANONICAL_FACTUAL_FORMAT = "layercake-canonical-factual-table/1"
MAX_FACTS = 4096
MAX_RELATION_CHARS = 256
MAX_ENTITY_CHARS = 512
MAX_VALUE_BYTES = 4096


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def normalize_facts(facts: Iterable[dict[str, Any]]) -> tuple[dict[str, str], ...]:
    """Validate and canonically order a bounded relation/entity/value table."""

    rows = []
    for item in facts:
        if not isinstance(item, dict) or set(item) != {"relation", "entity", "value"}:
            raise ValueError("canonical factual rows require relation, entity, and value")
        relation = item["relation"]
        entity = item["entity"]
        value = item["value"]
        if not all(isinstance(entry, str) and entry.strip() for entry in (relation, entity, value)):
            raise ValueError("canonical factual fields must be non-empty strings")
        if (
            len(relation) > MAX_RELATION_CHARS
            or len(entity) > MAX_ENTITY_CHARS
            or len(value.encode("utf-8")) > MAX_VALUE_BYTES
        ):
            raise ValueError("canonical factual field exceeds its safety boundary")
        rows.append({"relation": relation, "entity": entity, "value": value})
    if not rows or len(rows) > MAX_FACTS:
        raise ValueError("canonical factual table has an invalid row count")
    rows.sort(key=lambda row: (row["relation"], row["entity"].casefold(), row["value"]))
    keys = [(row["relation"], row["entity"].casefold()) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("canonical factual table contains a duplicate key")
    return tuple(rows)


def facts_sha256(facts: Iterable[dict[str, Any]]) -> str:
    normalized = normalize_facts(facts)
    return hashlib.sha256(_canonical_bytes(list(normalized))).hexdigest()


def canonical_factual_manifest_architecture(
    namespace: str, facts: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(namespace, str) or not namespace.strip() or len(namespace) > 512:
        raise ValueError("canonical factual namespace is invalid")
    normalized = normalize_facts(facts)
    return {
        "name": "canonical_factual_table",
        "format": CANONICAL_FACTUAL_FORMAT,
        "namespace": namespace,
        "facts": list(normalized),
        "facts_sha256": facts_sha256(normalized),
        "external_input_output": "UTF-8 bytes",
    }


class _RawByteTokenizer:
    @staticmethod
    def decode_actions(actions: Iterable[int], source_lexemes: Iterable[str]) -> bytes:
        del source_lexemes
        values = tuple(int(value) for value in actions)
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("canonical factual action is not a byte")
        return bytes(values)


@dataclass
class CanonicalFactualState:
    target: bytes
    generated_actions: list[int] = field(default_factory=list)
    source_lexemes: tuple[str, ...] = ()
    complete: bool = False


class CanonicalFactualDecoder(nn.Module):
    """Execute a selected factual table without receiver learning."""

    def __init__(self, namespace: str, facts: Iterable[dict[str, Any]]) -> None:
        super().__init__()
        architecture = canonical_factual_manifest_architecture(namespace, facts)
        self.namespace = str(architecture["namespace"])
        self.facts = tuple(architecture["facts"])
        digest = bytes.fromhex(str(architecture["facts_sha256"]))
        self.register_buffer("payload_digest", torch.tensor(tuple(digest), dtype=torch.uint8))
        self.tokenizer = _RawByteTokenizer()
        self.maximum_target_actions = max(
            len(str(row["value"]).encode("utf-8")) for row in self.facts
        )
        self._patterns = tuple(
            re.compile(rf"(?<!\w){re.escape(str(row['entity']))}(?!\w)", re.IGNORECASE)
            for row in self.facts
        )

    def _answer(self, prompt: bytes | str) -> bytes:
        if isinstance(prompt, bytes):
            try:
                text = prompt.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError("canonical factual prompt must be valid UTF-8") from exc
        elif isinstance(prompt, str):
            text = prompt
        else:
            raise TypeError("canonical factual prompt must be bytes or str")
        matches = [
            str(row["value"])
            for row, pattern in zip(self.facts, self._patterns, strict=True)
            if pattern.search(text)
        ]
        if len(matches) != 1:
            return b""
        return matches[0].encode("utf-8")

    def prefill_bytes(self, prompt: bytes | str) -> CanonicalFactualState:
        target = self._answer(prompt)
        return CanonicalFactualState(target=target, complete=not target)

    def decode_step(self, state: CanonicalFactualState) -> None:
        if state.complete:
            return
        position = len(state.generated_actions)
        if position >= len(state.target):
            state.complete = True
            return
        state.generated_actions.append(state.target[position])
        state.complete = len(state.generated_actions) >= len(state.target)
