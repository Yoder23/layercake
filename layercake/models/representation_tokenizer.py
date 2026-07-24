"""Versioned internal representation contracts for LayerCake token branches."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .baseline_transformer import BytePairTokenizer


HYBRID_FORMAT = "layercake-hybrid-token-byte/1"
HYBRID_CONTRACT_VERSION = "layercake-hybrid-fallback/1"
WORD_BYTE_HYBRID_FORMAT = "layercake-word-byte-hybrid/1"
WORD_BYTE_HYBRID_CONTRACT_VERSION = "layercake-word-byte-fallback/1"


class HybridTokenByteTokenizer:
    """BPE for ordinary text with deterministic neural raw-byte fallback spans.

    IDs 0..255 retain their universal byte meanings. Protected spans are emitted
    as those byte IDs without applying merges. The language model, rather than a
    replacement-character preprocessor, therefore predicts every fallback byte.
    """

    _CODE_IDENTIFIER = re.compile(rb"[A-Za-z0-9]*_[A-Za-z0-9_]*")
    _HEX_LITERAL = re.compile(rb"0[xX][0-9A-Fa-f]+")
    _PATH_LIKE = re.compile(rb"(?:[A-Za-z]:)?[\\/][^\s]+")
    _CODE_PUNCTUATION = frozenset(b"{}[]\\`")

    def __init__(self, base: BytePairTokenizer):
        self.base = base
        self.merges = base.merges
        self.pieces = base.pieces
        self.merge_ids = base.merge_ids

    @property
    def vocab_size(self) -> int:
        return self.base.vocab_size

    @staticmethod
    def _protected_mask(value: bytes) -> list[bool]:
        protected = [byte >= 0x80 for byte in value]
        for index, byte in enumerate(value):
            if byte in HybridTokenByteTokenizer._CODE_PUNCTUATION:
                protected[index] = True
        for pattern in (
            HybridTokenByteTokenizer._CODE_IDENTIFIER,
            HybridTokenByteTokenizer._HEX_LITERAL,
            HybridTokenByteTokenizer._PATH_LIKE,
        ):
            for match in pattern.finditer(value):
                protected[match.start():match.end()] = [True] * (
                    match.end() - match.start()
                )
        try:
            value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            protected[error.start:error.end] = [True] * max(
                1, error.end - error.start
            )
        return protected

    def encode(self, value: bytes | str) -> list[int]:
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not value:
            return []
        protected = self._protected_mask(value)
        encoded: list[int] = []
        start = 0
        while start < len(value):
            mode = protected[start]
            end = start + 1
            while end < len(value) and protected[end] == mode:
                end += 1
            span = value[start:end]
            encoded.extend(list(span) if mode else self.base.encode(span))
            start = end
        return encoded

    def decode(self, ids: list[int]) -> bytes:
        return self.base.decode(ids)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format": HYBRID_FORMAT,
            "merges": [list(pair) for pair in self.merges],
            "hybrid_contract": {
                "version": HYBRID_CONTRACT_VERSION,
                "external_input": "UTF-8 bytes",
                "external_output": "UTF-8 bytes",
                "ordinary_spans": "shared deterministic BPE",
                "fallback_unit_ids": [0, 255],
                "fallback_semantics": "identity byte values",
                "protected_spans": [
                    "non-ASCII UTF-8 bytes",
                    "malformed UTF-8 bytes",
                    "underscore identifiers",
                    "hex literals",
                    "path-like spans",
                    "code punctuation",
                ],
                "fallback_execution": "neural next-unit distribution over byte IDs",
            },
        }

    def hash(self) -> str:
        raw = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


class WordByteHybridTokenizer:
    """Whole lexical units for English with universal raw-byte fallback."""

    def __init__(self, forms: list[bytes]):
        if len(set(forms)) != len(forms):
            raise ValueError("word-byte vocabulary contains duplicate forms")
        if any(not form or len(form) < 2 for form in forms):
            raise ValueError("word-byte lexical forms must contain at least two bytes")
        self.forms = list(forms)
        self.pieces = {
            **{index: bytes([index]) for index in range(256)},
            **{index: form for index, form in enumerate(forms, start=256)},
        }
        self.form_ids = {
            form: index for index, form in enumerate(forms, start=256)
        }

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.forms)

    @staticmethod
    def _ascii_letter(value: int) -> bool:
        return 65 <= value <= 90 or 97 <= value <= 122

    def encode(self, value: bytes | str) -> list[int]:
        if isinstance(value, str):
            value = value.encode("utf-8")
        result: list[int] = []
        index = 0
        while index < len(value):
            start = index
            if (
                value[index] == 32
                and index + 1 < len(value)
                and self._ascii_letter(value[index + 1])
            ):
                index += 1
                while index < len(value) and self._ascii_letter(value[index]):
                    index += 1
                span = value[start:index]
            elif self._ascii_letter(value[index]):
                index += 1
                while index < len(value) and self._ascii_letter(value[index]):
                    index += 1
                span = value[start:index]
            else:
                result.append(value[index])
                index += 1
                continue
            token_id = self.form_ids.get(span)
            if token_id is None:
                result.extend(span)
            else:
                result.append(token_id)
        return result

    def decode(self, ids: list[int]) -> bytes:
        try:
            return b"".join(self.pieces[int(index)] for index in ids)
        except KeyError as error:
            raise ValueError(f"undefined word-byte token id: {error.args[0]}") from error

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format": WORD_BYTE_HYBRID_FORMAT,
            "forms_hex": [form.hex() for form in self.forms],
            "hybrid_contract": {
                "version": WORD_BYTE_HYBRID_CONTRACT_VERSION,
                "external_input": "UTF-8 bytes",
                "external_output": "UTF-8 bytes",
                "ordinary_spans": "frequency-locked whole English lexical forms",
                "fallback_unit_ids": [0, 255],
                "fallback_semantics": "identity byte values",
                "unknown_names_and_exact_strings": "raw byte units",
                "fallback_execution": "neural next-unit distribution over byte IDs",
            },
        }

    def hash(self) -> str:
        raw = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def tokenizer_from_document(document: dict[str, Any]):
    if document.get("format") == WORD_BYTE_HYBRID_FORMAT:
        contract = document.get("hybrid_contract", {})
        if contract.get("version") != WORD_BYTE_HYBRID_CONTRACT_VERSION:
            raise ValueError("unsupported word-byte hybrid contract")
        return WordByteHybridTokenizer(
            [bytes.fromhex(value) for value in document["forms_hex"]]
        )
    base = BytePairTokenizer([tuple(pair) for pair in document["merges"]])
    if document.get("format") == HYBRID_FORMAT:
        contract = document.get("hybrid_contract", {})
        if contract.get("version") != HYBRID_CONTRACT_VERSION:
            raise ValueError("unsupported hybrid token-byte contract")
        return HybridTokenByteTokenizer(base)
    return base
