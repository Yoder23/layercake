"""Versioned internal representation contracts for LayerCake token branches."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .baseline_transformer import BytePairTokenizer


HYBRID_FORMAT = "layercake-hybrid-token-byte/1"
HYBRID_CONTRACT_VERSION = "layercake-hybrid-fallback/1"


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


def tokenizer_from_document(document: dict[str, Any]):
    base = BytePairTokenizer([tuple(pair) for pair in document["merges"]])
    if document.get("format") == HYBRID_FORMAT:
        contract = document.get("hybrid_contract", {})
        if contract.get("version") != HYBRID_CONTRACT_VERSION:
            raise ValueError("unsupported hybrid token-byte contract")
        return HybridTokenByteTokenizer(base)
    return base
