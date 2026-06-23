"""Versioned text/byte interface contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal

InputMode = Literal["tokenized", "byte", "byte_patch"]


@dataclass(frozen=True)
class InputInterfaceSpec:
    mode: InputMode
    vocab_size: int | None = None
    byte_vocab_size: int = 256
    patching: str | None = None
    max_patch_size: int | None = None
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if self.mode == "tokenized" and not self.vocab_size:
            raise ValueError("tokenized mode requires vocab_size")
        if self.mode == "byte_patch" and not self.patching:
            raise ValueError("byte_patch mode requires a patching contract")
        if self.mode != "byte_patch" and self.patching is not None:
            raise ValueError("patching is only valid for byte_patch mode")
        if self.byte_vocab_size != 256:
            raise ValueError("LayerCake byte interfaces currently require 256 byte values")

    def canonical_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compatible_with(self, other: "InputInterfaceSpec", *, strict: bool = True) -> bool:
        if strict:
            return self == other
        if self.mode == other.mode:
            return self.encoding == other.encoding and self.patching == other.patching
        return {self.mode, other.mode} <= {"tokenized", "byte", "byte_patch"}
