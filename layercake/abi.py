"""Canonical LayerCake ABI contracts and compatibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from .input_interfaces import InputInterfaceSpec


class ABICompatibilityError(ValueError):
    """Raised when an artifact is unsafe to attach to an ABI."""


@dataclass(frozen=True)
class ABISpec:
    version: str
    d_abi: int
    normalization: str = "layernorm"
    basis: str = "canonical"
    input_interface: InputInterfaceSpec = InputInterfaceSpec(
        mode="tokenized", vocab_size=256
    )
    quantization: str = "fp32"
    compatible_bricks: tuple[str, ...] = (
        "dense",
        "low_rank",
        "sparse_low_rank",
        "gated_sparse",
    )

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("ABI version must be non-empty")
        if self.d_abi <= 0:
            raise ValueError("d_abi must be positive")

    def canonical_dict(self) -> dict:
        result = asdict(self)
        result["compatible_bricks"] = list(self.compatible_bricks)
        return result

    def hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compatibility_errors(
        self,
        other: "ABISpec",
        *,
        require_same_interface: bool = False,
        brick_type: str | None = None,
    ) -> list[str]:
        errors: list[str] = []
        if self.version != other.version:
            errors.append(f"version mismatch: {self.version!r} != {other.version!r}")
        if self.d_abi != other.d_abi:
            errors.append(f"d_abi mismatch: {self.d_abi} != {other.d_abi}")
        if self.normalization != other.normalization:
            errors.append("normalization contract mismatch")
        if self.basis != other.basis:
            errors.append("coordinate basis mismatch")
        if require_same_interface and self.input_interface != other.input_interface:
            errors.append("input-interface mismatch")
        if brick_type and (
            brick_type not in self.compatible_bricks
            or brick_type not in other.compatible_bricks
        ):
            errors.append(f"brick type {brick_type!r} is not supported by both ABIs")
        return errors

    def assert_compatible(
        self,
        other: "ABISpec",
        *,
        require_same_interface: bool = False,
        brick_type: str | None = None,
    ) -> None:
        errors = self.compatibility_errors(
            other,
            require_same_interface=require_same_interface,
            brick_type=brick_type,
        )
        if errors:
            raise ABICompatibilityError("; ".join(errors))


def common_abi(
    specs: Iterable[ABISpec], *, require_same_interface: bool = False
) -> ABISpec:
    specs = list(specs)
    if not specs:
        raise ValueError("at least one ABI spec is required")
    first = specs[0]
    for spec in specs[1:]:
        first.assert_compatible(spec, require_same_interface=require_same_interface)
    return first
