"""Generic selective-boundary BPE direct neural core host ABI v5."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.portable_token_plan import PortableTokenPlan
from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
from layercake_extensions.unicode_direct_neural_core import (
    DIRECT_NEURAL_CORE_COMPOSITION,
    DIRECT_NEURAL_CORE_ROLE,
    UNICODE_LEXEME_REGEX,
    UnicodeDirectNeuralCoreError,
    UnicodeSafeDirectNeuralCoreHost,
)


SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_VERSION = "lc-direct-neural-core/5"
SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_SHA256 = "6ac5e6fca6e21e54f7d7cd945f8179ba6c10ea772d3277ef502c89019be76dc5"
SELECTIVE_BPE_TOKENIZER_FORMAT = "layercake-selective-boundary-bpe/1"
SELECTIVE_BPE_TOKEN_PLAN_FORMAT = "layercake-selective-boundary-bpe-token-plan/1"
SELECTIVE_BOUNDARY_POLICY = "ASCII_ALNUM_OR_UNDERSCORE_WITH_DIGIT_OR_UNDERSCORE"
PROTECTED_UNIT = re.compile(r"^[A-Za-z0-9_]+$")


class SelectiveBoundaryBpeTokenizer(Utf8ConcatenativeBpeTokenizer):
    """Raw BPE with stable boundaries only around identifier-like units."""

    def split(self, value: bytes | str) -> list[bytes]:
        text = value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value
        if not isinstance(text, str):
            raise TypeError("selective BPE input must be bytes or str")
        units = UNICODE_LEXEME_REGEX.findall(text)
        if "".join(units) != text:
            raise RuntimeError("selective BPE lexical boundary split is not lossless")
        output: list[bytes] = []
        buffered: list[str] = []

        def flush() -> None:
            if buffered:
                output.extend(Utf8ConcatenativeBpeTokenizer.split(self, "".join(buffered)))
                buffered.clear()

        for unit in units:
            protected = PROTECTED_UNIT.fullmatch(unit) is not None and ("_" in unit or any(character.isdigit() for character in unit))
            if protected:
                flush()
                output.extend(Utf8ConcatenativeBpeTokenizer.split(self, unit))
            else:
                buffered.append(unit)
        flush()
        if b"".join(output) != text.encode("utf-8"):
            raise RuntimeError("selective BPE representation is not exact")
        return output

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format": SELECTIVE_BPE_TOKENIZER_FORMAT,
            "tokenizers_json": self.document,
            "tokenizers_json_sha256": hashlib.sha256(json.dumps(self.document, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "boundary_policy": SELECTIVE_BOUNDARY_POLICY,
            "piece_semantics": "UTF8_CONCATENATE_EXACTLY",
            "normalization": "NONE",
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "SelectiveBoundaryBpeTokenizer":
        if (
            set(document) != {"format", "tokenizers_json", "tokenizers_json_sha256", "boundary_policy", "piece_semantics", "normalization"}
            or document.get("format") != SELECTIVE_BPE_TOKENIZER_FORMAT
            or document.get("boundary_policy") != SELECTIVE_BOUNDARY_POLICY
            or document.get("piece_semantics") != "UTF8_CONCATENATE_EXACTLY"
            or document.get("normalization") != "NONE"
        ):
            raise ValueError("selective BPE tokenizer document changed")
        value = cls(document["tokenizers_json"])
        if value.canonical_dict() != document:
            raise ValueError("selective BPE tokenizer identity changed")
        return value

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def selective_bpe_token_plan_manifest_architecture(model: PortableTokenPlan, tokenizer: SelectiveBoundaryBpeTokenizer) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("model and selective BPE tokenizer vocabulary sizes differ")
    return {
        "name": "selective_boundary_bpe_portable_token_plan",
        "format": SELECTIVE_BPE_TOKEN_PLAN_FORMAT,
        "model": model.canonical_config(),
        "tokenizer": tokenizer.canonical_dict(),
        "tokenizer_sha256": tokenizer.hash(),
        "external_input_output": "UTF-8 bytes",
        "private_representation": "portable_token_plan_pointer_transformer",
        "action_validity": "every_fixed_and_pointer_action_complete_utf8",
    }


class SelectiveBoundaryBpeDirectNeuralCoreHost(UnicodeSafeDirectNeuralCoreHost):
    def __init__(self, registry_root: str | Path, *, trust_store: Mapping[str, bytes | str | Path], device: str | torch.device = "cpu") -> None:
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_VERSION,
                abi_hash=SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_SHA256,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=frozenset({"byte_input", "safe_tensors", "persistent_incremental_state", "unicode_atomic_actions", "strict_utf8_boundary", "selective_boundary_bpe"}),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )
        self.device = torch.device(device)
        self.module = None
        self.active_cake_id = None
        self.active_archive_hash = None
        self.active_payload_hash = None
        self.receiver_training_steps = 0
        self.receiver_calibration_runs = 0

    @staticmethod
    def _validate_role(package: CakePackage) -> None:
        manifest = package.manifest
        if not package.signed or manifest.cake_type != "portable_decoder" or manifest.abi_version != SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_VERSION or manifest.abi_hash != SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_SHA256:
            raise UnicodeDirectNeuralCoreError("selective BPE direct core identity mismatch")
        if manifest.domains != (DIRECT_NEURAL_CORE_ROLE,) or manifest.dependencies:
            raise UnicodeDirectNeuralCoreError("package is not an exclusive English core")
        architecture = manifest.architecture
        if architecture.get("name") != "selective_boundary_bpe_portable_token_plan" or architecture.get("format") != SELECTIVE_BPE_TOKEN_PLAN_FORMAT:
            raise UnicodeDirectNeuralCoreError("selective BPE architecture mismatch")

    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> PortableTokenPlan:
        architecture = package.manifest.architecture
        tokenizer = SelectiveBoundaryBpeTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]:
            raise UnicodeDirectNeuralCoreError("selective BPE tokenizer hash mismatch")
        model = PortableTokenPlan(**architecture["model"]).bind_tokenizer(tokenizer)
        model.load_state_dict(package.tensors, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
