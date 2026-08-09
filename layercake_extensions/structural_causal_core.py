"""Generic signed Phi-compatible structural causal English-core host ABI v7."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.structural_causal_core import StructuralCausalCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.unicode_direct_neural_core import (
    DIRECT_NEURAL_CORE_COMPOSITION,
    DIRECT_NEURAL_CORE_ROLE,
    UnicodeDirectNeuralCoreError,
    UnicodeSafeDirectNeuralCoreHost,
)


STRUCTURAL_CAUSAL_CORE_ABI_VERSION = "lc-direct-neural-core/7"
STRUCTURAL_CAUSAL_CORE_ABI_SHA256 = "e36ac987f0c3837ccaf56dee2d4e91542666a755657c3b688757a12aedfa5711"
STRUCTURAL_CAUSAL_CORE_FORMAT = "layercake-structural-causal-core/1"
STRUCTURAL_CAPABILITIES = frozenset(
    {
        "byte_input",
        "safe_tensors",
        "strict_utf8_boundary",
        "decoder_aware_external_tokenizer",
        "decoder_only_causal_execution",
        "persistent_rotary_kv_state",
        "structural_source_compatible_execution",
    }
)


def structural_core_manifest_architecture(model: StructuralCausalCore, tokenizer: DecoderAwareExternalTokenizer) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("structural model and tokenizer sizes differ")
    return {
        "name": "structural_causal_core",
        "format": STRUCTURAL_CAUSAL_CORE_FORMAT,
        "model": model.canonical_config(),
        "tokenizer": tokenizer.canonical_dict(),
        "tokenizer_sha256": tokenizer.hash(),
        "external_input_output": "UTF-8 bytes",
        "private_representation": "untied_phi_compatible_structural_causal_decoder",
        "action_validity": "external_sequence_decode_strict_utf8_pointer_prohibited",
    }


class StructuralCausalCoreHost(UnicodeSafeDirectNeuralCoreHost):
    """Install and execute one immutable v7 structural English core."""

    def __init__(self, registry_root: str | Path, *, trust_store: Mapping[str, bytes | str | Path], device: str | torch.device = "cpu") -> None:
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=STRUCTURAL_CAUSAL_CORE_ABI_VERSION,
                abi_hash=STRUCTURAL_CAUSAL_CORE_ABI_SHA256,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=STRUCTURAL_CAPABILITIES,
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
        if (
            not package.signed
            or manifest.cake_type != "portable_decoder"
            or manifest.abi_version != STRUCTURAL_CAUSAL_CORE_ABI_VERSION
            or manifest.abi_hash != STRUCTURAL_CAUSAL_CORE_ABI_SHA256
        ):
            raise UnicodeDirectNeuralCoreError("structural causal core identity mismatch")
        if manifest.domains != (DIRECT_NEURAL_CORE_ROLE,) or manifest.dependencies:
            raise UnicodeDirectNeuralCoreError("package is not an exclusive English core")
        if manifest.input_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "validity": "strict_utf8"}:
            raise UnicodeDirectNeuralCoreError("structural causal input contract mismatch")
        if manifest.output_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "composition": DIRECT_NEURAL_CORE_COMPOSITION, "validity": "strict_utf8"}:
            raise UnicodeDirectNeuralCoreError("structural causal output contract mismatch")
        architecture = manifest.architecture
        if architecture.get("name") != "structural_causal_core" or architecture.get("format") != STRUCTURAL_CAUSAL_CORE_FORMAT:
            raise UnicodeDirectNeuralCoreError("structural causal architecture mismatch")
        required = set(manifest.minimum_host_capabilities.get("features", []))
        if not STRUCTURAL_CAPABILITIES <= required:
            raise UnicodeDirectNeuralCoreError("structural causal capabilities are incomplete")

    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> StructuralCausalCore:
        architecture = package.manifest.architecture
        allowed = {"name", "format", "model", "tokenizer", "tokenizer_sha256", "external_input_output", "private_representation", "action_validity"}
        if (
            set(architecture) != allowed
            or architecture["external_input_output"] != "UTF-8 bytes"
            or architecture["private_representation"] != "untied_phi_compatible_structural_causal_decoder"
            or architecture["action_validity"] != "external_sequence_decode_strict_utf8_pointer_prohibited"
        ):
            raise UnicodeDirectNeuralCoreError("structural causal metadata is incomplete")
        tokenizer = DecoderAwareExternalTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]:
            raise UnicodeDirectNeuralCoreError("structural causal tokenizer hash mismatch")
        model = StructuralCausalCore(**architecture["model"]).bind_tokenizer(tokenizer)
        model.load_state_dict(package.tensors, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
