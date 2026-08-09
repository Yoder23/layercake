"""Generic signed decoder-only causal English-core host ABI v6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.native_causal_core import TiedNativeCausalCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.unicode_direct_neural_core import DIRECT_NEURAL_CORE_ROLE, UnicodeDirectNeuralCoreError, UnicodeSafeDirectNeuralCoreHost


CAUSAL_DIRECT_NEURAL_CORE_ABI_VERSION = "lc-direct-neural-core/6"
CAUSAL_DIRECT_NEURAL_CORE_ABI_SHA256 = "ea6e2149ceb010b1c78e53892e1bf0a5dc414b29577701d15ceaa909344f7b9e"
CAUSAL_TOKEN_PLAN_FORMAT = "layercake-tied-native-causal-core/1"


def causal_core_manifest_architecture(model: TiedNativeCausalCore, tokenizer: DecoderAwareExternalTokenizer) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("causal model and tokenizer sizes differ")
    return {
        "name": "tied_native_causal_core",
        "format": CAUSAL_TOKEN_PLAN_FORMAT,
        "model": model.canonical_config(),
        "tokenizer": tokenizer.canonical_dict(),
        "tokenizer_sha256": tokenizer.hash(),
        "external_input_output": "UTF-8 bytes",
        "private_representation": "tied_decoder_only_causal_transformer",
        "action_validity": "external_sequence_decode_strict_utf8_pointer_prohibited",
    }


class CausalDirectNeuralCoreHost(UnicodeSafeDirectNeuralCoreHost):
    def __init__(self, registry_root: str | Path, *, trust_store: Mapping[str, bytes | str | Path], device: str | torch.device = "cpu") -> None:
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=CAUSAL_DIRECT_NEURAL_CORE_ABI_VERSION,
                abi_hash=CAUSAL_DIRECT_NEURAL_CORE_ABI_SHA256,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=frozenset({"byte_input", "safe_tensors", "persistent_incremental_state", "strict_utf8_boundary", "decoder_aware_external_tokenizer", "decoder_only_causal_execution"}),
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
        if not package.signed or manifest.cake_type != "portable_decoder" or manifest.abi_version != CAUSAL_DIRECT_NEURAL_CORE_ABI_VERSION or manifest.abi_hash != CAUSAL_DIRECT_NEURAL_CORE_ABI_SHA256:
            raise UnicodeDirectNeuralCoreError("causal direct core identity mismatch")
        if manifest.domains != (DIRECT_NEURAL_CORE_ROLE,) or manifest.dependencies:
            raise UnicodeDirectNeuralCoreError("package is not an exclusive English core")
        architecture = manifest.architecture
        if architecture.get("name") != "tied_native_causal_core" or architecture.get("format") != CAUSAL_TOKEN_PLAN_FORMAT:
            raise UnicodeDirectNeuralCoreError("causal core architecture mismatch")

    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> TiedNativeCausalCore:
        architecture = package.manifest.architecture
        allowed = {"name", "format", "model", "tokenizer", "tokenizer_sha256", "external_input_output", "private_representation", "action_validity"}
        if set(architecture) != allowed or architecture["private_representation"] != "tied_decoder_only_causal_transformer":
            raise UnicodeDirectNeuralCoreError("causal core metadata is incomplete")
        tokenizer = DecoderAwareExternalTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]:
            raise UnicodeDirectNeuralCoreError("causal tokenizer hash mismatch")
        model = TiedNativeCausalCore(**architecture["model"]).bind_tokenizer(tokenizer)
        model.load_state_dict(package.tensors, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
