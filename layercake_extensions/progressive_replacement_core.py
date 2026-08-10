"""Generic signed progressive-replacement English-core host ABI v8."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.progressive_replacement_core import ProgressiveReplacementCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.unicode_direct_neural_core import (
    DIRECT_NEURAL_CORE_COMPOSITION,
    DIRECT_NEURAL_CORE_ROLE,
    UnicodeDirectNeuralCoreError,
    UnicodeSafeDirectNeuralCoreHost,
)


PROGRESSIVE_REPLACEMENT_CORE_ABI_VERSION = "lc-direct-neural-core/8"
PROGRESSIVE_REPLACEMENT_CORE_ABI_SHA256 = "0f86fc5e582d3d99bcf30ab94e918a1e1b7300d0e9f7c82c7866b6e5291ff972"
PROGRESSIVE_REPLACEMENT_CORE_FORMAT = "layercake-progressive-replacement-core/1"
PROGRESSIVE_REPLACEMENT_CAPABILITIES = frozenset(
    {
        "byte_input",
        "safe_tensors",
        "strict_utf8_boundary",
        "decoder_aware_external_tokenizer",
        "decoder_only_causal_execution",
        "persistent_rotary_kv_state",
        "full_residual_width_replacement_execution",
        "zero_source_transformer_blocks",
    }
)


def progressive_replacement_manifest_architecture(
    model: ProgressiveReplacementCore, tokenizer: DecoderAwareExternalTokenizer
) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("progressive replacement model and tokenizer sizes differ")
    return {
        "name": "progressive_replacement_core",
        "format": PROGRESSIVE_REPLACEMENT_CORE_FORMAT,
        "model": model.canonical_config(),
        "tokenizer": tokenizer.canonical_dict(),
        "tokenizer_sha256": tokenizer.hash(),
        "external_input_output": "UTF-8 bytes",
        "private_representation": "full_residual_width_progressive_bottleneck_replacement_decoder",
        "action_validity": "external_sequence_decode_strict_utf8_pointer_prohibited",
        "source_transformer_blocks": 0,
    }


class ProgressiveReplacementCoreHost(UnicodeSafeDirectNeuralCoreHost):
    """Install and execute one immutable v8 progressive replacement core."""

    def __init__(
        self,
        registry_root: str | Path,
        *,
        trust_store: Mapping[str, bytes | str | Path],
        device: str | torch.device = "cpu",
    ) -> None:
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=PROGRESSIVE_REPLACEMENT_CORE_ABI_VERSION,
                abi_hash=PROGRESSIVE_REPLACEMENT_CORE_ABI_SHA256,
                precisions=("fp32", "fp16", "bf16"),
                backends=("pytorch", "cuda"),
                capabilities=PROGRESSIVE_REPLACEMENT_CAPABILITIES,
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
            or manifest.abi_version != PROGRESSIVE_REPLACEMENT_CORE_ABI_VERSION
            or manifest.abi_hash != PROGRESSIVE_REPLACEMENT_CORE_ABI_SHA256
        ):
            raise UnicodeDirectNeuralCoreError("progressive replacement core identity mismatch")
        if manifest.domains != (DIRECT_NEURAL_CORE_ROLE,) or manifest.dependencies:
            raise UnicodeDirectNeuralCoreError("package is not an exclusive English core")
        if manifest.input_contract != {
            "external": "UTF-8 bytes",
            "role": DIRECT_NEURAL_CORE_ROLE,
            "validity": "strict_utf8",
        }:
            raise UnicodeDirectNeuralCoreError("progressive replacement input contract mismatch")
        if manifest.output_contract != {
            "external": "UTF-8 bytes",
            "role": DIRECT_NEURAL_CORE_ROLE,
            "composition": DIRECT_NEURAL_CORE_COMPOSITION,
            "validity": "strict_utf8",
        }:
            raise UnicodeDirectNeuralCoreError("progressive replacement output contract mismatch")
        architecture = manifest.architecture
        if (
            architecture.get("name") != "progressive_replacement_core"
            or architecture.get("format") != PROGRESSIVE_REPLACEMENT_CORE_FORMAT
            or architecture.get("source_transformer_blocks") != 0
        ):
            raise UnicodeDirectNeuralCoreError("progressive replacement architecture mismatch")
        required = set(manifest.minimum_host_capabilities.get("features", []))
        if not PROGRESSIVE_REPLACEMENT_CAPABILITIES <= required:
            raise UnicodeDirectNeuralCoreError("progressive replacement capabilities are incomplete")

    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> ProgressiveReplacementCore:
        architecture = package.manifest.architecture
        allowed = {
            "name", "format", "model", "tokenizer", "tokenizer_sha256",
            "external_input_output", "private_representation", "action_validity",
            "source_transformer_blocks",
        }
        if (
            set(architecture) != allowed
            or architecture["external_input_output"] != "UTF-8 bytes"
            or architecture["private_representation"]
            != "full_residual_width_progressive_bottleneck_replacement_decoder"
            or architecture["action_validity"]
            != "external_sequence_decode_strict_utf8_pointer_prohibited"
            or architecture["source_transformer_blocks"] != 0
        ):
            raise UnicodeDirectNeuralCoreError("progressive replacement metadata is incomplete")
        tokenizer = DecoderAwareExternalTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]:
            raise UnicodeDirectNeuralCoreError("progressive replacement tokenizer hash mismatch")
        model = ProgressiveReplacementCore(**architecture["model"]).bind_tokenizer(tokenizer)
        model.load_state_dict(package.tensors, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
