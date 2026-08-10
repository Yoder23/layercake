"""Signed source-aligned progressive-replacement English-core host ABI v9."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.source_aligned_progressive_replacement_core import SourceAlignedProgressiveReplacementCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.unicode_direct_neural_core import (
    DIRECT_NEURAL_CORE_COMPOSITION,
    DIRECT_NEURAL_CORE_ROLE,
    UnicodeDirectNeuralCoreError,
    UnicodeSafeDirectNeuralCoreHost,
)


SOURCE_ALIGNED_PROGRESSIVE_ABI_VERSION = "lc-direct-neural-core/9"
SOURCE_ALIGNED_PROGRESSIVE_ABI_SHA256 = "9d38094ff6be0fd9f37df485a023eade744128adf2d87d2fe3bb1453cd1eaf5c"
SOURCE_ALIGNED_PROGRESSIVE_FORMAT = "layercake-source-aligned-progressive-replacement-core/1"
SOURCE_ALIGNED_PROGRESSIVE_CAPABILITIES = frozenset(
    {
        "byte_input", "safe_tensors", "strict_utf8_boundary",
        "decoder_aware_external_tokenizer", "decoder_only_causal_execution",
        "persistent_rotary_kv_state", "full_residual_width_replacement_execution",
        "source_aligned_prompt_response_boundary", "zero_source_transformer_blocks",
    }
)


def source_aligned_progressive_manifest_architecture(
    model: SourceAlignedProgressiveReplacementCore,
    tokenizer: DecoderAwareExternalTokenizer,
) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("source-aligned model and tokenizer sizes differ")
    return {
        "name": "source_aligned_progressive_replacement_core",
        "format": SOURCE_ALIGNED_PROGRESSIVE_FORMAT,
        "model": model.canonical_config(),
        "tokenizer": tokenizer.canonical_dict(),
        "tokenizer_sha256": tokenizer.hash(),
        "external_input_output": "UTF-8 bytes",
        "private_representation": "source_aligned_full_residual_width_progressive_bottleneck_replacement_decoder",
        "causal_boundary": "first_response_from_final_prompt_no_injected_bos",
        "action_validity": "external_sequence_decode_strict_utf8_pointer_prohibited",
        "source_transformer_blocks": 0,
    }


class SourceAlignedProgressiveReplacementCoreHost(UnicodeSafeDirectNeuralCoreHost):
    def __init__(self, registry_root: str | Path, *, trust_store: Mapping[str, bytes | str | Path], device: str | torch.device = "cpu") -> None:
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=SOURCE_ALIGNED_PROGRESSIVE_ABI_VERSION,
                abi_hash=SOURCE_ALIGNED_PROGRESSIVE_ABI_SHA256,
                precisions=("fp32", "fp16", "bf16"),
                backends=("pytorch", "cuda"),
                capabilities=SOURCE_ALIGNED_PROGRESSIVE_CAPABILITIES,
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
            not package.signed or manifest.cake_type != "portable_decoder"
            or manifest.abi_version != SOURCE_ALIGNED_PROGRESSIVE_ABI_VERSION
            or manifest.abi_hash != SOURCE_ALIGNED_PROGRESSIVE_ABI_SHA256
            or manifest.domains != (DIRECT_NEURAL_CORE_ROLE,) or manifest.dependencies
        ):
            raise UnicodeDirectNeuralCoreError("source-aligned progressive core identity mismatch")
        if manifest.input_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "validity": "strict_utf8"}:
            raise UnicodeDirectNeuralCoreError("source-aligned progressive input contract mismatch")
        if manifest.output_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "composition": DIRECT_NEURAL_CORE_COMPOSITION, "validity": "strict_utf8"}:
            raise UnicodeDirectNeuralCoreError("source-aligned progressive output contract mismatch")
        architecture = manifest.architecture
        if (
            architecture.get("name") != "source_aligned_progressive_replacement_core"
            or architecture.get("format") != SOURCE_ALIGNED_PROGRESSIVE_FORMAT
            or architecture.get("causal_boundary") != "first_response_from_final_prompt_no_injected_bos"
            or architecture.get("source_transformer_blocks") != 0
        ):
            raise UnicodeDirectNeuralCoreError("source-aligned progressive architecture mismatch")
        if not SOURCE_ALIGNED_PROGRESSIVE_CAPABILITIES <= set(manifest.minimum_host_capabilities.get("features", [])):
            raise UnicodeDirectNeuralCoreError("source-aligned progressive capabilities incomplete")

    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> SourceAlignedProgressiveReplacementCore:
        architecture = package.manifest.architecture
        allowed = {
            "name", "format", "model", "tokenizer", "tokenizer_sha256",
            "external_input_output", "private_representation", "causal_boundary",
            "action_validity", "source_transformer_blocks",
        }
        if set(architecture) != allowed or architecture["private_representation"] != "source_aligned_full_residual_width_progressive_bottleneck_replacement_decoder":
            raise UnicodeDirectNeuralCoreError("source-aligned progressive metadata incomplete")
        tokenizer = DecoderAwareExternalTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]:
            raise UnicodeDirectNeuralCoreError("source-aligned progressive tokenizer hash mismatch")
        model = SourceAlignedProgressiveReplacementCore(**architecture["model"]).bind_tokenizer(tokenizer)
        model.load_state_dict(package.tensors, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
