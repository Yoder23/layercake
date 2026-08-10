"""Signed routed sparse rank-768 progressive host ABI v15."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.unicode_direct_neural_core import (
    DIRECT_NEURAL_CORE_COMPOSITION,
    DIRECT_NEURAL_CORE_ROLE,
    UnicodeDirectNeuralCoreError,
    UnicodeSafeDirectNeuralCoreHost,
)


ROUTED_SPARSE_RANK768_ABI_VERSION = "lc-direct-neural-core/15"
ROUTED_SPARSE_RANK768_ABI_SHA256 = "6fe3ff87c1bcaece3aa7a87bc0f2b92127ef9a10e2f778c838ff039c95265a3f"
ROUTED_SPARSE_RANK768_FORMAT = "layercake-routed-sparse-rank768-progressive-core/1"
ROUTED_SPARSE_RANK768_CAPABILITIES = frozenset(
    {
        "byte_input",
        "safe_tensors",
        "strict_utf8_boundary",
        "decoder_aware_external_tokenizer",
        "persistent_rotary_kv_state",
        "persistent_request_route",
        "source_aligned_prompt_response_boundary",
        "dual_compact_attention_execution",
        "rank768_linear_plus_sparse_residual_execution",
        "hard_top1_request_routing",
        "zero_source_transformer_blocks",
    }
)


def routed_sparse_rank768_manifest_architecture(
    model: RoutedSparseRank768ProgressiveCore,
    tokenizer: DecoderAwareExternalTokenizer,
) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer sizes differ")
    return {
        "name": "routed_sparse_rank768_progressive_core",
        "format": ROUTED_SPARSE_RANK768_FORMAT,
        "model": model.canonical_config(),
        "tokenizer": tokenizer.canonical_dict(),
        "tokenizer_sha256": tokenizer.hash(),
        "external_input_output": "UTF-8 bytes",
        "private_representation": "source_aligned_dual_compact_attention_rank768_linear_plus_sparse_three_route_residual",
        "causal_boundary": "first_response_from_final_prompt_no_injected_bos",
        "routing": "source_prompt_only_hard_top1_once_per_request_persistent",
        "action_validity": "external_sequence_decode_strict_utf8_pointer_prohibited",
        "source_transformer_blocks": 0,
    }


class RoutedSparseRank768ProgressiveCoreHost(UnicodeSafeDirectNeuralCoreHost):
    def __init__(
        self,
        registry_root: str | Path,
        *,
        trust_store: Mapping[str, bytes | str | Path],
        device: str | torch.device = "cpu",
    ):
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=ROUTED_SPARSE_RANK768_ABI_VERSION,
                abi_hash=ROUTED_SPARSE_RANK768_ABI_SHA256,
                precisions=("fp32", "fp16", "bf16"),
                backends=("pytorch", "cuda"),
                capabilities=ROUTED_SPARSE_RANK768_CAPABILITIES,
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
        architecture = manifest.architecture
        if (
            not package.signed
            or manifest.cake_type != "portable_decoder"
            or manifest.abi_version != ROUTED_SPARSE_RANK768_ABI_VERSION
            or manifest.abi_hash != ROUTED_SPARSE_RANK768_ABI_SHA256
            or manifest.domains != (DIRECT_NEURAL_CORE_ROLE,)
            or manifest.dependencies
        ):
            raise UnicodeDirectNeuralCoreError("routed sparse rank768 identity mismatch")
        if manifest.input_contract != {
            "external": "UTF-8 bytes",
            "role": DIRECT_NEURAL_CORE_ROLE,
            "validity": "strict_utf8",
        } or manifest.output_contract != {
            "external": "UTF-8 bytes",
            "role": DIRECT_NEURAL_CORE_ROLE,
            "composition": DIRECT_NEURAL_CORE_COMPOSITION,
            "validity": "strict_utf8",
        }:
            raise UnicodeDirectNeuralCoreError("routed sparse rank768 contract mismatch")
        if (
            architecture.get("name") != "routed_sparse_rank768_progressive_core"
            or architecture.get("format") != ROUTED_SPARSE_RANK768_FORMAT
            or architecture.get("source_transformer_blocks") != 0
            or architecture.get("routing")
            != "source_prompt_only_hard_top1_once_per_request_persistent"
            or not ROUTED_SPARSE_RANK768_CAPABILITIES
            <= set(manifest.minimum_host_capabilities.get("features", []))
        ):
            raise UnicodeDirectNeuralCoreError("routed sparse rank768 architecture mismatch")

    @staticmethod
    def _load_module(
        package: CakePackage, device: torch.device
    ) -> RoutedSparseRank768ProgressiveCore:
        architecture = package.manifest.architecture
        allowed = {
            "name",
            "format",
            "model",
            "tokenizer",
            "tokenizer_sha256",
            "external_input_output",
            "private_representation",
            "causal_boundary",
            "routing",
            "action_validity",
            "source_transformer_blocks",
        }
        if (
            set(architecture) != allowed
            or architecture["private_representation"]
            != "source_aligned_dual_compact_attention_rank768_linear_plus_sparse_three_route_residual"
        ):
            raise UnicodeDirectNeuralCoreError("routed sparse rank768 metadata incomplete")
        tokenizer = DecoderAwareExternalTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]:
            raise UnicodeDirectNeuralCoreError("tokenizer mismatch")
        model = RoutedSparseRank768ProgressiveCore(**architecture["model"]).bind_tokenizer(tokenizer)
        model.load_state_dict(package.tensors, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
