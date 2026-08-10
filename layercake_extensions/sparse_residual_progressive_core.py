"""Signed sparse-residual progressive host ABI v13."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import torch
from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.sparse_residual_progressive_core import SparseResidualProgressiveCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.unicode_direct_neural_core import DIRECT_NEURAL_CORE_COMPOSITION, DIRECT_NEURAL_CORE_ROLE, UnicodeDirectNeuralCoreError, UnicodeSafeDirectNeuralCoreHost

SPARSE_RESIDUAL_ABI_VERSION = "lc-direct-neural-core/13"
SPARSE_RESIDUAL_ABI_SHA256 = "684add397ac89442d6d29d2598c095074f21c4dc3ebe7f01c357e11290515540"
SPARSE_RESIDUAL_FORMAT = "layercake-sparse-residual-progressive-core/1"
SPARSE_RESIDUAL_CAPABILITIES = frozenset({
    "byte_input", "safe_tensors", "strict_utf8_boundary",
    "decoder_aware_external_tokenizer", "persistent_rotary_kv_state",
    "source_aligned_prompt_response_boundary", "hard_top1_sparse_residual_execution",
    "zero_source_transformer_blocks",
})


def sparse_residual_manifest_architecture(model: SparseResidualProgressiveCore, tokenizer: DecoderAwareExternalTokenizer) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer sizes differ")
    return {
        "name": "sparse_residual_progressive_core", "format": SPARSE_RESIDUAL_FORMAT,
        "model": model.canonical_config(), "tokenizer": tokenizer.canonical_dict(),
        "tokenizer_sha256": tokenizer.hash(), "external_input_output": "UTF-8 bytes",
        "private_representation": "source_aligned_compact_attention_hard_top1_rank192_residual_experts",
        "causal_boundary": "first_response_from_final_prompt_no_injected_bos",
        "action_validity": "external_sequence_decode_strict_utf8_pointer_prohibited",
        "source_transformer_blocks": 0,
    }


class SparseResidualProgressiveCoreHost(UnicodeSafeDirectNeuralCoreHost):
    def __init__(self, registry_root: str | Path, *, trust_store: Mapping[str, bytes | str | Path], device: str | torch.device = "cpu"):
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=SPARSE_RESIDUAL_ABI_VERSION, abi_hash=SPARSE_RESIDUAL_ABI_SHA256,
                precisions=("fp32", "fp16", "bf16"), backends=("pytorch", "cuda"),
                capabilities=SPARSE_RESIDUAL_CAPABILITIES,
            ), trust_store=trust_store, strict_signatures=True,
        )
        self.device = torch.device(device); self.module = None; self.active_cake_id = None
        self.active_archive_hash = None; self.active_payload_hash = None
        self.receiver_training_steps = 0; self.receiver_calibration_runs = 0

    @staticmethod
    def _validate_role(package: CakePackage) -> None:
        manifest = package.manifest; architecture = manifest.architecture
        if (
            not package.signed or manifest.cake_type != "portable_decoder"
            or manifest.abi_version != SPARSE_RESIDUAL_ABI_VERSION
            or manifest.abi_hash != SPARSE_RESIDUAL_ABI_SHA256
            or manifest.domains != (DIRECT_NEURAL_CORE_ROLE,) or manifest.dependencies
        ):
            raise UnicodeDirectNeuralCoreError("sparse-residual identity mismatch")
        if manifest.input_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "validity": "strict_utf8"} or manifest.output_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "composition": DIRECT_NEURAL_CORE_COMPOSITION, "validity": "strict_utf8"}:
            raise UnicodeDirectNeuralCoreError("sparse-residual contract mismatch")
        if architecture.get("name") != "sparse_residual_progressive_core" or architecture.get("format") != SPARSE_RESIDUAL_FORMAT or architecture.get("source_transformer_blocks") != 0 or not SPARSE_RESIDUAL_CAPABILITIES <= set(manifest.minimum_host_capabilities.get("features", [])):
            raise UnicodeDirectNeuralCoreError("sparse-residual architecture mismatch")

    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> SparseResidualProgressiveCore:
        architecture = package.manifest.architecture
        allowed = {"name", "format", "model", "tokenizer", "tokenizer_sha256", "external_input_output", "private_representation", "causal_boundary", "action_validity", "source_transformer_blocks"}
        if set(architecture) != allowed or architecture["private_representation"] != "source_aligned_compact_attention_hard_top1_rank192_residual_experts":
            raise UnicodeDirectNeuralCoreError("sparse-residual metadata incomplete")
        tokenizer = DecoderAwareExternalTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]:
            raise UnicodeDirectNeuralCoreError("tokenizer mismatch")
        model = SparseResidualProgressiveCore(**architecture["model"]).bind_tokenizer(tokenizer)
        model.load_state_dict(package.tensors, strict=True); model.to(device).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        return model
