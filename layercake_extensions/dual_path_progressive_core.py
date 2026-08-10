"""Signed source-aligned dual-path progressive host ABI v10."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import torch
from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.dual_path_progressive_core import DualPathProgressiveCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.unicode_direct_neural_core import DIRECT_NEURAL_CORE_COMPOSITION, DIRECT_NEURAL_CORE_ROLE, UnicodeDirectNeuralCoreError, UnicodeSafeDirectNeuralCoreHost

DUAL_PATH_PROGRESSIVE_ABI_VERSION = "lc-direct-neural-core/10"
DUAL_PATH_PROGRESSIVE_ABI_SHA256 = "951bdf91c6840e126ab598564c554f5e6e0bce9b07eafe1972c06435c542a1a0"
DUAL_PATH_PROGRESSIVE_FORMAT = "layercake-dual-path-progressive-core/1"
DUAL_PATH_PROGRESSIVE_CAPABILITIES = frozenset({"byte_input", "safe_tensors", "strict_utf8_boundary", "decoder_aware_external_tokenizer", "persistent_rotary_kv_state", "source_aligned_prompt_response_boundary", "dual_attention_mlp_replacement_execution", "zero_source_transformer_blocks"})

def dual_path_progressive_manifest_architecture(model: DualPathProgressiveCore, tokenizer: DecoderAwareExternalTokenizer) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size: raise ValueError("dual-path model and tokenizer sizes differ")
    return {"name": "dual_path_progressive_core", "format": DUAL_PATH_PROGRESSIVE_FORMAT, "model": model.canonical_config(), "tokenizer": tokenizer.canonical_dict(), "tokenizer_sha256": tokenizer.hash(), "external_input_output": "UTF-8 bytes", "private_representation": "source_aligned_dual_attention_mlp_progressive_replacement_decoder", "causal_boundary": "first_response_from_final_prompt_no_injected_bos", "action_validity": "external_sequence_decode_strict_utf8_pointer_prohibited", "source_transformer_blocks": 0}

class DualPathProgressiveCoreHost(UnicodeSafeDirectNeuralCoreHost):
    def __init__(self, registry_root: str | Path, *, trust_store: Mapping[str, bytes | str | Path], device: str | torch.device = "cpu") -> None:
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(self.registry, HostCapabilities(abi_version=DUAL_PATH_PROGRESSIVE_ABI_VERSION, abi_hash=DUAL_PATH_PROGRESSIVE_ABI_SHA256, precisions=("fp32", "fp16", "bf16"), backends=("pytorch", "cuda"), capabilities=DUAL_PATH_PROGRESSIVE_CAPABILITIES), trust_store=trust_store, strict_signatures=True)
        self.device = torch.device(device); self.module = None; self.active_cake_id = None; self.active_archive_hash = None; self.active_payload_hash = None; self.receiver_training_steps = 0; self.receiver_calibration_runs = 0

    @staticmethod
    def _validate_role(package: CakePackage) -> None:
        manifest = package.manifest
        if not package.signed or manifest.cake_type != "portable_decoder" or manifest.abi_version != DUAL_PATH_PROGRESSIVE_ABI_VERSION or manifest.abi_hash != DUAL_PATH_PROGRESSIVE_ABI_SHA256 or manifest.domains != (DIRECT_NEURAL_CORE_ROLE,) or manifest.dependencies: raise UnicodeDirectNeuralCoreError("dual-path progressive identity mismatch")
        if manifest.input_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "validity": "strict_utf8"}: raise UnicodeDirectNeuralCoreError("dual-path input mismatch")
        if manifest.output_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "composition": DIRECT_NEURAL_CORE_COMPOSITION, "validity": "strict_utf8"}: raise UnicodeDirectNeuralCoreError("dual-path output mismatch")
        architecture = manifest.architecture
        if architecture.get("name") != "dual_path_progressive_core" or architecture.get("format") != DUAL_PATH_PROGRESSIVE_FORMAT or architecture.get("causal_boundary") != "first_response_from_final_prompt_no_injected_bos" or architecture.get("source_transformer_blocks") != 0: raise UnicodeDirectNeuralCoreError("dual-path architecture mismatch")
        if not DUAL_PATH_PROGRESSIVE_CAPABILITIES <= set(manifest.minimum_host_capabilities.get("features", [])): raise UnicodeDirectNeuralCoreError("dual-path capabilities incomplete")

    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> DualPathProgressiveCore:
        architecture = package.manifest.architecture
        allowed = {"name", "format", "model", "tokenizer", "tokenizer_sha256", "external_input_output", "private_representation", "causal_boundary", "action_validity", "source_transformer_blocks"}
        if set(architecture) != allowed or architecture["private_representation"] != "source_aligned_dual_attention_mlp_progressive_replacement_decoder": raise UnicodeDirectNeuralCoreError("dual-path metadata incomplete")
        tokenizer = DecoderAwareExternalTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]: raise UnicodeDirectNeuralCoreError("dual-path tokenizer mismatch")
        model = DualPathProgressiveCore(**architecture["model"]).bind_tokenizer(tokenizer); model.load_state_dict(package.tensors, strict=True); model.to(device).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        return model
