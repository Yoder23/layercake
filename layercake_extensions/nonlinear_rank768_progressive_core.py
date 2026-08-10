"""Signed nonlinear rank-768 progressive host ABI v14."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import torch
from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.nonlinear_rank768_progressive_core import NonlinearRank768ProgressiveCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.unicode_direct_neural_core import DIRECT_NEURAL_CORE_COMPOSITION, DIRECT_NEURAL_CORE_ROLE, UnicodeDirectNeuralCoreError, UnicodeSafeDirectNeuralCoreHost

NONLINEAR_RANK768_ABI_VERSION = "lc-direct-neural-core/14"
NONLINEAR_RANK768_ABI_SHA256 = "d5354011e00ccf22b8ec1373155dfd4e61a07dbfa411994248baa93ef482ac7a"
NONLINEAR_RANK768_FORMAT = "layercake-nonlinear-rank768-progressive-core/1"
NONLINEAR_RANK768_CAPABILITIES = frozenset({"byte_input", "safe_tensors", "strict_utf8_boundary", "decoder_aware_external_tokenizer", "persistent_rotary_kv_state", "source_aligned_prompt_response_boundary", "gated_nonlinear_rank768_residual_execution", "zero_source_transformer_blocks"})


def nonlinear_rank768_manifest_architecture(model: NonlinearRank768ProgressiveCore, tokenizer: DecoderAwareExternalTokenizer) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size: raise ValueError("model and tokenizer sizes differ")
    return {"name": "nonlinear_rank768_progressive_core", "format": NONLINEAR_RANK768_FORMAT, "model": model.canonical_config(), "tokenizer": tokenizer.canonical_dict(), "tokenizer_sha256": tokenizer.hash(), "external_input_output": "UTF-8 bytes", "private_representation": "source_aligned_compact_attention_gated_nonlinear_rank768_residual", "causal_boundary": "first_response_from_final_prompt_no_injected_bos", "action_validity": "external_sequence_decode_strict_utf8_pointer_prohibited", "source_transformer_blocks": 0}


class NonlinearRank768ProgressiveCoreHost(UnicodeSafeDirectNeuralCoreHost):
    def __init__(self, registry_root: str | Path, *, trust_store: Mapping[str, bytes | str | Path], device: str | torch.device = "cpu"):
        self.registry = CakeRegistry(registry_root); self.installer = CakeInstaller(self.registry, HostCapabilities(abi_version=NONLINEAR_RANK768_ABI_VERSION, abi_hash=NONLINEAR_RANK768_ABI_SHA256, precisions=("fp32", "fp16", "bf16"), backends=("pytorch", "cuda"), capabilities=NONLINEAR_RANK768_CAPABILITIES), trust_store=trust_store, strict_signatures=True); self.device = torch.device(device); self.module = None; self.active_cake_id = None; self.active_archive_hash = None; self.active_payload_hash = None; self.receiver_training_steps = 0; self.receiver_calibration_runs = 0
    @staticmethod
    def _validate_role(package: CakePackage) -> None:
        m = package.manifest; a = m.architecture
        if not package.signed or m.cake_type != "portable_decoder" or m.abi_version != NONLINEAR_RANK768_ABI_VERSION or m.abi_hash != NONLINEAR_RANK768_ABI_SHA256 or m.domains != (DIRECT_NEURAL_CORE_ROLE,) or m.dependencies: raise UnicodeDirectNeuralCoreError("nonlinear rank768 identity mismatch")
        if m.input_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "validity": "strict_utf8"} or m.output_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "composition": DIRECT_NEURAL_CORE_COMPOSITION, "validity": "strict_utf8"}: raise UnicodeDirectNeuralCoreError("nonlinear rank768 contract mismatch")
        if a.get("name") != "nonlinear_rank768_progressive_core" or a.get("format") != NONLINEAR_RANK768_FORMAT or a.get("source_transformer_blocks") != 0 or not NONLINEAR_RANK768_CAPABILITIES <= set(m.minimum_host_capabilities.get("features", [])): raise UnicodeDirectNeuralCoreError("nonlinear rank768 architecture mismatch")
    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> NonlinearRank768ProgressiveCore:
        a = package.manifest.architecture; allowed = {"name", "format", "model", "tokenizer", "tokenizer_sha256", "external_input_output", "private_representation", "causal_boundary", "action_validity", "source_transformer_blocks"}
        if set(a) != allowed or a["private_representation"] != "source_aligned_compact_attention_gated_nonlinear_rank768_residual": raise UnicodeDirectNeuralCoreError("nonlinear rank768 metadata incomplete")
        t = DecoderAwareExternalTokenizer.from_document(a["tokenizer"])
        if t.hash() != a["tokenizer_sha256"]: raise UnicodeDirectNeuralCoreError("tokenizer mismatch")
        model = NonlinearRank768ProgressiveCore(**a["model"]).bind_tokenizer(t); model.load_state_dict(package.tensors, strict=True); model.to(device).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        return model
