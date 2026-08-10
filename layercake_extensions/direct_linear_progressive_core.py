"""Signed direct-linear progressive host ABI v12."""
from __future__ import annotations
from pathlib import Path
from typing import Any,Mapping
import torch
from layercake.cake.installer import CakeInstaller,HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.direct_linear_progressive_core import DirectLinearProgressiveCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.unicode_direct_neural_core import DIRECT_NEURAL_CORE_COMPOSITION,DIRECT_NEURAL_CORE_ROLE,UnicodeDirectNeuralCoreError,UnicodeSafeDirectNeuralCoreHost

DIRECT_LINEAR_ABI_VERSION="lc-direct-neural-core/12"
DIRECT_LINEAR_ABI_SHA256="e392333b89bce8a81379fe5e44fc2220564006522f40dd7434a3e02cb4760efc"
DIRECT_LINEAR_FORMAT="layercake-direct-linear-progressive-core/1"
DIRECT_LINEAR_CAPABILITIES=frozenset({"byte_input","safe_tensors","strict_utf8_boundary","decoder_aware_external_tokenizer","persistent_rotary_kv_state","source_aligned_prompt_response_boundary","direct_linear_rank192_mlp_execution","zero_source_transformer_blocks"})

def direct_linear_manifest_architecture(model:DirectLinearProgressiveCore,tokenizer:DecoderAwareExternalTokenizer)->dict[str,Any]:
    if model.fixed_vocab_size!=tokenizer.vocab_size: raise ValueError("model and tokenizer sizes differ")
    return {"name":"direct_linear_progressive_core","format":DIRECT_LINEAR_FORMAT,"model":model.canonical_config(),"tokenizer":tokenizer.canonical_dict(),"tokenizer_sha256":tokenizer.hash(),"external_input_output":"UTF-8 bytes","private_representation":"source_aligned_compact_attention_direct_linear_rank192_mlp_decoder","causal_boundary":"first_response_from_final_prompt_no_injected_bos","action_validity":"external_sequence_decode_strict_utf8_pointer_prohibited","source_transformer_blocks":0}

class DirectLinearProgressiveCoreHost(UnicodeSafeDirectNeuralCoreHost):
    def __init__(self,registry_root:str|Path,*,trust_store:Mapping[str,bytes|str|Path],device:str|torch.device="cpu"):
        self.registry=CakeRegistry(registry_root); self.installer=CakeInstaller(self.registry,HostCapabilities(abi_version=DIRECT_LINEAR_ABI_VERSION,abi_hash=DIRECT_LINEAR_ABI_SHA256,precisions=("fp32","fp16","bf16"),backends=("pytorch","cuda"),capabilities=DIRECT_LINEAR_CAPABILITIES),trust_store=trust_store,strict_signatures=True); self.device=torch.device(device); self.module=None; self.active_cake_id=None; self.active_archive_hash=None; self.active_payload_hash=None; self.receiver_training_steps=0; self.receiver_calibration_runs=0
    @staticmethod
    def _validate_role(package:CakePackage)->None:
        m=package.manifest; a=m.architecture
        if not package.signed or m.cake_type!="portable_decoder" or m.abi_version!=DIRECT_LINEAR_ABI_VERSION or m.abi_hash!=DIRECT_LINEAR_ABI_SHA256 or m.domains!=(DIRECT_NEURAL_CORE_ROLE,) or m.dependencies: raise UnicodeDirectNeuralCoreError("direct-linear identity mismatch")
        if m.input_contract!={"external":"UTF-8 bytes","role":DIRECT_NEURAL_CORE_ROLE,"validity":"strict_utf8"} or m.output_contract!={"external":"UTF-8 bytes","role":DIRECT_NEURAL_CORE_ROLE,"composition":DIRECT_NEURAL_CORE_COMPOSITION,"validity":"strict_utf8"}: raise UnicodeDirectNeuralCoreError("direct-linear contract mismatch")
        if a.get("name")!="direct_linear_progressive_core" or a.get("format")!=DIRECT_LINEAR_FORMAT or a.get("causal_boundary")!="first_response_from_final_prompt_no_injected_bos" or a.get("source_transformer_blocks")!=0 or not DIRECT_LINEAR_CAPABILITIES<=set(m.minimum_host_capabilities.get("features",[])): raise UnicodeDirectNeuralCoreError("direct-linear architecture mismatch")
    @staticmethod
    def _load_module(package:CakePackage,device:torch.device)->DirectLinearProgressiveCore:
        a=package.manifest.architecture; allowed={"name","format","model","tokenizer","tokenizer_sha256","external_input_output","private_representation","causal_boundary","action_validity","source_transformer_blocks"}
        if set(a)!=allowed or a["private_representation"]!="source_aligned_compact_attention_direct_linear_rank192_mlp_decoder": raise UnicodeDirectNeuralCoreError("direct-linear metadata incomplete")
        t=DecoderAwareExternalTokenizer.from_document(a["tokenizer"])
        if t.hash()!=a["tokenizer_sha256"]: raise UnicodeDirectNeuralCoreError("tokenizer mismatch")
        model=DirectLinearProgressiveCore(**a["model"]).bind_tokenizer(t); model.load_state_dict(package.tensors,strict=True); model.to(device).eval()
        for p in model.parameters(): p.requires_grad_(False)
        return model
