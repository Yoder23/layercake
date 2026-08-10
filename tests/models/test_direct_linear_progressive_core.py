from pathlib import Path
import pytest,torch
from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package,tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.direct_linear_progressive_core import DirectLinearProgressiveCore
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import EOS_ID
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.direct_linear_progressive_core import *
from tests.models.test_decoder_direct_neural_core import _doc

def _model(t): return DirectLinearProgressiveCore(fixed_vocab_size=t.vocab_size,full_width=24,bottleneck_width=8,attention_heads=2,replacement_layers=2,intermediate_size=16,maximum_source_actions=16,maximum_target_actions=8,maximum_sequence_actions=24).bind_tokenizer(t)
def _package(tmp:Path):
    torch.manual_seed(12); t=DecoderAwareExternalTokenizer(_doc()); m=_model(t).eval()
    with torch.no_grad():
        for p in m.parameters(): p.zero_()
        m.token_embedding.weight.fill_(1); m.final_norm.weight.fill_(1); m.lm_head.weight[EOS_ID].fill_(1)
    private,public,key_id=generate_keypair(); manifest=CakeManifest(schema_version="1",cake_id="direct-linear",name="Direct Linear",description="construct",version="12.0.0",publisher={"id":"construct","name":"Construct","key_id":key_id},abi_version=DIRECT_LINEAR_ABI_VERSION,abi_hash=DIRECT_LINEAR_ABI_SHA256,cake_type="portable_decoder",input_contract={"external":"UTF-8 bytes","role":"english-core","validity":"strict_utf8"},output_contract={"external":"UTF-8 bytes","role":"english-core","composition":"direct_core_only_no_router","validity":"strict_utf8"},architecture=direct_linear_manifest_architecture(m,t),supported_precisions=("fp32",),supported_backends=("pytorch","cuda"),minimum_host_capabilities={"features":sorted(DIRECT_LINEAR_CAPABILITIES)},tensor_payload_hash="",tensor_shapes=tensor_specs(m.state_dict()),package_hash="",training_data_provenance={"dataset":"construct-only"},evaluation_evidence={"status":"CONSTRUCT_ONLY"},license="Apache-2.0",dependencies=(),parent_version=None,signature={"algorithm":"ed25519","key_id":key_id},domains=("english-core",),permissions=("local-inference",))
    p=build_package(tmp/"direct.cake",manifest,m.state_dict(),private_key=private); return p,public,key_id,state_dict_hash(m.state_dict()),t

def test_count_and_incremental():
    t=DecoderAwareExternalTokenizer(_doc()); m=_model(t).eval(); assert sum(p.numel() for p in m.parameters())==m.parameter_count_for_config(fixed_vocab_size=t.vocab_size,full_width=24,bottleneck_width=8,replacement_layers=2,intermediate_size=16); assert m.parameter_count_for_config(fixed_vocab_size=32015,full_width=3072,bottleneck_width=192,replacement_layers=32,intermediate_size=768)==277220352
    ids,lex=t.encode_source("hello world"); s=m.prefill_ids(ids,lex); assert torch.allclose(s.next_logits,m(torch.tensor([ids]))[:,-1],atol=1e-5,rtol=1e-5); s.next_logits.zero_(); s.next_logits[0,4]=1; a,s=m.decode_step(s); assert torch.allclose(s.next_logits,m(torch.tensor([ids+[a]]))[:,-1],atol=1e-5,rtol=1e-5)
def test_lifecycle(tmp_path):
    p,pub,k,h,_=_package(tmp_path); host=DirectLinearProgressiveCoreHost(tmp_path/"r",trust_store={k:pub}); active=host.activate(p); assert active["state_dict_hash"]==h; assert host.generate("hello world")==b""; assert active["receiver_training_steps"]==0
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda(tmp_path):
    p,pub,k,h,_=_package(tmp_path); host=DirectLinearProgressiveCoreHost(tmp_path/"r",trust_store={k:pub},device="cuda"); assert host.activate(p)["state_dict_hash"]==h; assert host.generate("hello world")==b""
