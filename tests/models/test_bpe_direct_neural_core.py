from pathlib import Path
import pytest, torch
from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import PortableTokenPlan
from layercake_extensions.bpe_direct_neural_core import BPE_DIRECT_NEURAL_CORE_ABI_SHA256,BPE_DIRECT_NEURAL_CORE_ABI_VERSION,BpeDirectNeuralCoreHost,Utf8ConcatenativeBpeTokenizer,bpe_token_plan_manifest_architecture

def _doc():
 vocab={"[UNK]":0," ":1,"d":2,"e":3,"h":4,"l":5,"o":6,"r":7,"w":8,"he":9,"hel":10,"hell":11,"hello":12," w":13," wo":14," wor":15," worl":16," world":17}
 merges=[["h","e"],["he","l"],["hel","l"],["hell","o"],[" ","w"],[" w","o"],[" wo","r"],[" wor","l"],[" worl","d"]]
 return {"version":"1.0","truncation":None,"padding":None,"added_tokens":[],"normalizer":None,"pre_tokenizer":None,"post_processor":None,"decoder":None,"model":{"type":"BPE","dropout":None,"unk_token":"[UNK]","continuing_subword_prefix":None,"end_of_word_suffix":None,"fuse_unk":False,"byte_fallback":False,"ignore_merges":False,"vocab":vocab,"merges":merges}}

def _package(tmp_path:Path):
 torch.manual_seed(34003)
 tok=Utf8ConcatenativeBpeTokenizer(_doc());model=PortableTokenPlan(fixed_vocab_size=tok.vocab_size,model_width=24,attention_heads=4,encoder_layers=1,decoder_layers=1,feedforward_width=48,pointer_width=12,dropout=0.0,maximum_source_lexemes=16,maximum_target_actions=16).eval().bind_tokenizer(tok)
 with torch.no_grad():model.fixed_output.weight.zero_();model.fixed_output.bias.zero_();model.fixed_output.bias[2]=10;model.pointer_gate.weight.zero_();model.pointer_gate.bias.fill_(-10)
 private,public,key_id=generate_keypair();m=CakeManifest(schema_version="1",cake_id="bpe-core-construct",name="BPE Core",description="construct",version="3.0.0",publisher={"id":"construct","name":"Construct","key_id":key_id},abi_version=BPE_DIRECT_NEURAL_CORE_ABI_VERSION,abi_hash=BPE_DIRECT_NEURAL_CORE_ABI_SHA256,cake_type="portable_decoder",input_contract={"external":"UTF-8 bytes","role":"english-core","validity":"strict_utf8"},output_contract={"external":"UTF-8 bytes","role":"english-core","composition":"direct_core_only_no_router","validity":"strict_utf8"},architecture=bpe_token_plan_manifest_architecture(model,tok),supported_precisions=("fp32",),supported_backends=("pytorch","cuda"),minimum_host_capabilities={"features":["byte_input","safe_tensors","persistent_incremental_state","unicode_atomic_actions","strict_utf8_boundary","utf8_concatenative_bpe"]},tensor_payload_hash="",tensor_shapes=tensor_specs(model.state_dict()),package_hash="",training_data_provenance={"dataset":"construct-only","external_teacher":False},evaluation_evidence={"status":"CONSTRUCT_ONLY"},license="Apache-2.0",dependencies=(),parent_version=None,signature={"algorithm":"ed25519","key_id":key_id},domains=("english-core",),permissions=("local-inference",))
 path=build_package(tmp_path/"bpe-core.cake",m,model.state_dict(),private_key=private);return path,public,key_id,state_dict_hash(model.state_dict()),tok

def test_bpe_split_is_atomic_exact_and_identity_bound():
 tok=Utf8ConcatenativeBpeTokenizer(_doc());assert tok.split("hello world")==[b"hello",b" world"];assert b"".join(tok.split("hello world"))==b"hello world";assert Utf8ConcatenativeBpeTokenizer.from_document(tok.canonical_dict()).hash()==tok.hash()

def test_v3_signed_lifecycle_and_zero_learning(tmp_path):
 package,public,key_id,state,_=_package(tmp_path);archive=package.read_bytes();host=BpeDirectNeuralCoreHost(tmp_path/"registry",trust_store={key_id:public});first=host.activate(package);assert first["state_dict_hash"]==state;assert first["receiver_training_steps"]==first["receiver_calibration_runs"]==0;assert host.generate("hello world")==b"";host.remove();second=host.activate(package);assert package.read_bytes()==archive;assert second["archive_hash"]==first["archive_hash"]

@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_same_v3_package_executes_cuda(tmp_path):
 package,public,key_id,state,_=_package(tmp_path);host=BpeDirectNeuralCoreHost(tmp_path/"registry",trust_store={key_id:public},device="cuda");assert host.activate(package)["state_dict_hash"]==state;assert host.generate("hello world")==b""
