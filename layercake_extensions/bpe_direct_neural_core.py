"""Post-release UTF-8-concatenative BPE direct neural core host ABI v3."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Mapping
import torch
from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.portable_token_plan import LosslessLexemePointerTokenizer, PortableTokenPlan
from layercake_extensions.unicode_direct_neural_core import UnicodeSafeDirectNeuralCoreHost, UnicodeDirectNeuralCoreError, DIRECT_NEURAL_CORE_ROLE, DIRECT_NEURAL_CORE_COMPOSITION

BPE_DIRECT_NEURAL_CORE_ABI_VERSION="lc-direct-neural-core/3"
BPE_DIRECT_NEURAL_CORE_ABI_SHA256="4ec1f609cea9f42924ebcdc16fd6b76be1630351d629d77eaab9b956b7901ac2"
BPE_TOKENIZER_FORMAT="layercake-utf8-concatenative-bpe/1"
BPE_TOKEN_PLAN_FORMAT="layercake-utf8-bpe-token-plan/1"

class Utf8ConcatenativeBpeTokenizer(LosslessLexemePointerTokenizer):
 def __init__(self,document:Mapping[str,Any]):
  model=document.get("model",{}); allowed_top={"version","truncation","padding","added_tokens","normalizer","pre_tokenizer","post_processor","decoder","model"}
  if set(document)!=allowed_top or document.get("normalizer") is not None or document.get("pre_tokenizer") is not None or document.get("post_processor") is not None or document.get("decoder") is not None:raise ValueError("BPE tokenizer graph must be raw concatenative")
  if model.get("type")!="BPE" or model.get("dropout") is not None or model.get("unk_token")!="[UNK]" or model.get("continuing_subword_prefix") is not None or model.get("end_of_word_suffix") is not None or model.get("byte_fallback") is not False or model.get("ignore_merges") is not False:raise ValueError("unsupported BPE model settings")
  vocab=model.get("vocab");merges=model.get("merges")
  if not isinstance(vocab,dict) or not isinstance(merges,list) or vocab.get("[UNK]")!=0:raise ValueError("BPE vocabulary is incomplete")
  pieces=[value.encode("utf-8") for value in vocab if value!="[UNK]"]
  for piece in pieces:piece.decode("utf-8",errors="strict")
  super().__init__(sorted(pieces),format_version="layercake-lossless-lexeme-pointer/2")
  self.document=json.loads(json.dumps(document,sort_keys=True));self.merge_ranks={(str(pair[0]),str(pair[1])):index for index,pair in enumerate(merges)}
 def split(self,value:bytes|str)->list[bytes]:
  text=value.decode("utf-8",errors="strict") if isinstance(value,bytes) else value
  symbols=list(text)
  while len(symbols)>1:
   choices=[(self.merge_ranks[(symbols[i],symbols[i+1])],i) for i in range(len(symbols)-1) if (symbols[i],symbols[i+1]) in self.merge_ranks]
   if not choices:break
   _,index=min(choices);symbols[index:index+2]=[symbols[index]+symbols[index+1]]
  if "".join(symbols)!=text or any(symbol not in self.document["model"]["vocab"] for symbol in symbols):raise ValueError("BPE input is not representable")
  return [symbol.encode("utf-8") for symbol in symbols]
 def canonical_dict(self):return {"format":BPE_TOKENIZER_FORMAT,"tokenizers_json":self.document,"tokenizers_json_sha256":hashlib.sha256(json.dumps(self.document,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"piece_semantics":"UTF8_CONCATENATE_EXACTLY","normalization":"NONE"}
 @classmethod
 def from_document(cls,doc):
  if set(doc)!={"format","tokenizers_json","tokenizers_json_sha256","piece_semantics","normalization"} or doc.get("format")!=BPE_TOKENIZER_FORMAT or doc.get("piece_semantics")!="UTF8_CONCATENATE_EXACTLY" or doc.get("normalization")!="NONE":raise ValueError("BPE tokenizer document changed")
  value=cls(doc["tokenizers_json"])
  if value.canonical_dict()!=doc:raise ValueError("BPE tokenizer identity changed")
  return value
 def hash(self):return hashlib.sha256(json.dumps(self.canonical_dict(),sort_keys=True,separators=(",",":")).encode()).hexdigest()

def bpe_token_plan_manifest_architecture(model:PortableTokenPlan,tokenizer:Utf8ConcatenativeBpeTokenizer):
 if model.fixed_vocab_size!=tokenizer.vocab_size:raise ValueError("model and BPE tokenizer vocabulary sizes differ")
 return {"name":"utf8_bpe_portable_token_plan","format":BPE_TOKEN_PLAN_FORMAT,"model":model.canonical_config(),"tokenizer":tokenizer.canonical_dict(),"tokenizer_sha256":tokenizer.hash(),"external_input_output":"UTF-8 bytes","private_representation":"portable_token_plan_pointer_transformer","action_validity":"every_fixed_and_pointer_action_complete_utf8"}

class BpeDirectNeuralCoreHost(UnicodeSafeDirectNeuralCoreHost):
 def __init__(self,registry_root:str|Path,*,trust_store:Mapping[str,bytes|str|Path],device:str|torch.device="cpu"):
  self.registry=CakeRegistry(registry_root);self.installer=CakeInstaller(self.registry,HostCapabilities(abi_version=BPE_DIRECT_NEURAL_CORE_ABI_VERSION,abi_hash=BPE_DIRECT_NEURAL_CORE_ABI_SHA256,precisions=("fp32",),backends=("pytorch","cuda"),capabilities=frozenset({"byte_input","safe_tensors","persistent_incremental_state","unicode_atomic_actions","strict_utf8_boundary","utf8_concatenative_bpe"})),trust_store=trust_store,strict_signatures=True);self.device=torch.device(device);self.module=None;self.active_cake_id=None;self.active_archive_hash=None;self.active_payload_hash=None;self.receiver_training_steps=0;self.receiver_calibration_runs=0
 @staticmethod
 def _validate_role(package:CakePackage):
  m=package.manifest
  if not package.signed or m.cake_type!="portable_decoder" or m.abi_version!=BPE_DIRECT_NEURAL_CORE_ABI_VERSION or m.abi_hash!=BPE_DIRECT_NEURAL_CORE_ABI_SHA256:raise UnicodeDirectNeuralCoreError("BPE direct core identity mismatch")
  if m.domains!=(DIRECT_NEURAL_CORE_ROLE,) or m.dependencies:raise UnicodeDirectNeuralCoreError("package is not an exclusive English core")
  if m.architecture.get("name")!="utf8_bpe_portable_token_plan" or m.architecture.get("format")!=BPE_TOKEN_PLAN_FORMAT:raise UnicodeDirectNeuralCoreError("BPE architecture mismatch")
 @staticmethod
 def _load_module(package:CakePackage,device:torch.device)->PortableTokenPlan:
  a=package.manifest.architecture;t=Utf8ConcatenativeBpeTokenizer.from_document(a["tokenizer"])
  if t.hash()!=a["tokenizer_sha256"]:raise UnicodeDirectNeuralCoreError("BPE tokenizer hash mismatch")
  model=PortableTokenPlan(**a["model"]).bind_tokenizer(t);model.load_state_dict(package.tensors,strict=True);model.to(device).eval()
  for parameter in model.parameters():parameter.requires_grad_(False)
  return model
