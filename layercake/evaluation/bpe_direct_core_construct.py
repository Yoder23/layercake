"""Construct-certify the generic LayerCake UTF-8 BPE direct-core host v3."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, tempfile
from pathlib import Path
from typing import Any, Iterable
import torch
from layercake_extensions.bpe_direct_neural_core import BpeDirectNeuralCoreHost,Utf8ConcatenativeBpeTokenizer,BPE_DIRECT_NEURAL_CORE_ABI_SHA256,BPE_DIRECT_NEURAL_CORE_ABI_VERSION

def sha(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_protocol(root,path):
 p=json.loads(path.read_text(encoding="utf-8"))
 if p.get("format")!="layercake-postrelease-bpe-direct-core-construct/1" or p.get("status")!="PREREGISTERED_CONSTRUCT_ONLY":raise RuntimeError("BPE construct governance changed")
 for rel,want in p["bindings"].items():
  target=(root/rel).resolve()
  if not target.is_file() or sha(target)!=want:raise RuntimeError(f"BPE construct binding changed: {rel}")
 return p,sha(path)
def fixture_module(root):
 path=root/"tests/models/test_bpe_direct_neural_core.py";spec=importlib.util.spec_from_file_location("bpe_construct_fixture",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def execute(root:Path,protocol_path:Path)->dict[str,Any]:
 p,ph=load_protocol(root,protocol_path);fixture=fixture_module(root);tok=Utf8ConcatenativeBpeTokenizer(fixture._doc());sample="hello world";pieces=tok.split(sample)
 with tempfile.TemporaryDirectory(prefix="lc-bpe-v3-") as name:
  temp=Path(name);package,public,key_id,state,_=fixture._package(temp/"source");archive_sha=sha(package);cpu=BpeDirectNeuralCoreHost(temp/"cpu",trust_store={key_id:public});cpu_active=cpu.activate(package);prefill=cpu.prefill(sample);_,prefill=cpu.decode_step(prefill);realized=cpu.realize(prefill);cpu_generated=cpu.generate(sample);cpu_verify=cpu.verify();cpu.remove();cpu_reinstall=cpu.activate(package)
  cuda=None
  if torch.cuda.is_available():
   host=BpeDirectNeuralCoreHost(temp/"cuda",trust_store={key_id:public},device="cuda");active=host.activate(package);cuda={"archive_hash":active["archive_hash"],"payload_hash":active["payload_hash"],"state_dict_hash":active["state_dict_hash"],"generated_hex":host.generate(sample).hex()}
  checks={"abi_identity":BPE_DIRECT_NEURAL_CORE_ABI_VERSION=="lc-direct-neural-core/3" and BPE_DIRECT_NEURAL_CORE_ABI_SHA256==p["interface_sha256"],"bpe_exact_roundtrip":b"".join(pieces)==sample.encode(),"all_actions_valid_utf8":all(piece.decode("utf-8").encode()==piece for piece in pieces),"cpu_state_identity":cpu_active["state_dict_hash"]==state,"cpu_zero_learning":cpu_active["receiver_training_steps"]==cpu_active["receiver_calibration_runs"]==0,"persistent_state_realizes_bytes":isinstance(realized,bytes),"cpu_generation_strict":cpu_generated==b"","lifecycle_identity":archive_sha==sha(package) and cpu_reinstall["archive_hash"]==cpu_active["archive_hash"],"cpu_verifier_pass":cpu_verify["status"]=="PASS","cuda_available":torch.cuda.is_available(),"same_package_cuda":cuda is not None and cuda["archive_hash"]==cpu_active["archive_hash"] and cuda["payload_hash"]==cpu_active["payload_hash"] and cuda["state_dict_hash"]==cpu_active["state_dict_hash"] and cuda["generated_hex"]==cpu_generated.hex()}
 result={"format":"layercake-postrelease-bpe-direct-core-construct-result/1","status":"PASS" if all(checks.values()) else "FAIL","protocol":{"path":protocol_path.name,"sha256":ph},"checks":checks,"package":{"payload_hash":cpu_active["payload_hash"],"state_dict_hash":state},"receiver_training_steps":0,"receiver_calibration_runs":0,"external_artifact_used":False,"english_quality_tested":False,"performance_tested":False,"claim_boundary":"Generic host construct only; no ABI acquisition, English quality, or performance claim."};result["evidence_sha256"]=hashlib.sha256((json.dumps(result,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest();return result
def main(argv:Iterable[str]|None=None)->int:
 a=argparse.ArgumentParser();a.add_argument("command",choices=("execute","verify"));a.add_argument("--protocol",default="moonshot/postrelease_bpe_direct_neural_core_preregistration_v5.json");a.add_argument("--output",default="results/moonshot/postrelease/bpe_direct_neural_core_construct_v1.json");x=a.parse_args(argv);root=Path.cwd().resolve();e=execute(root,(root/x.protocol).resolve());o=(root/x.output).resolve()
 if x.command=="execute":
  if o.exists():raise RuntimeError("BPE construct result immutable")
  o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(e,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 elif json.loads(o.read_text(encoding="utf-8"))!=e:raise RuntimeError("stored BPE construct differs")
 print(json.dumps({"status":e["status"],"evidence_sha256":e["evidence_sha256"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
