"""Construct verifier for direct-linear host v12."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,tempfile
from pathlib import Path
import torch
from layercake.direct_linear_progressive_core import DirectLinearProgressiveCore
from layercake_extensions.direct_linear_progressive_core import DIRECT_LINEAR_ABI_SHA256,DIRECT_LINEAR_ABI_VERSION,DirectLinearProgressiveCoreHost
def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def execute(root:Path,protocol_path:Path):
    p=json.loads(protocol_path.read_text(encoding="utf-8"))
    if p.get("format")!="layercake-postrelease-direct-linear-progressive-construct/1" or p.get("status")!="PREREGISTERED_CONSTRUCT_ONLY": raise RuntimeError("governance changed")
    for n,h in p["bindings"].items():
        q=root/n
        if not q.is_file() or sha(q)!=h: raise RuntimeError(f"binding changed: {n}")
    spec=importlib.util.spec_from_file_location("fixture",root/"tests/models/test_direct_linear_progressive_core.py"); f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
    with tempfile.TemporaryDirectory(prefix="lc-v12-") as n:
        temp=Path(n); package,pub,key,state_hash,t=f._package(temp/"s"); archive=sha(package); cpu=DirectLinearProgressiveCoreHost(temp/"c",trust_store={key:pub}); active=cpu.activate(package); ids=t.encode_source("hello world")[0]; state=cpu.prefill("hello world"); first=torch.allclose(state.next_logits,cpu.module(torch.tensor([ids]))[:,-1],atol=1e-5,rtol=1e-5); before=tuple(x.shape[2] for x in state.layer_keys); state.next_logits.zero_(); state.next_logits[0,4]=1; action,state=cpu.decode_step(state); second=torch.allclose(state.next_logits,cpu.module(torch.tensor([ids+[action]]))[:,-1],atol=1e-5,rtol=1e-5); after=tuple(x.shape[2] for x in state.layer_keys); verified=cpu.verify(); cpu.remove(); reinstall=cpu.activate(package); cuda=None
        if torch.cuda.is_available(): cuda=DirectLinearProgressiveCoreHost(temp/"g",trust_store={key:pub},device="cuda").activate(package)
        count=DirectLinearProgressiveCore.parameter_count_for_config(fixed_vocab_size=32015,full_width=3072,bottleneck_width=192,replacement_layers=32,intermediate_size=768)
        checks={"abi":DIRECT_LINEAR_ABI_VERSION=="lc-direct-neural-core/12" and DIRECT_LINEAR_ABI_SHA256==p["interface_sha256"],"abi_hash":sha(root/p["canonical_abi"])==DIRECT_LINEAR_ABI_SHA256,"parameters":count==277220352,"direct_map":all(hasattr(x,"mlp_coefficient_projection") and not hasattr(x,"gate_up_proj") for x in cpu.module.layers),"first":bool(first),"second":bool(second),"cache":all(r==l+1 for l,r in zip(before,after)),"cpu":active["state_dict_hash"]==state_hash and verified["status"]=="PASS","zero_learning":active["receiver_training_steps"]==active["receiver_calibration_runs"]==0,"lifecycle":sha(package)==archive and reinstall["archive_hash"]==active["archive_hash"],"cuda":cuda is not None and cuda["archive_hash"]==active["archive_hash"] and cuda["state_dict_hash"]==state_hash,"zero_source":p["source_transformer_blocks"]==0}
    r={"format":"layercake-postrelease-direct-linear-progressive-result/1","status":"PASS" if all(checks.values()) else "FAIL","protocol":{"path":protocol_path.name,"sha256":sha(protocol_path)},"checks":checks,"target_parameters":count,"package":{"archive_hash":archive,"state_dict_hash":state_hash},"external_artifact_used":False,"english_quality_tested":False,"performance_tested":False,"claim_boundary":"Generic direct-linear host construct only; no acquisition, quality, or performance claim."}; r["evidence_sha256"]=hashlib.sha256((json.dumps(r,sort_keys=True,separators=(",",":"))+"\n").encode()).hexdigest(); return r
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("execute","verify")); ap.add_argument("--protocol",required=True); ap.add_argument("--output",required=True); a=ap.parse_args(); root=Path.cwd().resolve(); r=execute(root,root/a.protocol); out=root/a.output
    if a.command=="execute":
        if out.exists(): raise RuntimeError("output exists")
        out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    elif json.loads(out.read_text(encoding="utf-8"))!=r: raise RuntimeError("stored result differs")
    print(json.dumps({"status":r["status"],"evidence_sha256":r["evidence_sha256"]},indent=2)); return 0 if r["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
