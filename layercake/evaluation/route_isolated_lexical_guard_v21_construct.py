"""Construct-certify the v21 exact lexical-boundary universal guard host."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Iterable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import key_id
from layercake.models.shallow_sparse_english import ShallowSparseEnglishConfig, ShallowSparseEnglishCore
from layercake.portable_domain import canonical_json_hash
from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
from layercake_extensions.route_isolated_lexical_guard_core_v21 import (
    ARCHITECTURE_V21_FORMAT, EXACT_LEXICAL_BOUNDARY, EXACT_LEXICAL_GUARD_FEATURE,
    LexicalGuardPromptSpanCoreHost, ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256,
    ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION,
)
from layercake_extensions.route_isolated_prompt_span_core_v19 import PROMPT_SPAN_FEATURE, extract_prompt_segments, render_prompt_segments
from layercake_extensions.route_isolated_shallow_sparse_core import CAPABILITIES, CAPABILITY_TO_TASK_ROUTE, WEAK_CAPABILITIES, SparseCapabilityRouter, repetition_collapse
from layercake_extensions.route_isolated_shallow_sparse_core_v18 import ExplicitRouteResidual
from layercake_extensions.route_isolated_universal_guard_core_v20 import GUARD_PREDICATE, ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256, ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION, UNIVERSAL_GUARD_FEATURE


class ConstructError(RuntimeError): pass
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys():
    seed=hashlib.sha256(b"layercake-route-isolated-lexical-guard-v21-construct").digest(); private=Ed25519PrivateKey.from_private_bytes(seed)
    private_pem=private.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()); public=private.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)
    return private_pem,public,key_id(public)


def _router_document():
    vocab={"[UNK]":0}; vocab.update({chr(value):value-30 for value in range(32,127)})
    raw={"version":"1.0","truncation":None,"padding":None,"added_tokens":[{"id":0,"content":"[UNK]","single_word":False,"lstrip":False,"rstrip":False,"normalized":False,"special":True}],"normalizer":None,"pre_tokenizer":None,"post_processor":None,"decoder":None,"model":{"type":"BPE","dropout":None,"unk_token":"[UNK]","continuing_subword_prefix":None,"end_of_word_suffix":None,"fuse_unk":False,"byte_fallback":False,"ignore_merges":False,"vocab":vocab,"merges":[]}}
    return Utf8ConcatenativeBpeTokenizer(raw).canonical_dict()


def _fixture(directory: Path, *, capability: str, interface: str="v21", boundary: bool=True):
    torch.manual_seed(21021); vocab={"<eos>":0,"[UNK]":1,"hello":2,"FixI":3,"Fix":4,"##I":5}
    tokenizer=Tokenizer(WordPiece(vocab,unk_token="[UNK]")); tokenizer.pre_tokenizer=Whitespace(); tokenizer.decoder=WordPieceDecoder(prefix="##")
    tokenizer_doc=json.loads(tokenizer.to_str()); tokenizer_raw=json.dumps(tokenizer_doc,sort_keys=True,separators=(",", ":")).encode()
    config=ShallowSparseEnglishConfig(vocab_size=len(vocab),width=16,layers=3,heads=4,max_tokens=128,task_cakes=10,task_cake_rank=64); model=ShallowSparseEnglishCore(config).eval()
    router_doc=_router_document(); router_tokenizer=Utf8ConcatenativeBpeTokenizer.from_document(router_doc); router=SparseCapabilityRouter(router_tokenizer.vocab_size,32,len(CAPABILITIES)+1).eval(); residual=ExplicitRouteResidual(16,16,len(WEAK_CAPABILITIES)).eval()
    with torch.no_grad():
        for module in (model,router,residual):
            for parameter in module.parameters(): parameter.zero_()
        router.bias[CAPABILITIES.index(capability)]=10.0
    guard={"predicate":GUARD_PREDICATE,"scope":"all_capabilities","stop_before_collapsing_token":True,"abstention_markers":["cannot determine"],"abstention_clause":"I cannot determine that from the information given."}
    if boundary: guard["boundary"]=EXACT_LEXICAL_BOUNDARY
    architecture={"format":ARCHITECTURE_V21_FORMAT,"model":config.canonical_dict(),"model_tokenizer":{"format":"declarative-tokenizers-json/1","tokenizers_json":tokenizer_doc,"sha256":hashlib.sha256(tokenizer_raw).hexdigest(),"eos_token_id":0},"router":{"vocabulary":router_tokenizer.vocab_size,"character_hash_buckets":32,"character_ngram_minimum":2,"character_ngram_maximum":5,"hash_seed":450045,"classes":len(CAPABILITIES)+1},"router_tokenizer":router_doc,"residual":{"width":16,"rank":16,"routes":len(WEAK_CAPABILITIES),"reuse":"before_each_transformer_block"},"capabilities":list(CAPABILITIES),"capability_to_task_route":CAPABILITY_TO_TASK_ROUTE,"weak_capabilities":list(WEAK_CAPABILITIES),"guard":guard}
    tensors={prefix+name:value for prefix,state in (("model.",model.state_dict()),("router.",router.state_dict()),("residual.",residual.state_dict())) for name,value in state.items()}; private,public,signer=_keys(); v21=interface=="v21"
    abi_version=ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION if v21 else ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION; abi_hash=ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256 if v21 else ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256
    features=["byte_input","safe_tensors","persistent_incremental_state","physical_route_isolation","declarative_runtime_guard","strict_utf8_boundary",PROMPT_SPAN_FEATURE,UNIVERSAL_GUARD_FEATURE]+([EXACT_LEXICAL_GUARD_FEATURE] if v21 else [])
    manifest=CakeManifest(schema_version="1",cake_id=f"v21-{capability}-{interface}-{int(boundary)}",name="Lexical-guard construct",description="Generic v21 construct fixture",version="21.0.0",publisher={"id":"construct","name":"Construct","key_id":signer},abi_version=abi_version,abi_hash=abi_hash,cake_type="portable_decoder",input_contract={"external":"UTF-8 bytes","role":"english-core","validity":"strict_utf8"},output_contract={"external":"UTF-8 bytes","role":"english-core","composition":"direct_core_only_no_router","validity":"strict_utf8"},architecture=architecture,supported_precisions=("fp32",),supported_backends=("pytorch","cuda"),minimum_host_capabilities={"features":features},tensor_payload_hash="",tensor_shapes=tensor_specs(tensors),package_hash="",training_data_provenance={"dataset":"construct-only","external_teacher":False},evaluation_evidence={"status":"CONSTRUCT_ONLY"},license="Apache-2.0",dependencies=(),parent_version=None,signature={"algorithm":"ed25519","key_id":signer},domains=("english-core",),permissions=("local-inference",))
    path=build_package(directory/f"{manifest.cake_id}.cake",manifest,tensors,private_key=private); return path,public,signer,tensors


def _subtoken_execution(host: LexicalGuardPromptSpanCoreHost) -> dict[str, Any]:
    state=host.prefill("hello"); state["generated_ids"]=[3,3,3,4]; logits=torch.zeros_like(state["next_logits"]); logits[:,5]=1; state["next_logits"]=logits; past_before=state["past_key_values"]; before=host.model_tokenizer.decode(state["generated_ids"]); candidate=host.model_tokenizer.decode([*state["generated_ids"],5]); returned=host.decode_step(state); output=host.realize(state).decode("utf-8")
    return {"capability":state["capability"],"weak_route":int(state["weak_route"]),"before":before,"collapsing_candidate":candidate,"candidate_collapses":repetition_collapse(candidate),"returned":returned,"output":output,"output_collapses":repetition_collapse(output),"output_is_candidate_prefix":candidate.startswith(output),"generated_ids_unchanged":state["generated_ids"]==[3,3,3,4],"model_state_identity":state["past_key_values"] is past_before,"terminated_by_guard":bool(state["terminated_by_guard"]),"finished":bool(state["finished"]),"residual_route_values":[int(block._layercake_residual_routes.item()) for block in host.model.transformer.h]}


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format")!="layercake-postrelease-route-isolated-lexical-guard-v21-construct/1" or protocol.get("status")!="PREREGISTERED_CONSTRUCT_EXECUTION": raise ConstructError("v21 construct protocol changed")
    for relative,expected in protocol["bindings"].items():
        if _sha(root/relative)!=expected: raise ConstructError(f"v21 construct binding changed: {relative}")
    focused=subprocess.run(["C:\\Python310\\python.exe","-m","pytest","tests/test_route_isolated_shallow_sparse_core.py","-q"],cwd=root,check=True,capture_output=True,text=True); complete=subprocess.run(["C:\\Python310\\python.exe","-m","pytest","-q"],cwd=root,check=True,capture_output=True,text=True); sealed=subprocess.run(["C:\\Python310\\python.exe","-m","layercake.moonshot_campaign","verify-all"],cwd=root,check=True,capture_output=True,text=True); sealed_result=json.loads(sealed.stdout)
    prompt="Return the labels in order without commentary: [X-START] begin; [X-MIDDLE] continue; [X-END] finish."
    with tempfile.TemporaryDirectory(prefix="layercake-lexical-v21-") as raw:
        temp=Path(raw); strong,strong_public,strong_signer,strong_tensors=_fixture(temp/"strong",capability="supplied_text_summarization"); weak,weak_public,weak_signer,_=_fixture(temp/"weak",capability="coherence"); v20,v20_public,v20_signer,v20_tensors=_fixture(temp/"v20",capability="coherence",interface="v20",boundary=False); missing,missing_public,missing_signer,_=_fixture(temp/"missing",capability="coherence",boundary=False); executions={}
        for device in ["cpu"]+(["cuda"] if torch.cuda.is_available() else []):
            sh=LexicalGuardPromptSpanCoreHost(temp/f"strong-{device}",trust_store={strong_signer:strong_public},device=device); sa=sh.activate(strong); se=_subtoken_execution(sh); sv=sh.verify()
            wh=LexicalGuardPromptSpanCoreHost(temp/f"weak-{device}",trust_store={weak_signer:weak_public},device=device); wa=wh.activate(weak); we=_subtoken_execution(wh); generated=wh.generate(prompt,maximum_tokens=64); pointer=dict(wh.last_pointer_execution or {}); pointer.pop("wall_seconds",None); ordinary=wh.generate("hello",maximum_tokens=2); wv=wh.verify()
            executions[device]={"strong_active":sa,"strong_subtoken":se,"strong_verify":sv,"weak_active":wa,"weak_subtoken":we,"prompt_span_hex":generated.hex(),"pointer":pointer,"ordinary_hex":ordinary.hex(),"ordinary_used_pointer":wh.last_pointer_execution is not None,"weak_verify":wv}
        v20_rejected=False; missing_rejected=False
        try: LexicalGuardPromptSpanCoreHost(temp/"reject-v20",trust_store={v20_signer:v20_public}).activate(v20)
        except Exception: v20_rejected=True
        try: LexicalGuardPromptSpanCoreHost(temp/"reject-missing",trust_store={missing_signer:missing_public}).activate(missing)
        except Exception: missing_rejected=True
        expected={render_prompt_segments(p).encode().hex() for p in itertools.permutations(extract_prompt_segments(prompt))}; target="FixI FixI FixI"
        checks={"cuda_available":torch.cuda.is_available(),"canonical_abi_identity":_sha(root/protocol["canonical_abi"])==ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256,"v20_tensor_schema_unchanged":set(strong_tensors)==set(v20_tensors) and all(torch.equal(strong_tensors[name],v20_tensors[name]) for name in strong_tensors),"v20_manifest_rejected":v20_rejected,"missing_lexical_boundary_rejected":missing_rejected,"strong_exact_lexical_guard":all(value["strong_subtoken"]["capability"]=="supplied_text_summarization" and value["strong_subtoken"]["weak_route"]==-1 and value["strong_subtoken"]["output"]==target for value in executions.values()),"weak_exact_lexical_guard":all(value["weak_subtoken"]["capability"]=="coherence" and value["weak_subtoken"]["weak_route"]>=0 and value["weak_subtoken"]["output"]==target for value in executions.values()),"candidate_collapses_output_safe_prefix":all(item["candidate_collapses"] and not item["output_collapses"] and item["output_is_candidate_prefix"] for value in executions.values() for item in (value["strong_subtoken"],value["weak_subtoken"])),"model_state_not_advanced":all(item["returned"] is None and item["generated_ids_unchanged"] and item["model_state_identity"] and item["terminated_by_guard"] and item["finished"] for value in executions.values() for item in (value["strong_subtoken"],value["weak_subtoken"])),"physical_route_selection":all(value["strong_subtoken"]["residual_route_values"]==[-1,-1,-1] and len(set(value["weak_subtoken"]["residual_route_values"]))==1 and value["weak_subtoken"]["residual_route_values"][0]>=0 for value in executions.values()),"prompt_span_preserved":all(value["prompt_span_hex"] in expected and value["pointer"].get("candidate_count")==6 and value["pointer"].get("candidate_scoring_forward_passes")==1 and value["pointer"].get("persistent_prompt_state_reused") is True and value["pointer"].get("evaluator_used") is False for value in executions.values()),"ordinary_noncollapse_preserved":all(value["ordinary_hex"]=="" and value["ordinary_used_pointer"] is False for value in executions.values()),"cpu_cuda_identity":set(executions)=={"cpu","cuda"} and len({(value["strong_subtoken"]["output"],value["weak_subtoken"]["output"],value["prompt_span_hex"],value["ordinary_hex"]) for value in executions.values()})==1,"same_signed_packages_all_devices":len({value["strong_active"]["archive_hash"] for value in executions.values()})==1 and len({value["weak_active"]["archive_hash"] for value in executions.values()})==1,"receiver_learning_zero":all(value[name]["receiver_training_steps"]==value[name]["receiver_calibration_runs"]==0 for value in executions.values() for name in ("strong_active","weak_active")),"focused_tests_pass":"14 passed" in focused.stdout,"complete_tests_pass":complete.returncode==0,"sealed_campaign_unchanged":sealed_result.get("completed_phases_valid") is True}
        result={"format":"layercake-postrelease-route-isolated-lexical-guard-v21-construct-result/1","status":"PASS_CONSTRUCT_ONLY" if all(checks.values()) else "FAIL","protocol_sha256":_sha(protocol_path),"interface":ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION,"interface_sha256":ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256,"checks":checks,"devices":executions,"packages":{"strong_sha256":_sha(strong),"weak_sha256":_sha(weak)},"new_parameters":0,"teacher_present":False,"source_transformer_blocks":0,"historical_release_changed":False,"hardware":{"machine":platform.node(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},"claim_boundary":"Generic v21 lexical-guard host construct only; no external artifact, quality, information minimum, physical performance, phase, or superiority claim."}; result["evidence_sha256"]=canonical_json_hash(result); return result


def main(argv: Iterable[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("command",choices=("execute","verify")); parser.add_argument("--protocol",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(argv); root=Path.cwd().resolve(); expected=execute(root,root/args.protocol); output=root/args.output
    if args.command=="execute":
        if output.exists(): raise ConstructError(f"immutable output exists: {output}")
        if expected["status"]!="PASS_CONSTRUCT_ONLY": raise ConstructError(f"v21 construct failed: {expected['checks']}")
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(expected,indent=2,sort_keys=True)+"\n",encoding="utf-8"); result=expected
    else:
        stored=json.loads(output.read_text(encoding="utf-8"))
        if stored!=expected: raise ConstructError("stored v21 construct differs from recomputation")
        result={"status":"PASS","evidence_sha256":expected["evidence_sha256"],"construct_only":True}
    print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
