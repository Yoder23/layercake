"""Construct-certify the generic v20 universal-guard prompt-span host."""

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
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import key_id
from layercake.models.shallow_sparse_english import ShallowSparseEnglishConfig, ShallowSparseEnglishCore
from layercake.portable_domain import canonical_json_hash
from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
from layercake_extensions.route_isolated_prompt_span_core_v19 import (
    ARCHITECTURE_V19_FORMAT,
    PROMPT_SPAN_FEATURE,
    ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256,
    ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION,
    extract_prompt_segments,
    render_prompt_segments,
)
from layercake_extensions.route_isolated_shallow_sparse_core import CAPABILITIES, CAPABILITY_TO_TASK_ROUTE, WEAK_CAPABILITIES, SparseCapabilityRouter
from layercake_extensions.route_isolated_shallow_sparse_core_v18 import ExplicitRouteResidual
from layercake_extensions.route_isolated_universal_guard_core_v20 import (
    ARCHITECTURE_V20_FORMAT,
    GUARD_PREDICATE,
    ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256,
    ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION,
    UNIVERSAL_GUARD_FEATURE,
    UniversalGuardPromptSpanCoreHost,
)


class ConstructError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys():
    seed = hashlib.sha256(b"layercake-route-isolated-universal-guard-v20-construct").digest()
    private = Ed25519PrivateKey.from_private_bytes(seed)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return private_pem, public_pem, key_id(public_pem)


def _router_document():
    vocabulary = {"[UNK]": 0}
    vocabulary.update({chr(value): value - 30 for value in range(32, 127)})
    raw = {
        "version":"1.0","truncation":None,"padding":None,
        "added_tokens":[{"id":0,"content":"[UNK]","single_word":False,"lstrip":False,"rstrip":False,"normalized":False,"special":True}],
        "normalizer":None,"pre_tokenizer":None,"post_processor":None,"decoder":None,
        "model":{"type":"BPE","dropout":None,"unk_token":"[UNK]","continuing_subword_prefix":None,"end_of_word_suffix":None,"fuse_unk":False,"byte_fallback":False,"ignore_merges":False,"vocab":vocabulary,"merges":[]},
    }
    return Utf8ConcatenativeBpeTokenizer(raw).canonical_dict()


def _fixture(
    directory: Path,
    *,
    capability: str,
    repeating: bool,
    interface: str = "v20",
    guard_scope: str = "all_capabilities",
):
    torch.manual_seed(20020)
    vocabulary = {"<eos>":0,"[UNK]":1,"loop":2,"hello":3}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")); tokenizer.pre_tokenizer = Whitespace()
    tokenizer_doc = json.loads(tokenizer.to_str()); tokenizer_raw = json.dumps(tokenizer_doc, sort_keys=True, separators=(",", ":")).encode()
    config = ShallowSparseEnglishConfig(vocab_size=len(vocabulary), width=16, layers=3, heads=4, max_tokens=128, task_cakes=10, task_cake_rank=64)
    model = ShallowSparseEnglishCore(config).eval(); router_doc = _router_document(); router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(router_doc)
    router = SparseCapabilityRouter(router_tokenizer.vocab_size, 32, len(CAPABILITIES) + 1).eval(); residual = ExplicitRouteResidual(16, 16, len(WEAK_CAPABILITIES)).eval()
    with torch.no_grad():
        for module in (model, router, residual):
            for parameter in module.parameters(): parameter.zero_()
        router.bias[CAPABILITIES.index(capability)] = 10.0
        if repeating:
            model.transformer.wte.weight[2].copy_(torch.linspace(-1.0, 1.0, 16))
            model.transformer.ln_f.weight.fill_(1.0)
    architecture = {
        "format":ARCHITECTURE_V20_FORMAT,"model":config.canonical_dict(),
        "model_tokenizer":{"format":"declarative-tokenizers-json/1","tokenizers_json":tokenizer_doc,"sha256":hashlib.sha256(tokenizer_raw).hexdigest(),"eos_token_id":0},
        "router":{"vocabulary":router_tokenizer.vocab_size,"character_hash_buckets":32,"character_ngram_minimum":2,"character_ngram_maximum":5,"hash_seed":450045,"classes":len(CAPABILITIES)+1},
        "router_tokenizer":router_doc,"residual":{"width":16,"rank":16,"routes":len(WEAK_CAPABILITIES),"reuse":"before_each_transformer_block"},
        "capabilities":list(CAPABILITIES),"capability_to_task_route":CAPABILITY_TO_TASK_ROUTE,"weak_capabilities":list(WEAK_CAPABILITIES),
        "guard":{"predicate":GUARD_PREDICATE,"scope":guard_scope,"stop_before_collapsing_token":True,"abstention_markers":["cannot determine"],"abstention_clause":"I cannot determine that from the information given."},
    }
    tensors = {prefix+name:value for prefix,state in (("model.",model.state_dict()),("router.",router.state_dict()),("residual.",residual.state_dict())) for name,value in state.items()}
    private, public, signer = _keys()
    v20 = interface == "v20"
    abi_version = ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION if v20 else ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION
    abi_hash = ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256 if v20 else ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256
    features = ["byte_input","safe_tensors","persistent_incremental_state","physical_route_isolation","declarative_runtime_guard","strict_utf8_boundary",PROMPT_SPAN_FEATURE] + ([UNIVERSAL_GUARD_FEATURE] if v20 else [])
    manifest = CakeManifest(
        schema_version="1",cake_id=f"v20-{capability}-{int(repeating)}-{interface}-{guard_scope}",name="Universal-guard construct",description="Generic v20 construct fixture",version="20.0.0",publisher={"id":"construct","name":"Construct","key_id":signer},abi_version=abi_version,abi_hash=abi_hash,cake_type="portable_decoder",input_contract={"external":"UTF-8 bytes","role":"english-core","validity":"strict_utf8"},output_contract={"external":"UTF-8 bytes","role":"english-core","composition":"direct_core_only_no_router","validity":"strict_utf8"},architecture=architecture,supported_precisions=("fp32",),supported_backends=("pytorch","cuda"),minimum_host_capabilities={"features":features},tensor_payload_hash="",tensor_shapes=tensor_specs(tensors),package_hash="",training_data_provenance={"dataset":"construct-only","external_teacher":False},evaluation_evidence={"status":"CONSTRUCT_ONLY"},license="Apache-2.0",dependencies=(),parent_version=None,signature={"algorithm":"ed25519","key_id":signer},domains=("english-core",),permissions=("local-inference",),
    )
    path = build_package(directory / f"{manifest.cake_id}.cake", manifest, tensors, private_key=private)
    return path, public, signer, tensors


def _guard_execution(host: UniversalGuardPromptSpanCoreHost) -> dict[str, Any]:
    state = host.prefill("loop")
    initial_cache_layers = len(state["past_key_values"])
    for _ in range(8):
        if host.decode_step(state) is None:
            break
    return {
        "capability":state["capability"],"weak_route":int(state["weak_route"]),
        "terminated_by_guard":bool(state["terminated_by_guard"]),"finished":bool(state["finished"]),
        "generated_token_count":len(state["generated_ids"]),"output_hex":host.realize(state).hex(),
        "initial_cache_layers":initial_cache_layers,
        "residual_route_values":[int(block._layercake_residual_routes.item()) for block in host.model.transformer.h],
        "task_cake_calls":list(host.model.last_cake_calls),
    }


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-route-isolated-universal-guard-v20-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_EXECUTION":
        raise ConstructError("v20 construct protocol changed")
    for relative, expected in protocol["bindings"].items():
        if _sha(root / relative) != expected: raise ConstructError(f"v20 construct binding changed: {relative}")
    focused = subprocess.run(["C:\\Python310\\python.exe","-m","pytest","tests/test_route_isolated_shallow_sparse_core.py","-q"],cwd=root,check=True,capture_output=True,text=True)
    complete = subprocess.run(["C:\\Python310\\python.exe","-m","pytest","-q"],cwd=root,check=True,capture_output=True,text=True)
    sealed = subprocess.run(["C:\\Python310\\python.exe","-m","layercake.moonshot_campaign","verify-all"],cwd=root,check=True,capture_output=True,text=True); sealed_result=json.loads(sealed.stdout)
    prompt="Return the labels in order without commentary: [X-START] begin; [X-MIDDLE] continue; [X-END] finish."
    with tempfile.TemporaryDirectory(prefix="layercake-universal-v20-") as raw:
        temp=Path(raw)
        safe,public,signer,safe_tensors=_fixture(temp/"safe",capability="coherence",repeating=False)
        strong,strong_public,strong_signer,_=_fixture(temp/"strong",capability="supplied_text_summarization",repeating=True)
        weak,weak_public,weak_signer,_=_fixture(temp/"weak",capability="coherence",repeating=True)
        v19,v19_public,v19_signer,v19_tensors=_fixture(temp/"v19",capability="coherence",repeating=False,interface="v19",guard_scope="weak_capabilities_only")
        weak_scope,bad_public,bad_signer,_=_fixture(temp/"bad",capability="coherence",repeating=False,guard_scope="weak_capabilities_only")
        executions={}
        for device in ["cpu"]+(["cuda"] if torch.cuda.is_available() else []):
            safe_host=UniversalGuardPromptSpanCoreHost(temp/f"registry-safe-{device}",trust_store={signer:public},device=device); active=safe_host.activate(safe); generated=safe_host.generate(prompt,maximum_tokens=64); pointer=dict(safe_host.last_pointer_execution or {}); pointer.pop("wall_seconds",None); ordinary=safe_host.generate("hello",maximum_tokens=2); verified=safe_host.verify()
            strong_host=UniversalGuardPromptSpanCoreHost(temp/f"registry-strong-{device}",trust_store={strong_signer:strong_public},device=device); strong_active=strong_host.activate(strong); strong_guard=_guard_execution(strong_host)
            weak_host=UniversalGuardPromptSpanCoreHost(temp/f"registry-weak-{device}",trust_store={weak_signer:weak_public},device=device); weak_active=weak_host.activate(weak); weak_guard=_guard_execution(weak_host)
            executions[device]={"safe_active":active,"generated_hex":generated.hex(),"pointer":pointer,"ordinary_hex":ordinary.hex(),"ordinary_used_pointer":safe_host.last_pointer_execution is not None,"verify":verified,"strong_active":strong_active,"strong_guard":strong_guard,"weak_active":weak_active,"weak_guard":weak_guard}
        v19_rejected=False; weak_scope_rejected=False
        try: UniversalGuardPromptSpanCoreHost(temp/"registry-v19-reject",trust_store={v19_signer:v19_public}).activate(v19)
        except Exception: v19_rejected=True
        try: UniversalGuardPromptSpanCoreHost(temp/"registry-scope-reject",trust_store={bad_signer:bad_public}).activate(weak_scope)
        except Exception: weak_scope_rejected=True
        expected_candidates={render_prompt_segments(p).encode().hex() for p in itertools.permutations(extract_prompt_segments(prompt))}
        checks={
            "cuda_available":torch.cuda.is_available(),
            "canonical_abi_identity":_sha(root/protocol["canonical_abi"])==ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256,
            "v19_tensor_schema_unchanged":set(safe_tensors)==set(v19_tensors) and all(torch.equal(safe_tensors[name],v19_tensors[name]) for name in safe_tensors),
            "v19_manifest_rejected":v19_rejected,"weak_only_v20_guard_rejected":weak_scope_rejected,
            "strong_guard_actual":all(value["strong_guard"]["capability"]=="supplied_text_summarization" and value["strong_guard"]["weak_route"]==-1 and value["strong_guard"]["terminated_by_guard"] and value["strong_guard"]["generated_token_count"]==3 for value in executions.values()),
            "weak_guard_actual":all(value["weak_guard"]["capability"]=="coherence" and value["weak_guard"]["weak_route"]>=0 and value["weak_guard"]["terminated_by_guard"] and value["weak_guard"]["generated_token_count"]==3 for value in executions.values()),
            "guard_stops_before_collapse":all(bytes.fromhex(value[key]["output_hex"]).decode()=="loop loop loop" for value in executions.values() for key in ("strong_guard","weak_guard")),
            "persistent_state":all(value[key]["initial_cache_layers"]==3 for value in executions.values() for key in ("strong_guard","weak_guard")),
            "physical_route_selection":all(value["strong_guard"]["residual_route_values"]==[-1,-1,-1] and len(value["strong_guard"]["task_cake_calls"])==1 and len(set(value["weak_guard"]["residual_route_values"]))==1 and value["weak_guard"]["residual_route_values"][0]>=0 and len(value["weak_guard"]["task_cake_calls"])==1 for value in executions.values()),
            "prompt_span_preserved":all(value["generated_hex"] in expected_candidates and value["pointer"].get("candidate_count")==6 and value["pointer"].get("candidate_scoring_forward_passes")==1 and value["pointer"].get("persistent_prompt_state_reused") is True and value["pointer"].get("evaluator_used") is False for value in executions.values()),
            "ordinary_noncollapse_preserved":all(value["ordinary_hex"]=="" and value["ordinary_used_pointer"] is False for value in executions.values()),
            "cpu_cuda_identity":set(executions)=={"cpu","cuda"} and len({(value["generated_hex"],value["ordinary_hex"],value["strong_guard"]["output_hex"],value["weak_guard"]["output_hex"]) for value in executions.values()})==1,
            "same_signed_packages_all_devices":len({value["safe_active"]["archive_hash"] for value in executions.values()})==1 and len({value["strong_active"]["archive_hash"] for value in executions.values()})==1 and len({value["weak_active"]["archive_hash"] for value in executions.values()})==1,
            "receiver_learning_zero":all(value[name]["receiver_training_steps"]==value[name]["receiver_calibration_runs"]==0 for value in executions.values() for name in ("safe_active","strong_active","weak_active")),
            "focused_tests_pass":"10 passed" in focused.stdout,
            "complete_tests_pass":complete.returncode==0,
            "sealed_campaign_unchanged":sealed_result.get("completed_phases_valid") is True,
        }
        result={"format":"layercake-postrelease-route-isolated-universal-guard-v20-construct-result/1","status":"PASS_CONSTRUCT_ONLY" if all(checks.values()) else "FAIL","protocol_sha256":_sha(protocol_path),"interface":ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION,"interface_sha256":ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256,"checks":checks,"devices":executions,"packages":{"safe_sha256":_sha(safe),"strong_sha256":_sha(strong),"weak_sha256":_sha(weak)},"new_parameters":0,"teacher_present":False,"source_transformer_blocks":0,"historical_release_changed":False,"hardware":{"machine":platform.node(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},"claim_boundary":"Generic v20 universal-guard host construct only; no external artifact, English quality, information minimum, CPU/GPU performance, Phase recertification, or superiority claim."}
        result["evidence_sha256"]=canonical_json_hash(result)
        return result


def main(argv: Iterable[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("command",choices=("execute","verify")); parser.add_argument("--protocol",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(argv)
    root=Path.cwd().resolve(); protocol=root/args.protocol; output=root/args.output; expected=execute(root,protocol)
    if args.command=="execute":
        if output.exists(): raise ConstructError(f"immutable output exists: {output}")
        if expected["status"]!="PASS_CONSTRUCT_ONLY": raise ConstructError(f"v20 construct failed: {expected['checks']}")
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(expected,indent=2,sort_keys=True)+"\n",encoding="utf-8"); result=expected
    else:
        stored=json.loads(output.read_text(encoding="utf-8"))
        if stored!=expected: raise ConstructError("stored v20 construct differs from recomputation")
        result={"status":"PASS","evidence_sha256":expected["evidence_sha256"],"construct_only":True}
    print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
