"""Construct-certify the generic v19 model-ranked prompt-span host."""

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
from layercake_extensions.route_isolated_shallow_sparse_core import CAPABILITIES, CAPABILITY_TO_TASK_ROUTE, WEAK_CAPABILITIES, SparseCapabilityRouter
from layercake_extensions.route_isolated_shallow_sparse_core_v18 import ARCHITECTURE_V18_FORMAT, ROUTE_ISOLATED_CORE_V18_ABI_SHA256, ROUTE_ISOLATED_CORE_V18_ABI_VERSION, ExactRouteIsolatedShallowSparseCoreHost, ExplicitRouteResidual
from layercake_extensions.route_isolated_prompt_span_core_v19 import ARCHITECTURE_V19_FORMAT, PROMPT_SPAN_FEATURE, ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256, ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION, PromptSpanRouteIsolatedShallowSparseCoreHost, extract_prompt_segments, render_prompt_segments


class ConstructError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys():
    seed = hashlib.sha256(b"layercake-route-isolated-prompt-span-v19-construct").digest()
    private = Ed25519PrivateKey.from_private_bytes(seed)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return private_pem, public_pem, key_id(public_pem)


def _router_document():
    vocabulary = {"[UNK]": 0}
    vocabulary.update({chr(value): value - 30 for value in range(32, 127)})
    raw = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": 0, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}],
        "normalizer": None, "pre_tokenizer": None, "post_processor": None, "decoder": None,
        "model": {"type": "BPE", "dropout": None, "unk_token": "[UNK]", "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": vocabulary, "merges": []},
    }
    return Utf8ConcatenativeBpeTokenizer(raw).canonical_dict()


def _fixture(directory: Path, *, v18: bool = False):
    torch.manual_seed(19019)
    vocabulary = {"<eos>": 0, "[UNK]": 1, "hello": 2}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")); tokenizer.pre_tokenizer = Whitespace()
    tokenizer_doc = json.loads(tokenizer.to_str()); tokenizer_raw = json.dumps(tokenizer_doc, sort_keys=True, separators=(",", ":")).encode()
    config = ShallowSparseEnglishConfig(vocab_size=len(vocabulary), width=16, layers=3, heads=4, max_tokens=128, task_cakes=10, task_cake_rank=64)
    model = ShallowSparseEnglishCore(config).eval(); router_doc = _router_document(); router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(router_doc)
    router = SparseCapabilityRouter(router_tokenizer.vocab_size, 32, len(CAPABILITIES) + 1).eval(); residual = ExplicitRouteResidual(16, 16, len(WEAK_CAPABILITIES)).eval()
    with torch.no_grad():
        for module in (model, router, residual):
            for parameter in module.parameters(): parameter.zero_()
        router.bias[CAPABILITIES.index("coherence")] = 10.0
    architecture = {
        "format": ARCHITECTURE_V19_FORMAT,
        "model": config.canonical_dict(),
        "model_tokenizer": {"format": "declarative-tokenizers-json/1", "tokenizers_json": tokenizer_doc, "sha256": hashlib.sha256(tokenizer_raw).hexdigest(), "eos_token_id": 0},
        "router": {"vocabulary": router_tokenizer.vocab_size, "character_hash_buckets": 32, "character_ngram_minimum": 2, "character_ngram_maximum": 5, "hash_seed": 450045, "classes": len(CAPABILITIES) + 1},
        "router_tokenizer": router_doc,
        "residual": {"width": 16, "rank": 16, "routes": len(WEAK_CAPABILITIES), "reuse": "before_each_transformer_block"},
        "capabilities": list(CAPABILITIES), "capability_to_task_route": CAPABILITY_TO_TASK_ROUTE, "weak_capabilities": list(WEAK_CAPABILITIES),
        "guard": {"predicate": "contiguous_1_to_16_token_span_repeated_4_times_or_fourgram_diversity_below_0.35_at_32_tokens", "scope": "weak_capabilities_only", "stop_before_collapsing_token": True, "abstention_markers": ["cannot determine"], "abstention_clause": "I cannot determine that from the information given."},
    }
    tensors = {}
    for prefix, state in (("model.", model.state_dict()), ("router.", router.state_dict()), ("residual.", residual.state_dict())):
        tensors.update({prefix + name: value for name, value in state.items()})
    private, public, signer = _keys(); abi_version = ROUTE_ISOLATED_CORE_V18_ABI_VERSION if v18 else ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION; abi_hash = ROUTE_ISOLATED_CORE_V18_ABI_SHA256 if v18 else ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256
    features = ["byte_input", "safe_tensors", "persistent_incremental_state", "physical_route_isolation", "declarative_runtime_guard", "strict_utf8_boundary"] + ([] if v18 else [PROMPT_SPAN_FEATURE])
    manifest = CakeManifest(schema_version="1", cake_id="route-isolated-prompt-span-construct", name="Prompt-span construct", description="Generic construct-only model-ranked prompt-span core", version="19.0.0", publisher={"id": "construct", "name": "Construct", "key_id": signer}, abi_version=abi_version, abi_hash=abi_hash, cake_type="portable_decoder", input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"}, output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"}, architecture=architecture, supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"), minimum_host_capabilities={"features": features}, tensor_payload_hash="", tensor_shapes=tensor_specs(tensors), package_hash="", training_data_provenance={"dataset": "construct-only", "external_teacher": False}, evaluation_evidence={"status": "CONSTRUCT_ONLY"}, license="Apache-2.0", dependencies=(), parent_version=None, signature={"algorithm": "ed25519", "key_id": signer}, domains=("english-core",), permissions=("local-inference",))
    package = build_package(directory / ("v18.cake" if v18 else "v19.cake"), manifest, tensors, private_key=private)
    return package, public, signer, tensors


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-route-isolated-prompt-span-v19-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise ConstructError("v19 construct protocol changed")
    for relative, expected in protocol["bindings"].items():
        if _sha(root / relative) != expected: raise ConstructError(f"v19 construct binding changed: {relative}")
    focused = subprocess.run(["C:\\Python310\\python.exe", "-m", "pytest", "tests/test_route_isolated_shallow_sparse_core.py", "-q"], cwd=root, check=True, capture_output=True, text=True)
    sealed = subprocess.run(["C:\\Python310\\python.exe", "-m", "layercake.moonshot_campaign", "verify-all"], cwd=root, check=True, capture_output=True, text=True); sealed_result = json.loads(sealed.stdout)
    prompt = "Return the labels in order without commentary: [X-START] begin; [X-MIDDLE] continue; [X-END] finish."
    with tempfile.TemporaryDirectory(prefix="layercake-prompt-span-v19-") as raw:
        temp = Path(raw); package, public, signer, tensors = _fixture(temp); executions = {}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            host = PromptSpanRouteIsolatedShallowSparseCoreHost(temp / f"registry-{device}", trust_store={signer: public}, device=device); active = host.activate(package); generated = host.generate(prompt, maximum_tokens=64); pointer = dict(host.last_pointer_execution or {}); ordinary = host.generate("hello", maximum_tokens=2); verified = host.verify(); executions[device] = {"active": active, "generated_hex": generated.hex(), "pointer": pointer, "ordinary_hex": ordinary.hex(), "ordinary_used_pointer": host.last_pointer_execution is not None, "verify": verified}
        v18, v18_public, v18_signer, _ = _fixture(temp / "v18", v18=True); v18_rejected = False
        try: PromptSpanRouteIsolatedShallowSparseCoreHost(temp / "registry-v18", trust_store={v18_signer: v18_public}).activate(v18)
        except Exception: v18_rejected = True
        tensor_shapes = {name: list(value.shape) for name, value in tensors.items() if name.startswith("residual.")}
        expected_candidates = {
            render_prompt_segments(permutation).encode("utf-8").hex()
            for permutation in itertools.permutations(extract_prompt_segments(prompt))
        }
        checks = {
            "cuda_available": torch.cuda.is_available(),
            "canonical_abi_identity": _sha(root / protocol["canonical_abi"]) == ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256,
            "v18_tensor_schema_unchanged": set(tensor_shapes) == {"residual.down", "residual.up", "residual.norm.weight", "residual.norm.bias"},
            "literal_spans_preserved": all(all(segment.encode() in bytes.fromhex(value["generated_hex"]) for segment in extract_prompt_segments(prompt)) for value in executions.values()),
            "exact_candidate_rendering": all(value["generated_hex"] in expected_candidates for value in executions.values()),
            "model_ranked_six_candidates": all(
                value["pointer"].get("candidate_count") == 6
                and value["pointer"].get("evaluator_used") is False
                and value["pointer"].get("selected_index")
                == max(
                    range(6),
                    key=lambda index: (
                        value["pointer"]["model_log_probability_sums"][index],
                        -index,
                    ),
                )
                for value in executions.values()
            ),
            "persistent_prompt_state_reused": all(value["pointer"].get("persistent_prompt_state_reused") is True and value["pointer"].get("candidate_scoring_forward_passes") == 1 for value in executions.values()),
            "physical_one_route": all(value["pointer"].get("active_residual_routes") == 1 for value in executions.values()),
            "cpu_cuda_identity": set(executions) == {"cpu", "cuda"} and len({value["generated_hex"] for value in executions.values()}) == 1,
            "ordinary_fallback_unchanged": all(value["ordinary_hex"] == "" and value["ordinary_used_pointer"] is False for value in executions.values()),
            "same_signed_package_all_devices": len({value["active"]["archive_hash"] for value in executions.values()}) == 1,
            "receiver_learning_zero": all(value["active"]["receiver_training_steps"] == value["active"]["receiver_calibration_runs"] == 0 for value in executions.values()),
            "v18_manifest_rejected": v18_rejected,
            "focused_tests_pass": "7 passed" in focused.stdout,
            "sealed_campaign_unchanged": sealed_result.get("completed_phases_valid") is True,
        }
        result = {"format": "layercake-postrelease-route-isolated-prompt-span-v19-construct-result/1", "status": "PASS_CONSTRUCT_ONLY" if all(checks.values()) else "FAIL", "protocol_sha256": _sha(protocol_path), "interface": ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION, "interface_sha256": ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256, "checks": checks, "devices": executions, "package_sha256": _sha(package), "package_bytes": package.stat().st_size, "new_parameters": 0, "teacher_present": False, "source_transformer_blocks": 0, "historical_release_changed": False, "hardware": {"machine": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, "claim_boundary": "Generic v19 constrained neural decoding construct only; no external artifact, English quality, information minimum, CPU/GPU performance, broad neural-generation replacement, or superiority claim."}
        result["evidence_sha256"] = canonical_json_hash(result); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("execute", "verify")); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(argv)
    root = Path.cwd().resolve(); protocol = root / args.protocol; output = root / args.output; expected = execute(root, protocol)
    if args.command == "execute":
        if output.exists(): raise ConstructError(f"immutable output exists: {output}")
        if expected["status"] != "PASS_CONSTRUCT_ONLY": raise ConstructError("v19 construct failed")
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"); result = expected
    else:
        stored = json.loads(output.read_text(encoding="utf-8"))
        if stored != expected: raise ConstructError("stored v19 construct differs from recomputation")
        result = {"status": "PASS", "evidence_sha256": expected["evidence_sha256"], "construct_only": True}
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
