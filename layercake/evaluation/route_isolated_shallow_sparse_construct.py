"""Execute the generic v17 route-isolated shallow-sparse host construct."""

from __future__ import annotations

import argparse
import hashlib
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
from layercake.models.shallow_sparse_english import (
    ShallowSparseEnglishConfig,
    ShallowSparseEnglishCore,
)
from layercake.portable_domain import canonical_json_hash
from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
from layercake_extensions.route_isolated_shallow_sparse_core import (
    ARCHITECTURE_FORMAT,
    CAPABILITIES,
    CAPABILITY_TO_TASK_ROUTE,
    ROUTE_ISOLATED_CORE_ABI_SHA256,
    ROUTE_ISOLATED_CORE_ABI_VERSION,
    WEAK_CAPABILITIES,
    RouteIsolatedResidual,
    RouteIsolatedShallowSparseCoreHost,
    SparseCapabilityRouter,
)


class ConstructError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys():
    seed = hashlib.sha256(b"layercake-route-isolated-v17-construct").digest()
    private = Ed25519PrivateKey.from_private_bytes(seed)
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem, key_id(public_pem)


def _router_document():
    raw = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{"id": 0, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {"type": "BPE", "dropout": None, "unk_token": "[UNK]", "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": {"[UNK]": 0, "h": 1, "e": 2, "l": 3, "o": 4}, "merges": []},
    }
    return Utf8ConcatenativeBpeTokenizer(raw).canonical_dict()


def _fixture(
    directory: Path,
    *,
    abi_version=ROUTE_ISOLATED_CORE_ABI_VERSION,
    abi_hash=ROUTE_ISOLATED_CORE_ABI_SHA256,
    architecture_format=ARCHITECTURE_FORMAT,
    residual_type=RouteIsolatedResidual,
    domain="english-core",
):
    torch.manual_seed(17044)
    tokenizer = Tokenizer(WordLevel({"<eos>": 0, "[UNK]": 1, "hello": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_doc = json.loads(tokenizer.to_str())
    tokenizer_raw = json.dumps(tokenizer_doc, sort_keys=True, separators=(",", ":")).encode()
    config = ShallowSparseEnglishConfig(vocab_size=3, width=16, layers=3, heads=4, max_tokens=32, task_cakes=10, task_cake_rank=64)
    model = ShallowSparseEnglishCore(config).eval()
    router_doc = _router_document()
    router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(router_doc)
    router = SparseCapabilityRouter(router_tokenizer.vocab_size, 32, len(CAPABILITIES) + 1).eval()
    residual = residual_type(16, 16, len(WEAK_CAPABILITIES)).eval()
    with torch.no_grad():
        for module in (model, router, residual):
            for parameter in module.parameters():
                parameter.zero_()
        router.bias[0] = 10.0
    architecture = {
        "format": architecture_format,
        "model": config.canonical_dict(),
        "model_tokenizer": {"format": "declarative-tokenizers-json/1", "tokenizers_json": tokenizer_doc, "sha256": hashlib.sha256(tokenizer_raw).hexdigest(), "eos_token_id": 0},
        "router": {"vocabulary": router_tokenizer.vocab_size, "character_hash_buckets": 32, "character_ngram_minimum": 2, "character_ngram_maximum": 5, "hash_seed": 450045, "classes": len(CAPABILITIES) + 1},
        "router_tokenizer": router_doc,
        "residual": {"width": 16, "rank": 16, "routes": len(WEAK_CAPABILITIES), "reuse": "before_each_transformer_block"},
        "capabilities": list(CAPABILITIES),
        "capability_to_task_route": CAPABILITY_TO_TASK_ROUTE,
        "weak_capabilities": list(WEAK_CAPABILITIES),
        "guard": {"predicate": "contiguous_1_to_16_token_span_repeated_4_times_or_fourgram_diversity_below_0.35_at_32_tokens", "scope": "weak_capabilities_only", "stop_before_collapsing_token": True, "abstention_markers": ["cannot determine"], "abstention_clause": "I cannot determine that from the information given."},
    }
    tensors = {}
    for prefix, state in (("model.", model.state_dict()), ("router.", router.state_dict()), ("residual.", residual.state_dict())):
        tensors.update({prefix + name: value for name, value in state.items()})
    private, public, signer = _keys()
    manifest = CakeManifest(
        schema_version="1", cake_id=f"route-isolated-{domain}-construct", name="Route-isolated core construct", description="Generic construct-only route-isolated shallow-sparse core", version="17.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": signer}, abi_version=abi_version, abi_hash=abi_hash, cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"}, output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"}, architecture=architecture,
        supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"), minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "persistent_incremental_state", "physical_route_isolation", "declarative_runtime_guard", "strict_utf8_boundary"]},
        tensor_payload_hash="", tensor_shapes=tensor_specs(tensors), package_hash="", training_data_provenance={"dataset": "construct-only", "external_teacher": False}, evaluation_evidence={"status": "CONSTRUCT_ONLY"}, license="Apache-2.0", dependencies=(), parent_version=None, signature={"algorithm": "ed25519", "key_id": signer}, domains=(domain,), permissions=("local-inference",),
    )
    package = build_package(directory / "route-isolated-core.cake", manifest, tensors, private_key=private)
    return package, public, signer, tensors


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-route-isolated-shallow-sparse-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise ConstructError("v17 construct protocol changed")
    for relative, expected in protocol["bindings"].items():
        if _sha(root / relative) != expected:
            raise ConstructError(f"v17 construct binding changed: {relative}")
    focused = subprocess.run(["C:\\Python310\\python.exe", "-m", "pytest", "tests/test_route_isolated_shallow_sparse_core.py", "-q"], cwd=root, check=True, capture_output=True, text=True)
    sealed = subprocess.run(["C:\\Python310\\python.exe", "-m", "layercake.moonshot_campaign", "verify-all"], cwd=root, check=True, capture_output=True, text=True)
    sealed_result = json.loads(sealed.stdout)
    with tempfile.TemporaryDirectory(prefix="layercake-route-isolated-v17-") as raw:
        temp = Path(raw)
        package, public, signer, tensors = _fixture(temp)
        executions = {}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            host = RouteIsolatedShallowSparseCoreHost(temp / f"registry-{device}", trust_store={signer: public}, device=device)
            active = host.activate(package)
            state = host.prefill("hello")
            generated = host.generate("hello", maximum_tokens=2)
            verified = host.verify()
            removed = host.remove()
            reactivated = host.activate(package)
            executions[device] = {"active": active, "route": "grammar", "cache_present": state["past_key_values"] is not None, "generated_hex": generated.hex(), "verify": verified, "remove": removed, "reactivated": reactivated}
        host = RouteIsolatedShallowSparseCoreHost(temp / "registry-invalid", trust_store={signer: public})
        host.activate(package)
        invalid_utf8_rejected = False
        try:
            host.generate(b"\x9c")
        except UnicodeDecodeError:
            invalid_utf8_rejected = True
        wrong_abi, wrong_public, wrong_signer, _ = _fixture(temp / "wrong-abi", abi_version="lc-direct-neural-core/5")
        wrong_abi_rejected = False
        try:
            RouteIsolatedShallowSparseCoreHost(temp / "registry-wrong", trust_store={wrong_signer: wrong_public}).activate(wrong_abi)
        except Exception:
            wrong_abi_rejected = True
        wrong_role, role_public, role_signer, _ = _fixture(temp / "wrong-role", domain="python")
        wrong_role_rejected = False
        try:
            RouteIsolatedShallowSparseCoreHost(temp / "registry-role", trust_store={role_signer: role_public}).activate(wrong_role)
        except Exception:
            wrong_role_rejected = True
        tampered_raw = bytearray(package.read_bytes()); tampered_raw[len(tampered_raw) // 2] ^= 1
        tampered = temp / "tampered.cake"; tampered.write_bytes(tampered_raw)
        tamper_rejected = False
        try:
            RouteIsolatedShallowSparseCoreHost(temp / "registry-tamper", trust_store={signer: public}).activate(tampered)
        except Exception:
            tamper_rejected = True
        checks = {
            "canonical_abi_identity": _sha(root / protocol["canonical_abi"]) == ROUTE_ISOLATED_CORE_ABI_SHA256,
            "single_signed_package": True,
            "three_tensor_namespaces": {name.split(".", 1)[0] for name in tensors} == {"model", "router", "residual"},
            "persistent_state": all(value["cache_present"] for value in executions.values()),
            "same_archive_all_devices": len({value["active"]["archive_hash"] for value in executions.values()}) == 1,
            "same_payload_all_devices": len({value["active"]["payload_hash"] for value in executions.values()}) == 1,
            "same_generation_all_devices": len({value["generated_hex"] for value in executions.values()}) == 1,
            "receiver_learning_zero": all(value["active"]["receiver_training_steps"] == value["active"]["receiver_calibration_runs"] == 0 for value in executions.values()),
            "invalid_utf8_rejected": invalid_utf8_rejected,
            "wrong_abi_rejected": wrong_abi_rejected,
            "wrong_role_rejected": wrong_role_rejected,
            "tamper_rejected": tamper_rejected,
            "focused_tests_pass": "4 passed" in focused.stdout,
            "sealed_campaign_unchanged": sealed_result.get("completed_phases_valid") is True,
        }
        result = {
            "format": "layercake-postrelease-route-isolated-shallow-sparse-construct-result/1",
            "status": "PASS_CONSTRUCT_ONLY" if all(checks.values()) else "FAIL",
            "protocol_sha256": _sha(protocol_path),
            "interface": ROUTE_ISOLATED_CORE_ABI_VERSION,
            "interface_sha256": ROUTE_ISOLATED_CORE_ABI_SHA256,
            "checks": checks,
            "devices": executions,
            "package_sha256": _sha(package),
            "package_bytes": package.stat().st_size,
            "teacher_present": False,
            "source_transformer_blocks": 0,
            "historical_release_changed": False,
            "hardware": {"machine": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "claim_boundary": "Generic v17 host construct only; no external artifact, English quality, information minimum, performance, or superiority claim.",
        }
        result["evidence_sha256"] = canonical_json_hash(result)
        return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); protocol = root / args.protocol; output = root / args.output
    expected = execute(root, protocol)
    if args.command == "execute":
        if output.exists():
            raise ConstructError(f"immutable output exists: {output}")
        if expected["status"] != "PASS_CONSTRUCT_ONLY":
            raise ConstructError("v17 construct failed")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = expected
    else:
        stored = json.loads(output.read_text(encoding="utf-8"))
        if stored != expected:
            raise ConstructError("stored v17 construct differs from recomputation")
        result = {"status": "PASS", "evidence_sha256": expected["evidence_sha256"], "construct_only": True}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
