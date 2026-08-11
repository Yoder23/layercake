import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest
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
from layercake_extensions.bpe_direct_neural_core import (
    Utf8ConcatenativeBpeTokenizer,
)
from layercake_extensions.route_isolated_shallow_sparse_core import (
    ARCHITECTURE_FORMAT,
    CAPABILITIES,
    CAPABILITY_TO_TASK_ROUTE,
    ROUTE_ISOLATED_CORE_ABI_SHA256,
    ROUTE_ISOLATED_CORE_ABI_VERSION,
    WEAK_CAPABILITIES,
    RouteIsolatedCoreError,
    RouteIsolatedResidual,
    RouteIsolatedShallowSparseCoreHost,
    SparseCapabilityRouter,
    repetition_collapse,
)


def _keys():
    seed = hashlib.sha256(b"route-isolated-shallow-sparse-v17-tests").digest()
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


def _router_tokenizer_document():
    raw = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "[UNK]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {"[UNK]": 0, "h": 1, "e": 2, "l": 3, "o": 4},
            "merges": [],
        },
    }
    return Utf8ConcatenativeBpeTokenizer(raw).canonical_dict()


def _fixture(directory: Path, *, abi_version=ROUTE_ISOLATED_CORE_ABI_VERSION):
    torch.manual_seed(17017)
    tokenizer = Tokenizer(WordLevel({"<eos>": 0, "[UNK]": 1, "hello": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_doc = json.loads(tokenizer.to_str())
    tokenizer_raw = json.dumps(tokenizer_doc, sort_keys=True, separators=(",", ":")).encode()
    config = ShallowSparseEnglishConfig(
        vocab_size=3,
        width=16,
        layers=3,
        heads=4,
        max_tokens=32,
        task_cakes=10,
        task_cake_rank=64,
    )
    model = ShallowSparseEnglishCore(config).eval()
    router_document = _router_tokenizer_document()
    router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(router_document)
    router = SparseCapabilityRouter(router_tokenizer.vocab_size, 32, len(CAPABILITIES) + 1).eval()
    residual = RouteIsolatedResidual(16, 16, len(WEAK_CAPABILITIES)).eval()
    with torch.no_grad():
        for module in (model, router, residual):
            for parameter in module.parameters():
                parameter.zero_()
        router.bias[0] = 10.0
    architecture = {
        "format": ARCHITECTURE_FORMAT,
        "model": config.canonical_dict(),
        "model_tokenizer": {
            "format": "declarative-tokenizers-json/1",
            "tokenizers_json": tokenizer_doc,
            "sha256": hashlib.sha256(tokenizer_raw).hexdigest(),
            "eos_token_id": 0,
        },
        "router": {
            "vocabulary": router_tokenizer.vocab_size,
            "character_hash_buckets": 32,
            "character_ngram_minimum": 2,
            "character_ngram_maximum": 5,
            "hash_seed": 450045,
            "classes": len(CAPABILITIES) + 1,
        },
        "router_tokenizer": router_document,
        "residual": {
            "width": 16,
            "rank": 16,
            "routes": len(WEAK_CAPABILITIES),
            "reuse": "before_each_transformer_block",
        },
        "capabilities": list(CAPABILITIES),
        "capability_to_task_route": CAPABILITY_TO_TASK_ROUTE,
        "weak_capabilities": list(WEAK_CAPABILITIES),
        "guard": {
            "predicate": "contiguous_1_to_16_token_span_repeated_4_times_or_fourgram_diversity_below_0.35_at_32_tokens",
            "scope": "weak_capabilities_only",
            "stop_before_collapsing_token": True,
            "abstention_markers": ["cannot determine"],
            "abstention_clause": "I cannot determine that from the information given.",
        },
    }
    tensors = {}
    for prefix, state in (
        ("model.", model.state_dict()),
        ("router.", router.state_dict()),
        ("residual.", residual.state_dict()),
    ):
        tensors.update({prefix + name: value for name, value in state.items()})
    private, public, signer = _keys()
    manifest = CakeManifest(
        schema_version="1",
        cake_id="route-isolated-construct",
        name="Route-isolated construct",
        description="Construct-only guarded shallow sparse core",
        version="17.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": signer},
        abi_version=abi_version,
        abi_hash=ROUTE_ISOLATED_CORE_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=architecture,
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "persistent_incremental_state", "physical_route_isolation", "declarative_runtime_guard", "strict_utf8_boundary"]},
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(tensors),
        package_hash="",
        training_data_provenance={"dataset": "construct-only", "external_teacher": False},
        evaluation_evidence={"status": "CONSTRUCT_ONLY"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=("english-core",),
        permissions=("local-inference",),
    )
    path = build_package(directory / "core.cake", manifest, tensors, private_key=private)
    return path, public, signer


def test_signed_package_lifecycle_and_persistent_state(tmp_path):
    package, public, signer = _fixture(tmp_path)
    host = RouteIsolatedShallowSparseCoreHost(tmp_path / "registry", trust_store={signer: public})
    active = host.activate(package)
    assert active["receiver_training_steps"] == 0
    assert host.route("hello") == "grammar"
    state = host.prefill("hello")
    assert state["past_key_values"] is not None
    assert host.generate("hello", maximum_tokens=2) == b""
    assert host.verify()["status"] == "PASS"
    assert host.remove()["status"] == "REMOVED"


def test_wrong_abi_and_invalid_utf8_fail_closed(tmp_path):
    package, public, signer = _fixture(tmp_path, abi_version="lc-direct-neural-core/5")
    host = RouteIsolatedShallowSparseCoreHost(tmp_path / "registry", trust_store={signer: public})
    with pytest.raises(Exception):
        host.activate(package)
    valid, public, signer = _fixture(tmp_path / "valid")
    host = RouteIsolatedShallowSparseCoreHost(tmp_path / "registry-valid", trust_store={signer: public})
    host.activate(valid)
    with pytest.raises(UnicodeDecodeError):
        host.generate(b"\x9c")


def test_route_residual_executes_only_selected_slice():
    residual = RouteIsolatedResidual(16, 16, 4)
    with torch.no_grad():
        for parameter in residual.parameters():
            parameter.zero_()
        residual.norm.weight.fill_(1.0)
        residual.down.weight[:16].fill_(0.1)
        residual.up.weight[:, :16].fill_(0.1)
    hidden = torch.arange(32, dtype=torch.float32).reshape(2, 1, 16)
    output = residual.delta(hidden, torch.tensor([0, 1]))
    assert not torch.equal(output[0], torch.zeros_like(output[0]))
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_repetition_guard_contract():
    assert repetition_collapse("loop loop loop loop")
    assert not repetition_collapse("This concise answer has no repeated local loop.")
