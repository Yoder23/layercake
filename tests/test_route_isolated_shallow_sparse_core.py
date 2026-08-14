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
from layercake_extensions.route_isolated_shallow_sparse_core_v18 import (
    ARCHITECTURE_V18_FORMAT,
    ROUTE_ISOLATED_CORE_V18_ABI_SHA256,
    ROUTE_ISOLATED_CORE_V18_ABI_VERSION,
    ExactRouteIsolatedShallowSparseCoreHost,
    ExplicitRouteResidual,
)
from layercake_extensions.route_isolated_prompt_span_core_v19 import (
    ARCHITECTURE_V19_FORMAT,
    PROMPT_SPAN_FEATURE,
    ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256,
    ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION,
    PromptSpanRouteIsolatedShallowSparseCoreHost,
    extract_prompt_segments,
)
from layercake_extensions.route_isolated_universal_guard_core_v20 import (
    ARCHITECTURE_V20_FORMAT,
    ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256,
    ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION,
    UNIVERSAL_GUARD_FEATURE,
    UniversalGuardPromptSpanCoreHost,
)
from layercake_extensions.route_isolated_lexical_guard_core_v21 import (
    ARCHITECTURE_V21_FORMAT,
    EXACT_LEXICAL_BOUNDARY,
    EXACT_LEXICAL_GUARD_FEATURE,
    LexicalGuardPromptSpanCoreHost,
    ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256,
    ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION,
    maximal_safe_lexical_prefix,
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
    vocabulary = {"[UNK]": 0}
    vocabulary.update({chr(value): value - 30 for value in range(32, 127)})
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
            "vocab": vocabulary,
            "merges": [],
        },
    }
    return Utf8ConcatenativeBpeTokenizer(raw).canonical_dict()


def _fixture(
    directory: Path,
    *,
    abi_version=ROUTE_ISOLATED_CORE_ABI_VERSION,
    abi_hash=ROUTE_ISOLATED_CORE_ABI_SHA256,
    architecture_format=ARCHITECTURE_FORMAT,
    residual_type=RouteIsolatedResidual,
    router_class=0,
    extra_host_features=(),
    guard_scope="weak_capabilities_only",
    guard_boundary=None,
):
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
        max_tokens=128,
        task_cakes=10,
        task_cake_rank=64,
    )
    model = ShallowSparseEnglishCore(config).eval()
    router_document = _router_tokenizer_document()
    router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(router_document)
    router = SparseCapabilityRouter(router_tokenizer.vocab_size, 32, len(CAPABILITIES) + 1).eval()
    residual = residual_type(16, 16, len(WEAK_CAPABILITIES)).eval()
    with torch.no_grad():
        for module in (model, router, residual):
            for parameter in module.parameters():
                parameter.zero_()
        router.bias[router_class] = 10.0
    architecture = {
        "format": architecture_format,
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
            "scope": guard_scope,
            "stop_before_collapsing_token": True,
            "abstention_markers": ["cannot determine"],
            "abstention_clause": "I cannot determine that from the information given.",
        },
    }
    if guard_boundary is not None:
        architecture["guard"]["boundary"] = guard_boundary
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
        abi_hash=abi_hash,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=architecture,
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "persistent_incremental_state", "physical_route_isolation", "declarative_runtime_guard", "strict_utf8_boundary", *extra_host_features]},
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


def test_v18_accepts_exact_explicit_route_schema_and_v17_rejects_it(tmp_path):
    package, public, signer = _fixture(
        tmp_path,
        abi_version=ROUTE_ISOLATED_CORE_V18_ABI_VERSION,
        abi_hash=ROUTE_ISOLATED_CORE_V18_ABI_SHA256,
        architecture_format=ARCHITECTURE_V18_FORMAT,
        residual_type=ExplicitRouteResidual,
    )
    host = ExactRouteIsolatedShallowSparseCoreHost(
        tmp_path / "registry-v18", trust_store={signer: public}
    )
    active = host.activate(package)
    assert active["status"] == "ACTIVE"
    assert set(host.residual.state_dict()) == {"down", "up", "norm.weight", "norm.bias"}
    assert host.generate("hello", maximum_tokens=2) == b""
    with pytest.raises(Exception):
        RouteIsolatedShallowSparseCoreHost(
            tmp_path / "registry-v17", trust_store={signer: public}
        ).activate(package)


def test_v19_prompt_span_mode_is_model_ranked_literal_and_persistent(tmp_path):
    package, public, signer = _fixture(
        tmp_path,
        abi_version=ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION,
        abi_hash=ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256,
        architecture_format=ARCHITECTURE_V19_FORMAT,
        residual_type=ExplicitRouteResidual,
        router_class=CAPABILITIES.index("coherence"),
        extra_host_features=(PROMPT_SPAN_FEATURE,),
    )
    host = PromptSpanRouteIsolatedShallowSparseCoreHost(
        tmp_path / "registry-v19", trust_store={signer: public}
    )
    host.activate(package)
    prompt = (
        "Return the labels in order without commentary: "
        "[X-START] begin; [X-MIDDLE] continue; [X-END] finish."
    )
    result = host.generate(prompt, maximum_tokens=64).decode("utf-8")
    assert all(segment in result for segment in extract_prompt_segments(prompt))
    assert host.last_pointer_execution is not None
    assert host.last_pointer_execution["candidate_count"] == 6
    assert host.last_pointer_execution["persistent_prompt_state_reused"] is True
    assert host.last_pointer_execution["candidate_scoring_forward_passes"] == 1
    assert host.last_pointer_execution["evaluator_used"] is False


def test_v19_rejects_v18_manifest_and_falls_back_for_ordinary_prompts(tmp_path):
    v18, public, signer = _fixture(
        tmp_path / "v18",
        abi_version=ROUTE_ISOLATED_CORE_V18_ABI_VERSION,
        abi_hash=ROUTE_ISOLATED_CORE_V18_ABI_SHA256,
        architecture_format=ARCHITECTURE_V18_FORMAT,
        residual_type=ExplicitRouteResidual,
    )
    with pytest.raises(Exception):
        PromptSpanRouteIsolatedShallowSparseCoreHost(
            tmp_path / "registry-reject", trust_store={signer: public}
        ).activate(v18)
    v19, public, signer = _fixture(
        tmp_path / "v19",
        abi_version=ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION,
        abi_hash=ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256,
        architecture_format=ARCHITECTURE_V19_FORMAT,
        residual_type=ExplicitRouteResidual,
        extra_host_features=(PROMPT_SPAN_FEATURE,),
    )
    host = PromptSpanRouteIsolatedShallowSparseCoreHost(
        tmp_path / "registry-fallback", trust_store={signer: public}
    )
    host.activate(v19)
    assert host.generate("hello", maximum_tokens=2) == b""
    assert host.last_pointer_execution is None


def test_v20_accepts_only_declared_universal_guard_package(tmp_path):
    weak_only, public, signer = _fixture(
        tmp_path / "weak-only",
        abi_version=ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION,
        abi_hash=ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256,
        architecture_format=ARCHITECTURE_V20_FORMAT,
        residual_type=ExplicitRouteResidual,
        extra_host_features=(PROMPT_SPAN_FEATURE, UNIVERSAL_GUARD_FEATURE),
    )
    with pytest.raises(RouteIsolatedCoreError):
        UniversalGuardPromptSpanCoreHost(
            tmp_path / "registry-weak-only", trust_store={signer: public}
        ).activate(weak_only)
    universal, public, signer = _fixture(
        tmp_path / "universal",
        abi_version=ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_VERSION,
        abi_hash=ROUTE_ISOLATED_UNIVERSAL_GUARD_CORE_V20_ABI_SHA256,
        architecture_format=ARCHITECTURE_V20_FORMAT,
        residual_type=ExplicitRouteResidual,
        extra_host_features=(PROMPT_SPAN_FEATURE, UNIVERSAL_GUARD_FEATURE),
        guard_scope="all_capabilities",
    )
    host = UniversalGuardPromptSpanCoreHost(
        tmp_path / "registry-universal", trust_store={signer: public}
    )
    assert host.activate(universal)["receiver_training_steps"] == 0
    assert host.verify()["status"] == "PASS"


class _RepeatingTokenizer:
    eos_token_id = 0

    @staticmethod
    def decode(values):
        return " ".join("loop" for _ in values)


class _RepeatingModel:
    class _Transformer:
        h = []

    transformer = _Transformer()

    def __call__(self, token, **_kwargs):
        logits = torch.zeros((1, 1, 3))
        logits[..., 2] = 1
        return {"past_key_values": object(), "logits": logits}


@pytest.mark.parametrize("weak_route", [-1, 0])
def test_v20_guard_stops_strong_and_weak_routes_before_collapse(tmp_path, weak_route):
    host = UniversalGuardPromptSpanCoreHost(tmp_path / f"registry-{weak_route}", trust_store={})
    host.model = _RepeatingModel()
    host.router = object()
    host.residual = object()
    host.model_tokenizer = _RepeatingTokenizer()
    host.router_tokenizer = object()
    logits = torch.zeros((1, 3))
    logits[:, 2] = 1
    state = {
        "past_key_values": object(),
        "task_route": torch.tensor([0]),
        "weak_route": weak_route,
        "next_logits": logits,
        "generated_ids": [],
        "terminated_by_guard": False,
        "finished": False,
    }
    assert host.decode_step(state) == 2
    assert host.decode_step(state) == 2
    assert host.decode_step(state) == 2
    assert host.decode_step(state) is None
    assert state["generated_ids"] == [2, 2, 2]
    assert state["terminated_by_guard"] and state["finished"]


def test_v21_exact_lexical_prefix_removes_partial_subtoken():
    value = 'Please update it.FixI.FixI.FixI.FixI'
    assert repetition_collapse(value)
    assert maximal_safe_lexical_prefix(value) == 'Please update it.FixI.FixI.FixI.'


def test_v21_package_requires_lexical_boundary_declaration(tmp_path):
    package, public, signer = _fixture(
        tmp_path / "v21",
        abi_version=ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION,
        abi_hash=ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256,
        architecture_format=ARCHITECTURE_V21_FORMAT,
        residual_type=ExplicitRouteResidual,
        extra_host_features=(
            PROMPT_SPAN_FEATURE,
            UNIVERSAL_GUARD_FEATURE,
            EXACT_LEXICAL_GUARD_FEATURE,
        ),
        guard_scope="all_capabilities",
        guard_boundary=EXACT_LEXICAL_BOUNDARY,
    )
    host = LexicalGuardPromptSpanCoreHost(
        tmp_path / "registry-v21", trust_store={signer: public}
    )
    assert host.activate(package)["receiver_training_steps"] == 0
    missing, public, signer = _fixture(
        tmp_path / "missing",
        abi_version=ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION,
        abi_hash=ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256,
        architecture_format=ARCHITECTURE_V21_FORMAT,
        residual_type=ExplicitRouteResidual,
        extra_host_features=(
            PROMPT_SPAN_FEATURE,
            UNIVERSAL_GUARD_FEATURE,
            EXACT_LEXICAL_GUARD_FEATURE,
        ),
        guard_scope="all_capabilities",
    )
    with pytest.raises(RouteIsolatedCoreError):
        LexicalGuardPromptSpanCoreHost(
            tmp_path / "registry-missing", trust_store={signer: public}
        ).activate(missing)


class _PartialLexicalTokenizer:
    eos_token_id = 0

    @staticmethod
    def decode(values):
        return {
            0: "",
            1: "Please update it.FixI.",
            2: "Please update it.FixI.FixI.",
            3: "Please update it.FixI.FixI.FixI.Fix",
            4: "Please update it.FixI.FixI.FixI.FixI",
        }[len(values)]


@pytest.mark.parametrize("weak_route", [-1, 0])
def test_v21_realizes_exact_lexical_prefix_without_advancing_state(tmp_path, weak_route):
    host = LexicalGuardPromptSpanCoreHost(tmp_path / f"v21-{weak_route}", trust_store={})
    host.model = _RepeatingModel()
    host.router = object()
    host.residual = object()
    host.model_tokenizer = _PartialLexicalTokenizer()
    host.router_tokenizer = object()
    host.guard = {"abstention_markers":["cannot determine"],"abstention_clause":"I cannot determine that from the information given."}
    logits = torch.zeros((1, 3)); logits[:, 2] = 1
    state = {"past_key_values":object(),"task_route":torch.tensor([0]),"capability":"grammar","weak_route":weak_route,"next_logits":logits,"generated_ids":[],"terminated_by_guard":False,"finished":False,"guard_realization_override":None}
    assert host.decode_step(state) == 2
    assert host.decode_step(state) == 2
    assert host.decode_step(state) == 2
    past_before = state["past_key_values"]
    assert host.decode_step(state) is None
    assert state["past_key_values"] is past_before
    assert state["generated_ids"] == [2, 2, 2]
    assert host.realize(state) == b"Please update it.FixI.FixI.FixI."
    assert state["terminated_by_guard"] and state["finished"]
