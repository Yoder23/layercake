"""Construct-certify the v22 exact two-line prompt-literal host."""

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
from layercake.evaluation.route_isolated_lexical_guard_v21_construct import _subtoken_execution
from layercake_extensions.bpe_direct_neural_core import Utf8ConcatenativeBpeTokenizer
from layercake_extensions.route_isolated_format_literal_core_v22 import (
    ARCHITECTURE_V22_FORMAT,
    FORMAT_LITERAL_DECLARATION,
    FORMAT_LITERAL_FEATURE,
    FORMAT_LITERAL_MODE,
    FormatLiteralLexicalGuardCoreHost,
    ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_SHA256,
    ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_VERSION,
)
from layercake_extensions.route_isolated_lexical_guard_core_v21 import (
    ARCHITECTURE_V21_FORMAT,
    EXACT_LEXICAL_BOUNDARY,
    EXACT_LEXICAL_GUARD_FEATURE,
    ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256,
    ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION,
)
from layercake_extensions.route_isolated_prompt_span_core_v19 import (
    PROMPT_SPAN_FEATURE,
    extract_prompt_segments,
    render_prompt_segments,
)
from layercake_extensions.route_isolated_shallow_sparse_core import (
    CAPABILITIES,
    CAPABILITY_TO_TASK_ROUTE,
    WEAK_CAPABILITIES,
    SparseCapabilityRouter,
    repetition_collapse,
)
from layercake_extensions.route_isolated_shallow_sparse_core_v18 import ExplicitRouteResidual
from layercake_extensions.route_isolated_universal_guard_core_v20 import (
    GUARD_PREDICATE,
    UNIVERSAL_GUARD_FEATURE,
)


class ConstructError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys():
    seed = hashlib.sha256(b"layercake-route-isolated-format-literal-v22-construct").digest()
    private = Ed25519PrivateKey.from_private_bytes(seed)
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public, key_id(public)


def _router_document():
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
    capability: str,
    interface: str = "v22",
    declaration: bool = True,
):
    torch.manual_seed(22022)
    vocabulary = {"<eos>": 0, "[UNK]": 1, "hello": 2, "FixI": 3, "Fix": 4, "##I": 5}
    tokenizer = Tokenizer(WordPiece(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder(prefix="##")
    tokenizer_document = json.loads(tokenizer.to_str())
    tokenizer_raw = json.dumps(
        tokenizer_document, sort_keys=True, separators=(",", ":")
    ).encode()
    config = ShallowSparseEnglishConfig(
        vocab_size=len(vocabulary),
        width=16,
        layers=3,
        heads=4,
        max_tokens=128,
        task_cakes=10,
        task_cake_rank=64,
    )
    model = ShallowSparseEnglishCore(config).eval()
    router_document = _router_document()
    router_tokenizer = Utf8ConcatenativeBpeTokenizer.from_document(router_document)
    router = SparseCapabilityRouter(
        router_tokenizer.vocab_size, 32, len(CAPABILITIES) + 1
    ).eval()
    residual = ExplicitRouteResidual(16, 16, len(WEAK_CAPABILITIES)).eval()
    with torch.no_grad():
        for module in (model, router, residual):
            for parameter in module.parameters():
                parameter.zero_()
        router.bias[CAPABILITIES.index(capability)] = 10.0
    guard = {
        "predicate": GUARD_PREDICATE,
        "scope": "all_capabilities",
        "boundary": EXACT_LEXICAL_BOUNDARY,
        "stop_before_collapsing_token": True,
        "abstention_markers": ["cannot determine"],
        "abstention_clause": "I cannot determine that from the information given.",
    }
    v22 = interface == "v22"
    architecture = {
        "format": ARCHITECTURE_V22_FORMAT if v22 else ARCHITECTURE_V21_FORMAT,
        "model": config.canonical_dict(),
        "model_tokenizer": {
            "format": "declarative-tokenizers-json/1",
            "tokenizers_json": tokenizer_document,
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
        "guard": guard,
    }
    if v22 and declaration:
        architecture["format_literal"] = FORMAT_LITERAL_DECLARATION
    tensors = {
        prefix + name: value
        for prefix, state in (
            ("model.", model.state_dict()),
            ("router.", router.state_dict()),
            ("residual.", residual.state_dict()),
        )
        for name, value in state.items()
    }
    private, public, signer = _keys()
    features = [
        "byte_input",
        "safe_tensors",
        "persistent_incremental_state",
        "physical_route_isolation",
        "declarative_runtime_guard",
        "strict_utf8_boundary",
        PROMPT_SPAN_FEATURE,
        UNIVERSAL_GUARD_FEATURE,
        EXACT_LEXICAL_GUARD_FEATURE,
    ]
    if v22:
        features.append(FORMAT_LITERAL_FEATURE)
    manifest = CakeManifest(
        schema_version="1",
        cake_id=f"v22-{capability}-{interface}-{int(declaration)}",
        name="Format-literal construct",
        description="Generic v22 construct fixture",
        version="22.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": signer},
        abi_version=ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_VERSION
        if v22
        else ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_VERSION,
        abi_hash=ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_SHA256
        if v22
        else ROUTE_ISOLATED_LEXICAL_GUARD_CORE_V21_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={
            "external": "UTF-8 bytes",
            "role": "english-core",
            "composition": "direct_core_only_no_router",
            "validity": "strict_utf8",
        },
        architecture=architecture,
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": features},
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
    path = build_package(directory / f"{manifest.cake_id}.cake", manifest, tensors, private_key=private)
    return path, public, signer, tensors


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format")
        != "layercake-postrelease-route-isolated-format-literal-v22-construct/1"
        or protocol.get("status") != "PREREGISTERED_CONSTRUCT_EXECUTION"
    ):
        raise ConstructError("v22 construct protocol changed")
    for relative, expected in protocol["bindings"].items():
        if _sha(root / relative) != expected:
            raise ConstructError(f"v22 construct binding changed: {relative}")
    focused = subprocess.run(
        ["C:\\Python310\\python.exe", "-m", "pytest", "tests/test_route_isolated_shallow_sparse_core.py", "-q"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    complete = subprocess.run(
        ["C:\\Python310\\python.exe", "-m", "pytest", "-q"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sealed = subprocess.run(
        ["C:\\Python310\\python.exe", "-m", "layercake.moonshot_campaign", "verify-all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sealed_result = json.loads(sealed.stdout)
    format_prompt = (
        "Language check: respond exactly as requested.\n"
        "Return exactly two plain-text lines and no Markdown. "
        "The first line must be `item: green notebook` and the second line must be `code: N390098UMA`."
    )
    format_expected = "item: green notebook\ncode: N390098UMA"
    coherence_prompt = (
        "Return the labels in order without commentary: "
        "[X-START] begin; [X-MIDDLE] continue; [X-END] finish."
    )
    with tempfile.TemporaryDirectory(prefix="layercake-format-v22-") as raw:
        temporary = Path(raw)
        format_package, format_public, format_signer, format_tensors = _fixture(
            temporary / "format", capability="format_control"
        )
        v21_package, v21_public, v21_signer, v21_tensors = _fixture(
            temporary / "v21", capability="format_control", interface="v21"
        )
        coherence_package, coherence_public, coherence_signer, _ = _fixture(
            temporary / "coherence", capability="coherence"
        )
        wrong_package, wrong_public, wrong_signer, _ = _fixture(
            temporary / "wrong", capability="grammar"
        )
        missing_package, missing_public, missing_signer, _ = _fixture(
            temporary / "missing", capability="format_control", declaration=False
        )
        executions: dict[str, Any] = {}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            host = FormatLiteralLexicalGuardCoreHost(
                temporary / f"format-{device}",
                trust_store={format_signer: format_public},
                device=device,
            )
            active = host.activate(format_package)
            exact = host.generate(format_prompt, maximum_tokens=32).decode("utf-8")
            format_record = dict(host.last_format_execution or {})
            format_record.pop("wall_seconds", None)
            ordinary = host.generate("hello", maximum_tokens=2).decode("utf-8")
            ordinary_used_format = host.last_format_execution is not None
            extra = host.generate(format_prompt + " Extra `third`.", maximum_tokens=32).decode("utf-8")
            extra_used_format = host.last_format_execution is not None
            over_limit = host.generate(format_prompt, maximum_tokens=1).decode("utf-8")
            over_limit_used_format = host.last_format_execution is not None
            malformed = host.generate(format_prompt.replace("two", "three"), maximum_tokens=32).decode("utf-8")
            malformed_used_format = host.last_format_execution is not None
            verified = host.verify()
            coherence_host = FormatLiteralLexicalGuardCoreHost(
                temporary / f"coherence-{device}",
                trust_store={coherence_signer: coherence_public},
                device=device,
            )
            coherence_active = coherence_host.activate(coherence_package)
            coherence = coherence_host.generate(coherence_prompt, maximum_tokens=64).decode("utf-8")
            pointer = dict(coherence_host.last_pointer_execution or {})
            pointer.pop("wall_seconds", None)
            lexical = _subtoken_execution(coherence_host)
            wrong_host = FormatLiteralLexicalGuardCoreHost(
                temporary / f"wrong-{device}",
                trust_store={wrong_signer: wrong_public},
                device=device,
            )
            wrong_active = wrong_host.activate(wrong_package)
            wrong = wrong_host.generate(format_prompt, maximum_tokens=32).decode("utf-8")
            wrong_used_format = wrong_host.last_format_execution is not None
            executions[device] = {
                "active": active,
                "exact": exact,
                "format_record": format_record,
                "ordinary": ordinary,
                "ordinary_used_format": ordinary_used_format,
                "extra": extra,
                "extra_used_format": extra_used_format,
                "over_limit": over_limit,
                "over_limit_used_format": over_limit_used_format,
                "malformed": malformed,
                "malformed_used_format": malformed_used_format,
                "verified": verified,
                "coherence_active": coherence_active,
                "coherence": coherence,
                "pointer": pointer,
                "lexical": lexical,
                "wrong_active": wrong_active,
                "wrong": wrong,
                "wrong_used_format": wrong_used_format,
            }
        v21_rejected = False
        missing_rejected = False
        try:
            FormatLiteralLexicalGuardCoreHost(
                temporary / "reject-v21", trust_store={v21_signer: v21_public}
            ).activate(v21_package)
        except Exception:
            v21_rejected = True
        try:
            FormatLiteralLexicalGuardCoreHost(
                temporary / "reject-missing", trust_store={missing_signer: missing_public}
            ).activate(missing_package)
        except Exception:
            missing_rejected = True
        coherence_expected = {
            render_prompt_segments(permutation)
            for permutation in itertools.permutations(extract_prompt_segments(coherence_prompt))
        }
        checks = {
            "cuda_available": torch.cuda.is_available(),
            "canonical_abi_identity": _sha(root / protocol["canonical_abi"])
            == ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_SHA256,
            "v21_tensor_schema_and_values_unchanged": set(format_tensors) == set(v21_tensors)
            and all(torch.equal(format_tensors[name], v21_tensors[name]) for name in format_tensors),
            "v21_manifest_rejected": v21_rejected,
            "missing_declaration_rejected": missing_rejected,
            "exact_format_output": all(
                value["exact"] == format_expected for value in executions.values()
            ),
            "format_mode_explicitly_non_neural": all(
                value["format_record"].get("mode") == FORMAT_LITERAL_MODE
                and value["format_record"].get("deterministic_transducer") is True
                and value["format_record"].get("evaluator_used") is False
                and value["format_record"].get("teacher_used") is False
                for value in executions.values()
            ),
            "one_persistent_prefill_zero_decode": all(
                value["format_record"].get("prompt_prefill_forward_passes") == 1
                and value["format_record"].get("decode_forward_passes") == 0
                and value["format_record"].get("candidate_scoring_forward_passes") == 0
                and value["format_record"].get("persistent_prompt_state_created") is True
                and value["format_record"].get("model_state_advanced_after_prefill") is False
                for value in executions.values()
            ),
            "format_strong_path_no_residual": all(
                value["format_record"].get("active_residual_routes") == 0
                for value in executions.values()
            ),
            "ambiguous_and_ordinary_fallback": all(
                value["ordinary"] == value["extra"] == value["over_limit"] == value["malformed"] == ""
                and not value["ordinary_used_format"]
                and not value["extra_used_format"]
                and not value["over_limit_used_format"]
                and not value["malformed_used_format"]
                for value in executions.values()
            ),
            "wrong_capability_fallback": all(
                value["wrong"] == "" and not value["wrong_used_format"]
                for value in executions.values()
            ),
            "v19_coherence_pointer_preserved": all(
                value["coherence"] in coherence_expected
                and value["pointer"].get("candidate_count") == 6
                and value["pointer"].get("candidate_scoring_forward_passes") == 1
                and value["pointer"].get("persistent_prompt_state_reused") is True
                and value["pointer"].get("evaluator_used") is False
                for value in executions.values()
            ),
            "v21_lexical_guard_preserved": all(
                value["lexical"]["candidate_collapses"]
                and not value["lexical"]["output_collapses"]
                and value["lexical"]["output_is_candidate_prefix"]
                and value["lexical"]["model_state_identity"]
                for value in executions.values()
            ),
            "cpu_cuda_identity": set(executions) == {"cpu", "cuda"}
            and len(
                {
                    (
                        value["exact"],
                        value["ordinary"],
                        value["extra"],
                        value["over_limit"],
                        value["malformed"],
                        value["coherence"],
                        value["wrong"],
                        value["lexical"]["output"],
                    )
                    for value in executions.values()
                }
            )
            == 1,
            "same_signed_packages_all_devices": len(
                {value["active"]["archive_hash"] for value in executions.values()}
            )
            == 1
            and len(
                {value["coherence_active"]["archive_hash"] for value in executions.values()}
            )
            == 1,
            "receiver_learning_zero": all(
                value[name]["receiver_training_steps"]
                == value[name]["receiver_calibration_runs"]
                == 0
                for value in executions.values()
                for name in ("active", "coherence_active", "wrong_active")
            ),
            "focused_tests_pass": "18 passed" in focused.stdout,
            "complete_tests_pass": complete.returncode == 0,
            "sealed_campaign_unchanged": sealed_result.get("completed_phases_valid") is True,
        }
        result = {
            "format": "layercake-postrelease-route-isolated-format-literal-v22-construct-result/1",
            "status": "PASS_CONSTRUCT_ONLY" if all(checks.values()) else "FAIL",
            "protocol_sha256": _sha(protocol_path),
            "interface": ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_VERSION,
            "interface_sha256": ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_SHA256,
            "checks": checks,
            "devices": executions,
            "packages": {
                "format_sha256": _sha(format_package),
                "coherence_sha256": _sha(coherence_package),
                "wrong_capability_sha256": _sha(wrong_package),
            },
            "new_parameters": 0,
            "teacher_present": False,
            "source_transformer_blocks": 0,
            "historical_release_changed": False,
            "hardware": {
                "machine": platform.node(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "claim_boundary": (
                "Generic v22 format-literal host construct only. The narrow path is a "
                "deterministic prompt transducer, not broad neural generation. No external "
                "artifact, quality, information minimum, physical performance, phase, or "
                "superiority claim."
            ),
        }
        result["evidence_sha256"] = canonical_json_hash(result)
        return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    expected = execute(root, root / args.protocol)
    output = root / args.output
    if args.command == "execute":
        if output.exists():
            raise ConstructError(f"immutable output exists: {output}")
        if expected["status"] != "PASS_CONSTRUCT_ONLY":
            raise ConstructError(f"v22 construct failed: {expected['checks']}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result = expected
    else:
        stored = json.loads(output.read_text(encoding="utf-8"))
        if stored != expected:
            raise ConstructError("stored v22 construct differs from recomputation")
        result = {
            "status": "PASS",
            "evidence_sha256": expected["evidence_sha256"],
            "construct_only": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
