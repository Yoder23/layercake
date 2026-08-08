"""Post-release decoder-aware external-token direct English-core host ABI v4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from tokenizers import Tokenizer

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage
from layercake.cake.registry import CakeRegistry
from layercake.portable_token_plan import BOS_ID, EOS_ID, PAD_ID, SPECIAL_COUNT, UNK_ID, PortableTokenPlan
from layercake_extensions.unicode_direct_neural_core import (
    DIRECT_NEURAL_CORE_COMPOSITION,
    DIRECT_NEURAL_CORE_ROLE,
    UnicodeDirectNeuralCoreError,
    UnicodeSafeDirectNeuralCoreHost,
)


DECODER_DIRECT_NEURAL_CORE_ABI_VERSION = "lc-direct-neural-core/4"
DECODER_DIRECT_NEURAL_CORE_ABI_SHA256 = "ad51284179bc8aa379a5b168746456882b8720114784c077c2642cf27e4c1745"
DECODER_TOKENIZER_FORMAT = "layercake-decoder-aware-external-tokenizer/1"
DECODER_TOKEN_PLAN_FORMAT = "layercake-decoder-aware-token-plan/1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class DecoderAwareExternalTokenizer:
    """Collision-free wrapper over a declarative tokenizers JSON graph."""

    def __init__(self, document: Mapping[str, Any]):
        allowed = {"version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder", "model"}
        model = document.get("model", {})
        if set(document) != allowed or document.get("truncation") is not None or document.get("padding") is not None:
            raise ValueError("external tokenizer document shape changed")
        if document.get("normalizer") is None or document.get("decoder") is None:
            raise ValueError("decoder-aware tokenizer requires normalization and decoding")
        if document.get("pre_tokenizer") is not None:
            raise ValueError("v4 does not admit an external pre-tokenizer")
        post = document.get("post_processor")
        if post is not None and (post.get("type") != "TemplateProcessing" or post.get("special_tokens") not in ({}, [])):
            raise ValueError("v4 post-processor must be identity TemplateProcessing")
        if model.get("type") != "BPE" or model.get("dropout") is not None or model.get("continuing_subword_prefix") is not None or model.get("end_of_word_suffix") is not None:
            raise ValueError("unsupported external BPE model")
        vocab = model.get("vocab")
        added = document.get("added_tokens")
        if not isinstance(vocab, dict) or not vocab or not isinstance(added, list):
            raise ValueError("external tokenizer action inventory is incomplete")
        ids = [int(value) for value in vocab.values()]
        ids.extend(int(token["id"]) for token in added)
        if min(ids) != 0 or set(range(max(ids) + 1)) - set(ids):
            raise ValueError("external tokenizer IDs must be dense from zero")
        self.document = json.loads(json.dumps(document, sort_keys=True))
        self.external_action_count = max(ids) + 1
        self.fixed_lexemes = tuple(str(index).encode("ascii") for index in range(self.external_action_count))
        self.backend = Tokenizer.from_str(json.dumps(self.document, ensure_ascii=False))

    @property
    def vocab_size(self) -> int:
        return SPECIAL_COUNT + self.external_action_count

    def encode_source(self, value: bytes | str) -> tuple[list[int], list[bytes]]:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="strict")
        elif isinstance(value, str):
            text = value
        else:
            raise TypeError("external tokenizer input must be bytes or str")
        encoded = self.backend.encode(text, add_special_tokens=False)
        if self.backend.decode(encoded.ids, skip_special_tokens=False) != text:
            raise ValueError("external tokenizer cannot round-trip this prompt exactly")
        return [SPECIAL_COUNT + int(value) for value in encoded.ids], [token.encode("utf-8") for token in encoded.tokens]

    def encode_fixed_target(self, value: bytes | str) -> list[int]:
        text = value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value
        encoded = self.backend.encode(text, add_special_tokens=False)
        if self.backend.decode(encoded.ids, skip_special_tokens=False) != text:
            raise ValueError("external tokenizer cannot round-trip this target exactly")
        return [SPECIAL_COUNT + int(value) for value in encoded.ids] + [EOS_ID]

    def decode_actions(self, actions: Iterable[int], source_lexemes: list[bytes]) -> bytes:
        del source_lexemes
        external: list[int] = []
        for raw in actions:
            action = int(raw)
            if action == EOS_ID:
                break
            if action in {PAD_ID, BOS_ID, UNK_ID}:
                raise ValueError("host special action cannot be emitted")
            if action >= self.vocab_size:
                raise ValueError("pointer actions are prohibited by v4")
            external.append(action - SPECIAL_COUNT)
        text = self.backend.decode(external, skip_special_tokens=False)
        return text.encode("utf-8", errors="strict")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format": DECODER_TOKENIZER_FORMAT,
            "tokenizers_json": self.document,
            "tokenizers_json_sha256": hashlib.sha256(_canonical(self.document)).hexdigest(),
            "external_id_offset": SPECIAL_COUNT,
            "host_special_ids": {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unknown_source": UNK_ID},
            "fixed_actions_decode_as_one_sequence": True,
            "pointer_actions": "PROHIBITED",
            "external_input_output": "UTF-8 bytes",
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "DecoderAwareExternalTokenizer":
        allowed = {"format", "tokenizers_json", "tokenizers_json_sha256", "external_id_offset", "host_special_ids", "fixed_actions_decode_as_one_sequence", "pointer_actions", "external_input_output"}
        if (
            set(document) != allowed
            or document.get("format") != DECODER_TOKENIZER_FORMAT
            or document.get("external_id_offset") != SPECIAL_COUNT
            or document.get("host_special_ids") != {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unknown_source": UNK_ID}
            or document.get("fixed_actions_decode_as_one_sequence") is not True
            or document.get("pointer_actions") != "PROHIBITED"
            or document.get("external_input_output") != "UTF-8 bytes"
            or hashlib.sha256(_canonical(document.get("tokenizers_json"))).hexdigest() != document.get("tokenizers_json_sha256")
        ):
            raise ValueError("decoder-aware tokenizer document changed")
        value = cls(document["tokenizers_json"])
        if value.canonical_dict() != document:
            raise ValueError("decoder-aware tokenizer identity changed")
        return value

    def hash(self) -> str:
        return hashlib.sha256(_canonical(self.canonical_dict())).hexdigest()


def decoder_token_plan_manifest_architecture(model: PortableTokenPlan, tokenizer: DecoderAwareExternalTokenizer) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("model and decoder-aware tokenizer sizes differ")
    return {
        "name": "decoder_aware_portable_token_plan",
        "format": DECODER_TOKEN_PLAN_FORMAT,
        "model": model.canonical_config(),
        "tokenizer": tokenizer.canonical_dict(),
        "tokenizer_sha256": tokenizer.hash(),
        "external_input_output": "UTF-8 bytes",
        "private_representation": "portable_token_plan_fixed_action_transformer",
        "action_validity": "fixed_sequence_decode_strict_utf8_pointer_prohibited",
    }


class DecoderAwareDirectNeuralCoreHost(UnicodeSafeDirectNeuralCoreHost):
    """Install and execute one v4 decoder-aware English core."""

    def __init__(self, registry_root: str | Path, *, trust_store: Mapping[str, bytes | str | Path], device: str | torch.device = "cpu"):
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=DECODER_DIRECT_NEURAL_CORE_ABI_VERSION,
                abi_hash=DECODER_DIRECT_NEURAL_CORE_ABI_SHA256,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=frozenset({"byte_input", "safe_tensors", "persistent_incremental_state", "strict_utf8_boundary", "external_tokenizer_decoder", "fixed_actions_only"}),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )
        self.device = torch.device(device)
        self.module = None
        self.active_cake_id = None
        self.active_archive_hash = None
        self.active_payload_hash = None
        self.receiver_training_steps = 0
        self.receiver_calibration_runs = 0

    @staticmethod
    def _validate_role(package: CakePackage) -> None:
        manifest = package.manifest
        if not package.signed or manifest.cake_type != "portable_decoder" or manifest.abi_version != DECODER_DIRECT_NEURAL_CORE_ABI_VERSION or manifest.abi_hash != DECODER_DIRECT_NEURAL_CORE_ABI_SHA256:
            raise UnicodeDirectNeuralCoreError("decoder-aware direct core identity mismatch")
        if manifest.domains != (DIRECT_NEURAL_CORE_ROLE,) or manifest.dependencies:
            raise UnicodeDirectNeuralCoreError("package is not an exclusive English core")
        if manifest.input_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "validity": "strict_utf8"}:
            raise UnicodeDirectNeuralCoreError("decoder-aware input contract mismatch")
        if manifest.output_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "composition": DIRECT_NEURAL_CORE_COMPOSITION, "validity": "strict_utf8"}:
            raise UnicodeDirectNeuralCoreError("decoder-aware output contract mismatch")
        architecture = manifest.architecture
        if architecture.get("name") != "decoder_aware_portable_token_plan" or architecture.get("format") != DECODER_TOKEN_PLAN_FORMAT or architecture.get("action_validity") != "fixed_sequence_decode_strict_utf8_pointer_prohibited":
            raise UnicodeDirectNeuralCoreError("decoder-aware architecture mismatch")
        required = set(manifest.minimum_host_capabilities.get("features", []))
        expected = {"byte_input", "safe_tensors", "persistent_incremental_state", "strict_utf8_boundary", "external_tokenizer_decoder", "fixed_actions_only"}
        if not expected <= required:
            raise UnicodeDirectNeuralCoreError("decoder-aware host capabilities are incomplete")

    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> PortableTokenPlan:
        architecture = package.manifest.architecture
        allowed = {"name", "format", "model", "tokenizer", "tokenizer_sha256", "external_input_output", "private_representation", "action_validity"}
        if set(architecture) != allowed or architecture["external_input_output"] != "UTF-8 bytes" or architecture["private_representation"] != "portable_token_plan_fixed_action_transformer":
            raise UnicodeDirectNeuralCoreError("decoder-aware token-plan metadata is incomplete")
        tokenizer = DecoderAwareExternalTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]:
            raise UnicodeDirectNeuralCoreError("decoder-aware tokenizer hash mismatch")
        model = PortableTokenPlan(**architecture["model"]).bind_tokenizer(tokenizer)
        model.load_state_dict(package.tensors, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model
