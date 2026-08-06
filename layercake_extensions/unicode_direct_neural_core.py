"""Post-release Unicode-atomic direct neural English-core host ABI v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import CakePackage, load_package
from layercake.cake.registry import CakeRegistry
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import BOS_ID, EOS_ID, PAD_ID, SPECIAL_COUNT, UNK_ID, LosslessLexemePointerTokenizer, PortableTokenPlan


UNICODE_DIRECT_NEURAL_CORE_ABI_VERSION = "lc-direct-neural-core/2"
UNICODE_DIRECT_NEURAL_CORE_ABI_SHA256 = "d34ba8a41815b06820988f501aab920adef627b6929a058cc9fe40528c92624f"
UNICODE_TOKENIZER_FORMAT = "layercake-unicode-atomic-lexeme-pointer/1"
UNICODE_TOKEN_PLAN_FORMAT = "layercake-unicode-atomic-token-plan/1"
UNICODE_LEXEME_PATTERN = r"[^\W\d]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|//|\*\*|->|\s+|."
UNICODE_LEXEME_REGEX = re.compile(UNICODE_LEXEME_PATTERN, re.DOTALL | re.UNICODE)
DIRECT_NEURAL_CORE_ROLE = "english-core"
DIRECT_NEURAL_CORE_COMPOSITION = "direct_core_only_no_router"


class UnicodeDirectNeuralCoreError(ValueError):
    """Raised when a v2 package or UTF-8 boundary is invalid."""


class UnicodeAtomicLexemePointerTokenizer(LosslessLexemePointerTokenizer):
    """Lossless lexemes where every selectable action is complete UTF-8."""

    def __init__(self, fixed_lexemes: Iterable[bytes], *, format_version: str = UNICODE_TOKENIZER_FORMAT) -> None:
        values = tuple(fixed_lexemes)
        if format_version != UNICODE_TOKENIZER_FORMAT:
            raise ValueError("unsupported Unicode-atomic tokenizer format")
        if any(not value for value in values) or len(values) != len(set(values)) or tuple(sorted(values)) != values:
            raise ValueError("fixed Unicode lexemes must be non-empty, unique, and byte-sorted")
        for value in values:
            value.decode("utf-8", errors="strict")
        self.fixed_lexemes = values
        self.format_version = format_version
        self.lexeme_to_id = {value: index for index, value in enumerate(values, start=SPECIAL_COUNT)}
        self.id_to_lexeme = {index: value for index, value in enumerate(values, start=SPECIAL_COUNT)}

    @staticmethod
    def split(value: bytes | str) -> list[bytes]:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="strict")
        elif isinstance(value, str):
            text = value
        else:
            raise TypeError("Unicode tokenizer input must be bytes or str")
        if not text:
            return []
        pieces = UNICODE_LEXEME_REGEX.findall(text)
        if "".join(pieces) != text:
            raise RuntimeError("Unicode lexeme representation is not lossless")
        encoded = [piece.encode("utf-8") for piece in pieces]
        for piece in encoded:
            piece.decode("utf-8", errors="strict")
        return encoded

    @classmethod
    def build_generic(cls, rows: Iterable[Mapping[str, Any]]) -> "UnicodeAtomicLexemePointerTokenizer":
        rows = list(rows)
        excluded: set[bytes] = set()
        for row in rows:
            copy_lexemes = row.get("copy_lexemes")
            if not isinstance(copy_lexemes, list) or not copy_lexemes or any(not isinstance(value, str) or not value for value in copy_lexemes):
                raise ValueError("generic rows require non-empty string copy_lexemes")
            for value in copy_lexemes:
                pieces = cls.split(value)
                if pieces != [value.encode("utf-8")]:
                    raise ValueError("each copy lexeme must be exactly one Unicode lexeme")
                excluded.add(pieces[0])
        values: set[bytes] = set()
        for row in rows:
            for field in ("prompt", "response"):
                value = row.get(field)
                if not isinstance(value, str):
                    raise ValueError(f"generic row {field} must be a string")
                values.update(piece for piece in cls.split(value) if piece not in excluded)
        return cls(sorted(values))

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format": UNICODE_TOKENIZER_FORMAT,
            "unicode_lexeme_pattern": UNICODE_LEXEME_PATTERN,
            "unicode_normalization": "PRESERVE_EXACT_NO_NORMALIZATION",
            "special_ids": {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unknown_source": UNK_ID},
            "fixed_lexemes_hex": [value.hex() for value in self.fixed_lexemes],
            "fixed_vocabulary_order": "ascending_utf8_bytes",
            "every_action_valid_utf8": True,
            "external_input_output": "UTF-8 bytes",
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "UnicodeAtomicLexemePointerTokenizer":
        allowed = {"format", "unicode_lexeme_pattern", "unicode_normalization", "special_ids", "fixed_lexemes_hex", "fixed_vocabulary_order", "every_action_valid_utf8", "external_input_output"}
        expected_specials = {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unknown_source": UNK_ID}
        if (
            set(document) != allowed
            or document.get("format") != UNICODE_TOKENIZER_FORMAT
            or document.get("unicode_lexeme_pattern") != UNICODE_LEXEME_PATTERN
            or document.get("unicode_normalization") != "PRESERVE_EXACT_NO_NORMALIZATION"
            or document.get("special_ids") != expected_specials
            or document.get("fixed_vocabulary_order") != "ascending_utf8_bytes"
            or document.get("every_action_valid_utf8") is not True
            or document.get("external_input_output") != "UTF-8 bytes"
        ):
            raise ValueError("Unicode-atomic tokenizer document changed")
        values = document.get("fixed_lexemes_hex")
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("Unicode fixed lexemes must be hex strings")
        return cls([bytes.fromhex(value) for value in values])

    def hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def unicode_token_plan_manifest_architecture(model: PortableTokenPlan, tokenizer: UnicodeAtomicLexemePointerTokenizer) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("model and Unicode tokenizer vocabulary sizes differ")
    return {
        "name": "unicode_atomic_portable_token_plan",
        "format": UNICODE_TOKEN_PLAN_FORMAT,
        "model": model.canonical_config(),
        "tokenizer": tokenizer.canonical_dict(),
        "tokenizer_sha256": tokenizer.hash(),
        "external_input_output": "UTF-8 bytes",
        "private_representation": "portable_token_plan_pointer_transformer",
        "action_validity": "every_fixed_and_pointer_action_complete_utf8",
    }


class UnicodeSafeDirectNeuralCoreHost:
    """Install and execute one v2 Unicode-atomic English core."""

    def __init__(self, registry_root: str | Path, *, trust_store: Mapping[str, bytes | str | Path], device: str | torch.device = "cpu") -> None:
        self.registry = CakeRegistry(registry_root)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=UNICODE_DIRECT_NEURAL_CORE_ABI_VERSION,
                abi_hash=UNICODE_DIRECT_NEURAL_CORE_ABI_SHA256,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=frozenset({"byte_input", "safe_tensors", "persistent_incremental_state", "unicode_atomic_actions", "strict_utf8_boundary"}),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )
        self.device = torch.device(device)
        self.module: nn.Module | None = None
        self.active_cake_id: str | None = None
        self.active_archive_hash: str | None = None
        self.active_payload_hash: str | None = None
        self.receiver_training_steps = 0
        self.receiver_calibration_runs = 0

    @staticmethod
    def _validate_role(package: CakePackage) -> None:
        manifest = package.manifest
        if not package.signed or manifest.cake_type != "portable_decoder":
            raise UnicodeDirectNeuralCoreError("Unicode direct core must be a signed portable decoder")
        if manifest.abi_version != UNICODE_DIRECT_NEURAL_CORE_ABI_VERSION or manifest.abi_hash != UNICODE_DIRECT_NEURAL_CORE_ABI_SHA256:
            raise UnicodeDirectNeuralCoreError("Unicode direct core ABI identity mismatch")
        if manifest.domains != (DIRECT_NEURAL_CORE_ROLE,) or manifest.dependencies:
            raise UnicodeDirectNeuralCoreError("package is not an exclusive dependency-free English core")
        if manifest.input_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "validity": "strict_utf8"}:
            raise UnicodeDirectNeuralCoreError("Unicode direct core input contract mismatch")
        if manifest.output_contract != {"external": "UTF-8 bytes", "role": DIRECT_NEURAL_CORE_ROLE, "composition": DIRECT_NEURAL_CORE_COMPOSITION, "validity": "strict_utf8"}:
            raise UnicodeDirectNeuralCoreError("Unicode direct core output contract mismatch")
        architecture = manifest.architecture
        if architecture.get("name") != "unicode_atomic_portable_token_plan" or architecture.get("format") != UNICODE_TOKEN_PLAN_FORMAT or architecture.get("action_validity") != "every_fixed_and_pointer_action_complete_utf8":
            raise UnicodeDirectNeuralCoreError("Unicode direct core architecture mismatch")
        required = set(manifest.minimum_host_capabilities.get("features", []))
        expected = {"byte_input", "safe_tensors", "persistent_incremental_state", "unicode_atomic_actions", "strict_utf8_boundary"}
        if not expected <= required:
            raise UnicodeDirectNeuralCoreError("Unicode direct core capabilities are incomplete")

    @staticmethod
    def _load_module(package: CakePackage, device: torch.device) -> PortableTokenPlan:
        architecture = package.manifest.architecture
        allowed = {"name", "format", "model", "tokenizer", "tokenizer_sha256", "external_input_output", "private_representation", "action_validity"}
        if set(architecture) != allowed or architecture["external_input_output"] != "UTF-8 bytes" or architecture["private_representation"] != "portable_token_plan_pointer_transformer":
            raise UnicodeDirectNeuralCoreError("Unicode token-plan metadata is incomplete")
        tokenizer = UnicodeAtomicLexemePointerTokenizer.from_document(architecture["tokenizer"])
        if tokenizer.hash() != architecture["tokenizer_sha256"]:
            raise UnicodeDirectNeuralCoreError("Unicode tokenizer hash mismatch")
        model = PortableTokenPlan(**architecture["model"]).bind_tokenizer(tokenizer)
        model.load_state_dict(package.tensors, strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def activate(self, source: str | Path) -> dict[str, Any]:
        inspected = self.installer.inspect(source)
        self._validate_role(inspected)
        record = self.installer.install(source)
        installed = load_package(record["blob"], trust_store=self.installer.trust_store, require_signature=True)
        self._validate_role(installed)
        module = self._load_module(installed, self.device)
        self.module = module
        self.active_cake_id = installed.manifest.cake_id
        self.active_archive_hash = installed.archive_hash
        self.active_payload_hash = installed.manifest.tensor_payload_hash
        return {"status": "ACTIVE", "cake_id": self.active_cake_id, "archive_hash": self.active_archive_hash, "payload_hash": self.active_payload_hash, "state_dict_hash": state_dict_hash(module.state_dict()), "device": str(self.device), "receiver_training_steps": 0, "receiver_calibration_runs": 0}

    def _require_active(self) -> PortableTokenPlan:
        if self.module is None or self.active_cake_id is None:
            raise UnicodeDirectNeuralCoreError("no Unicode direct neural core is active")
        return self.module  # type: ignore[return-value]

    def prefill(self, prompt: bytes | str):
        if isinstance(prompt, bytes):
            prompt.decode("utf-8", errors="strict")
        else:
            prompt.encode("utf-8", errors="strict")
        return self._require_active().prefill_bytes(prompt)

    def decode_step(self, state):
        return self._require_active().decode_step(state)

    def realize(self, state) -> bytes:
        module = self._require_active()
        raw = module.tokenizer.decode_actions(state.generated_actions, state.source_lexemes)
        raw.decode("utf-8", errors="strict")
        return raw

    def generate(self, prompt: bytes | str, *, maximum_actions: int | None = None) -> bytes:
        if isinstance(prompt, bytes):
            prompt.decode("utf-8", errors="strict")
        else:
            prompt.encode("utf-8", errors="strict")
        raw = self._require_active().generate_bytes(prompt, maximum_actions=maximum_actions)
        try:
            raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise UnicodeDirectNeuralCoreError("core attempted invalid UTF-8 output") from exc
        return raw

    def verify(self) -> dict[str, Any]:
        if self.active_cake_id is None:
            raise UnicodeDirectNeuralCoreError("no Unicode direct neural core is active")
        result = self.installer.verify(self.active_cake_id)
        if result["archive_hash"] != self.active_archive_hash or result["payload_hash"] != self.active_payload_hash:
            raise UnicodeDirectNeuralCoreError("active Unicode direct core identity changed")
        return {**result, "role": DIRECT_NEURAL_CORE_ROLE, "utf8": "STRICT"}

    def remove(self) -> dict[str, Any]:
        if self.active_cake_id is None:
            raise UnicodeDirectNeuralCoreError("no Unicode direct neural core is active")
        result = self.installer.remove(self.active_cake_id)
        self.module = None
        self.active_cake_id = None
        self.active_archive_hash = None
        self.active_payload_hash = None
        return result
