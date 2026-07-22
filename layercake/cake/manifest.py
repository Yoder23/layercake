"""Strict manifest schema for non-executable ``.cake`` packages."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, replace
import hashlib
import json
import re
from typing import Any, Mapping


SCHEMA_VERSION = "1"
CAKE_TYPES = {"portable_decoder", "host_residual", "portable_fusion"}
PRECISIONS = {"fp32", "fp16", "bf16", "int8"}
BACKENDS = {"pytorch", "torchscript", "onnx", "coreml", "executorch", "cuda"}
_ID = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,240}$")


class ManifestError(ValueError):
    """Raised when package metadata is unsafe or semantically incomplete."""


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _plain_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ManifestError(f"{field} must be a JSON object with string keys")
    # Round trip rejects non-JSON values and non-finite floats.
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must contain only canonical JSON values") from exc


@dataclass(frozen=True)
class CakeManifest:
    schema_version: str
    cake_id: str
    name: str
    description: str
    version: str
    publisher: dict[str, Any]
    abi_version: str
    abi_hash: str
    cake_type: str
    input_contract: dict[str, Any]
    output_contract: dict[str, Any]
    architecture: dict[str, Any]
    supported_precisions: tuple[str, ...]
    supported_backends: tuple[str, ...]
    minimum_host_capabilities: dict[str, Any]
    tensor_payload_hash: str
    tensor_shapes: dict[str, dict[str, Any]]
    package_hash: str
    training_data_provenance: dict[str, Any]
    evaluation_evidence: dict[str, Any]
    license: str
    dependencies: tuple[str, ...]
    parent_version: str | None
    signature: dict[str, Any]
    domains: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestError(f"unsupported manifest schema: {self.schema_version!r}")
        if not _ID.fullmatch(self.cake_id):
            raise ManifestError("cake_id must be a lowercase, path-free package identifier")
        if not self.name.strip() or not self.description.strip():
            raise ManifestError("name and description are required")
        if not _SEMVER.fullmatch(self.version):
            raise ManifestError("version must be semantic versioning")
        if self.parent_version is not None and not _SEMVER.fullmatch(self.parent_version):
            raise ManifestError("parent_version must be semantic versioning")
        if not self.abi_version or not _HEX64.fullmatch(self.abi_hash):
            raise ManifestError("ABI version and a lowercase SHA-256 ABI hash are required")
        if self.cake_type not in CAKE_TYPES:
            raise ManifestError(f"unsupported cake_type: {self.cake_type!r}")
        if not self.supported_precisions or not set(self.supported_precisions) <= PRECISIONS:
            raise ManifestError("supported_precisions contains an unsupported value")
        if not self.supported_backends or not set(self.supported_backends) <= BACKENDS:
            raise ManifestError("supported_backends contains an unsupported value")
        if self.tensor_payload_hash and not _HEX64.fullmatch(self.tensor_payload_hash):
            raise ManifestError("tensor_payload_hash must be a lowercase SHA-256 digest")
        if self.package_hash and not _HEX64.fullmatch(self.package_hash):
            raise ManifestError("package_hash must be a lowercase SHA-256 digest")
        if not self.license.strip():
            raise ManifestError("license is required")
        publisher = _plain_mapping(self.publisher, "publisher")
        if not all(publisher.get(key) for key in ("id", "name", "key_id")):
            raise ManifestError("publisher requires id, name, and key_id")
        signature = _plain_mapping(self.signature, "signature")
        if signature.get("algorithm") not in {"ed25519", "none"}:
            raise ManifestError("signature algorithm must be ed25519 or none")
        if signature.get("algorithm") == "ed25519" and not signature.get("key_id"):
            raise ManifestError("signed packages require a signature key_id")
        for field, value in (
            ("input_contract", self.input_contract),
            ("output_contract", self.output_contract),
            ("architecture", self.architecture),
            ("minimum_host_capabilities", self.minimum_host_capabilities),
            ("training_data_provenance", self.training_data_provenance),
            ("evaluation_evidence", self.evaluation_evidence),
        ):
            _plain_mapping(value, field)
        if not self.tensor_shapes:
            raise ManifestError("tensor_shapes must declare at least one tensor")
        for tensor_name, spec in self.tensor_shapes.items():
            if not _TENSOR_NAME.fullmatch(tensor_name) or tensor_name.startswith("."):
                raise ManifestError(f"unsafe tensor name: {tensor_name!r}")
            if set(spec) != {"shape", "dtype"}:
                raise ManifestError(f"tensor {tensor_name!r} requires only shape and dtype")
            shape = spec["shape"]
            if not isinstance(shape, list) or any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
            ):
                raise ManifestError(f"invalid tensor shape for {tensor_name!r}")
            if not isinstance(spec["dtype"], str) or not spec["dtype"]:
                raise ManifestError(f"invalid tensor dtype for {tensor_name!r}")
        for sequence_name, values in (
            ("dependencies", self.dependencies),
            ("domains", self.domains),
            ("keywords", self.keywords),
            ("permissions", self.permissions),
        ):
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ManifestError(f"{sequence_name} must contain non-empty strings")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ManifestError("dependencies must not contain duplicates")

    def canonical_dict(self, *, blank_package_hash: bool = False) -> dict[str, Any]:
        result = asdict(self)
        for field in (
            "supported_precisions",
            "supported_backends",
            "dependencies",
            "domains",
            "keywords",
            "permissions",
        ):
            result[field] = list(result[field])
        if blank_package_hash:
            result["package_hash"] = ""
        return result

    def canonical_bytes(self, *, blank_package_hash: bool = False) -> bytes:
        return canonical_json(self.canonical_dict(blank_package_hash=blank_package_hash))

    def with_integrity(self, *, tensor_payload_hash: str, package_hash: str) -> "CakeManifest":
        return replace(
            self,
            tensor_payload_hash=tensor_payload_hash,
            package_hash=package_hash,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CakeManifest":
        if not isinstance(data, dict):
            raise ManifestError("manifest must be a JSON object")
        fields = set(cls.__dataclass_fields__)
        unknown = set(data) - fields
        missing = {
            name for name, item in cls.__dataclass_fields__.items()
            if item.default is MISSING and item.default_factory is MISSING and name not in data
        }
        if unknown:
            raise ManifestError(f"unknown manifest fields: {sorted(unknown)}")
        if missing:
            raise ManifestError(f"missing manifest fields: {sorted(missing)}")
        converted = dict(data)
        for name in (
            "supported_precisions",
            "supported_backends",
            "dependencies",
            "domains",
            "keywords",
            "permissions",
        ):
            value = converted.get(name, [])
            if not isinstance(value, list):
                raise ManifestError(f"{name} must be a JSON array")
            converted[name] = tuple(value)
        return cls(**converted)

    @classmethod
    def from_json(cls, raw: bytes | str) -> "CakeManifest":
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError("manifest is not valid UTF-8 JSON") from exc
        return cls.from_dict(data)


def abi_hash(version: str, contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        b"LAYERCAKE-ABI-V1\0" + version.encode("utf-8") + b"\0" + canonical_json(contract)
    ).hexdigest()
