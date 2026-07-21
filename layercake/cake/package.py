"""Build and validate safe ``.cake`` archives."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Mapping
import zipfile

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from .manifest import CakeManifest
from .signing import SignatureEnvelope, SignatureError, sign_hash, verify_hash


MANIFEST_NAME = "manifest.json"
TENSORS_NAME = "tensors.safetensors"
SIGNATURE_NAME = "signature.json"
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
_CONTEXT = b"LAYERCAKE-CAKE-CONTENT-V1\0"


class PackageError(ValueError):
    pass


@dataclass(frozen=True)
class CakePackage:
    path: Path
    manifest: CakeManifest
    tensors: dict[str, torch.Tensor]
    signed: bool
    signature_key_id: str | None
    archive_hash: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_specs(tensors: Mapping[str, torch.Tensor]) -> dict[str, dict]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
        for name, tensor in sorted(tensors.items())
    }


def content_hash(manifest: CakeManifest, payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(_CONTEXT)
    manifest_bytes = manifest.canonical_bytes(blank_package_hash=True)
    for name, data in ((MANIFEST_NAME, manifest_bytes), (TENSORS_NAME, payload)):
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _safe_tensor_payload(tensors: Mapping[str, torch.Tensor]) -> bytes:
    if not tensors:
        raise PackageError("a cake requires at least one tensor")
    clean: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise PackageError("tensor payload must map string names to tensors")
        if tensor.layout != torch.strided or tensor.is_sparse or tensor.is_quantized:
            raise PackageError(f"unsupported tensor layout for {name!r}")
        clean[name] = tensor.detach().cpu().contiguous()
    try:
        return save_safetensors(clean, metadata={"format": "layercake-cake-tensors/1"})
    except Exception as exc:
        raise PackageError("failed to encode safe tensor payload") from exc


def build_package(
    path: str | Path,
    manifest: CakeManifest,
    tensors: Mapping[str, torch.Tensor],
    *,
    private_key: bytes | str | Path | None = None,
) -> Path:
    """Write a deterministic package. Unsigned output is explicitly local-development only."""
    path = Path(path)
    if path.suffix != ".cake":
        raise PackageError("cake packages must use the .cake extension")
    payload = _safe_tensor_payload(tensors)
    payload_hash = sha256_bytes(payload)
    shapes = tensor_specs(tensors)
    if manifest.tensor_shapes != shapes:
        raise PackageError("manifest tensor_shapes do not match payload tensors")
    unsigned = replace(manifest, tensor_payload_hash=payload_hash, package_hash="")
    digest = content_hash(unsigned, payload)
    final_manifest = unsigned.with_integrity(
        tensor_payload_hash=payload_hash,
        package_hash=digest,
    )
    signature_bytes: bytes | None = None
    algorithm = final_manifest.signature.get("algorithm")
    if private_key is not None:
        if algorithm != "ed25519":
            raise PackageError("private_key requires signature.algorithm=ed25519")
        envelope = sign_hash(
            digest,
            private_key,
            signer_key_id=str(final_manifest.signature["key_id"]),
        )
        signature_bytes = envelope.canonical_bytes()
    elif algorithm != "none":
        raise PackageError("a published cake requires its Ed25519 private key")

    members = [
        (MANIFEST_NAME, final_manifest.canonical_bytes()),
        (TENSORS_NAME, payload),
    ]
    if signature_bytes is not None:
        members.append((SIGNATURE_NAME, signature_bytes))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in members:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o600 << 16
                info.compress_type = zipfile.ZIP_STORED
                archive.writestr(info, data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _read_members(path: Path) -> tuple[dict[str, bytes], str]:
    raw = path.read_bytes()
    if len(raw) > MAX_PACKAGE_BYTES:
        raise PackageError("package exceeds the configured size limit")
    archive_hash = sha256_bytes(raw)
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            folded = [name.casefold() for name in names]
            if len(names) != len(set(names)) or len(folded) != len(set(folded)):
                raise PackageError("duplicate or case-ambiguous archive entries")
            allowed = {MANIFEST_NAME, TENSORS_NAME, SIGNATURE_NAME}
            if not {MANIFEST_NAME, TENSORS_NAME} <= set(names) or not set(names) <= allowed:
                raise PackageError("package contains missing or unsupported entries")
            total = 0
            for info in infos:
                pure = PurePosixPath(info.filename)
                if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
                    raise PackageError("path traversal or nested entries are forbidden")
                if info.is_dir() or info.flag_bits & 0x1:
                    raise PackageError("directories and encrypted entries are forbidden")
                mode = (info.external_attr >> 16) & 0o170000
                if mode not in {0, 0o100000}:
                    raise PackageError("non-regular archive entries are forbidden")
                total += info.file_size
                if total > MAX_PACKAGE_BYTES:
                    raise PackageError("expanded package exceeds the size limit")
            return {name: archive.read(name) for name in names}, archive_hash
    except (zipfile.BadZipFile, OSError) as exc:
        raise PackageError("invalid cake ZIP container") from exc


def load_package(
    path: str | Path,
    *,
    trust_store: Mapping[str, bytes | str | Path] | None = None,
    require_signature: bool = True,
    allow_local_development: bool = False,
) -> CakePackage:
    path = Path(path)
    if path.suffix != ".cake" or not path.is_file():
        raise PackageError("cake package path must be an existing .cake file")
    members, archive_hash = _read_members(path)
    manifest = CakeManifest.from_json(members[MANIFEST_NAME])
    payload = members[TENSORS_NAME]
    if sha256_bytes(payload) != manifest.tensor_payload_hash:
        raise PackageError("tensor payload hash mismatch")
    expected_content_hash = content_hash(manifest, payload)
    if manifest.package_hash != expected_content_hash:
        raise PackageError("authenticated package content hash mismatch")
    signed = SIGNATURE_NAME in members
    signature_key_id: str | None = None
    if signed:
        try:
            envelope = SignatureEnvelope.from_bytes(members[SIGNATURE_NAME])
            if envelope.key_id != manifest.signature.get("key_id"):
                raise SignatureError("manifest and signature key identifiers differ")
            verify_hash(envelope, manifest.package_hash, trust_store or {})
            signature_key_id = envelope.key_id
        except SignatureError as exc:
            raise PackageError(str(exc)) from exc
    elif require_signature or not allow_local_development:
        raise PackageError("unsigned cake rejected; explicitly enable trusted local development")
    elif manifest.signature.get("algorithm") != "none":
        raise PackageError("missing signature envelope")
    try:
        tensors = load_safetensors(payload)
    except Exception as exc:
        raise PackageError("invalid safetensors payload") from exc
    actual_specs = tensor_specs(tensors)
    if actual_specs != manifest.tensor_shapes:
        raise PackageError("tensor names, shapes, or dtypes do not match the manifest")
    return CakePackage(
        path=path.resolve(),
        manifest=manifest,
        tensors=tensors,
        signed=signed,
        signature_key_id=signature_key_id,
        archive_hash=archive_hash,
    )
