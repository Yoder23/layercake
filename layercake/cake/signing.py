"""Ed25519 signing for published cakes; no package code is ever executed."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .manifest import canonical_json


SIGNING_CONTEXT = b"LAYERCAKE-CAKE-SIGNATURE-V1\0"


class SignatureError(ValueError):
    pass


@dataclass(frozen=True)
class SignatureEnvelope:
    algorithm: str
    key_id: str
    signed_hash: str
    signature: str

    def canonical_bytes(self) -> bytes:
        return canonical_json(self.__dict__)

    @classmethod
    def from_bytes(cls, raw: bytes) -> "SignatureEnvelope":
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SignatureError("signature envelope is not valid JSON") from exc
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise SignatureError("signature envelope has an invalid schema")
        envelope = cls(**data)
        if envelope.algorithm != "ed25519":
            raise SignatureError("unsupported signature algorithm")
        return envelope


def key_id(public_key_pem: bytes) -> str:
    public = load_public_key(public_key_pem)
    raw = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:32]


def generate_keypair() -> tuple[bytes, bytes, str]:
    private = Ed25519PrivateKey.generate()
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


def load_private_key(value: bytes | str | Path) -> Ed25519PrivateKey:
    raw = Path(value).read_bytes() if isinstance(value, Path) else (
        Path(value).read_bytes() if isinstance(value, str) and "BEGIN" not in value else
        value.encode("utf-8") if isinstance(value, str) else value
    )
    key = serialization.load_pem_private_key(raw, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise SignatureError("private key is not Ed25519")
    return key


def load_public_key(value: bytes | str | Path) -> Ed25519PublicKey:
    raw = Path(value).read_bytes() if isinstance(value, Path) else (
        Path(value).read_bytes() if isinstance(value, str) and "BEGIN" not in value else
        value.encode("utf-8") if isinstance(value, str) else value
    )
    key = serialization.load_pem_public_key(raw)
    if not isinstance(key, Ed25519PublicKey):
        raise SignatureError("public key is not Ed25519")
    return key


def sign_hash(package_hash: str, private_key: bytes | str | Path, *, signer_key_id: str) -> SignatureEnvelope:
    if len(package_hash) != 64:
        raise SignatureError("package hash must be SHA-256")
    signature = load_private_key(private_key).sign(SIGNING_CONTEXT + package_hash.encode("ascii"))
    return SignatureEnvelope(
        algorithm="ed25519",
        key_id=signer_key_id,
        signed_hash=package_hash,
        signature=base64.b64encode(signature).decode("ascii"),
    )


def verify_hash(
    envelope: SignatureEnvelope,
    package_hash: str,
    trust_store: Mapping[str, bytes | str | Path],
) -> None:
    if envelope.signed_hash != package_hash:
        raise SignatureError("signature does not cover this package hash")
    try:
        public_value = trust_store[envelope.key_id]
    except KeyError as exc:
        raise SignatureError(f"untrusted publisher key: {envelope.key_id}") from exc
    try:
        signature = base64.b64decode(envelope.signature, validate=True)
        load_public_key(public_value).verify(
            signature, SIGNING_CONTEXT + package_hash.encode("ascii")
        )
    except (InvalidSignature, ValueError) as exc:
        raise SignatureError("invalid package signature") from exc
