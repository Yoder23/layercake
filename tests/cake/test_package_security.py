import io
import json
from pathlib import Path
import zipfile

import pytest
import torch

from layercake.cake.cli import DEFAULT_ABI_HASH, DEFAULT_ABI_VERSION
from layercake.cake.manifest import CakeManifest, ManifestError
from layercake.cake.package import PackageError, build_package, load_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.portable_domain import PortableDomainDecoder
from layercake.models.portable_decoder import portable_decoder_manifest_architecture


def manifest(model, *, signed=False):
    state = model.state_dict()
    return CakeManifest(
        schema_version="1", cake_id="python", name="Python", description="Neural Python specialist",
        version="1.2.3", publisher={"id": "test", "name": "Test", "key_id": "placeholder"},
        abi_version=DEFAULT_ABI_VERSION, abi_hash=DEFAULT_ABI_HASH, cake_type="portable_decoder",
        input_contract={"mode": "causal_bytes"}, output_contract={"mode": "next_byte_logits"},
        architecture=portable_decoder_manifest_architecture(feature_width=8, hidden_width=16),
        supported_precisions=("fp32",), supported_backends=("pytorch",),
        minimum_host_capabilities={"features": ["byte_input"]}, tensor_payload_hash="",
        tensor_shapes=tensor_specs(state), package_hash="",
        training_data_provenance={"dataset": "test", "sha256": "0" * 64},
        evaluation_evidence={"status": "TEST"}, license="Apache-2.0", dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519" if signed else "none", "key_id": "placeholder"},
        domains=("python",), keywords=("generator",), permissions=(),
    )


def test_signed_package_authenticates_manifest_and_payload(tmp_path):
    private, public, identifier = generate_keypair()
    model = PortableDomainDecoder(feature_width=8, hidden_width=16)
    item = manifest(model, signed=True)
    item = CakeManifest.from_dict({
        **item.canonical_dict(),
        "publisher": {"id": "test", "name": "Test", "key_id": identifier},
        "signature": {"algorithm": "ed25519", "key_id": identifier},
    })
    path = build_package(tmp_path / "python.cake", item, model.state_dict(), private_key=private)
    loaded = load_package(path, trust_store={identifier: public})
    assert loaded.signed
    assert loaded.manifest.package_hash
    assert set(loaded.tensors) == set(model.state_dict())


def test_unsigned_requires_explicit_local_trust(tmp_path):
    model = PortableDomainDecoder(feature_width=8, hidden_width=16)
    path = build_package(tmp_path / "python.cake", manifest(model), model.state_dict())
    with pytest.raises(PackageError, match="unsigned"):
        load_package(path)
    assert not load_package(path, require_signature=False, allow_local_development=True).signed


def _rewrite(path: Path, mutation):
    with zipfile.ZipFile(path, "r") as source:
        members = [(info.filename, source.read(info.filename)) for info in source.infolist()]
    mutation(members)
    with zipfile.ZipFile(path, "w") as target:
        for name, value in members:
            target.writestr(name, value)


def test_payload_tampering_is_rejected(tmp_path):
    model = PortableDomainDecoder(feature_width=8, hidden_width=16)
    path = build_package(tmp_path / "python.cake", manifest(model), model.state_dict())
    def mutate(members):
        for index, (name, value) in enumerate(members):
            if name == "tensors.safetensors":
                members[index] = (name, value[:-1] + bytes([value[-1] ^ 1]))
    _rewrite(path, mutate)
    with pytest.raises(PackageError, match="payload hash"):
        load_package(path, require_signature=False, allow_local_development=True)


@pytest.mark.parametrize("bad_name", ["../escape", "/absolute", "nested/value"])
def test_path_traversal_and_nested_entries_are_rejected(tmp_path, bad_name):
    model = PortableDomainDecoder(feature_width=8, hidden_width=16)
    path = build_package(tmp_path / "python.cake", manifest(model), model.state_dict())
    _rewrite(path, lambda members: members.append((bad_name, b"bad")))
    with pytest.raises(PackageError):
        load_package(path, require_signature=False, allow_local_development=True)


def test_duplicate_entries_are_rejected(tmp_path):
    model = PortableDomainDecoder(feature_width=8, hidden_width=16)
    path = build_package(tmp_path / "python.cake", manifest(model), model.state_dict())
    _rewrite(path, lambda members: members.append(members[0]))
    with pytest.raises(PackageError, match="duplicate"):
        load_package(path, require_signature=False, allow_local_development=True)


def test_unknown_manifest_fields_fail_closed():
    model = PortableDomainDecoder(feature_width=8, hidden_width=16)
    raw = manifest(model).canonical_dict()
    raw["execute"] = "payload.py"
    with pytest.raises(ManifestError, match="unknown"):
        CakeManifest.from_dict(raw)
