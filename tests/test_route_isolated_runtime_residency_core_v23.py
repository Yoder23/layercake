import json
from pathlib import Path

import pytest
import torch

import layercake.cake.installer as installer_module
from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, load_package
from layercake.evaluation.route_isolated_format_literal_v22_construct import (
    _fixture,
    _keys,
)
from layercake_extensions.route_isolated_runtime_residency_core_v23 import (
    ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256,
    ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_VERSION,
    SINGLE_PARSE_ACTIVATION_FEATURE,
    RuntimeResidencyFormatLiteralCoreHost,
)


FORMAT_PROMPT = (
    "Return exactly two plain-text lines and no Markdown. "
    "The first line must be `item: green notebook` and the second line must be "
    "`code: N390098UMA`."
)
FORMAT_EXPECTED = b"item: green notebook\ncode: N390098UMA"


def _v23_fixture(directory: Path, *, include_feature: bool = True):
    v22, public, signer, tensors = _fixture(
        directory / "source-v22", capability="format_control"
    )
    loaded = load_package(v22, trust_store={signer: public}, require_signature=True)
    document = loaded.manifest.canonical_dict()
    features = list(document["minimum_host_capabilities"]["features"])
    if include_feature:
        features.append(SINGLE_PARSE_ACTIVATION_FEATURE)
    document.update(
        {
            "cake_id": "runtime-residency-v23",
            "version": "23.0.0",
            "abi_version": ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_VERSION,
            "abi_hash": ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256,
            "minimum_host_capabilities": {"features": features},
            "tensor_payload_hash": "",
            "package_hash": "",
        }
    )
    private, _, _ = _keys()
    path = build_package(
        directory / f"v23-{int(include_feature)}.cake",
        CakeManifest.from_dict(document),
        tensors,
        private_key=private,
    )
    return path, v22, public, signer, tensors, loaded


def test_v23_authenticates_once_preserves_tensors_and_rejects_v22(
    tmp_path, monkeypatch
):
    package, v22, public, signer, tensors, loaded_v22 = _v23_fixture(tmp_path)
    original = installer_module.load_package
    calls = []

    def counted(*args, **kwargs):
        calls.append(Path(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(installer_module, "load_package", counted)
    host = RuntimeResidencyFormatLiteralCoreHost(
        tmp_path / "registry", trust_store={signer: public}
    )
    active = host.activate(package)
    assert len(calls) == 1
    assert active["authenticated_package_parses"] == 1
    assert active["payload_hash"] == loaded_v22.manifest.tensor_payload_hash
    assert set(host.model.state_dict()) == {
        name.removeprefix("model.") for name in tensors if name.startswith("model.")
    }
    assert host.generate(FORMAT_PROMPT, maximum_tokens=32) == FORMAT_EXPECTED
    assert host.verify()["status"] == "PASS"
    assert host.remove()["status"] == "REMOVED"

    rejected = RuntimeResidencyFormatLiteralCoreHost(
        tmp_path / "reject-v22", trust_store={signer: public}
    )
    with pytest.raises(Exception):
        rejected.activate(v22)
    assert rejected.registry.list() == []


def test_v23_missing_feature_rejects_before_registry_mutation(tmp_path):
    package, _, public, signer, _, _ = _v23_fixture(
        tmp_path, include_feature=False
    )
    host = RuntimeResidencyFormatLiteralCoreHost(
        tmp_path / "registry", trust_store={signer: public}
    )
    with pytest.raises(Exception, match="single-parse"):
        host.activate(package)
    assert host.registry.list() == []


def test_v23_reuses_only_identical_route_and_preserves_device_outputs(tmp_path):
    package, _, public, signer, _, _ = _v23_fixture(tmp_path)
    outputs = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        host = RuntimeResidencyFormatLiteralCoreHost(
            tmp_path / f"registry-{device}",
            trust_store={signer: public},
            device=device,
        )
        host.activate(package)
        host._set_residual_route(-1)
        first = host.model.transformer.h[0]._layercake_residual_routes
        host._set_residual_route(-1)
        assert host.model.transformer.h[0]._layercake_residual_routes is first
        host._set_residual_route_batch(-1, 6)
        batch = host.model.transformer.h[0]._layercake_residual_routes
        assert batch is not first and batch.shape == (6,)
        host._set_residual_route(0)
        changed = host.model.transformer.h[0]._layercake_residual_routes
        assert changed is not batch and changed.tolist() == [0]
        outputs[device] = host.generate(FORMAT_PROMPT, maximum_tokens=32)
    assert set(outputs.values()) == {FORMAT_EXPECTED}
    if torch.cuda.is_available():
        assert set(outputs) == {"cpu", "cuda"}


def test_v23_canonical_abi_hash_is_bound():
    path = (
        Path(__file__).resolve().parents[1]
        / "moonshot/canonical_route_isolated_format_literal_core_abi_v23.json"
    )
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256
    )
    assert json.loads(path.read_text())["abi_version"] == (
        ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_VERSION
    )
