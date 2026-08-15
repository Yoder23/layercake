import hashlib
import json
from pathlib import Path

import pytest
import torch

import layercake.cake.installer as installer_module
from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, load_package
from layercake.evaluation.route_isolated_format_literal_v22_construct import _keys
from layercake.evaluation.route_isolated_runtime_residency_v23_construct import (
    _v23_fixture,
)
from layercake_extensions.route_isolated_allocation_bounded_core_v24 import (
    ALLOCATION_BOUNDED_ADOPTION_FEATURE,
    ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_SHA256,
    ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_VERSION,
    AllocationBoundedRuntimeResidencyCoreHost,
)


FORMAT_PROMPT = (
    "Return exactly two plain-text lines and no Markdown. "
    "The first line must be `item: green notebook` and the second line must be "
    "`code: N390098UMA`."
)
FORMAT_EXPECTED = b"item: green notebook\ncode: N390098UMA"


def _v24_fixture(directory: Path, *, include_feature: bool = True):
    v23, v22, public, signer, tensors, loaded_v22 = _v23_fixture(
        directory / "source-v23", capability="format_control"
    )
    loaded_v23 = load_package(v23, trust_store={signer: public}, require_signature=True)
    document = loaded_v23.manifest.canonical_dict()
    features = list(document["minimum_host_capabilities"]["features"])
    if include_feature:
        features.append(ALLOCATION_BOUNDED_ADOPTION_FEATURE)
    document.update(
        {
            "cake_id": f"allocation-bounded-v24-{int(include_feature)}",
            "version": "24.0.0",
            "abi_version": ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_VERSION,
            "abi_hash": ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_SHA256,
            "minimum_host_capabilities": {"features": features},
            "tensor_payload_hash": "",
            "package_hash": "",
        }
    )
    private, _, _ = _keys()
    package = build_package(
        directory / f"v24-{int(include_feature)}.cake",
        CakeManifest.from_dict(document),
        tensors,
        private_key=private,
    )
    return package, v23, v22, public, signer, tensors, loaded_v23, loaded_v22


def test_v24_adopts_authenticated_storage_once_and_rejects_v23(tmp_path, monkeypatch):
    package, v23, _, public, signer, tensors, loaded_v23, _ = _v24_fixture(tmp_path)
    original = installer_module.load_package
    calls = []

    def counted(*args, **kwargs):
        calls.append(Path(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(installer_module, "load_package", counted)
    host = AllocationBoundedRuntimeResidencyCoreHost(
        tmp_path / "registry", trust_store={signer: public}
    )
    active = host.activate(package)
    assert len(calls) == 1
    assert active["authenticated_package_parses"] == 1
    assert active["payload_hash"] == loaded_v23.manifest.tensor_payload_hash
    assert active["strict_assigned_tensor_count"] == len(tensors)
    assert active["authenticated_tensor_count"] == len(tensors)
    assert active["meta_tensors_after_adoption"] == 0
    assert active["reconstructed_nonpersistent_buffers"] == 6
    assert not any(
        value.device.type == "meta"
        for module in (host.model, host.router, host.residual)
        for value in module.state_dict().values()
    )
    assert host.generate(FORMAT_PROMPT, maximum_tokens=32) == FORMAT_EXPECTED

    rejected = AllocationBoundedRuntimeResidencyCoreHost(
        tmp_path / "reject-v23", trust_store={signer: public}
    )
    with pytest.raises(Exception):
        rejected.activate(v23)
    assert rejected.registry.list() == []


def test_v24_missing_feature_rejects_before_registry_mutation(tmp_path):
    package, _, _, public, signer, _, _, _ = _v24_fixture(
        tmp_path, include_feature=False
    )
    host = AllocationBoundedRuntimeResidencyCoreHost(
        tmp_path / "registry", trust_store={signer: public}
    )
    with pytest.raises(Exception, match="allocation-bounded"):
        host.activate(package)
    assert host.registry.list() == []


def test_v24_preserves_route_reuse_and_device_outputs(tmp_path):
    package, _, _, public, signer, tensors, _, _ = _v24_fixture(tmp_path)
    outputs = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        host = AllocationBoundedRuntimeResidencyCoreHost(
            tmp_path / f"registry-{device}", trust_store={signer: public}, device=device
        )
        active = host.activate(package)
        assert active["strict_assigned_tensor_count"] == len(tensors)
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


def test_v24_canonical_abi_hash_is_bound():
    path = (
        Path(__file__).resolve().parents[1]
        / "moonshot/canonical_route_isolated_format_literal_core_abi_v24.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_SHA256
    )
    assert json.loads(path.read_text())["abi_version"] == (
        ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_VERSION
    )
