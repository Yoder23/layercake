import hashlib
import json
from pathlib import Path

import pytest
import torch

from layercake.evaluation.route_isolated_clarification_v25_construct import _v25_fixture
from layercake_extensions.route_isolated_clarification_core_v25 import (
    ARCHITECTURE_V25_FORMAT,
    CLARIFICATION_ROUTE_ISOLATION_FEATURE,
    CLARIFICATION_ROUTE_V25,
    RESIDUAL_CAPABILITIES_V25,
    ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256,
    ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION,
    ClarificationRouteAllocationBoundedCoreHost,
)


def test_v25_physically_selects_only_clarification_route_four(tmp_path):
    package, _, public, signer, tensors, _ = _v25_fixture(tmp_path)
    outputs = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        host = ClarificationRouteAllocationBoundedCoreHost(
            tmp_path / f"registry-{device}", trust_store={signer: public}, device=device
        )
        active = host.activate(package)
        state = host.prefill("Could you clarify the request?")
        assert state["capability"] == "clarification"
        assert state["weak_route"] == CLARIFICATION_ROUTE_V25
        assert all(
            block._layercake_residual_routes.tolist() == [CLARIFICATION_ROUTE_V25]
            for block in host.model.transformer.h
        )
        assert active["strict_assigned_tensor_count"] == len(tensors)
        assert active["authenticated_tensor_count"] == len(tensors)
        assert active["meta_tensors_after_adoption"] == 0
        outputs[device] = host.generate("Could you clarify the request?", maximum_tokens=8)
    assert len(set(outputs.values())) == 1


def test_v25_rejects_v24_missing_feature_and_wrong_route_count(tmp_path):
    package, v22, public, signer, _, _ = _v25_fixture(tmp_path / "valid")
    del package
    host = ClarificationRouteAllocationBoundedCoreHost(
        tmp_path / "reject-v22", trust_store={signer: public}
    )
    with pytest.raises(Exception):
        host.activate(v22)
    assert host.registry.list() == []

    missing, _, public, signer, _, _ = _v25_fixture(tmp_path / "missing", feature=False)
    host = ClarificationRouteAllocationBoundedCoreHost(
        tmp_path / "reject-feature", trust_store={signer: public}
    )
    with pytest.raises(Exception, match="clarification-route"):
        host.activate(missing)
    assert host.registry.list() == []

    wrong, _, public, signer, _, _ = _v25_fixture(tmp_path / "wrong", routes=4)
    host = ClarificationRouteAllocationBoundedCoreHost(
        tmp_path / "reject-routes", trust_store={signer: public}
    )
    with pytest.raises(Exception, match="residual geometry"):
        host.activate(wrong)
    assert host.registry.list() == []


def test_v25_canonical_abi_hash_is_bound():
    path = (
        Path(__file__).resolve().parents[1]
        / "moonshot/canonical_route_isolated_clarification_core_abi_v25.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256
    )
    assert json.loads(path.read_text())["abi_version"] == (
        ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION
    )
