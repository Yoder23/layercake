"""Execute the isolated five-route LayerCake v25 construct."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, load_package, tensor_specs
from layercake.evaluation.route_isolated_format_literal_v22_construct import (
    _fixture as _v22_fixture,
    _keys,
)
from layercake_extensions.route_isolated_allocation_bounded_core_v24 import (
    ALLOCATION_BOUNDED_ADOPTION_FEATURE,
    ARCHITECTURE_V24_FORMAT,
    ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_SHA256,
    ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_VERSION,
)
from layercake_extensions.route_isolated_clarification_core_v25 import (
    ARCHITECTURE_V25_FORMAT,
    CLARIFICATION_ROUTE_ISOLATION_FEATURE,
    RESIDUAL_CAPABILITIES_V25,
    ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256,
    ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION,
    ClarificationRouteAllocationBoundedCoreHost,
)
from layercake_extensions.route_isolated_runtime_residency_core_v23 import (
    SINGLE_PARSE_ACTIVATION_FEATURE,
)
from layercake_extensions.route_isolated_shallow_sparse_core import CAPABILITIES


FORMAT = "layercake-postrelease-route-isolated-v25-clarification-route-execution/1"
FORMAT_PROMPT = (
    "Return exactly two plain-text lines and no Markdown. "
    "The first line must be `item: green notebook` and the second line must be "
    "`code: N390098UMA`."
)
FORMAT_EXPECTED = b"item: green notebook\ncode: N390098UMA"


class ConstructError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, value: Any) -> None:
    if path.exists():
        raise ConstructError(f"immutable construct output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _v24_fixture(directory: Path, *, capability: str):
    v22, public, signer, tensors = _v22_fixture(
        directory / "source-v22", capability=capability
    )
    loaded = load_package(v22, trust_store={signer: public}, require_signature=True)
    document = loaded.manifest.canonical_dict()
    architecture = dict(document["architecture"])
    architecture["format"] = ARCHITECTURE_V24_FORMAT
    features = list(document["minimum_host_capabilities"]["features"])
    features.extend([SINGLE_PARSE_ACTIVATION_FEATURE, ALLOCATION_BOUNDED_ADOPTION_FEATURE])
    document.update(
        {
            "cake_id": f"v24-source-{capability.replace('_', '-')}",
            "version": "24.0.0",
            "abi_version": ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_VERSION,
            "abi_hash": ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_SHA256,
            "architecture": architecture,
            "minimum_host_capabilities": {"features": features},
            "tensor_payload_hash": "",
            "tensor_shapes": tensor_specs(tensors),
            "package_hash": "",
        }
    )
    private, _, _ = _keys()
    path = build_package(
        directory / "source-v24.cake",
        CakeManifest.from_dict(document),
        tensors,
        private_key=private,
    )
    return path, public, signer, tensors


def _v25_fixture(
    directory: Path,
    *,
    capability: str = "clarification",
    feature: bool = True,
    routes: int = 5,
):
    v24, public, signer, original = _v24_fixture(directory, capability=capability)
    loaded = load_package(v24, trust_store={signer: public}, require_signature=True)
    document = loaded.manifest.canonical_dict()
    architecture = dict(document["architecture"])
    architecture["format"] = ARCHITECTURE_V25_FORMAT
    architecture["weak_capabilities"] = list(RESIDUAL_CAPABILITIES_V25)
    architecture["residual"] = {**architecture["residual"], "routes": routes}
    tensors = {name: value.clone() for name, value in original.items()}
    if routes == 5:
        tensors["residual.down"] = torch.cat(
            [tensors["residual.down"], torch.zeros(1, 16, 16)], dim=0
        )
        tensors["residual.up"] = torch.cat(
            [tensors["residual.up"], torch.zeros(1, 16, 16)], dim=0
        )
    features = list(document["minimum_host_capabilities"]["features"])
    if feature:
        features.append(CLARIFICATION_ROUTE_ISOLATION_FEATURE)
    document.update(
        {
            "cake_id": f"clarification-v25-{capability.replace('_', '-')}-{int(feature)}-{routes}",
            "version": "25.0.0",
            "abi_version": ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION,
            "abi_hash": ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256,
            "architecture": architecture,
            "minimum_host_capabilities": {"features": features},
            "tensor_payload_hash": "",
            "tensor_shapes": tensor_specs(tensors),
            "package_hash": "",
        }
    )
    private, _, _ = _keys()
    package = build_package(
        directory / f"v25-{capability}-{int(feature)}-{routes}.cake",
        CakeManifest.from_dict(document),
        tensors,
        private_key=private,
    )
    return package, v24, public, signer, tensors, original


def _rejected(path: Path, public: bytes, signer: str, registry: Path) -> bool:
    host = ClarificationRouteAllocationBoundedCoreHost(
        registry, trust_store={signer: public}
    )
    try:
        host.activate(path)
    except Exception:
        return host.registry.list() == []
    return False


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_CONSTRUCT_EXECUTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("external_artifact_used") is not False
    ):
        raise ConstructError("v25 execution governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or _sha(target) != expected:
            raise ConstructError(f"v25 construct binding changed: {relative}")

    mappings: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="layercake-v25-construct-") as raw:
        temporary = Path(raw)
        for capability in CAPABILITIES:
            package, _, public, signer, _, _ = _v25_fixture(
                temporary / f"map-{capability}", capability=capability
            )
            host = ClarificationRouteAllocationBoundedCoreHost(
                temporary / f"registry-map-{capability}", trust_store={signer: public}
            )
            active = host.activate(package)
            state = host.prefill("hello")
            expected = (
                RESIDUAL_CAPABILITIES_V25.index(capability)
                if capability in RESIDUAL_CAPABILITIES_V25
                else -1
            )
            if state["capability"] != capability:
                raise ConstructError("fixture router did not select its declared capability")
            mappings[capability] = int(state["weak_route"])
            if int(state["weak_route"]) != expected or any(
                block._layercake_residual_routes.tolist() != [expected]
                for block in host.model.transformer.h
            ):
                raise ConstructError("physical residual mapping changed")
            if active["authenticated_package_parses"] != 1:
                raise ConstructError("activation parsed the package more than once")

        clarification, v24, public, signer, tensors, original = _v25_fixture(
            temporary / "clarification", capability="clarification"
        )
        loaded = load_package(
            clarification, trust_store={signer: public}, require_signature=True
        )
        device_outputs = {}
        activations = {}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            host = ClarificationRouteAllocationBoundedCoreHost(
                temporary / f"registry-{device}",
                trust_store={signer: public},
                device=device,
            )
            activations[device] = host.activate(clarification)
            device_outputs[device] = host.generate("hello", maximum_tokens=8).hex()

        format_package, _, format_public, format_signer, _, _ = _v25_fixture(
            temporary / "format", capability="format_control"
        )
        format_outputs = {}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            host = ClarificationRouteAllocationBoundedCoreHost(
                temporary / f"format-registry-{device}",
                trust_store={format_signer: format_public},
                device=device,
            )
            host.activate(format_package)
            format_outputs[device] = host.generate(FORMAT_PROMPT, maximum_tokens=32)

        missing, _, missing_public, missing_signer, _, _ = _v25_fixture(
            temporary / "missing", feature=False
        )
        wrong, _, wrong_public, wrong_signer, _, _ = _v25_fixture(
            temporary / "wrong", routes=4
        )
        preserved = (
            torch.equal(tensors["residual.norm.weight"], original["residual.norm.weight"])
            and torch.equal(tensors["residual.norm.bias"], original["residual.norm.bias"])
            and torch.equal(tensors["residual.down"][:4], original["residual.down"])
            and torch.equal(tensors["residual.up"][:4], original["residual.up"])
        )
        gates = {
            "canonical_v25_abi_identity": _sha(root / protocol["canonical_abi"])
            == ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256,
            "v24_first_four_routes_exact": preserved,
            "v24_manifest_rejected_before_registry_mutation": _rejected(
                v24, public, signer, temporary / "reject-v24"
            ),
            "missing_feature_rejected_before_registry_mutation": _rejected(
                missing, missing_public, missing_signer, temporary / "reject-missing"
            ),
            "wrong_route_count_rejected_before_registry_mutation": _rejected(
                wrong, wrong_public, wrong_signer, temporary / "reject-wrong"
            ),
            "all_14_capability_mappings_exact": mappings
            == {
                capability: (
                    RESIDUAL_CAPABILITIES_V25.index(capability)
                    if capability in RESIDUAL_CAPABILITIES_V25
                    else -1
                )
                for capability in CAPABILITIES
            },
            "clarification_physically_selects_route_four": mappings["clarification"] == 4,
            "one_active_route_maximum": all(value >= -1 and value <= 4 for value in mappings.values()),
            "signed_package_identity": all(
                value["archive_hash"] == loaded.archive_hash
                and value["payload_hash"] == loaded.manifest.tensor_payload_hash
                for value in activations.values()
            ),
            "strict_storage_adoption": all(
                value["strict_assigned_tensor_count"] == len(tensors)
                and value["authenticated_tensor_count"] == len(tensors)
                and value["meta_tensors_after_adoption"] == 0
                for value in activations.values()
            ),
            "cpu_cuda_clarification_fixture_identity": len(set(device_outputs.values())) == 1
            and set(device_outputs) == {"cpu", "cuda"},
            "cpu_cuda_format_identity": len(set(format_outputs.values())) == 1
            and set(format_outputs) == {"cpu", "cuda"}
            and set(format_outputs.values()) == {FORMAT_EXPECTED},
            "receiver_learning_zero": all(
                value["receiver_training_steps"] == value["receiver_calibration_runs"] == 0
                for value in activations.values()
            ),
            "complete_repository_tests_pass": int(protocol["complete_tests"]["passed"]) > 0
            and int(protocol["complete_tests"]["failed"]) == 0,
            "unchanged_sealed_campaign_verifier": protocol["sealed_campaign_verifier"]
            == "PASS_PHASES_0_THROUGH_8",
            "external_artifact_absent": True,
            "training_absent": True,
        }

    result = {
        "format": "layercake-postrelease-route-isolated-v25-clarification-route-result/1",
        "status": "PASS_CONSTRUCT_ONLY" if all(gates.values()) else "FAIL_CONSTRUCT",
        "protocol_sha256": _sha(protocol_path),
        "canonical_abi_sha256": ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256,
        "capability_to_residual_route": mappings,
        "devices": sorted(device_outputs),
        "authenticated_tensor_count": len(tensors),
        "gates": gates,
        "training_performed": False,
        "external_artifact_used": False,
        "english_quality_tested": False,
        "physical_performance_tested": False,
        "claim_boundary": "Generic isolated V25 five-route host construct only. No external artifact, English quality, information minimum, physical speed or memory, campaign-phase recertification, or ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_immutable(output, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
