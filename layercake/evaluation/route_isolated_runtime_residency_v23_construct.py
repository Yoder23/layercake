"""Construct-certify the isolated v23 single-parse route-isolated host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import torch

import layercake.cake.installer as installer_module
from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, load_package
from layercake.evaluation.route_isolated_format_literal_v22_construct import (
    _fixture,
    _keys,
)
from layercake.portable_domain import canonical_json_hash
from layercake_extensions.route_isolated_runtime_residency_core_v23 import (
    ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256,
    ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_VERSION,
    SINGLE_PARSE_ACTIVATION_FEATURE,
    RuntimeResidencyFormatLiteralCoreHost,
)


class ConstructError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v23_fixture(directory: Path, *, capability: str, include_feature: bool = True):
    v22, public, signer, tensors = _fixture(
        directory / "source-v22", capability=capability
    )
    loaded = load_package(v22, trust_store={signer: public}, require_signature=True)
    document = loaded.manifest.canonical_dict()
    features = list(document["minimum_host_capabilities"]["features"])
    if include_feature:
        features.append(SINGLE_PARSE_ACTIVATION_FEATURE)
    document.update(
        {
            "cake_id": f"runtime-residency-v23-{capability}-{int(include_feature)}",
            "version": "23.0.0",
            "abi_version": ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_VERSION,
            "abi_hash": ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256,
            "minimum_host_capabilities": {"features": features},
            "tensor_payload_hash": "",
            "package_hash": "",
        }
    )
    private, _, _ = _keys()
    package = build_package(
        directory / f"v23-{capability}-{int(include_feature)}.cake",
        CakeManifest.from_dict(document),
        tensors,
        private_key=private,
    )
    return package, v22, public, signer, tensors, loaded


def _activate_once(host, package) -> tuple[dict[str, Any], int]:
    original = installer_module.load_package
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    installer_module.load_package = counted
    try:
        active = host.activate(package)
    finally:
        installer_module.load_package = original
    return active, calls


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format")
        != "layercake-postrelease-route-isolated-v23-runtime-residency-execution/1"
        or protocol.get("status") != "PREREGISTERED_CONSTRUCT_EXECUTION"
    ):
        raise ConstructError("v23 construct protocol changed")
    for relative, expected in protocol["bindings"].items():
        if _sha(root / relative) != expected:
            raise ConstructError(f"v23 construct binding changed: {relative}")
    focused = subprocess.run(
        [
            "C:\\Python310\\python.exe",
            "-m",
            "pytest",
            "tests/test_route_isolated_runtime_residency_core_v23.py",
            "-q",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sealed = subprocess.run(
        [
            "C:\\Python310\\python.exe",
            "-m",
            "layercake.moonshot_campaign",
            "verify-all",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sealed_result = json.loads(sealed.stdout)
    junit = ET.parse(root / protocol["complete_junit"]).getroot()
    tests = sum(int(node.attrib.get("tests", 0)) for node in junit.iter("testsuite"))
    failures = sum(int(node.attrib.get("failures", 0)) for node in junit.iter("testsuite"))
    errors = sum(int(node.attrib.get("errors", 0)) for node in junit.iter("testsuite"))
    format_prompt = (
        "Return exactly two plain-text lines and no Markdown. "
        "The first line must be `item: green notebook` and the second line must be "
        "`code: N390098UMA`."
    )
    expected = b"item: green notebook\ncode: N390098UMA"
    with tempfile.TemporaryDirectory(prefix="layercake-runtime-residency-v23-") as raw:
        temporary = Path(raw)
        package, v22, public, signer, tensors, loaded_v22 = _v23_fixture(
            temporary / "valid", capability="format_control"
        )
        missing, _, missing_public, missing_signer, _, _ = _v23_fixture(
            temporary / "missing",
            capability="format_control",
            include_feature=False,
        )
        executions = {}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            host = RuntimeResidencyFormatLiteralCoreHost(
                temporary / f"registry-{device}",
                trust_store={signer: public},
                device=device,
            )
            active, parses = _activate_once(host, package)
            output = host.generate(format_prompt, maximum_tokens=32)
            host._set_residual_route(-1)
            first = host.model.transformer.h[0]._layercake_residual_routes
            host._set_residual_route(-1)
            reused = host.model.transformer.h[0]._layercake_residual_routes is first
            host._set_residual_route_batch(-1, 6)
            batch = host.model.transformer.h[0]._layercake_residual_routes
            host._set_residual_route(0)
            changed = host.model.transformer.h[0]._layercake_residual_routes
            executions[device] = {
                "active": active,
                "parses": parses,
                "output_hex": output.hex(),
                "route_reused": reused,
                "batch_changed": batch is not first and tuple(batch.shape) == (6,),
                "route_changed": changed is not batch and changed.tolist() == [0],
            }
        v22_rejected = False
        v22_registry_empty = False
        reject_v22 = RuntimeResidencyFormatLiteralCoreHost(
            temporary / "reject-v22", trust_store={signer: public}
        )
        try:
            reject_v22.activate(v22)
        except Exception:
            v22_rejected = True
            v22_registry_empty = reject_v22.registry.list() == []
        missing_rejected = False
        missing_registry_empty = False
        reject_missing = RuntimeResidencyFormatLiteralCoreHost(
            temporary / "reject-missing",
            trust_store={missing_signer: missing_public},
        )
        try:
            reject_missing.activate(missing)
        except Exception:
            missing_rejected = True
            missing_registry_empty = reject_missing.registry.list() == []
        loaded_v23 = load_package(
            package, trust_store={signer: public}, require_signature=True
        )
        checks = {
            "cuda_available": torch.cuda.is_available(),
            "canonical_v23_abi_identity": _sha(root / protocol["canonical_abi"])
            == ROUTE_ISOLATED_RUNTIME_RESIDENCY_CORE_V23_ABI_SHA256,
            "sealed_shared_sources_unchanged": all(
                _sha(root / relative) == expected_hash
                for relative, expected_hash in protocol["sealed_shared_sources"].items()
            ),
            "v22_tensor_schema_and_values_unchanged": set(loaded_v23.tensors)
            == set(tensors)
            and all(
                torch.equal(loaded_v23.tensors[name], tensors[name]) for name in tensors
            )
            and loaded_v23.manifest.tensor_payload_hash
            == loaded_v22.manifest.tensor_payload_hash,
            "v22_manifest_rejected_before_registry_mutation": v22_rejected
            and v22_registry_empty,
            "missing_feature_rejected_before_registry_mutation": missing_rejected
            and missing_registry_empty,
            "one_authenticated_parse_per_activation": all(
                value["parses"] == value["active"]["authenticated_package_parses"] == 1
                for value in executions.values()
            ),
            "same_cpu_cuda_output": set(executions) == {"cpu", "cuda"}
            and {value["output_hex"] for value in executions.values()}
            == {expected.hex()},
            "same_package_and_payload_all_devices": len(
                {value["active"]["archive_hash"] for value in executions.values()}
            )
            == 1
            and len({value["active"]["payload_hash"] for value in executions.values()})
            == 1,
            "route_reuse_and_changes": all(
                value["route_reused"]
                and value["batch_changed"]
                and value["route_changed"]
                for value in executions.values()
            ),
            "receiver_learning_zero": all(
                value["active"]["receiver_training_steps"]
                == value["active"]["receiver_calibration_runs"]
                == 0
                for value in executions.values()
            ),
            "focused_tests_pass": "4 passed" in focused.stdout,
            "complete_tests_pass": tests == 682 and failures == errors == 0,
            "sealed_campaign_unchanged": sealed_result.get("completed_phases_valid")
            is True,
        }
        result = {
            "format": "layercake-postrelease-route-isolated-v23-runtime-residency-construct-result/1",
            "status": "PASS_CONSTRUCT_ONLY" if all(checks.values()) else "FAIL",
            "protocol_sha256": _sha(protocol_path),
            "checks": checks,
            "executions": executions,
            "complete_tests": {"tests": tests, "failures": failures, "errors": errors},
            "sealed_campaign": sealed_result,
            "tensor_payload_hash": loaded_v23.manifest.tensor_payload_hash,
            "external_artifact_used": False,
            "english_quality_tested": False,
            "performance_tested": False,
            "receiver_training_steps": 0,
            "receiver_calibration_runs": 0,
            "claim_boundary": "Generic isolated v23 construct only; no external acquisition, English quality, information minimum, physical performance, phase recertification, or ABI-superiority claim.",
        }
        result["evidence_sha256"] = canonical_json_hash(result)
        return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise ConstructError(f"immutable v23 construct output exists: {output}")
    result = execute(root, (root / args.protocol).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
