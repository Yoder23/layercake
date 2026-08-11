"""Execute the exact explicit-route residual v18 host construct."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Iterable

import torch

from layercake.evaluation.route_isolated_shallow_sparse_construct import _fixture
from layercake.portable_domain import canonical_json_hash
from layercake_extensions.route_isolated_shallow_sparse_core import WEAK_CAPABILITIES
from layercake_extensions.route_isolated_shallow_sparse_core_v18 import (
    ARCHITECTURE_V18_FORMAT,
    ROUTE_ISOLATED_CORE_V18_ABI_SHA256,
    ROUTE_ISOLATED_CORE_V18_ABI_VERSION,
    ExactRouteIsolatedShallowSparseCoreHost,
    ExplicitRouteResidual,
)


class ConstructError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-route-isolated-shallow-sparse-v18-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise ConstructError("v18 construct protocol changed")
    for relative, expected in protocol["bindings"].items():
        if _sha(root / relative) != expected:
            raise ConstructError(f"v18 construct binding changed: {relative}")
    focused = subprocess.run(["C:\\Python310\\python.exe", "-m", "pytest", "tests/test_route_isolated_shallow_sparse_core.py", "-q"], cwd=root, check=True, capture_output=True, text=True)
    sealed = subprocess.run(["C:\\Python310\\python.exe", "-m", "layercake.moonshot_campaign", "verify-all"], cwd=root, check=True, capture_output=True, text=True)
    sealed_result = json.loads(sealed.stdout)
    with tempfile.TemporaryDirectory(prefix="layercake-route-isolated-v18-") as raw:
        temp = Path(raw)
        package, public, signer, tensors = _fixture(
            temp,
            abi_version=ROUTE_ISOLATED_CORE_V18_ABI_VERSION,
            abi_hash=ROUTE_ISOLATED_CORE_V18_ABI_SHA256,
            architecture_format=ARCHITECTURE_V18_FORMAT,
            residual_type=ExplicitRouteResidual,
        )
        residual_shapes = {name: list(value.shape) for name, value in tensors.items() if name.startswith("residual.")}
        executions = {}
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            host = ExactRouteIsolatedShallowSparseCoreHost(temp / f"registry-{device}", trust_store={signer: public}, device=device)
            active = host.activate(package)
            state = host.prefill("hello")
            generated = host.generate("hello", maximum_tokens=2)
            verified = host.verify()
            removed = host.remove()
            reactivated = host.activate(package)
            executions[device] = {"active": active, "cache_present": state["past_key_values"] is not None, "generated_hex": generated.hex(), "verify": verified, "remove": removed, "reactivated": reactivated}
        checks = {
            "canonical_abi_identity": _sha(root / protocol["canonical_abi"]) == ROUTE_ISOLATED_CORE_V18_ABI_SHA256,
            "explicit_route_tensor_keys": set(residual_shapes) == {"residual.down", "residual.up", "residual.norm.weight", "residual.norm.bias"},
            "explicit_route_axes": residual_shapes["residual.down"][:2] == [len(WEAK_CAPABILITIES), 16] and residual_shapes["residual.up"][0] == len(WEAK_CAPABILITIES) and residual_shapes["residual.up"][2] == 16,
            "persistent_state": all(value["cache_present"] for value in executions.values()),
            "same_archive_all_devices": len({value["active"]["archive_hash"] for value in executions.values()}) == 1,
            "same_payload_all_devices": len({value["active"]["payload_hash"] for value in executions.values()}) == 1,
            "same_generation_all_devices": len({value["generated_hex"] for value in executions.values()}) == 1,
            "receiver_learning_zero": all(value["active"]["receiver_training_steps"] == value["active"]["receiver_calibration_runs"] == 0 for value in executions.values()),
            "focused_tests_pass": "5 passed" in focused.stdout,
            "sealed_campaign_unchanged": sealed_result.get("completed_phases_valid") is True,
        }
        result = {
            "format": "layercake-postrelease-route-isolated-shallow-sparse-v18-construct-result/1",
            "status": "PASS_CONSTRUCT_ONLY" if all(checks.values()) else "FAIL",
            "protocol_sha256": _sha(protocol_path),
            "interface": ROUTE_ISOLATED_CORE_V18_ABI_VERSION,
            "interface_sha256": ROUTE_ISOLATED_CORE_V18_ABI_SHA256,
            "checks": checks,
            "residual_shapes": residual_shapes,
            "devices": executions,
            "package_sha256": _sha(package),
            "package_bytes": package.stat().st_size,
            "teacher_present": False,
            "source_transformer_blocks": 0,
            "historical_release_changed": False,
            "hardware": {"machine": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "claim_boundary": "Generic exact-schema v18 host construct only; no external artifact, English quality, information minimum, performance, or superiority claim.",
        }
        result["evidence_sha256"] = canonical_json_hash(result)
        return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); protocol = root / args.protocol; output = root / args.output
    expected = execute(root, protocol)
    if args.command == "execute":
        if output.exists():
            raise ConstructError(f"immutable output exists: {output}")
        if expected["status"] != "PASS_CONSTRUCT_ONLY":
            raise ConstructError("v18 construct failed")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = expected
    else:
        stored = json.loads(output.read_text(encoding="utf-8"))
        if stored != expected:
            raise ConstructError("stored v18 construct differs from recomputation")
        result = {"status": "PASS", "evidence_sha256": expected["evidence_sha256"], "construct_only": True}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
