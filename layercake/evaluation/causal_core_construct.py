"""Construct-certify the generic tied native causal direct-core host v6."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch

from layercake.portable_token_plan import BOS_ID
from layercake_extensions.causal_direct_neural_core import (
    CAUSAL_DIRECT_NEURAL_CORE_ABI_SHA256,
    CAUSAL_DIRECT_NEURAL_CORE_ABI_VERSION,
    CausalDirectNeuralCoreHost,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path, path: Path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-causal-core-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise RuntimeError("causal core construct governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"causal core construct binding changed: {relative}")
    return protocol, sha(path)


def fixture_module(root: Path):
    path = root / "tests/models/test_causal_direct_neural_core.py"
    spec = importlib.util.spec_from_file_location("causal_core_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    fixture = fixture_module(root)
    with tempfile.TemporaryDirectory(prefix="lc-causal-core-v6-") as name:
        temp = Path(name)
        package, public, key_id, state_hash, tokenizer = fixture._package(temp / "source")
        archive_sha = sha(package)
        cpu = CausalDirectNeuralCoreHost(temp / "cpu", trust_store={key_id: public})
        active = cpu.activate(package)
        state = cpu.prefill("hello world")
        packed = torch.tensor([tokenizer.encode_source("hello world")[0] + [BOS_ID]], dtype=torch.long)
        full_logits = cpu.module(packed)[:, -1]
        incremental_match = torch.allclose(state.next_logits, full_logits, atol=1e-6, rtol=1e-6)
        initial_cache = tuple(value.shape[1] for value in state.layer_inputs)
        _, state = cpu.decode_step(state)
        realized = cpu.realize(state)
        generated = cpu.generate("hello world")
        verified = cpu.verify()
        cpu.remove()
        reinstall = cpu.activate(package)
        cuda = None
        if torch.cuda.is_available():
            host = CausalDirectNeuralCoreHost(temp / "cuda", trust_store={key_id: public}, device="cuda")
            cuda_active = host.activate(package)
            cuda = {
                "archive_hash": cuda_active["archive_hash"],
                "payload_hash": cuda_active["payload_hash"],
                "state_dict_hash": cuda_active["state_dict_hash"],
                "generated_hex": host.generate("hello world").hex(),
            }
        checks = {
            "abi_identity": CAUSAL_DIRECT_NEURAL_CORE_ABI_VERSION == "lc-direct-neural-core/6" and CAUSAL_DIRECT_NEURAL_CORE_ABI_SHA256 == protocol["interface_sha256"],
            "canonical_abi_hash": sha(root / protocol["canonical_abi"]) == CAUSAL_DIRECT_NEURAL_CORE_ABI_SHA256,
            "cpu_state_identity": active["state_dict_hash"] == state_hash,
            "cpu_zero_learning": active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0,
            "full_incremental_first_step_identity": bool(incremental_match),
            "persistent_layer_state": len(initial_cache) == 2 and all(value > 1 for value in initial_cache),
            "persistent_state_realizes_bytes": isinstance(realized, bytes),
            "cpu_generation_strict": generated == b"",
            "lifecycle_identity": archive_sha == sha(package) and reinstall["archive_hash"] == active["archive_hash"],
            "cpu_verifier_pass": verified["status"] == "PASS",
            "cuda_available": torch.cuda.is_available(),
            "same_package_cuda": cuda is not None and cuda["archive_hash"] == active["archive_hash"] and cuda["payload_hash"] == active["payload_hash"] and cuda["state_dict_hash"] == active["state_dict_hash"] and cuda["generated_hex"] == generated.hex(),
        }
    result = {
        "format": "layercake-postrelease-causal-core-construct-result/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "checks": checks,
        "package": {"payload_hash": active["payload_hash"], "state_dict_hash": state_hash},
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "external_artifact_used": False,
        "english_quality_tested": False,
        "performance_tested": False,
        "claim_boundary": "Generic decoder-only causal host construct only; no external acquisition, English quality, or performance claim.",
    }
    result["evidence_sha256"] = hashlib.sha256((json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol)
    output = root / args.output
    if args.command == "execute":
        if output.exists():
            raise RuntimeError("causal core construct result immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result:
        raise RuntimeError("stored causal core construct differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
