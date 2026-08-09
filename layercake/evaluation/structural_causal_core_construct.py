"""Construct-certify the generic Phi-compatible structural causal host v7."""

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
from layercake.structural_causal_core import StructuralCausalCore
from layercake_extensions.structural_causal_core import (
    STRUCTURAL_CAUSAL_CORE_ABI_SHA256,
    STRUCTURAL_CAUSAL_CORE_ABI_VERSION,
    StructuralCausalCoreHost,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path, path: Path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-structural-causal-core-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise RuntimeError("structural causal construct governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"structural causal construct binding changed: {relative}")
    return protocol, sha(path)


def fixture_module(root: Path):
    path = root / "tests/models/test_structural_causal_core.py"
    spec = importlib.util.spec_from_file_location("structural_causal_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    fixture = fixture_module(root)
    with tempfile.TemporaryDirectory(prefix="lc-structural-causal-v7-") as name:
        temp = Path(name)
        package, public, key_id, state_hash, tokenizer = fixture._package(temp / "source")
        archive_sha = sha(package)
        cpu = StructuralCausalCoreHost(temp / "cpu", trust_store={key_id: public})
        active = cpu.activate(package)
        source_ids = tokenizer.encode_source("hello world")[0]
        state = cpu.prefill("hello world")
        packed = torch.tensor([source_ids + [BOS_ID]], dtype=torch.long)
        first_match = torch.allclose(state.next_logits, cpu.module(packed)[:, -1], atol=1e-5, rtol=1e-5)
        initial_cache = tuple(value.shape[2] for value in state.layer_keys)
        # The lifecycle fixture intentionally makes EOS the first action.  Use a
        # valid nonterminal stimulus here so the construct verifier exercises a
        # real incremental cache append instead of terminating at the boundary.
        state.next_logits.zero_()
        state.next_logits[0, 4] = 1.0
        action, state = cpu.decode_step(state)
        packed = torch.tensor([source_ids + [BOS_ID, action]], dtype=torch.long)
        second_match = torch.allclose(state.next_logits, cpu.module(packed)[:, -1], atol=1e-5, rtol=1e-5)
        expanded_cache = tuple(value.shape[2] for value in state.layer_keys)
        realized = cpu.realize(state)
        generated = cpu.generate("hello world")
        verified = cpu.verify()
        cpu.remove()
        reinstall = cpu.activate(package)
        cuda = None
        if torch.cuda.is_available():
            host = StructuralCausalCoreHost(temp / "cuda", trust_store={key_id: public}, device="cuda")
            cuda_active = host.activate(package)
            cuda = {
                "archive_hash": cuda_active["archive_hash"],
                "payload_hash": cuda_active["payload_hash"],
                "state_dict_hash": cuda_active["state_dict_hash"],
                "generated_hex": host.generate("hello world").hex(),
            }
        target_parameters = StructuralCausalCore(fixed_vocab_size=32_015).parameter_count()
        checks = {
            "abi_identity": STRUCTURAL_CAUSAL_CORE_ABI_VERSION == "lc-direct-neural-core/7" and STRUCTURAL_CAUSAL_CORE_ABI_SHA256 == protocol["interface_sha256"],
            "canonical_abi_hash": sha(root / protocol["canonical_abi"]) == STRUCTURAL_CAUSAL_CORE_ABI_SHA256,
            "preregistered_target_parameter_count": target_parameters == 14_654_784,
            "cpu_state_identity": active["state_dict_hash"] == state_hash,
            "cpu_zero_learning": active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0,
            "full_incremental_first_step_identity": bool(first_match),
            "full_incremental_second_step_identity": bool(second_match),
            "persistent_rotary_kv_state": len(initial_cache) == 2 and all(after == before + 1 for before, after in zip(initial_cache, expanded_cache)),
            "persistent_state_realizes_bytes": isinstance(realized, bytes),
            "cpu_generation_strict": generated == b"",
            "lifecycle_identity": archive_sha == sha(package) and reinstall["archive_hash"] == active["archive_hash"],
            "cpu_verifier_pass": verified["status"] == "PASS",
            "cuda_available": torch.cuda.is_available(),
            "same_package_cuda": cuda is not None and cuda["archive_hash"] == active["archive_hash"] and cuda["payload_hash"] == active["payload_hash"] and cuda["state_dict_hash"] == active["state_dict_hash"] and cuda["generated_hex"] == generated.hex(),
        }
    result = {
        "format": "layercake-postrelease-structural-causal-core-construct-result/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "checks": checks,
        "package": {"payload_hash": active["payload_hash"], "state_dict_hash": state_hash},
        "target_parameters": target_parameters,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "external_artifact_used": False,
        "english_quality_tested": False,
        "performance_tested": False,
        "claim_boundary": "Generic source-compatible causal host construct only; no external acquisition, English quality, or performance claim.",
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
            raise RuntimeError("structural causal construct result immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result:
        raise RuntimeError("stored structural causal construct differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
