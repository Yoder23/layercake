"""Construct-certify the generic selective-boundary BPE direct-core host v5."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch

from layercake_extensions.selective_boundary_bpe_direct_neural_core import (
    SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_SHA256,
    SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_VERSION,
    SelectiveBoundaryBpeDirectNeuralCoreHost,
    SelectiveBoundaryBpeTokenizer,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path, path: Path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-selective-boundary-bpe-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise RuntimeError("selective BPE construct governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"selective BPE construct binding changed: {relative}")
    return protocol, sha(path)


def fixture_module(root: Path):
    path = root / "tests/models/test_selective_boundary_bpe_direct_neural_core.py"
    spec = importlib.util.spec_from_file_location("selective_bpe_fixture", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    fixture = fixture_module(root)
    tokenizer = SelectiveBoundaryBpeTokenizer(fixture._doc())
    sample = "Alpha_77 Beta"
    pieces = tokenizer.split(sample)
    with tempfile.TemporaryDirectory(prefix="lc-selective-bpe-v5-") as name:
        temp = Path(name)
        package, public, key_id, state, _ = fixture._package(temp / "source")
        archive_sha = sha(package)
        cpu = SelectiveBoundaryBpeDirectNeuralCoreHost(temp / "cpu", trust_store={key_id: public})
        active = cpu.activate(package)
        incremental = cpu.prefill(sample); _, incremental = cpu.decode_step(incremental)
        realized = cpu.realize(incremental); generated = cpu.generate(sample); verified = cpu.verify()
        cpu.remove(); reinstall = cpu.activate(package)
        cuda = None
        if torch.cuda.is_available():
            host = SelectiveBoundaryBpeDirectNeuralCoreHost(temp / "cuda", trust_store={key_id: public}, device="cuda")
            cuda_active = host.activate(package)
            cuda = {"archive_hash": cuda_active["archive_hash"], "payload_hash": cuda_active["payload_hash"], "state_dict_hash": cuda_active["state_dict_hash"], "generated_hex": host.generate(sample).hex()}
        checks = {
            "abi_identity": SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_VERSION == "lc-direct-neural-core/5" and SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_SHA256 == protocol["interface_sha256"],
            "selective_split_exact": b"".join(pieces) == sample.encode("utf-8"),
            "identifier_boundary_stable": pieces[0] == b"Alpha_77",
            "all_actions_valid_utf8": all(piece.decode("utf-8").encode("utf-8") == piece for piece in pieces),
            "cpu_state_identity": active["state_dict_hash"] == state,
            "cpu_zero_learning": active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0,
            "persistent_state_realizes_bytes": isinstance(realized, bytes),
            "cpu_generation_strict": generated == b"",
            "lifecycle_identity": archive_sha == sha(package) and reinstall["archive_hash"] == active["archive_hash"],
            "cpu_verifier_pass": verified["status"] == "PASS",
            "cuda_available": torch.cuda.is_available(),
            "same_package_cuda": cuda is not None and cuda["archive_hash"] == active["archive_hash"] and cuda["payload_hash"] == active["payload_hash"] and cuda["state_dict_hash"] == active["state_dict_hash"] and cuda["generated_hex"] == generated.hex(),
        }
    result = {"format": "layercake-postrelease-selective-boundary-bpe-construct-result/1", "status": "PASS" if all(checks.values()) else "FAIL", "protocol": {"path": protocol_path.name, "sha256": protocol_sha}, "checks": checks, "package": {"payload_hash": active["payload_hash"], "state_dict_hash": state}, "receiver_training_steps": 0, "receiver_calibration_runs": 0, "external_artifact_used": False, "english_quality_tested": False, "performance_tested": False, "claim_boundary": "Generic host construct only; no ABI acquisition, English quality, or performance claim."}
    result["evidence_sha256"] = hashlib.sha256((json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("execute", "verify")); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(argv)
    root = Path.cwd().resolve(); result = execute(root, root / args.protocol); output = root / args.output
    if args.command == "execute":
        if output.exists(): raise RuntimeError("selective BPE construct result immutable")
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result:
        raise RuntimeError("stored selective BPE construct differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2)); return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
