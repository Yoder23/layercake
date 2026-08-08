"""Construct-certify the decoder-aware LayerCake direct-core host v4."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch

from layercake_extensions.decoder_direct_neural_core import (
    DECODER_DIRECT_NEURAL_CORE_ABI_SHA256,
    DECODER_DIRECT_NEURAL_CORE_ABI_VERSION,
    DecoderAwareDirectNeuralCoreHost,
    DecoderAwareExternalTokenizer,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(root: Path, path: Path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-decoder-direct-core-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise RuntimeError("decoder-aware construct governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"decoder-aware construct binding changed: {relative}")
    return protocol, sha(path)


def fixture_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_hash = load_protocol(root, protocol_path)
    fixture = fixture_module(root / "tests/models/test_decoder_direct_neural_core.py", "decoder_construct_fixture")
    v3_fixture = fixture_module(root / "tests/models/test_bpe_direct_neural_core.py", "bpe_construct_fixture_v4")
    tokenizer = DecoderAwareExternalTokenizer(fixture._doc())
    sample = "hello world"
    source_actions, _ = tokenizer.encode_source(sample)
    target_actions = tokenizer.encode_fixed_target(sample)
    pointer_rejected = False
    try:
        tokenizer.decode_actions([tokenizer.vocab_size], [])
    except ValueError:
        pointer_rejected = True
    with tempfile.TemporaryDirectory(prefix="lc-decoder-v4-") as name:
        temp = Path(name)
        package, public, key_id, state, _ = fixture._package(temp / "source")
        archive_hash = sha(package)
        cpu = DecoderAwareDirectNeuralCoreHost(temp / "cpu", trust_store={key_id: public})
        cpu_active = cpu.activate(package)
        prefill = cpu.prefill(sample)
        _, prefill = cpu.decode_step(prefill)
        realized = cpu.realize(prefill)
        generated = cpu.generate(sample, maximum_actions=1)
        verified = cpu.verify()
        cpu.remove()
        reinstalled = cpu.activate(package)
        v3_path, v3_public, v3_key_id, _, _ = v3_fixture._package(temp / "v3")
        v3_rejected = False
        try:
            DecoderAwareDirectNeuralCoreHost(temp / "reject-v3", trust_store={v3_key_id: v3_public}).activate(v3_path)
        except Exception:
            v3_rejected = True
        cuda = None
        if torch.cuda.is_available():
            cuda_host = DecoderAwareDirectNeuralCoreHost(temp / "cuda", trust_store={key_id: public}, device="cuda")
            active = cuda_host.activate(package)
            cuda = {"archive_hash": active["archive_hash"], "payload_hash": active["payload_hash"], "state_dict_hash": active["state_dict_hash"], "generated_hex": cuda_host.generate(sample, maximum_actions=1).hex()}
        checks = {
            "abi_identity": DECODER_DIRECT_NEURAL_CORE_ABI_VERSION == "lc-direct-neural-core/4" and DECODER_DIRECT_NEURAL_CORE_ABI_SHA256 == protocol["interface_sha256"],
            "decoder_sequence_roundtrip": tokenizer.decode_actions(target_actions, []) == sample.encode("utf-8"),
            "collision_free_external_actions": min(source_actions) >= 4,
            "pointer_actions_rejected": pointer_rejected,
            "cpu_state_identity": cpu_active["state_dict_hash"] == state,
            "cpu_zero_learning": cpu_active["receiver_training_steps"] == cpu_active["receiver_calibration_runs"] == 0,
            "persistent_state_realizes_utf8": bool(realized.decode("utf-8")),
            "cpu_generation_strict_utf8": bool(generated.decode("utf-8")),
            "lifecycle_identity": archive_hash == sha(package) and reinstalled["archive_hash"] == cpu_active["archive_hash"],
            "cpu_verifier_pass": verified["status"] == "PASS",
            "v3_identity_rejected": v3_rejected,
            "cuda_available": torch.cuda.is_available(),
            "same_package_cuda": cuda is not None and cuda["archive_hash"] == cpu_active["archive_hash"] and cuda["payload_hash"] == cpu_active["payload_hash"] and cuda["state_dict_hash"] == cpu_active["state_dict_hash"] and cuda["generated_hex"] == generated.hex(),
        }
    result = {
        "format": "layercake-postrelease-decoder-direct-core-construct-result/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": {"path": protocol_path.name, "sha256": protocol_hash},
        "checks": checks,
        "package": {"payload_hash": cpu_active["payload_hash"], "state_dict_hash": state},
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "external_artifact_used": False,
        "english_quality_tested": False,
        "performance_tested": False,
        "claim_boundary": "Generic decoder-aware host construct only; no ABI acquisition, English quality, or performance claim.",
    }
    result["evidence_sha256"] = hashlib.sha256((json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", default="moonshot/postrelease_decoder_direct_neural_core_preregistration_v9.json")
    parser.add_argument("--output", default="results/moonshot/postrelease/decoder_direct_neural_core_construct_v4.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    evidence = execute(root, (root / args.protocol).resolve())
    output = (root / args.output).resolve()
    if args.command == "execute":
        if output.exists():
            raise RuntimeError("decoder-aware construct result immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != evidence:
        raise RuntimeError("stored decoder-aware construct differs")
    print(json.dumps({"status": evidence["status"], "evidence_sha256": evidence["evidence_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
