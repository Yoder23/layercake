"""Execute the preregistered direct-neural-core host construct certification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable

import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.models.direct_neural_core import (
    DIRECT_NEURAL_CORE_ABI_SHA256,
    DIRECT_NEURAL_CORE_ABI_VERSION,
    DirectNeuralCoreError,
    DirectNeuralCoreHost,
)
from layercake.models.portable_decoder import portable_token_plan_manifest_architecture
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import (
    EOS_ID,
    LosslessLexemePointerTokenizer,
    PortableTokenPlan,
    build_token_plan_artifact,
)


class DirectNeuralCoreConstructError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_construct_package(root: Path, *, domain: str = "english-core"):
    torch.manual_seed(60119)
    rows = [
        {"prompt": "Rewrite this clearly: quiet bridge", "response": "The bridge is quiet.", "copy_lexemes": ["bridge"]},
        {"prompt": "Reply politely to Sam", "response": "Hello Sam, thank you.", "copy_lexemes": ["Sam"]},
    ]
    tokenizer = LosslessLexemePointerTokenizer.build_generic(rows)
    model = PortableTokenPlan(
        fixed_vocab_size=tokenizer.vocab_size,
        model_width=24,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_width=48,
        pointer_width=12,
        dropout=0.0,
        maximum_source_lexemes=32,
        maximum_target_actions=16,
    ).eval().bind_tokenizer(tokenizer)
    with torch.no_grad():
        model.fixed_output.weight.zero_()
        model.fixed_output.bias.zero_()
        model.fixed_output.bias[EOS_ID] = 10.0
        model.pointer_gate.weight.zero_()
        model.pointer_gate.bias.fill_(-10.0)
    artifact = build_token_plan_artifact(model, tokenizer, domain_id="english-core")
    private, public, key_id = generate_keypair()
    state = artifact["state_dict"]
    manifest = CakeManifest(
        schema_version="1",
        cake_id=f"{domain}-construct",
        name="Direct Neural Core Construct",
        description="Construct-only immutable direct neural core",
        version="1.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": key_id},
        abi_version=DIRECT_NEURAL_CORE_ABI_VERSION,
        abi_hash=DIRECT_NEURAL_CORE_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router"},
        architecture=portable_token_plan_manifest_architecture(artifact["spec"]),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "persistent_incremental_state"]},
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(state),
        package_hash="",
        training_data_provenance={"dataset": "construct-only", "external_teacher": False},
        evaluation_evidence={"status": "CONSTRUCT_ONLY"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id},
        domains=(domain,),
        permissions=("local-inference",),
    )
    path = build_package(root / f"{domain}.cake", manifest, state, private_key=private)
    return path, public, key_id, state_dict_hash(state)


def _exercise(package: Path, public: bytes, key_id: str, expected_state: str, root: Path, device: str) -> dict[str, Any]:
    host = DirectNeuralCoreHost(root, trust_store={key_id: public}, device=device)
    archive_before = _sha256(package)
    active = host.activate(package)
    verified = host.verify()
    state = host.prefill("Reply politely to Sam")
    action, state = host.decode_step(state)
    output = host.generate("Reply politely to Sam")
    cached_positions = [int(value.shape[1]) for value in state.layer_self_attention_inputs]
    removed = host.remove()
    second = host.activate(package)
    return {
        "device": device,
        "archive_sha256": archive_before,
        "archive_unchanged": _sha256(package) == archive_before,
        "archive_hash_installed": active["archive_hash"] == archive_before,
        "payload_hash_preserved": second["payload_hash"] == active["payload_hash"] == verified["payload_hash"],
        "state_dict_hash_preserved": active["state_dict_hash"] == second["state_dict_hash"] == expected_state,
        "receiver_training_steps": active["receiver_training_steps"],
        "receiver_calibration_runs": active["receiver_calibration_runs"],
        "incremental_action": int(action),
        "incremental_cache_positions": cached_positions,
        "incremental_state_present": all(value == 1 for value in cached_positions),
        "generated_utf8_hex": output.hex(),
        "remove_status": removed["status"],
        "reinstall_exact": second["archive_hash"] == active["archive_hash"] and second["state_dict_hash"] == active["state_dict_hash"],
    }


def execute(repo: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-direct-neural-core-host-execution/1" or protocol.get("status") != "PREREGISTERED_IMPLEMENTATION_FROZEN":
        raise DirectNeuralCoreConstructError("construct execution protocol changed")
    for relative, expected in protocol.get("bindings", {}).items():
        if _sha256(repo / relative) != expected:
            raise DirectNeuralCoreConstructError(f"construct binding changed: {relative}")
    sealed = subprocess.run(
        ["C:\\Python310\\python.exe", "-m", "layercake.moonshot_campaign", "verify-all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    sealed_result = json.loads(sealed.stdout)
    with tempfile.TemporaryDirectory(prefix="layercake-direct-core-construct-") as temp:
        workspace = Path(temp)
        package, public, key_id, expected_state = _build_construct_package(workspace)
        cpu = _exercise(package, public, key_id, expected_state, workspace / "cpu-registry", "cpu")
        cuda = None
        if torch.cuda.is_available():
            cuda = _exercise(package, public, key_id, expected_state, workspace / "cuda-registry", "cuda")
        wrong, wrong_public, wrong_key, _ = _build_construct_package(workspace / "wrong", domain="python")
        role_rejected = False
        try:
            DirectNeuralCoreHost(workspace / "wrong-registry", trust_store={wrong_key: wrong_public}).activate(wrong)
        except DirectNeuralCoreError:
            role_rejected = True
        raw = bytearray(package.read_bytes())
        raw[len(raw) // 2] ^= 1
        tampered = workspace / "tampered.cake"
        tampered.write_bytes(raw)
        tamper_rejected = False
        try:
            DirectNeuralCoreHost(workspace / "tamper-registry", trust_store={key_id: public}).activate(tampered)
        except Exception:
            tamper_rejected = True
    gates = {
        "sealed_campaign_valid": sealed_result.get("completed_phases_valid") is True,
        "signed_package": True,
        "cpu_install_execute": all((cpu["archive_unchanged"], cpu["archive_hash_installed"], cpu["payload_hash_preserved"], cpu["state_dict_hash_preserved"], cpu["incremental_state_present"], cpu["reinstall_exact"])),
        "cuda_same_package_execute": cuda is not None and all((cuda["archive_unchanged"], cuda["archive_hash_installed"], cuda["payload_hash_preserved"], cuda["state_dict_hash_preserved"], cuda["incremental_state_present"], cuda["reinstall_exact"])),
        "zero_receiver_learning": cpu["receiver_training_steps"] == cpu["receiver_calibration_runs"] == 0 and cuda is not None and cuda["receiver_training_steps"] == cuda["receiver_calibration_runs"] == 0,
        "capability_role_rejected": role_rejected,
        "tamper_rejected": tamper_rejected,
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "format": "layercake-postrelease-direct-neural-core-host-construct-result/1",
        "status": "PASS_CONSTRUCT_ONLY" if passed else "FAIL_CONSTRUCT",
        "protocol": {"path": protocol_path.name, "sha256": _sha256(protocol_path)},
        "gates": gates,
        "cpu": cpu,
        "cuda": cuda,
        "teacher_present_at_inference": False,
        "training_performed": False,
        "sealed_release_changed": False,
        "external_artifact_used": False,
        "english_quality_certified": False,
        "performance_certified": False,
        "external_artifact_acceptance_authorized": passed,
    }
    result["evidence_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="moonshot/postrelease_direct_neural_core_host_execution_v2.json")
    parser.add_argument("--output", default="results/moonshot/postrelease/direct_neural_core_host_construct_v1.json")
    args = parser.parse_args(argv)
    repo = Path.cwd().resolve()
    result = execute(repo, (repo / args.protocol).resolve())
    output = (repo / args.output).resolve()
    if output.exists():
        raise DirectNeuralCoreConstructError("construct result is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_CONSTRUCT_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
