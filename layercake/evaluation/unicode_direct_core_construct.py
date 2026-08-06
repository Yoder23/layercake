"""Execute the post-release Unicode-atomic direct-core v2 construct gates."""

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
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import key_id
from layercake.portable_domain import canonical_json_hash, state_dict_hash
from layercake.portable_token_plan import PortableTokenPlan
from layercake_extensions.unicode_direct_neural_core import UNICODE_DIRECT_NEURAL_CORE_ABI_SHA256, UNICODE_DIRECT_NEURAL_CORE_ABI_VERSION, UnicodeAtomicLexemePointerTokenizer, UnicodeDirectNeuralCoreError, UnicodeSafeDirectNeuralCoreHost, unicode_token_plan_manifest_architecture


class UnicodeConstructError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deterministic_keypair() -> tuple[bytes, bytes, str]:
    seed = hashlib.sha256(b"layercake-unicode-direct-core-construct-v1").digest()
    private = Ed25519PrivateKey.from_private_bytes(seed)
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    return private_pem, public_pem, key_id(public_pem)


def _fixture(directory: Path, *, abi_version: str = UNICODE_DIRECT_NEURAL_CORE_ABI_VERSION, domain: str = "english-core"):
    torch.manual_seed(80261)
    rows = [
        {"prompt": "Reply to Mira: “café 🙂”", "response": "Hello Mira — café 🙂", "copy_lexemes": ["Mira"]},
        {"prompt": "Rewrite 東京 clearly", "response": "東京 is clear.", "copy_lexemes": ["東京"]},
    ]
    tokenizer = UnicodeAtomicLexemePointerTokenizer.build_generic(rows)
    model = PortableTokenPlan(fixed_vocab_size=tokenizer.vocab_size, model_width=24, attention_heads=4, encoder_layers=1, decoder_layers=1, feedforward_width=48, pointer_width=12, dropout=0.0, maximum_source_lexemes=32, maximum_target_actions=24).eval().bind_tokenizer(tokenizer)
    with torch.no_grad():
        model.fixed_output.weight.zero_(); model.fixed_output.bias.zero_(); model.fixed_output.bias[2] = 10.0
        model.pointer_gate.weight.zero_(); model.pointer_gate.bias.fill_(-10.0)
    private, public, signer_key_id = _deterministic_keypair()
    manifest = CakeManifest(
        schema_version="1", cake_id=f"unicode-{domain}-construct", name="Unicode Core Construct", description="Construct-only Unicode core", version="2.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": signer_key_id}, abi_version=abi_version, abi_hash=UNICODE_DIRECT_NEURAL_CORE_ABI_SHA256,
        cake_type="portable_decoder", input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=unicode_token_plan_manifest_architecture(model, tokenizer), supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "persistent_incremental_state", "unicode_atomic_actions", "strict_utf8_boundary"]},
        tensor_payload_hash="", tensor_shapes=tensor_specs(model.state_dict()), package_hash="", training_data_provenance={"dataset": "construct-only", "external_teacher": False},
        evaluation_evidence={"status": "CONSTRUCT_ONLY"}, license="Apache-2.0", dependencies=(), parent_version=None, signature={"algorithm": "ed25519", "key_id": signer_key_id}, domains=(domain,), permissions=("local-inference",),
    )
    path = build_package(directory / "unicode-core.cake", manifest, model.state_dict(), private_key=private)
    return path, public, signer_key_id, tokenizer, state_dict_hash(model.state_dict())


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-unicode-direct-core-implementation/1" or protocol.get("status") != "PREREGISTERED_IMPLEMENTATION_FROZEN":
        raise UnicodeConstructError("Unicode construct protocol changed")
    for relative, expected in protocol.get("bindings", {}).items():
        if _sha256(root / relative) != expected:
            raise UnicodeConstructError(f"Unicode construct binding changed: {relative}")
    result_protocol_name = str(protocol.get("result_protocol", protocol_path.name))
    result_protocol_sha256 = str(protocol.get("result_protocol_sha256", _sha256(protocol_path)))
    sealed = subprocess.run(["C:\\Python310\\python.exe", "-m", "layercake.moonshot_campaign", "verify-all"], cwd=root, check=True, capture_output=True, text=True)
    sealed_result = json.loads(sealed.stdout)
    samples = ["“café 🙂”", "東京", "e\u0301", "مرحبا", "नमस्ते", "🏳️‍🌈"]
    sample_rows = []
    for text in samples:
        pieces = UnicodeAtomicLexemePointerTokenizer.split(text)
        sample_rows.append({"text": text, "utf8_hex": text.encode("utf-8").hex(), "piece_hex": [piece.hex() for piece in pieces], "exact_roundtrip": b"".join(pieces) == text.encode("utf-8"), "all_pieces_valid_utf8": all(piece.decode("utf-8").encode("utf-8") == piece for piece in pieces)})
    with tempfile.TemporaryDirectory(prefix="layercake-unicode-core-") as raw:
        temp = Path(raw)
        package, public, key_id, tokenizer, expected_state = _fixture(temp)
        archive = package.read_bytes()
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        executions = {}
        for device in devices:
            host = UnicodeSafeDirectNeuralCoreHost(temp / f"registry-{device}", trust_store={key_id: public}, device=device)
            active = host.activate(package)
            state = host.prefill("Reply to Mira: “café 🙂”")
            action, state = host.decode_step(state)
            realized = host.realize(state)
            generated = host.generate("Rewrite 東京 clearly")
            verified = host.verify()
            removed = host.remove()
            reactivated = host.activate(package)
            executions[device] = {"active": active, "first_action": action, "cached_positions": len(state.generated_actions), "realized_hex": realized.hex(), "generated_hex": generated.hex(), "generated_valid_utf8": generated.decode("utf-8") == "", "verify": verified, "remove": removed, "reactivated": reactivated}
        invalid_input_rejected = False
        host = UnicodeSafeDirectNeuralCoreHost(temp / "registry-invalid", trust_store={key_id: public})
        host.activate(package)
        try:
            host.generate(b"\x9c")
        except UnicodeDecodeError:
            invalid_input_rejected = True
        class InvalidModule:
            def generate_bytes(self, prompt, *, maximum_actions=None):
                return b"\x9c"
        host.module = InvalidModule()
        invalid_output_rejected = False
        try:
            host.generate("valid")
        except UnicodeDirectNeuralCoreError:
            invalid_output_rejected = True
        v1_package, v1_public, v1_key, _, _ = _fixture(temp / "v1", abi_version="lc-direct-neural-core/1")
        v1_package_rejected = False
        try:
            UnicodeSafeDirectNeuralCoreHost(temp / "registry-v1", trust_store={v1_key: v1_public}).activate(v1_package)
        except Exception:
            v1_package_rejected = True
        wrong_role, wrong_public, wrong_key, _, _ = _fixture(temp / "wrong-role", domain="python")
        wrong_role_rejected = False
        try:
            UnicodeSafeDirectNeuralCoreHost(temp / "registry-wrong-role", trust_store={wrong_key: wrong_public}).activate(wrong_role)
        except Exception:
            wrong_role_rejected = True
        tampered_raw = bytearray(package.read_bytes()); tampered_raw[len(tampered_raw) // 2] ^= 1
        tampered = temp / "tampered.cake"; tampered.write_bytes(tampered_raw)
        tamper_rejected = False
        try:
            UnicodeSafeDirectNeuralCoreHost(temp / "registry-tamper", trust_store={key_id: public}).activate(tampered)
        except Exception:
            tamper_rejected = True
        result: dict[str, Any] = {
            "format": "layercake-postrelease-unicode-direct-core-construct/1",
            "status": "PASS_CONSTRUCT_ONLY" if sealed_result.get("completed_phases_valid") is True and all(row["exact_roundtrip"] and row["all_pieces_valid_utf8"] for row in sample_rows) and invalid_input_rejected and invalid_output_rejected and v1_package_rejected and wrong_role_rejected and tamper_rejected else "FAIL",
            "protocol": {"path": result_protocol_name, "sha256": result_protocol_sha256},
            "sealed_campaign_valid": sealed_result.get("completed_phases_valid") is True,
            "interface": UNICODE_DIRECT_NEURAL_CORE_ABI_VERSION,
            "interface_sha256": UNICODE_DIRECT_NEURAL_CORE_ABI_SHA256,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "archive_bytes": len(archive),
            "expected_state_dict_hash": expected_state,
            "tokenizer_sha256": tokenizer.hash(),
            "unicode_samples": sample_rows,
            "all_fixed_actions_valid_utf8": all(value.decode("utf-8").encode("utf-8") == value for value in tokenizer.fixed_lexemes),
            "invalid_input_rejected": invalid_input_rejected,
            "invalid_output_rejected": invalid_output_rejected,
            "v1_package_rejected": v1_package_rejected,
            "wrong_role_rejected": wrong_role_rejected,
            "tamper_rejected": tamper_rejected,
            "devices": executions,
            "same_signed_package_all_devices": len({value["active"]["archive_hash"] for value in executions.values()}) == 1,
            "same_state_all_devices": len({value["active"]["state_dict_hash"] for value in executions.values()}) == 1,
            "receiver_training_steps": 0,
            "receiver_calibration_runs": 0,
            "teacher_present": False,
            "hardware": {"machine": platform.node(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "historical_release_changed": False,
            "claim_boundary": "Unicode-safe host construct only; no English quality, external acquisition, or performance claim."
        }
        result["evidence_sha256"] = canonical_json_hash(result)
        return result


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    stored = json.loads(output.read_text(encoding="utf-8"))
    expected = execute(root, protocol_path)
    if stored != expected:
        raise UnicodeConstructError("stored Unicode construct differs from clean recomputation")
    return {"status": "PASS", "evidence_sha256": expected["evidence_sha256"], "construct_only": True}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", default="moonshot/postrelease_unicode_direct_neural_core_implementation_v2.json")
    parser.add_argument("--output", default="results/moonshot/postrelease/unicode_direct_neural_core_construct_v1.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); output = (root / args.output).resolve(); protocol = (root / args.protocol).resolve()
    if args.command == "execute":
        if output.exists():
            raise UnicodeConstructError(f"construct output is immutable: {output}")
        result = execute(root, protocol)
        if result["status"] != "PASS_CONSTRUCT_ONLY":
            raise UnicodeConstructError("Unicode direct core construct failed")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        result = verify(root, protocol, output)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
