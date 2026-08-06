from __future__ import annotations

from pathlib import Path

import pytest
import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake_extensions.direct_neural_core import (
    DIRECT_NEURAL_CORE_ABI_SHA256,
    DIRECT_NEURAL_CORE_ABI_VERSION,
    DirectNeuralCoreError,
    DirectNeuralCoreHost,
)
from layercake.models.portable_decoder import portable_token_plan_manifest_architecture
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import (
    LosslessLexemePointerTokenizer,
    PortableTokenPlan,
    build_token_plan_artifact,
)


def _package(tmp_path: Path, *, domain: str = "english-core"):
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
        model.fixed_output.bias[2] = 10.0
        model.pointer_gate.weight.zero_()
        model.pointer_gate.bias.fill_(-10.0)
    artifact = build_token_plan_artifact(model, tokenizer, domain_id="english-core")
    private, public, key_id = generate_keypair()
    manifest = CakeManifest(
        schema_version="1",
        cake_id=f"{domain}-construct",
        name="Direct Neural Core Construct",
        description="Test-only immutable direct neural core",
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
        tensor_shapes=tensor_specs(artifact["state_dict"]),
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
    path = build_package(tmp_path / f"{domain}.cake", manifest, artifact["state_dict"], private_key=private)
    return path, public, key_id, state_dict_hash(model.state_dict())


def test_signed_direct_core_install_incremental_remove_reinstall(tmp_path):
    package, public, key_id, expected_state = _package(tmp_path)
    archive = package.read_bytes()
    host = DirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public})
    first = host.activate(package)
    assert first["state_dict_hash"] == expected_state
    assert first["receiver_training_steps"] == 0
    assert first["receiver_calibration_runs"] == 0
    assert host.verify()["status"] == "PASS"
    state = host.prefill("Reply politely to Sam")
    action, state = host.decode_step(state)
    assert isinstance(action, int)
    assert len(state.generated_actions) == 1
    assert isinstance(host.generate("Reply politely to Sam"), bytes)
    removed = host.remove()
    assert removed["status"] == "REMOVED"
    second = host.activate(package)
    assert package.read_bytes() == archive
    assert second["archive_hash"] == first["archive_hash"]
    assert second["payload_hash"] == first["payload_hash"]
    assert second["state_dict_hash"] == first["state_dict_hash"]


def test_direct_core_rejects_capability_role_and_tampering(tmp_path):
    package, public, key_id, _ = _package(tmp_path, domain="python")
    host = DirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public})
    with pytest.raises(DirectNeuralCoreError, match="English core"):
        host.activate(package)
    good, public, key_id, _ = _package(tmp_path / "good")
    raw = bytearray(good.read_bytes())
    raw[len(raw) // 2] ^= 1
    tampered = tmp_path / "tampered.cake"
    tampered.write_bytes(raw)
    other = DirectNeuralCoreHost(tmp_path / "other", trust_store={key_id: public})
    with pytest.raises(Exception):
        other.activate(tampered)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_signed_core_package_executes_on_cuda(tmp_path):
    package, public, key_id, expected_state = _package(tmp_path)
    host = DirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public}, device="cuda")
    result = host.activate(package)
    assert result["state_dict_hash"] == expected_state
    state = host.prefill("Rewrite this clearly: quiet bridge")
    _, state = host.decode_step(state)
    assert len(state.generated_actions) == 1
