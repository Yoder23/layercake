from __future__ import annotations

from pathlib import Path

import pytest
import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import PortableTokenPlan
from layercake_extensions.unicode_direct_neural_core import (
    UNICODE_DIRECT_NEURAL_CORE_ABI_SHA256,
    UNICODE_DIRECT_NEURAL_CORE_ABI_VERSION,
    UnicodeAtomicLexemePointerTokenizer,
    UnicodeDirectNeuralCoreError,
    UnicodeSafeDirectNeuralCoreHost,
    unicode_token_plan_manifest_architecture,
)


def _package(tmp_path: Path, *, abi_version: str = UNICODE_DIRECT_NEURAL_CORE_ABI_VERSION):
    torch.manual_seed(80261)
    rows = [
        {"prompt": "Reply to Mira: “café 🙂”", "response": "Hello Mira — café 🙂", "copy_lexemes": ["Mira"]},
        {"prompt": "Rewrite 東京 clearly", "response": "東京 is clear.", "copy_lexemes": ["東京"]},
    ]
    tokenizer = UnicodeAtomicLexemePointerTokenizer.build_generic(rows)
    model = PortableTokenPlan(fixed_vocab_size=tokenizer.vocab_size, model_width=24, attention_heads=4, encoder_layers=1, decoder_layers=1, feedforward_width=48, pointer_width=12, dropout=0.0, maximum_source_lexemes=32, maximum_target_actions=24).eval().bind_tokenizer(tokenizer)
    with torch.no_grad():
        model.fixed_output.weight.zero_()
        model.fixed_output.bias.zero_()
        model.fixed_output.bias[2] = 10.0
        model.pointer_gate.weight.zero_()
        model.pointer_gate.bias.fill_(-10.0)
    private, public, key_id = generate_keypair()
    architecture = unicode_token_plan_manifest_architecture(model, tokenizer)
    manifest = CakeManifest(
        schema_version="1", cake_id="unicode-core-construct", name="Unicode Core Construct", description="Test-only Unicode-atomic core", version="2.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": key_id}, abi_version=abi_version, abi_hash=UNICODE_DIRECT_NEURAL_CORE_ABI_SHA256,
        cake_type="portable_decoder", input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"}, architecture=architecture,
        supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "persistent_incremental_state", "unicode_atomic_actions", "strict_utf8_boundary"]},
        tensor_payload_hash="", tensor_shapes=tensor_specs(model.state_dict()), package_hash="", training_data_provenance={"dataset": "construct-only", "external_teacher": False},
        evaluation_evidence={"status": "CONSTRUCT_ONLY"}, license="Apache-2.0", dependencies=(), parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id}, domains=("english-core",), permissions=("local-inference",),
    )
    path = build_package(tmp_path / "unicode-core.cake", manifest, model.state_dict(), private_key=private)
    return path, public, key_id, state_dict_hash(model.state_dict()), tokenizer


def test_unicode_actions_are_atomic_and_lossless():
    text = "“café🙂東京e\u0301”"
    pieces = UnicodeAtomicLexemePointerTokenizer.split(text)
    assert b"".join(pieces) == text.encode("utf-8")
    assert all(piece.decode("utf-8").encode("utf-8") == piece for piece in pieces)
    with pytest.raises(UnicodeDecodeError):
        UnicodeAtomicLexemePointerTokenizer.split(b"\x9c")


def test_v2_signed_install_strict_boundary_remove_reinstall(tmp_path):
    package, public, key_id, expected_state, tokenizer = _package(tmp_path)
    archive = package.read_bytes()
    host = UnicodeSafeDirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public})
    first = host.activate(package)
    assert first["state_dict_hash"] == expected_state
    assert first["receiver_training_steps"] == first["receiver_calibration_runs"] == 0
    assert host.verify()["utf8"] == "STRICT"
    state = host.prefill("Reply to Mira: “café 🙂”")
    _, state = host.decode_step(state)
    assert isinstance(host.realize(state), bytes)
    assert host.generate("Rewrite 東京 clearly").decode("utf-8") == ""
    with pytest.raises(UnicodeDecodeError):
        host.generate(b"\x9c")
    host.remove()
    second = host.activate(package)
    assert package.read_bytes() == archive
    assert second["archive_hash"] == first["archive_hash"]
    assert all(value.decode("utf-8").encode("utf-8") == value for value in tokenizer.fixed_lexemes)


def test_v2_rejects_v1_identity_and_invalid_module_output(tmp_path):
    package, public, key_id, _, _ = _package(tmp_path / "v1", abi_version="lc-direct-neural-core/1")
    host = UnicodeSafeDirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public})
    with pytest.raises(Exception):
        host.activate(package)

    class InvalidModule:
        def generate_bytes(self, prompt, *, maximum_actions=None):
            return b"\x9c"

    host.module = InvalidModule()
    host.active_cake_id = "fixture"
    with pytest.raises(UnicodeDirectNeuralCoreError, match="invalid UTF-8"):
        host.generate("valid")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_v2_signed_package_executes_on_cuda(tmp_path):
    package, public, key_id, expected_state, _ = _package(tmp_path)
    host = UnicodeSafeDirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public}, device="cuda")
    assert host.activate(package)["state_dict_hash"] == expected_state
    assert host.generate("Reply to Mira: “café 🙂”").decode("utf-8") == ""
