from __future__ import annotations

import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, load_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.models.portable_decoder import (
    load_cake_module,
    portable_token_plan_manifest_architecture,
)
from layercake.portable_token_plan import (
    LosslessLexemePointerTokenizer,
    PortableTokenPlan,
    build_token_plan_artifact,
)


def test_signed_token_plan_package_round_trip_is_exact(tmp_path):
    rows = [
        {
            "function_name": "novel_name",
            "prompt": "Define novel_name(value).",
            "response": "def novel_name(value):\n    return value\n",
        }
    ]
    tokenizer = LosslessLexemePointerTokenizer.build(rows)
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
    artifact = build_token_plan_artifact(model, tokenizer)
    private, public, key_id = generate_keypair()
    state = artifact["state_dict"]
    manifest = CakeManifest(
        schema_version="1",
        cake_id="python-plan-test",
        name="Python Plan Test",
        description="Test-only lossless token-plan package",
        version="1.0.0",
        publisher={"id": "test", "name": "Test", "key_id": key_id},
        abi_version="lc-semantic-gpt2-768/1",
        abi_hash="d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761",
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes"},
        output_contract={"external": "UTF-8 bytes"},
        architecture=portable_token_plan_manifest_architecture(
            artifact["spec"]
        ),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input"]},
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(state),
        package_hash="",
        training_data_provenance={"dataset": "test"},
        evaluation_evidence={"status": "TEST"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id},
        domains=("python",),
    )
    path = build_package(
        tmp_path / "python-plan.cake",
        manifest,
        state,
        private_key=private,
    )
    package = load_package(path, trust_store={key_id: public})
    loaded = load_cake_module(package)
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    state_before = loaded.prefill_bytes(rows[0]["prompt"])
    assert state_before.encoded.shape[-1] == model.model_width
