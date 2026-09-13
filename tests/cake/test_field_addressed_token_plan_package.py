from __future__ import annotations

import hashlib

import pytest
import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.models.direct_cake_host import DirectCakeHost
from layercake.models.portable_decoder import (
    field_addressed_token_plan_manifest_architecture,
)
from layercake.portable_token_plan import (
    EOS_ID,
    FieldAddressedPointerTokenizer,
    PortableTokenPlan,
)


ABI_VERSION = "lc-field-addressed-test/1"
ABI_HASH = hashlib.sha256(ABI_VERSION.encode()).hexdigest()


def _tokenizer() -> FieldAddressedPointerTokenizer:
    fields = ("name", "value")
    fixed = sorted(
        {
            b"<FIELD:name>",
            b"<FIELD:value>",
            b"Result",
            b" ",
            b"=",
            b"\n",
        }
    )
    return FieldAddressedPointerTokenizer(fields, fixed, field_width=4)


def test_field_addressed_codec_is_exact_and_fail_closed():
    tokenizer = _tokenizer()
    prompt = (
        "INSTRUCTION: report the supplied material\n"
        "SUPPLIED MATERIAL:\nname=coordinator35\nvalue=917"
    )
    ids, lexemes = tokenizer.encode_source(prompt)
    assert len(ids) == 10
    assert b"coordinator" in lexemes
    assert b"35" in lexemes
    coordinator = tokenizer.vocab_size + lexemes.index(b"coordinator")
    suffix = tokenizer.vocab_size + lexemes.index(b"35")
    assert tokenizer.decode_actions([coordinator, suffix, EOS_ID], lexemes) == b"coordinator35"
    with pytest.raises(ValueError, match="schema changed"):
        tokenizer.encode_source(
            "SUPPLIED MATERIAL:\nname=coordinator35\nextra=917"
        )
    with pytest.raises(ValueError, match="available field position"):
        tokenizer.decode_actions([tokenizer.vocab_size + 3], lexemes)


def test_signed_field_addressed_package_runs_through_direct_host(tmp_path):
    tokenizer = _tokenizer()
    model = PortableTokenPlan(
        fixed_vocab_size=tokenizer.vocab_size,
        model_width=16,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_width=32,
        pointer_width=8,
        dropout=0.0,
        maximum_source_lexemes=16,
        maximum_target_actions=4,
    ).eval().bind_tokenizer(tokenizer)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fixed_output.bias[EOS_ID] = 20.0

    private, public, key_id = generate_keypair()
    architecture = field_addressed_token_plan_manifest_architecture(
        model=model.canonical_config(),
        tokenizer=tokenizer.canonical_dict(),
        tokenizer_sha256=tokenizer.hash(),
    )
    manifest = CakeManifest(
        schema_version="1",
        cake_id="field-addressed-test",
        name="Field addressed test",
        description="Generic field-addressed host contract test",
        version="1.0.0",
        publisher={"id": "test", "name": "Test", "key_id": key_id},
        abi_version=ABI_VERSION,
        abi_hash=ABI_HASH,
        cake_type="portable_decoder",
        input_contract={
            "external": "UTF-8 bytes",
            "mode": "direct_selected_portable_decoder",
        },
        output_contract={"external": "UTF-8 bytes"},
        architecture=architecture,
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={
            "features": ["byte_input", "safe_tensors", "incremental"]
        },
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(model.state_dict()),
        package_hash="",
        training_data_provenance={"dataset": "test-only"},
        evaluation_evidence={"status": "TEST_ONLY"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id},
        domains=("test",),
        permissions=("local-inference",),
    )
    package = build_package(
        tmp_path / "field.cake",
        manifest,
        model.state_dict(),
        private_key=private,
    )
    host = DirectCakeHost(
        tmp_path / "registry",
        abi_version=ABI_VERSION,
        abi_hash=ABI_HASH,
        trust_store={key_id: public},
    )
    host.install(package)
    result = host.generate(
        "field-addressed-test",
        "SUPPLIED MATERIAL:\nname=coordinator35\nvalue=917",
    )
    assert result.output == b""
    assert result.actions == (EOS_ID,)
    assert result.prefill_calls == 1
    assert result.decode_step_calls == 1
    assert host.telemetry()["field-addressed-test"]["module_load_calls"] == 1
    host.remove("field-addressed-test")
    with pytest.raises(KeyError, match="not installed"):
        host.generate(
            "field-addressed-test",
            "SUPPLIED MATERIAL:\nname=coordinator35\nvalue=917",
        )


def test_field_addressed_document_rejects_contract_changes():
    document = _tokenizer().canonical_dict()
    document["instruction_in_neural_source"] = True
    with pytest.raises(ValueError, match="contract mismatch"):
        FieldAddressedPointerTokenizer.from_document(document)
