from pathlib import Path

import pytest
import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import BOS_ID, EOS_ID
from layercake.structural_causal_core import StructuralCausalCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.structural_causal_core import (
    STRUCTURAL_CAPABILITIES,
    STRUCTURAL_CAUSAL_CORE_ABI_SHA256,
    STRUCTURAL_CAUSAL_CORE_ABI_VERSION,
    StructuralCausalCoreHost,
    structural_core_manifest_architecture,
)
from tests.models.test_decoder_direct_neural_core import _doc


def _model(tokenizer: DecoderAwareExternalTokenizer) -> StructuralCausalCore:
    return StructuralCausalCore(
        fixed_vocab_size=tokenizer.vocab_size,
        model_width=24,
        attention_heads=2,
        decoder_layers=2,
        intermediate_size=48,
        maximum_source_actions=16,
        maximum_target_actions=8,
        maximum_sequence_actions=24,
    ).bind_tokenizer(tokenizer)


def _package(tmp_path: Path):
    torch.manual_seed(77007)
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = _model(tokenizer).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.token_embedding.weight.fill_(1.0)
        model.final_norm.weight.fill_(1.0)
        model.lm_head.weight[EOS_ID].fill_(1.0)
    private, public, key_id = generate_keypair()
    manifest = CakeManifest(
        schema_version="1",
        cake_id="structural-causal-core-construct",
        name="Structural Causal Core",
        description="construct",
        version="7.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": key_id},
        abi_version=STRUCTURAL_CAUSAL_CORE_ABI_VERSION,
        abi_hash=STRUCTURAL_CAUSAL_CORE_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=structural_core_manifest_architecture(model, tokenizer),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": sorted(STRUCTURAL_CAPABILITIES)},
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(model.state_dict()),
        package_hash="",
        training_data_provenance={"dataset": "construct-only", "external_teacher": False},
        evaluation_evidence={"status": "CONSTRUCT_ONLY"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id},
        domains=("english-core",),
        permissions=("local-inference",),
    )
    path = build_package(tmp_path / "structural-causal-core.cake", manifest, model.state_dict(), private_key=private)
    return path, public, key_id, state_dict_hash(model.state_dict()), tokenizer


def test_structural_parameter_accounting_matches_preregistered_target():
    model = StructuralCausalCore(fixed_vocab_size=32_015)
    assert model.parameter_count() == 14_654_784


def test_structural_incremental_logits_match_full_causal_forward():
    torch.manual_seed(101)
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = _model(tokenizer).eval()
    source, lexemes = tokenizer.encode_source("hello world")
    state = model.prefill_ids(source, lexemes)
    packed = torch.tensor([source + [BOS_ID]], dtype=torch.long)
    assert torch.allclose(state.next_logits, model(packed)[:, -1], atol=1e-5, rtol=1e-5)
    assert all(key.shape[2] == len(source) + 1 for key in state.layer_keys)
    action, state = model.decode_step(state)
    packed = torch.tensor([source + [BOS_ID, action]], dtype=torch.long)
    assert torch.allclose(state.next_logits, model(packed)[:, -1], atol=1e-5, rtol=1e-5)
    assert all(key.shape[2] == len(source) + 2 for key in state.layer_keys)


def test_v7_signed_lifecycle_persistent_state_and_zero_learning(tmp_path):
    package, public, key_id, state_hash, _ = _package(tmp_path)
    archive = package.read_bytes()
    host = StructuralCausalCoreHost(tmp_path / "registry", trust_store={key_id: public})
    first = host.activate(package)
    assert first["state_dict_hash"] == state_hash
    state = host.prefill("hello world")
    action, state = host.decode_step(state)
    assert action == EOS_ID and state.complete
    assert host.realize(state) == b""
    assert first["receiver_training_steps"] == first["receiver_calibration_runs"] == 0
    host.remove()
    second = host.activate(package)
    assert package.read_bytes() == archive
    assert second["archive_hash"] == first["archive_hash"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_v7_package_executes_cuda(tmp_path):
    package, public, key_id, state_hash, _ = _package(tmp_path)
    host = StructuralCausalCoreHost(tmp_path / "registry", trust_store={key_id: public}, device="cuda")
    assert host.activate(package)["state_dict_hash"] == state_hash
    assert host.generate("hello world") == b""
