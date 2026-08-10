from pathlib import Path

import pytest
import torch

from layercake.basis_aligned_progressive_core import BasisAlignedProgressiveCore
from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import EOS_ID
from layercake_extensions.basis_aligned_progressive_core import (
    BASIS_ALIGNED_PROGRESSIVE_ABI_SHA256,
    BASIS_ALIGNED_PROGRESSIVE_ABI_VERSION,
    BASIS_ALIGNED_PROGRESSIVE_CAPABILITIES,
    BasisAlignedProgressiveCoreHost,
    basis_aligned_progressive_manifest_architecture,
)
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from tests.models.test_decoder_direct_neural_core import _doc


def _model(tokenizer: DecoderAwareExternalTokenizer) -> BasisAlignedProgressiveCore:
    return BasisAlignedProgressiveCore(
        fixed_vocab_size=tokenizer.vocab_size,
        full_width=24,
        bottleneck_width=8,
        attention_heads=2,
        replacement_layers=2,
        intermediate_size=16,
        maximum_source_actions=16,
        maximum_target_actions=8,
        maximum_sequence_actions=24,
    ).bind_tokenizer(tokenizer)


def _package(tmp_path: Path):
    torch.manual_seed(110011)
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
        cake_id="basis-aligned-progressive-construct",
        name="Basis Aligned Progressive Core",
        description="construct",
        version="11.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": key_id},
        abi_version=BASIS_ALIGNED_PROGRESSIVE_ABI_VERSION,
        abi_hash=BASIS_ALIGNED_PROGRESSIVE_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=basis_aligned_progressive_manifest_architecture(model, tokenizer),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": sorted(BASIS_ALIGNED_PROGRESSIVE_CAPABILITIES)},
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
    path = build_package(
        tmp_path / "basis-aligned-progressive.cake",
        manifest,
        model.state_dict(),
        private_key=private,
    )
    return path, public, key_id, state_dict_hash(model.state_dict()), tokenizer


def test_basis_aligned_parameter_accounting_and_residual_mean() -> None:
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = _model(tokenizer).eval()
    expected = BasisAlignedProgressiveCore.parameter_count_for_config(
        fixed_vocab_size=tokenizer.vocab_size,
        full_width=24,
        bottleneck_width=8,
        replacement_layers=2,
        intermediate_size=16,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    assert BasisAlignedProgressiveCore.parameter_count_for_config(
        fixed_vocab_size=32_015,
        full_width=3_072,
        bottleneck_width=192,
        replacement_layers=32,
        intermediate_size=768,
    ) == 291_382_272
    hidden = torch.randn(1, 3, 24)
    with torch.no_grad():
        for parameter in model.layers[0].parameters():
            parameter.zero_()
        model.layers[0].mlp_residual_mean.fill_(0.25)
    delta = model.layers[0]._mlp_delta(hidden)
    assert torch.equal(delta, torch.full_like(hidden, 0.25))


def test_basis_aligned_incremental_matches_full_forward() -> None:
    torch.manual_seed(505)
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = _model(tokenizer).eval()
    source, lexemes = tokenizer.encode_source("hello world")
    state = model.prefill_ids(source, lexemes)
    assert torch.allclose(
        state.next_logits,
        model(torch.tensor([source]))[:, -1],
        atol=1e-5,
        rtol=1e-5,
    )
    before = tuple(key.shape[2] for key in state.layer_keys)
    state.next_logits.zero_()
    state.next_logits[0, 4] = 1.0
    action, state = model.decode_step(state)
    assert torch.allclose(
        state.next_logits,
        model(torch.tensor([source + [action]]))[:, -1],
        atol=1e-5,
        rtol=1e-5,
    )
    assert all(key.shape[2] == length + 1 for key, length in zip(state.layer_keys, before))


def test_v11_signed_lifecycle_and_zero_learning(tmp_path: Path) -> None:
    package, public, key_id, state_hash, _ = _package(tmp_path)
    host = BasisAlignedProgressiveCoreHost(tmp_path / "registry", trust_store={key_id: public})
    active = host.activate(package)
    assert active["state_dict_hash"] == state_hash
    assert host.generate("hello world") == b""
    assert active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_v11_package_executes_cuda(tmp_path: Path) -> None:
    package, public, key_id, state_hash, _ = _package(tmp_path)
    host = BasisAlignedProgressiveCoreHost(
        tmp_path / "registry", trust_store={key_id: public}, device="cuda"
    )
    assert host.activate(package)["state_dict_hash"] == state_hash
    assert host.generate("hello world") == b""
