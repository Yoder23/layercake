from pathlib import Path
import pytest
import torch
from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import EOS_ID
from layercake.sparse_residual_progressive_core import SparseResidualProgressiveCore, SparseResidualProgressiveLayer
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.sparse_residual_progressive_core import *
from tests.models.test_decoder_direct_neural_core import _doc


def _model(tokenizer):
    return SparseResidualProgressiveCore(
        fixed_vocab_size=tokenizer.vocab_size, full_width=24, bottleneck_width=8,
        attention_heads=2, replacement_layers=2, intermediate_size=16,
        residual_experts=4, maximum_source_actions=16, maximum_target_actions=8,
        maximum_sequence_actions=24,
    ).bind_tokenizer(tokenizer)


def _package(tmp: Path):
    torch.manual_seed(13); tokenizer = DecoderAwareExternalTokenizer(_doc()); model = _model(tokenizer).eval()
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
        model.token_embedding.weight.fill_(1); model.final_norm.weight.fill_(1); model.lm_head.weight[EOS_ID].fill_(1)
    private, public, key_id = generate_keypair()
    manifest = CakeManifest(
        schema_version="1", cake_id="sparse-residual", name="Sparse Residual",
        description="construct", version="13.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": key_id},
        abi_version=SPARSE_RESIDUAL_ABI_VERSION, abi_hash=SPARSE_RESIDUAL_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=sparse_residual_manifest_architecture(model, tokenizer),
        supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": sorted(SPARSE_RESIDUAL_CAPABILITIES)},
        tensor_payload_hash="", tensor_shapes=tensor_specs(model.state_dict()), package_hash="",
        training_data_provenance={"dataset": "construct-only"}, evaluation_evidence={"status": "CONSTRUCT_ONLY"},
        license="Apache-2.0", dependencies=(), parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id}, domains=("english-core",), permissions=("local-inference",),
    )
    package = build_package(tmp / "sparse.cake", manifest, model.state_dict(), private_key=private)
    return package, public, key_id, state_dict_hash(model.state_dict()), tokenizer


def test_count_incremental_and_top1_physical_selection():
    tokenizer = DecoderAwareExternalTokenizer(_doc()); model = _model(tokenizer).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == model.parameter_count_for_config(
        fixed_vocab_size=tokenizer.vocab_size, full_width=24, bottleneck_width=8,
        replacement_layers=2, intermediate_size=16, residual_experts=4,
    )
    assert model.parameter_count_for_config(
        fixed_vocab_size=32015, full_width=3072, bottleneck_width=192,
        replacement_layers=32, intermediate_size=768, residual_experts=4,
    ) == 391_154_688
    layer = SparseResidualProgressiveLayer(8, 2, 1, 4, residual_experts=4, rms_epsilon=1e-5, rope_theta=10000.0)
    with torch.no_grad():
        layer.post_attention_norm.weight.fill_(1); layer.residual_router.weight.zero_(); layer.residual_router.weight[2].fill_(1)
        layer.expert_coefficient_weights.fill_(float("nan")); layer.expert_output_bases.fill_(float("nan")); layer.expert_residual_means.fill_(float("nan"))
        layer.expert_coefficient_weights[2].zero_(); layer.expert_output_bases[2].zero_(); layer.expert_residual_means[2].fill_(1)
    output = layer._mlp_delta(torch.ones(1, 3, 8))
    assert torch.isfinite(output).all() and torch.equal(output, torch.ones_like(output))
    assert layer.last_active_expert_counts == (0, 0, 3, 0)
    ids, lexemes = tokenizer.encode_source("hello world"); state = model.prefill_ids(ids, lexemes)
    assert torch.allclose(state.next_logits, model(torch.tensor([ids]))[:, -1], atol=1e-5, rtol=1e-5)
    state.next_logits.zero_(); state.next_logits[0, 4] = 1; action, state = model.decode_step(state)
    assert torch.allclose(state.next_logits, model(torch.tensor([ids + [action]]))[:, -1], atol=1e-5, rtol=1e-5)


def test_lifecycle(tmp_path):
    package, public, key_id, expected_hash, _ = _package(tmp_path)
    host = SparseResidualProgressiveCoreHost(tmp_path / "registry", trust_store={key_id: public})
    active = host.activate(package)
    assert active["state_dict_hash"] == expected_hash
    assert host.generate("hello world") == b""
    assert active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda(tmp_path):
    package, public, key_id, expected_hash, _ = _package(tmp_path)
    host = SparseResidualProgressiveCoreHost(tmp_path / "registry", trust_store={key_id: public}, device="cuda")
    assert host.activate(package)["state_dict_hash"] == expected_hash
