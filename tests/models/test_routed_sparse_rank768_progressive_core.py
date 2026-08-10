from __future__ import annotations
from pathlib import Path
import torch
from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.routed_sparse_rank768_progressive_core import (
    RoutedSparseRank768ProgressiveCore,
)
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import EOS_ID
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.routed_sparse_rank768_progressive_core import (
    ROUTED_SPARSE_RANK768_ABI_SHA256,
    ROUTED_SPARSE_RANK768_ABI_VERSION,
    ROUTED_SPARSE_RANK768_CAPABILITIES,
    RoutedSparseRank768ProgressiveCoreHost,
    routed_sparse_rank768_manifest_architecture,
)
from tests.models.test_decoder_direct_neural_core import _doc


def _model(tokenizer):
    return RoutedSparseRank768ProgressiveCore(
        fixed_vocab_size=tokenizer.vocab_size,
        full_width=24,
        bottleneck_width=8,
        attention_heads=2,
        replacement_layers=2,
        intermediate_size=16,
        residual_rank=16,
        sparse_width=8,
        maximum_source_actions=16,
        maximum_target_actions=8,
        maximum_sequence_actions=24,
    ).bind_tokenizer(tokenizer)


def _package(tmp: Path):
    torch.manual_seed(15)
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = _model(tokenizer).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.token_embedding.weight.fill_(1)
        model.final_norm.weight.fill_(1)
        model.lm_head.weight[EOS_ID].fill_(1)
        model.router.bias[0] = 1
    private, public, key_id = generate_keypair()
    manifest = CakeManifest(
        schema_version="1",
        cake_id="routed-sparse-rank768",
        name="Routed Sparse Rank768",
        description="construct",
        version="15.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": key_id},
        abi_version=ROUTED_SPARSE_RANK768_ABI_VERSION,
        abi_hash=ROUTED_SPARSE_RANK768_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=routed_sparse_rank768_manifest_architecture(model, tokenizer),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": sorted(ROUTED_SPARSE_RANK768_CAPABILITIES)},
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(model.state_dict()),
        package_hash="",
        training_data_provenance={"dataset": "construct-only"},
        evaluation_evidence={"status": "CONSTRUCT_ONLY"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id},
        domains=("english-core",),
        permissions=("local-inference",),
    )
    package = build_package(tmp / "routed.cake", manifest, model.state_dict(), private_key=private)
    return package, public, key_id, state_dict_hash(model.state_dict()), tokenizer


def test_count_route_and_incremental_identity() -> None:
    torch.manual_seed(15)
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = _model(tokenizer).eval()
    assert model.parameter_count() == model.parameter_count_for_config(
        fixed_vocab_size=tokenizer.vocab_size,
        full_width=24,
        bottleneck_width=8,
        replacement_layers=2,
        intermediate_size=16,
        residual_rank=16,
        sparse_width=8,
    )
    assert model.parameter_count_for_config(
        fixed_vocab_size=32015,
        full_width=3072,
        bottleneck_width=192,
        replacement_layers=32,
        intermediate_size=768,
        residual_rank=768,
        sparse_width=384,
    ) == 536_758_275
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.copy_(torch.tensor([3.0, 2.0, 1.0]))
    ids, lexemes = tokenizer.encode_source("hello world")
    state = model.prefill_ids(ids, lexemes)
    assert state.route_index == 0
    assert all(value.shape[1] == 2 * model.attention_heads for value in state.layer_keys)
    assert torch.allclose(
        state.next_logits,
        model.forward_routed(torch.tensor([ids]), 0)[:, -1],
        atol=1e-5,
        rtol=1e-5,
    )
    state.next_logits.zero_()
    state.next_logits[0, 4] = 1
    action, state = model.decode_step(state)
    assert torch.allclose(
        state.next_logits,
        model.forward_routed(torch.tensor([ids + [action]]), 0)[:, -1],
        atol=1e-5,
        rtol=1e-5,
    )


def test_unselected_route_outputs_are_physically_inactive() -> None:
    torch.manual_seed(15)
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = _model(tokenizer).eval()
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.copy_(torch.tensor([3.0, 2.0, 1.0]))
        for layer in model.layers:
            layer.route_coefficient_projections[1].weight.fill_(float("nan"))
            layer.route_coefficient_projections[2].weight.fill_(float("nan"))
    ids, _ = tokenizer.encode_source("hello world")
    output = model(torch.tensor([ids]))
    assert torch.isfinite(output).all()


def test_signed_lifecycle_is_zero_learning(tmp_path: Path) -> None:
    package, public, key_id, expected_hash, _ = _package(tmp_path)
    host = RoutedSparseRank768ProgressiveCoreHost(
        tmp_path / "registry", trust_store={key_id: public}
    )
    active = host.activate(package)
    assert active["state_dict_hash"] == expected_hash
    assert host.generate("hello world") == b""
    assert active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0
