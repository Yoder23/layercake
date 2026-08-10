from pathlib import Path

import pytest
import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import EOS_ID
from layercake.source_aligned_progressive_replacement_core import SourceAlignedProgressiveReplacementCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from layercake_extensions.source_aligned_progressive_replacement_core import (
    SOURCE_ALIGNED_PROGRESSIVE_ABI_SHA256,
    SOURCE_ALIGNED_PROGRESSIVE_ABI_VERSION,
    SOURCE_ALIGNED_PROGRESSIVE_CAPABILITIES,
    SourceAlignedProgressiveReplacementCoreHost,
    source_aligned_progressive_manifest_architecture,
)
from tests.models.test_decoder_direct_neural_core import _doc


def _model(tokenizer: DecoderAwareExternalTokenizer) -> SourceAlignedProgressiveReplacementCore:
    return SourceAlignedProgressiveReplacementCore(
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
    torch.manual_seed(99009)
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
        schema_version="1", cake_id="source-aligned-progressive-construct",
        name="Source Aligned Progressive Core", description="construct", version="9.0.0",
        publisher={"id": "construct", "name": "Construct", "key_id": key_id},
        abi_version=SOURCE_ALIGNED_PROGRESSIVE_ABI_VERSION,
        abi_hash=SOURCE_ALIGNED_PROGRESSIVE_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=source_aligned_progressive_manifest_architecture(model, tokenizer),
        supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": sorted(SOURCE_ALIGNED_PROGRESSIVE_CAPABILITIES)},
        tensor_payload_hash="", tensor_shapes=tensor_specs(model.state_dict()), package_hash="",
        training_data_provenance={"dataset": "construct-only", "external_teacher": False},
        evaluation_evidence={"status": "CONSTRUCT_ONLY"}, license="Apache-2.0", dependencies=(),
        parent_version=None, signature={"algorithm": "ed25519", "key_id": key_id},
        domains=("english-core",), permissions=("local-inference",),
    )
    path = build_package(tmp_path / "source-aligned-progressive.cake", manifest, model.state_dict(), private_key=private)
    return path, public, key_id, state_dict_hash(model.state_dict()), tokenizer


def test_source_aligned_prefill_has_no_injected_boundary_action() -> None:
    torch.manual_seed(303)
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = _model(tokenizer).eval()
    source, lexemes = tokenizer.encode_source("hello world")
    state = model.prefill_ids(source, lexemes)
    packed = torch.tensor([source], dtype=torch.long)
    assert state.sequence_length == len(source)
    assert state.source_ids.tolist() == [source]
    assert torch.allclose(state.next_logits, model(packed)[:, -1], atol=1e-5, rtol=1e-5)
    action, state = model.decode_step(state)
    packed = torch.tensor([source + [action]], dtype=torch.long)
    assert torch.allclose(state.next_logits, model(packed)[:, -1], atol=1e-5, rtol=1e-5)


def test_v9_signed_lifecycle_and_zero_learning(tmp_path: Path) -> None:
    package, public, key_id, state_hash, _ = _package(tmp_path)
    host = SourceAlignedProgressiveReplacementCoreHost(tmp_path / "registry", trust_store={key_id: public})
    active = host.activate(package)
    assert active["state_dict_hash"] == state_hash
    state = host.prefill("hello world")
    action, state = host.decode_step(state)
    assert action == EOS_ID and state.complete
    assert active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_v9_package_executes_cuda(tmp_path: Path) -> None:
    package, public, key_id, state_hash, _ = _package(tmp_path)
    host = SourceAlignedProgressiveReplacementCoreHost(tmp_path / "registry", trust_store={key_id: public}, device="cuda")
    assert host.activate(package)["state_dict_hash"] == state_hash
    assert host.generate("hello world") == b""
