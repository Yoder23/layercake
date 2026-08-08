from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, normalizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import PortableTokenPlan
from layercake_extensions.decoder_direct_neural_core import (
    DECODER_DIRECT_NEURAL_CORE_ABI_SHA256,
    DECODER_DIRECT_NEURAL_CORE_ABI_VERSION,
    DecoderAwareDirectNeuralCoreHost,
    DecoderAwareExternalTokenizer,
    decoder_token_plan_manifest_architecture,
)


def _doc():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    tokenizer.post_processor = processors.TemplateProcessing(single="$A", special_tokens=[])
    tokenizer.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.Fuse(), decoders.Strip(" ", 1, 0)])
    tokenizer.train_from_iterator(["hello world", "hello moon", "warm world"], BpeTrainer(vocab_size=64, special_tokens=["[UNK]"], initial_alphabet=list("▁helowarmnd")))
    return __import__("json").loads(tokenizer.to_str())


def _package(tmp_path: Path):
    torch.manual_seed(44004)
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = PortableTokenPlan(fixed_vocab_size=tokenizer.vocab_size, model_width=24, attention_heads=4, encoder_layers=1, decoder_layers=1, feedforward_width=48, pointer_width=12, dropout=0.0, maximum_source_lexemes=16, maximum_target_actions=1).eval().bind_tokenizer(tokenizer)
    first_action = tokenizer.encode_fixed_target("hello")[0]
    with torch.no_grad():
        model.fixed_output.weight.zero_()
        model.fixed_output.bias.zero_()
        model.fixed_output.bias[first_action] = 10
        model.pointer_gate.weight.zero_()
        model.pointer_gate.bias.fill_(-10)
    private, public, key_id = generate_keypair()
    manifest = CakeManifest(schema_version="1", cake_id="decoder-core-construct", name="Decoder Core", description="construct", version="4.0.0", publisher={"id": "construct", "name": "Construct", "key_id": key_id}, abi_version=DECODER_DIRECT_NEURAL_CORE_ABI_VERSION, abi_hash=DECODER_DIRECT_NEURAL_CORE_ABI_SHA256, cake_type="portable_decoder", input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"}, output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"}, architecture=decoder_token_plan_manifest_architecture(model, tokenizer), supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"), minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "persistent_incremental_state", "strict_utf8_boundary", "external_tokenizer_decoder", "fixed_actions_only"]}, tensor_payload_hash="", tensor_shapes=tensor_specs(model.state_dict()), package_hash="", training_data_provenance={"dataset": "construct-only", "external_teacher": False}, evaluation_evidence={"status": "CONSTRUCT_ONLY"}, license="Apache-2.0", dependencies=(), parent_version=None, signature={"algorithm": "ed25519", "key_id": key_id}, domains=("english-core",), permissions=("local-inference",))
    path = build_package(tmp_path / "decoder-core.cake", manifest, model.state_dict(), private_key=private)
    return path, public, key_id, state_dict_hash(model.state_dict()), tokenizer


def test_decoder_tokenizer_collision_free_exact_sequence():
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    source, _ = tokenizer.encode_source("hello world")
    assert min(source) >= 4
    target = tokenizer.encode_fixed_target("hello world")
    assert tokenizer.decode_actions(target, []) == b"hello world"
    assert DecoderAwareExternalTokenizer.from_document(tokenizer.canonical_dict()).hash() == tokenizer.hash()
    with pytest.raises(ValueError, match="pointer actions"):
        tokenizer.decode_actions([tokenizer.vocab_size], [])


def test_v4_signed_lifecycle_and_zero_learning(tmp_path):
    package, public, key_id, state, _ = _package(tmp_path)
    archive = package.read_bytes()
    host = DecoderAwareDirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public})
    first = host.activate(package)
    assert first["state_dict_hash"] == state
    assert first["receiver_training_steps"] == first["receiver_calibration_runs"] == 0
    assert host.generate("hello world", maximum_actions=1).decode("utf-8")
    host.remove()
    second = host.activate(package)
    assert package.read_bytes() == archive
    assert second["archive_hash"] == first["archive_hash"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_v4_package_executes_cuda(tmp_path):
    package, public, key_id, state, _ = _package(tmp_path)
    host = DecoderAwareDirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public}, device="cuda")
    assert host.activate(package)["state_dict_hash"] == state
    assert host.generate("hello world", maximum_actions=1).decode("utf-8")
