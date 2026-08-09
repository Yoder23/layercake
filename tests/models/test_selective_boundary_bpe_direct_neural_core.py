from pathlib import Path

import pytest
import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import PortableTokenPlan
from layercake_extensions.selective_boundary_bpe_direct_neural_core import (
    SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_SHA256,
    SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_VERSION,
    SelectiveBoundaryBpeDirectNeuralCoreHost,
    SelectiveBoundaryBpeTokenizer,
    selective_bpe_token_plan_manifest_architecture,
)


def _doc():
    vocab = {"[UNK]": 0, " ": 1, ".": 2, "7": 3, "A": 4, "B": 5, "_": 6, "a": 7, "e": 8, "h": 9, "l": 10, "p": 11, "t": 12, "Alpha": 13, "Alpha_": 14, "Alpha_7": 15, "Alpha_77": 16, " Beta": 17}
    merges = [["A", "l"], ["Al", "p"], ["Alp", "h"], ["Alph", "a"], ["Alpha", "_"], ["Alpha_", "7"], ["Alpha_7", "7"], [" ", "B"], [" B", "e"], [" Be", "t"], [" Bet", "a"]]
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None, "pre_tokenizer": None, "post_processor": None, "decoder": None, "model": {"type": "BPE", "dropout": None, "unk_token": "[UNK]", "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": vocab, "merges": merges}}


def _package(tmp_path: Path):
    torch.manual_seed(35001)
    tokenizer = SelectiveBoundaryBpeTokenizer(_doc())
    model = PortableTokenPlan(fixed_vocab_size=tokenizer.vocab_size, model_width=24, attention_heads=4, encoder_layers=1, decoder_layers=1, feedforward_width=48, pointer_width=12, dropout=0.0, maximum_source_lexemes=32, maximum_target_actions=16).eval().bind_tokenizer(tokenizer)
    with torch.no_grad():
        model.fixed_output.weight.zero_(); model.fixed_output.bias.zero_(); model.fixed_output.bias[2] = 10; model.pointer_gate.weight.zero_(); model.pointer_gate.bias.fill_(-10)
    private, public, key_id = generate_keypair()
    manifest = CakeManifest(schema_version="1", cake_id="selective-bpe-core-construct", name="Selective BPE Core", description="construct", version="5.0.0", publisher={"id": "construct", "name": "Construct", "key_id": key_id}, abi_version=SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_VERSION, abi_hash=SELECTIVE_BPE_DIRECT_NEURAL_CORE_ABI_SHA256, cake_type="portable_decoder", input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"}, output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"}, architecture=selective_bpe_token_plan_manifest_architecture(model, tokenizer), supported_precisions=("fp32",), supported_backends=("pytorch", "cuda"), minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "persistent_incremental_state", "unicode_atomic_actions", "strict_utf8_boundary", "selective_boundary_bpe"]}, tensor_payload_hash="", tensor_shapes=tensor_specs(model.state_dict()), package_hash="", training_data_provenance={"dataset": "construct-only", "external_teacher": False}, evaluation_evidence={"status": "CONSTRUCT_ONLY"}, license="Apache-2.0", dependencies=(), parent_version=None, signature={"algorithm": "ed25519", "key_id": key_id}, domains=("english-core",), permissions=("local-inference",))
    path = build_package(tmp_path / "selective-bpe-core.cake", manifest, model.state_dict(), private_key=private)
    return path, public, key_id, state_dict_hash(model.state_dict()), tokenizer


def test_selective_split_is_exact_and_identifier_stable():
    tokenizer = SelectiveBoundaryBpeTokenizer(_doc())
    assert tokenizer.split("Alpha_77 Beta") == [b"Alpha_77", b" Beta"]
    assert b"".join(tokenizer.split("Alpha_77 Beta")) == b"Alpha_77 Beta"
    assert SelectiveBoundaryBpeTokenizer.from_document(tokenizer.canonical_dict()).hash() == tokenizer.hash()


def test_v5_signed_lifecycle_and_zero_learning(tmp_path):
    package, public, key_id, state, _ = _package(tmp_path)
    archive = package.read_bytes()
    host = SelectiveBoundaryBpeDirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public})
    first = host.activate(package)
    assert first["state_dict_hash"] == state
    assert first["receiver_training_steps"] == first["receiver_calibration_runs"] == 0
    assert host.generate("Alpha_77 Beta") == b""
    host.remove(); second = host.activate(package)
    assert package.read_bytes() == archive
    assert second["archive_hash"] == first["archive_hash"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_v5_package_executes_cuda(tmp_path):
    package, public, key_id, state, _ = _package(tmp_path)
    host = SelectiveBoundaryBpeDirectNeuralCoreHost(tmp_path / "registry", trust_store={key_id: public}, device="cuda")
    assert host.activate(package)["state_dict_hash"] == state
    assert host.generate("Alpha_77 Beta") == b""
