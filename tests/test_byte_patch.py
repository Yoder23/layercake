import pytest
import torch

from layercake.abi import ABISpec
from layercake.byte_patch import (
    ByteCodec,
    BytePatchEncoder,
    DifficultyPatcher,
    FixedBytePatcher,
    WhitespaceBytePatcher,
)
from layercake.domain_bricks import LowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec
from layercake.model_v2 import BytePatchLayerCake


def test_utf8_byte_round_trip():
    text = "LayerCake: café λ 🚀"
    ids = ByteCodec.encode_text(text)
    assert ByteCodec.decode_bytes(ids) == text


def test_fixed_patch_shape_and_compression():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    encoder = BytePatchEncoder(d_model=16, d_byte=8, patcher=FixedBytePatcher(4))
    patches, metadata = encoder(ids)
    assert patches.shape == (1, 2, 16)
    assert metadata[0].boundaries == ((0, 4), (4, 8))
    assert metadata[0].compression_ratio == 4.0


def test_whitespace_patcher_is_stable():
    ids = ByteCodec.encode_text("abc def\nx")
    patcher = WhitespaceBytePatcher(max_patch_size=8)
    a = patcher.boundaries(ids)
    b = patcher.boundaries(ids)
    assert a == b
    assert a.boundaries == ((0, 4), (4, 8), (8, 9))


def test_difficulty_patcher_is_explicit_stub():
    with pytest.raises(NotImplementedError):
        DifficultyPatcher().boundaries([1, 2, 3])


def test_byte_patch_forward_and_domain_brick():
    interface = InputInterfaceSpec(
        mode="byte_patch", patching="fixed:4", max_patch_size=4
    )
    abi = ABISpec(version="lc-abi/2", d_abi=32, input_interface=interface)
    brick = LowRankDomainOperator(abi, rank=4, alpha_init=0.0)
    model = BytePatchLayerCake(
        abi, d_model=32, n_layers=1, n_heads=4, patcher=FixedBytePatcher(4), domain_brick=brick
    )
    byte_ids = torch.tensor([ByteCodec.encode_text("abcdefgh")], dtype=torch.long)
    logits, abi_states, metadata = model(byte_ids)
    assert logits.shape == (1, 8, 256)
    assert abi_states.shape == (1, 2, 32)
    assert metadata[0].compression_ratio == 4.0
