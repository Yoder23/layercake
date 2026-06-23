import pytest

from layercake.abi import ABISpec, ABICompatibilityError
from layercake.input_interfaces import InputInterfaceSpec


def token_spec():
    return InputInterfaceSpec(mode="tokenized", vocab_size=1000)


def test_abi_hash_is_stable_and_order_independent():
    a = ABISpec(version="lc-abi/2", d_abi=64, input_interface=token_spec())
    b = ABISpec(version="lc-abi/2", d_abi=64, input_interface=token_spec())
    assert a.hash() == b.hash()
    assert len(a.hash()) == 64


def test_incompatible_abi_rejected():
    source = ABISpec(version="lc-abi/2", d_abi=64, input_interface=token_spec())
    target = ABISpec(version="lc-abi/3", d_abi=32, input_interface=token_spec())
    with pytest.raises(ABICompatibilityError, match="version mismatch"):
        source.assert_compatible(target)


def test_cross_interface_allowed_only_when_requested():
    token = ABISpec(version="lc-abi/2", d_abi=64, input_interface=token_spec())
    byte_patch = ABISpec(
        version="lc-abi/2",
        d_abi=64,
        input_interface=InputInterfaceSpec(
            mode="byte_patch", patching="fixed:4", max_patch_size=4
        ),
    )
    token.assert_compatible(byte_patch)
    with pytest.raises(ABICompatibilityError, match="input-interface"):
        token.assert_compatible(byte_patch, require_same_interface=True)
