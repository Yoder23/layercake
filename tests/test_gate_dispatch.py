import pytest

from layercake.rolling.gates import (
    ABIDriftGate,
    DominanceGate,
    ProtectedCapabilityGate,
    TransferExactnessGate,
    gate_from_config,
)


def test_gate_dispatch_explicit_types():
    assert isinstance(gate_from_config({"type": "abi_drift", "metric": "abi_drift"}), ABIDriftGate)
    assert isinstance(gate_from_config({"type": "transfer_exactness", "metric": "diff"}), TransferExactnessGate)
    assert isinstance(gate_from_config({"type": "protected_capabilities"}), ProtectedCapabilityGate)
    assert isinstance(gate_from_config({"type": "dominance"}), DominanceGate)


def test_unknown_gate_fails_loudly():
    with pytest.raises(ValueError, match="unknown gate type"):
        gate_from_config({"type": "not_a_real_gate", "metric": "score"})
