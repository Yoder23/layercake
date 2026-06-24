from layercake.rolling.gates import (
    ABIHashCompatibilityGate,
    BytePatchCompatibilityGate,
    InputInterfaceCompatibilityGate,
    MaxMetricGate,
    MinMetricGate,
    RegressionGate,
    run_gates,
)


def test_metric_and_compatibility_gates():
    context = {
        "score": 1.0,
        "loss": 0.1,
        "parent": {"loss": 0.2},
        "commit": {"abi_hash": "a", "input_interface_hash": "i", "byte_patch_hash": "p"},
        "parent_commit": {"abi_hash": "a", "input_interface_hash": "i", "byte_patch_hash": "p"},
    }
    gates = [
        MinMetricGate("score_gate", "score", 0.5),
        MaxMetricGate("loss_gate", "loss", 0.2),
        RegressionGate("regression_gate", "loss", max_delta=0.0),
        ABIHashCompatibilityGate(),
        InputInterfaceCompatibilityGate(),
        BytePatchCompatibilityGate(),
    ]
    assert all(result.passed for result in run_gates(gates, context))
