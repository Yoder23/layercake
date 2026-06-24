from layercake.rolling.scaling_gates import run_dominance_suite


def test_dominance_gate_schema_valid():
    result = run_dominance_suite({
        "layercake_training_seconds": 1,
        "transformer_training_seconds": 2,
        "layercake_bpb": 1,
        "transformer_bpb": 2,
        "layercake_trainable_params": 1,
        "transformer_trainable_params": 2,
    })
    assert result["passed"]
    assert "gates" in result and "dimensions" in result
