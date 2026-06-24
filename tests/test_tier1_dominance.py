from layercake.rolling.tier1 import run_tier1_dominance_smoke


def test_tier1_dominance_smoke_schema_and_gates(tmp_path):
    result = run_tier1_dominance_smoke(
        steps=2,
        data_path=tmp_path / "data.txt",
        output_path=tmp_path / "tier1.json",
    )
    assert result["status"] in {"PASS", "FAIL"}
    assert result["scope"].startswith("Tier 0/Tier 1 smoke")
    assert "layercake_preview_guided" in result
    assert "tiny_byte_transformer" in result
    gates = result["dominance"]["gates"]
    for gate in (
        "lower_training_time",
        "lower_final_bpb",
        "fewer_trainable_params",
        "preview_beats_blind_bpb",
        "layercake_faster_cpu_generation",
        "generation_printable",
    ):
        assert gate in gates
