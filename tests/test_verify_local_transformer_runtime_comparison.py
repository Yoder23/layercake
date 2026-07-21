from scripts.verify_local_transformer_runtime_comparison import build_comparison


def _passing_evidence() -> dict:
    return {
        "runtime_metadata": {
            "model": "qwen2.5:1.5b",
            "runtime": "ollama",
            "runtime_version": "0.0-test",
            "quantization": "q4_K_M",
            "prompt_set": "prompt_packs/local_runtime_v1.jsonl",
            "hardware": "RTX 3080 Laptop + one-thread CPU",
            "cpu_threads": 1,
            "gpu_settings": {"device": "cuda:0"},
        },
        "same_prompt_pack": True,
        "artifacts": {
            "layercake_raw_generations": "results/local_runtime/layercake.jsonl",
            "transformer_raw_generations": "results/local_runtime/transformer.jsonl",
        },
        "ratios": {
            "cpu_generation_speed_ratio": 5.2,
            "gpu_generation_speed_ratio": 5.1,
        },
        "quality": {"noninferior_or_better": True},
    }


def test_local_runtime_comparison_stays_open_without_evidence(tmp_path):
    result = build_comparison(None, output_path=tmp_path / "comparison.json", root=tmp_path)
    assert result["status"] == "OPEN"
    assert result["gates"]["evidence_present"] is False
    assert result["ratios"]["cpu_generation_speed_ratio"] == 0.0


def test_local_runtime_comparison_passes_with_pinned_5x_evidence(tmp_path):
    result = build_comparison(
        _passing_evidence(),
        output_path=tmp_path / "comparison.json",
        root=tmp_path,
    )
    assert result["status"] == "PASS"
    assert result["gates"]["pinned_runtime_metadata"] is True
    assert result["gates"]["quality_noninferior"] is True


def test_local_runtime_comparison_fails_on_weak_or_unpinned_evidence(tmp_path):
    evidence = _passing_evidence()
    evidence["runtime_metadata"]["runtime_version"] = ""
    evidence["ratios"]["gpu_generation_speed_ratio"] = 4.9
    result = build_comparison(
        evidence,
        output_path=tmp_path / "comparison.json",
        root=tmp_path,
    )
    assert result["status"] == "FAIL"
    assert result["gates"]["pinned_runtime_metadata"] is False
    assert result["gates"]["gpu_generation_5x"] is False
