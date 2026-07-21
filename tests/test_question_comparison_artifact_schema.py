import json
from pathlib import Path


def test_repeat5_question_comparison_artifacts_include_raw_trials():
    for path in [
        Path("results/breakthrough_equal/question_comparison_lmonly1000_cpu_stateful_repeats5.json"),
        Path("results/breakthrough_equal/question_comparison_lmonly1000_gpu_draft_repeats5.json"),
    ]:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        assert data["repeats"] == 5
        assert len(data["samples"]) == 2
        for sample in data["samples"]:
            assert len(sample["layercake"]["timing"]["trials"]) == 5
            assert len(sample["transformer"]["timing"]["trials"]) == 5
            assert sample["speed_ratio_layercake_over_transformer"] > 0
