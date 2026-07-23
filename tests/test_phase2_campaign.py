from layercake.phase2_campaign import _long_context_prompts, _observations, _quality


def test_phase2_locked_observation_depths_are_promotable() -> None:
    functional = _observations(sustained=False)
    sustained = _observations(sustained=True)
    assert len(functional) == 120
    assert len({prompt["id"] for prompt, _ in functional}) == 100
    assert sum(trial == 2 for _, trial in functional) == 20
    assert len(sustained) == 40
    assert len({prompt["id"] for prompt, _ in sustained}) == 20
    assert sum(trial == 2 for _, trial in sustained) == 20


def test_phase2_long_context_suite_is_frozen_and_depthful() -> None:
    prompts = _long_context_prompts()
    observations = _observations(long_context=True)
    assert len(prompts) == len(observations) == 20
    assert len({prompt["sha256"] for prompt in prompts}) == 20
    assert {prompt["filler_word_count"] for prompt in prompts} == {32, 64, 96, 128, 160}
    assert all(prompt["expected_codeword"] in prompt["text"] for prompt in prompts)


def test_phase2_quality_diagnostic_detects_repetition() -> None:
    repetitive = _quality(b"abcd" * 200)
    varied = _quality(bytes(range(256)) * 3)
    assert repetitive["repetition_rate"] > varied["repetition_rate"]
    assert repetitive["unique_4gram_rate"] < varied["unique_4gram_rate"]
