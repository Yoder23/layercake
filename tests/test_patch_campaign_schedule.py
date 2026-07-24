import pytest

from layercake.training.patch_campaign import _learning_rate_multiplier


def test_cosine_tail_learning_rate_is_constant_then_decays() -> None:
    training = {
        "learning_rate_schedule": "cosine_tail",
        "decay_start_fraction": 0.75,
        "minimum_learning_rate_ratio": 0.05,
    }
    assert _learning_rate_multiplier(training, 1, 101) == 1.0
    assert _learning_rate_multiplier(training, 76, 101) == 1.0
    midpoint = _learning_rate_multiplier(training, 88, 101)
    assert 0.05 < midpoint < 1.0
    assert _learning_rate_multiplier(training, 101, 101) == pytest.approx(
        0.05
    )


def test_unknown_learning_rate_schedule_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _learning_rate_multiplier(
            {"learning_rate_schedule": "invented"}, 1, 10
        )
