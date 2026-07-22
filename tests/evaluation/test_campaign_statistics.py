from __future__ import annotations

import pytest

from layercake.evaluation.campaign_statistics import (
    bootstrap_confidence_interval,
    mean,
    median,
    non_inferiority_check,
    p50,
    p95,
    p99,
    paired_bootstrap_difference,
    seed_level_confidence_interval,
    superiority_check,
)


def test_descriptive_statistics_use_frozen_nearest_rank_method() -> None:
    values = list(range(1, 101))
    assert mean(values) == 50.5
    assert median(values) == 50.5
    assert p50(values) == 50
    assert p95(values) == 95
    assert p99(values) == 99


def test_bootstrap_is_deterministic_and_records_method() -> None:
    first = bootstrap_confidence_interval([1, 2, 3, 4], resamples=500, seed=7)
    second = bootstrap_confidence_interval([1, 2, 3, 4], resamples=500, seed=7)
    assert first == second
    assert first.lower <= first.estimate <= first.upper
    assert first.method == "percentile_bootstrap"


def test_paired_bootstrap_requires_exact_pairing_keys() -> None:
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap_difference({"a": 1.0}, {"b": 1.0}, resamples=100)
    interval = paired_bootstrap_difference(
        {"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}, resamples=500
    )
    assert interval.estimate == -2.0
    assert interval.method == "paired_percentile_bootstrap"


def test_seed_level_interval_operates_on_seed_aggregates() -> None:
    interval = seed_level_confidence_interval(
        {1: [1.0, 3.0], 2: [2.0, 4.0], 3: [3.0, 5.0]}, resamples=500
    )
    assert interval.estimate == 3.0
    assert interval.method == "seed_level_percentile_bootstrap"


def test_noninferiority_and_superiority_have_directional_semantics() -> None:
    candidate = {index: value for index, value in enumerate([0.8, 0.9, 1.0, 1.1])}
    reference = {index: value for index, value in enumerate([1.2, 1.3, 1.4, 1.5])}
    noninferior = non_inferiority_check(
        candidate, reference, margin=0.05, lower_is_better=True, resamples=500
    )
    superior = superiority_check(
        candidate, reference, lower_is_better=True, resamples=500
    )
    assert noninferior.passed
    assert superior.passed
