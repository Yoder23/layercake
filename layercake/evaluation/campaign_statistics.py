"""Frozen statistical methods for the Moonshot benchmark campaign."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Callable, Hashable, Mapping, Sequence


@dataclass(frozen=True)
class ConfidenceInterval:
    confidence: float
    lower: float
    estimate: float
    upper: float
    resamples: int
    seed: int
    method: str


@dataclass(frozen=True)
class ComparisonResult:
    estimate: float
    interval: ConfidenceInterval
    margin: float
    passed: bool
    alternative: str


def _finite(values: Sequence[float]) -> list[float]:
    converted = [float(value) for value in values]
    if not converted or not all(math.isfinite(value) for value in converted):
        raise ValueError("statistics require one or more finite values")
    return converted


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(_finite(values)))


def median(values: Sequence[float]) -> float:
    return float(statistics.median(_finite(values)))


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(_finite(values))
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def p50(values: Sequence[float]) -> float:
    return nearest_rank(values, 0.50)


def p95(values: Sequence[float]) -> float:
    return nearest_rank(values, 0.95)


def p99(values: Sequence[float]) -> float:
    return nearest_rank(values, 0.99)


def _percentile_interval(samples: Sequence[float], confidence: float) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    alpha = (1.0 - confidence) / 2.0
    return nearest_rank(samples, alpha), nearest_rank(samples, 1.0 - alpha)


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260722,
) -> ConfidenceInterval:
    observed = _finite(values)
    if resamples < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
    generator = random.Random(seed)
    draws = [
        float(statistic([observed[generator.randrange(len(observed))] for _ in observed]))
        for _ in range(resamples)
    ]
    lower, upper = _percentile_interval(draws, confidence)
    return ConfidenceInterval(
        confidence=confidence,
        lower=lower,
        estimate=float(statistic(observed)),
        upper=upper,
        resamples=resamples,
        seed=seed,
        method="percentile_bootstrap",
    )


def paired_bootstrap_difference(
    left: Mapping[Hashable, float],
    right: Mapping[Hashable, float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260722,
) -> ConfidenceInterval:
    if set(left) != set(right) or not left:
        raise ValueError("paired bootstrap requires identical non-empty pairing keys")
    differences = [float(left[key]) - float(right[key]) for key in sorted(left, key=str)]
    interval = bootstrap_confidence_interval(
        differences,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    return ConfidenceInterval(**{**interval.__dict__, "method": "paired_percentile_bootstrap"})


def seed_level_confidence_interval(
    observations: Mapping[int, Sequence[float]],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260722,
) -> ConfidenceInterval:
    if len(observations) < 2:
        raise ValueError("seed-level confidence intervals require at least two seeds")
    seed_means = [mean(observations[key]) for key in sorted(observations)]
    interval = bootstrap_confidence_interval(
        seed_means,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    return ConfidenceInterval(**{**interval.__dict__, "method": "seed_level_percentile_bootstrap"})


def non_inferiority_check(
    candidate: Mapping[Hashable, float],
    reference: Mapping[Hashable, float],
    *,
    margin: float,
    lower_is_better: bool,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260722,
) -> ComparisonResult:
    interval = paired_bootstrap_difference(
        candidate,
        reference,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    passed = interval.upper <= margin if lower_is_better else interval.lower >= -margin
    return ComparisonResult(
        estimate=interval.estimate,
        interval=interval,
        margin=float(margin),
        passed=passed,
        alternative="candidate_is_non_inferior",
    )


def superiority_check(
    candidate: Mapping[Hashable, float],
    reference: Mapping[Hashable, float],
    *,
    lower_is_better: bool,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20260722,
) -> ComparisonResult:
    interval = paired_bootstrap_difference(
        candidate,
        reference,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    passed = interval.upper < 0.0 if lower_is_better else interval.lower > 0.0
    return ComparisonResult(
        estimate=interval.estimate,
        interval=interval,
        margin=0.0,
        passed=passed,
        alternative="candidate_is_superior",
    )
