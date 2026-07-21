from __future__ import annotations

import statistics


def latency_summary(samples_milliseconds: list[float], *, useful_units: int = 1) -> dict:
    if not samples_milliseconds or useful_units <= 0:
        raise ValueError("latency samples and useful units must be positive")
    ordered = sorted(float(value) for value in samples_milliseconds)
    def percentile(q: float) -> float:
        return ordered[round((len(ordered) - 1) * q)]
    p50 = statistics.median(ordered)
    return {
        "p50_milliseconds": p50,
        "p95_milliseconds": percentile(0.95),
        "p99_milliseconds": percentile(0.99),
        "mean_milliseconds": statistics.fmean(ordered),
        "useful_units_per_second": useful_units / (p50 / 1000),
        "samples": ordered,
    }
