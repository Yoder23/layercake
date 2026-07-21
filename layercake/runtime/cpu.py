from __future__ import annotations

import statistics
import time
from typing import Callable

import torch
from torch import nn


def configure_cpu(threads: int = 1, *, interop_threads: int | None = None) -> dict:
    if threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(int(threads))
    # PyTorch permits setting interop threads only before parallel work starts.
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except RuntimeError:
            pass
    return {
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "mkldnn": bool(torch.backends.mkldnn.enabled),
    }


def parameter_bytes(module: nn.Module) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in list(module.parameters()) + list(module.buffers())
    )


def quantize_dynamic(module: nn.Module) -> nn.Module:
    return torch.ao.quantization.quantize_dynamic(
        module.cpu().eval(), {nn.Linear, nn.GRU}, dtype=torch.qint8
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def benchmark_callable(
    function: Callable[[], object], *, warmup: int = 3, repeats: int = 20,
    useful_units: int = 1,
) -> dict[str, float | int]:
    if warmup < 0 or repeats <= 0 or useful_units <= 0:
        raise ValueError("invalid benchmark counts")
    for _ in range(warmup):
        function()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    median = statistics.median(samples)
    return {
        "warmup_runs_excluded": warmup,
        "measured_runs": repeats,
        "p50_milliseconds": median,
        "p95_milliseconds": _percentile(samples, 0.95),
        "p99_milliseconds": _percentile(samples, 0.99),
        "mean_milliseconds": statistics.fmean(samples),
        "useful_units_per_second": useful_units / (median / 1000),
        "raw_milliseconds": samples,
    }
