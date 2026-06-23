"""Stable benchmark record schema and JSONL helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable


@dataclass
class BenchmarkRecord:
    benchmark: str
    model: str
    input_mode: str
    wall_seconds: float
    trainable_parameters: int
    total_parameters: int
    units_processed: int
    unit: str
    throughput: float
    installed_bricks: int = 0
    active_bricks: int = 0
    patch_compression_ratio: float | None = None
    loss: float | None = None
    validation_ppl: float | None = None
    domain_ppl: float | None = None
    general_ppl: float | None = None
    peak_memory_bytes: int | None = None

    def validate(self) -> None:
        if self.wall_seconds < 0 or self.units_processed < 0:
            raise ValueError("timing and work counts must be non-negative")
        if self.active_bricks > self.installed_bricks:
            raise ValueError("active bricks cannot exceed installed bricks")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def timed_run(fn: Callable[[], None], units: int) -> tuple[float, float]:
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    return elapsed, units / max(elapsed, 1e-12)


def append_jsonl(path: str | Path, record: BenchmarkRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
