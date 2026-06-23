import json

import pytest

from layercake.benchmarks import BenchmarkRecord, append_jsonl


def test_benchmark_output_schema(tmp_path):
    record = BenchmarkRecord(
        benchmark="smoke",
        model="tiny",
        input_mode="byte_patch",
        wall_seconds=0.1,
        trainable_parameters=10,
        total_parameters=100,
        units_processed=1000,
        unit="bytes",
        throughput=10000.0,
        installed_bricks=8,
        active_bricks=2,
        patch_compression_ratio=4.0,
    )
    path = tmp_path / "bench.jsonl"
    append_jsonl(path, record)
    loaded = json.loads(path.read_text().strip())
    assert loaded["active_bricks"] == 2
    assert loaded["unit"] == "bytes"


def test_active_cannot_exceed_installed():
    record = BenchmarkRecord(
        benchmark="bad",
        model="tiny",
        input_mode="byte",
        wall_seconds=1,
        trainable_parameters=1,
        total_parameters=1,
        units_processed=1,
        unit="bytes",
        throughput=1,
        installed_bricks=1,
        active_bricks=2,
    )
    with pytest.raises(ValueError):
        record.validate()
