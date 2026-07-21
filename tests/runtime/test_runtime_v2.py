import torch

from layercake.portable_domain import PortableDomainDecoder
from layercake.runtime.cpu import benchmark_callable, parameter_bytes
from layercake.runtime.mobile_export import export_mobile_runtime


def test_cpu_benchmark_reports_declared_warmup_and_percentiles():
    result = benchmark_callable(lambda: sum(range(10)), warmup=2, repeats=5, useful_units=10)
    assert result["warmup_runs_excluded"] == 2
    assert result["measured_runs"] == 5
    assert result["p50_milliseconds"] <= result["p99_milliseconds"]


def test_mobile_export_reloads_exact_portable_path(tmp_path):
    torch.manual_seed(9)
    model = PortableDomainDecoder(feature_width=8, hidden_width=16).eval()
    result = export_mobile_runtime(
        model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long), tmp_path / "portable.pt"
    )
    assert result["overall_status"] == "PASS"
    assert result["max_logit_difference"] == 0
    assert result["physical_mobile_inference"] == "NOT_RUN_NO_HARDWARE"
    assert parameter_bytes(model) > 0
