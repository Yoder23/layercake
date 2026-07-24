"""Runtime APIs, imported lazily so native-only processes stay native-only."""

from importlib import import_module

__all__ = [
    "benchmark_callable", "configure_cpu", "cuda_available", "export_mobile_runtime",
    "parameter_bytes", "prepare_cuda_model", "quantize_dynamic",
]

_LAZY_EXPORTS = {
    "benchmark_callable": (".cpu", "benchmark_callable"),
    "configure_cpu": (".cpu", "configure_cpu"),
    "parameter_bytes": (".cpu", "parameter_bytes"),
    "quantize_dynamic": (".cpu", "quantize_dynamic"),
    "cuda_available": (".cuda", "cuda_available"),
    "prepare_cuda_model": (".cuda", "prepare_cuda_model"),
    "export_mobile_runtime": (".mobile_export", "export_mobile_runtime"),
}


def __getattr__(name: str):
    try:
        module_name, symbol_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    symbol = getattr(import_module(module_name, __name__), symbol_name)
    globals()[name] = symbol
    return symbol
