"""LayerCake v2 research primitives with lazy public imports.

Keeping the package initializer dependency-free is required by the native
runtime: importing an ONNX entry point must not silently load PyTorch.
"""

from importlib import import_module

__all__ = [
    "ABISpec",
    "ABICompatibilityError",
    "InputInterfaceSpec",
    "LayerCakeRuntime",
    "PortableDomainDecoder",
    "PortableDomainSpec",
]

_LAZY_EXPORTS = {
    "ABISpec": (".abi", "ABISpec"),
    "ABICompatibilityError": (".abi", "ABICompatibilityError"),
    "InputInterfaceSpec": (".input_interfaces", "InputInterfaceSpec"),
    "LayerCakeRuntime": (".portable_domain", "LayerCakeRuntime"),
    "PortableDomainDecoder": (".portable_domain", "PortableDomainDecoder"),
    "PortableDomainSpec": (".portable_domain", "PortableDomainSpec"),
}


def __getattr__(name: str):
    try:
        module_name, symbol_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    symbol = getattr(import_module(module_name, __name__), symbol_name)
    globals()[name] = symbol
    return symbol
