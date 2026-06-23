"""LayerCake v2 research primitives.

The original flat ``model.py`` API remains supported.  This package contains
the versioned ABI, byte-patch, sparse-brick, alignment, and orchestration work.
"""

from .abi import ABISpec, ABICompatibilityError
from .input_interfaces import InputInterfaceSpec
from .portable_domain import LayerCakeRuntime, PortableDomainDecoder, PortableDomainSpec

__all__ = [
    "ABISpec",
    "ABICompatibilityError",
    "InputInterfaceSpec",
    "LayerCakeRuntime",
    "PortableDomainDecoder",
    "PortableDomainSpec",
]
