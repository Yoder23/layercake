"""Native CPU capability dispatch metadata.

The current Windows proof runtime uses PyTorch oneDNN/MKL kernels; this module
keeps ISA selection explicit so a future fused extension cannot silently change
the benchmark path.
"""

from __future__ import annotations

import platform

import torch


def native_capabilities() -> dict:
    capability = "UNKNOWN"
    getter = getattr(torch.backends.cpu, "get_cpu_capability", None)
    if getter is not None:
        capability = str(getter())
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_cpu_capability": capability,
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "backend": "pytorch-onednn-reference-native-dispatch",
    }

