"""Native CPU capability dispatch metadata, loaded only when requested."""

from __future__ import annotations


def native_capabilities() -> dict:
    import platform
    import torch

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
