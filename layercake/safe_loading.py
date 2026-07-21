"""Restricted legacy checkpoint loading.

New release artifacts use safetensors. This helper exists only for historical
``.pt`` checkpoints and forbids arbitrary pickle globals on every supported
PyTorch version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointError(ValueError):
    pass


def _validate_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, torch.Tensor) or value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise CheckpointError(f"unsupported checkpoint key at {path}")
            _validate_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_tree(item, f"{path}[{index}]")
        return
    raise CheckpointError(f"unsupported checkpoint value at {path}: {type(value).__name__}")


def safe_torch_load(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    require_mapping: bool = True,
) -> Any:
    try:
        value = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"restricted checkpoint load failed: {path}") from exc
    _validate_tree(value)
    if require_mapping and not isinstance(value, dict):
        raise CheckpointError("checkpoint root must be a mapping")
    return value
