"""AOT-friendly TorchScript export plus fail-closed physical-device status."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn


def export_mobile_runtime(
    module: nn.Module,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...],
    output: str | Path,
    *,
    physical_device_measured: bool = False,
) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    module = module.cpu().eval()
    if isinstance(example_inputs, torch.Tensor):
        trace_inputs: torch.Tensor | tuple[torch.Tensor, ...] = example_inputs.cpu().long()
        call_inputs = (trace_inputs,)
    else:
        call_inputs = tuple(value.detach().cpu() for value in example_inputs)
        trace_inputs = call_inputs
    with torch.inference_mode():
        expected = module(*call_inputs)
        if isinstance(expected, tuple):
            expected = expected[0]
        traced = torch.jit.trace(module, trace_inputs, strict=True)
        traced = torch.jit.freeze(traced)
        traced.save(str(output))
        loaded = torch.jit.load(str(output)).eval()
        actual = loaded(*call_inputs)
        if isinstance(actual, tuple):
            actual = actual[0]
        max_difference = float((expected - actual).abs().max().item())
    raw = output.read_bytes()
    smoke_passed = max_difference == 0.0
    certificate = {
        "format": "layercake-mobile-export/1",
        "created_at_unix": time.time(),
        "artifact": str(output),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_bytes": len(raw),
        "torchscript_smoke": "PASS" if smoke_passed else "FAIL",
        "max_logit_difference": max_difference,
        "host_platform": platform.platform(),
        "physical_mobile_inference": "PASS" if physical_device_measured else "NOT_RUN_NO_HARDWARE",
        "overall_status": "PASS" if smoke_passed else "FAIL",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return certificate
