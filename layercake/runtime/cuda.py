from __future__ import annotations

import torch
from torch import nn


def cuda_available() -> bool:
    return torch.cuda.is_available()


def prepare_cuda_model(
    module: nn.Module, *, precision: str = "fp16", compile_model: bool = False
) -> tuple[nn.Module, dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("CUDA precision must be fp32, fp16, or bf16")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    module = module.eval().to(device="cuda", dtype=dtype)
    compiled = False
    if compile_model:
        try:
            module = torch.compile(module, mode="reduce-overhead")
            compiled = True
        except Exception:
            compiled = False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return module, {
        "device": properties.name,
        "total_memory_bytes": properties.total_memory,
        "precision": precision,
        "compiled": compiled,
        "cuda_version": torch.version.cuda,
    }


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
