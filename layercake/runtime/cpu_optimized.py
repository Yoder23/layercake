"""Batch-one CPU runtime built around persistent LayerCake state."""

from __future__ import annotations

import time

import torch
from torch import nn

from .cpu import configure_cpu, parameter_bytes, quantize_dynamic


class CPUOptimizedRuntime:
    def __init__(
        self,
        model: nn.Module,
        *,
        threads: int = 1,
        route: int = 0,
        quantized: bool = False,
    ):
        self.environment = configure_cpu(threads)
        self.route = int(route)
        self.quantized = bool(quantized)
        self.model = quantize_dynamic(model) if quantized else model.cpu().eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: bytes | torch.Tensor,
        count: int,
        *,
        fusion_cake: nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict]:
        started = time.perf_counter_ns()
        kwargs = {"route": self.route}
        if fusion_cake is not None:
            kwargs["fusion_cake"] = fusion_cake
        state = self.model.prefill(prompt, capture_generated=True, **kwargs)
        prefill_ns = time.perf_counter_ns() - started
        decode_started = time.perf_counter_ns()
        logits, state = self.model.decode_many(state, count, **(
            {"fusion_cake": fusion_cake} if fusion_cake is not None else {}
        ))
        decode_ns = time.perf_counter_ns() - decode_started
        return state.generated_bytes, {
            "prefill_milliseconds": prefill_ns / 1_000_000,
            "decode_milliseconds": decode_ns / 1_000_000,
            "time_to_first_byte_milliseconds": (
                prefill_ns / 1_000_000 if count == 0 else
                prefill_ns / 1_000_000 + decode_ns / 1_000_000 / count
            ),
            "bytes_per_second": count / (decode_ns / 1_000_000_000) if decode_ns else 0.0,
            "state_bytes": state.state_bytes,
            "resident_parameter_bytes": parameter_bytes(self.model),
            "threads": self.environment["threads"],
            "quantized": self.quantized,
            "incremental": True,
            "logit_steps": int(logits.shape[1]),
        }
