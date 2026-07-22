"""Optimized CUDA setup and cached transformer generation helpers."""

from __future__ import annotations

import time

import torch

from .cuda import prepare_cuda_model, synchronize


class CUDAOptimizedRuntime:
    def __init__(self, model: torch.nn.Module, *, precision: str = "fp16", compile_model: bool = False):
        self.model, self.environment = prepare_cuda_model(
            model, precision=precision, compile_model=compile_model
        )

    @torch.inference_mode()
    def benchmark_cached_tokens(self, prompt_ids: torch.Tensor, count: int) -> dict:
        prompt_ids = prompt_ids.cuda()
        synchronize()
        started = time.perf_counter_ns()
        state = self.model.prefill(prompt_ids)
        synchronize()
        prefill = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        _, state = self.model.decode_many(state, count)
        synchronize()
        decode = time.perf_counter_ns() - started
        return {
            "prefill_milliseconds": prefill / 1_000_000,
            "decode_milliseconds": decode / 1_000_000,
            "tokens_per_second": count / (decode / 1_000_000_000),
            "cached": True,
            "generated_tokens": int(state.generated_ids.shape[1]),
            **self.environment,
        }

