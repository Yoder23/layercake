"""Paired full-core LayerCake/transformer training-throughput benchmark.

This benchmark measures complete optimizer steps (zero-grad, forward, loss,
backward, gradient clipping, and AdamW update) for the 15M-class foundation
architectures used by the North Star v22 lineage.  Both models receive the
same logical raw-byte volume per step.  The transformer's sequence length is derived
from its measured training-corpus bytes/token ratio.

The result is deliberately fail-closed: a 5x gate is reported only when the
minimum repeat ratio clears 5x on every requested device.  Model/tokenizer
initialization, checkpoint I/O, and evaluation are disclosed but excluded
from the steady-state optimizer-step timing.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_schema_action_generation import _execution_environment
from scripts.train_bpe_transformer_from_config import BPETokenTransformerLM
from scripts.train_byte_core_from_config import (
    _build_model,
    _load_config_with_extends,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "breakthrough_equal"
    / "northstar_v22_training_speed.json"
)
LAYERCAKE_CONFIG = ROOT / "configs" / "northstar_v6_patch4_grounded_mix_layercake.json"
ROUTED_LAYERCAKE_CONFIG = ROOT / "configs" / "northstar_v23_shared3_routed_tail.json"
TRANSFORMER_CONFIG = ROOT / "configs" / "northstar_v22_fair_corrected_bpe.json"
TRANSFORMER_METRICS = (
    ROOT
    / "runs_experiment"
    / "northstar_v22_fair_corrected_bpe"
    / "training_metrics.json"
)


@dataclass(frozen=True)
class Workload:
    raw_sequence_bytes: int
    batch_size: int
    transformer_tokens: int
    transformer_bytes_per_token: float

    @property
    def layercake_bytes_per_step(self) -> float:
        return float(self.raw_sequence_bytes * self.batch_size)

    @property
    def transformer_bytes_per_step(self) -> float:
        return float(
            self.transformer_tokens
            * self.transformer_bytes_per_token
            * self.batch_size
        )


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = int(
        getattr(
            model,
            "_logical_total_parameters",
            sum(parameter.numel() for parameter in model.parameters()),
        )
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


def _tensor_state_bytes(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, int]:
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    gradient_bytes = sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                optimizer_bytes += value.numel() * value.element_size()
    return {
        "parameters": int(parameter_bytes),
        "gradients": int(gradient_bytes),
        "optimizer": int(optimizer_bytes),
        "total": int(parameter_bytes + gradient_bytes + optimizer_bytes),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _layercake_loss(model: torch.nn.Module, rows: torch.Tensor) -> torch.Tensor:
    x = rows[:, :-1]
    y = rows[:, 1:]
    output = model(
        x,
        return_aux=True,
        return_patch_prediction=True,
    )
    logits = output[0][:, : y.shape[1], :]
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())

    patch_predictions = output[3]
    prediction_tensor = torch.stack(patch_predictions, dim=2)
    targets = model.patch_prediction_targets(x)[
        :, :: model.patch_prediction_stride
    ]
    targets = targets[
        :, : prediction_tensor.shape[1], : prediction_tensor.shape[2]
    ]
    patch_loss = F.cross_entropy(
        prediction_tensor.reshape(-1, prediction_tensor.shape[-1]),
        targets.reshape(-1),
    )
    return loss + 0.12 * patch_loss


def _layercake_next_byte_loss(
    model: torch.nn.Module, rows: torch.Tensor
) -> torch.Tensor:
    x = rows[:, :-1]
    y = rows[:, 1:]
    logits = model(x)[0][:, : y.shape[1], :]
    return F.cross_entropy(logits.flatten(0, 1), y.flatten())


def _layercake_domain_cake_loss(
    model: torch.nn.Module, rows: torch.Tensor
) -> torch.Tensor:
    inputs = rows[:, :-1]
    context_indices = torch.full(
        (inputs.shape[0],),
        inputs.shape[1] // model.patch_size - 2,
        dtype=torch.long,
        device=inputs.device,
    )
    predictions, targets = model.domain_cake_patch_predictions(
        inputs,
        context_indices=context_indices,
    )
    prediction_tensor = torch.stack(predictions, dim=2)
    targets = targets[
        :, : prediction_tensor.shape[1], : prediction_tensor.shape[2]
    ]
    return F.cross_entropy(
        prediction_tensor.reshape(-1, prediction_tensor.shape[-1]),
        targets.reshape(-1),
    )


def _transformer_loss(model: torch.nn.Module, rows: torch.Tensor) -> torch.Tensor:
    logits = model(rows[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten())


def _optimizer(
    model: torch.nn.Module,
    device: torch.device,
    *,
    sparse_route: int | None = None,
) -> torch.optim.AdamW:
    kwargs: dict[str, Any] = {
        "lr": 1.0e-4,
        "betas": (0.9, 0.95),
        "weight_decay": 0.01,
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    parameters = (
        list(model.sparse_cake_parameters(sparse_route))
        if sparse_route is not None
        else list(model.parameters())
    )
    return torch.optim.AdamW(parameters, **kwargs)


def _run_steps(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: torch.Tensor,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    device: torch.device,
    steps: int,
    scaler: torch.amp.GradScaler,
) -> tuple[float, float, int]:
    optimized_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    _synchronize(device)
    started = time.perf_counter()
    final_loss = float("nan")
    gradient_parameters = 0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = loss_fn(model, rows)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
        gradient_parameters = sum(
            parameter.numel()
            for parameter in optimized_parameters
            if parameter.grad is not None
        )
        scaler.step(optimizer)
        scaler.update()
        final_loss = float(loss.detach().float().item())
    _synchronize(device)
    return time.perf_counter() - started, final_loss, int(gradient_parameters)


def _build_layercake(device: torch.device) -> torch.nn.Module:
    config = _load_config_with_extends(LAYERCAKE_CONFIG)
    return _build_model(config["model"], device)


def _build_layercake_next_byte_only(device: torch.device) -> torch.nn.Module:
    """Build the favorable core-only variant and remove dormant state from AdamW."""
    model = _build_layercake(device)
    dormant_prefixes = ("patch_generator.", "to_abi.", "from_abi.")
    for name, parameter in model.named_parameters():
        if name == "bos_context" or name.startswith(dormant_prefixes):
            parameter.requires_grad_(False)
    return model


def _build_layercake_routed_top1(device: torch.device) -> torch.nn.Module:
    config = _load_config_with_extends(LAYERCAKE_CONFIG)
    model_config = dict(config["model"])
    model_config.update(
        {
            "d_model": 320,
            "d_abi": 128,
            "layers": 1,
            "local_width": 320,
            "local_layers": 1,
            "routed_cake_experts": 6,
        }
    )
    model = _build_model(model_config, device)
    model.set_cake_route(0)
    return model


def _build_layercake_routed_top1_next_byte(
    device: torch.device,
) -> torch.nn.Module:
    model = _build_layercake_routed_top1(device)
    dormant_prefixes = ("patch_generator.", "to_abi.", "from_abi.")
    for name, parameter in model.named_parameters():
        if name == "bos_context" or name.startswith(dormant_prefixes):
            parameter.requires_grad_(False)
    return model


def _build_layercake_routed_parallel(device: torch.device) -> torch.nn.Module:
    config = _load_config_with_extends(LAYERCAKE_CONFIG)
    model_config = dict(config["model"])
    model_config.update(
        {
            "d_model": 320,
            "d_abi": 128,
            "layers": 1,
            "local_decoder": "routed_window_transformer",
            "local_width": 320,
            "local_layers": 0,
            "routed_cake_experts": 12,
        }
    )
    model = _build_model(model_config, device)
    model.set_cake_route(0)
    return model


def _build_layercake_routed_parallel_next_byte(
    device: torch.device,
) -> torch.nn.Module:
    model = _build_layercake_routed_parallel(device)
    dormant_prefixes = ("patch_generator.", "to_abi.", "from_abi.")
    for name, parameter in model.named_parameters():
        if name == "bos_context" or name.startswith(dormant_prefixes):
            parameter.requires_grad_(False)
    return model


def _build_layercake_routed_depth4_parallel(
    device: torch.device,
) -> torch.nn.Module:
    """Two full four-layer cakes: dense-v22 depth with one active route."""
    config = _load_config_with_extends(LAYERCAKE_CONFIG)
    model_config = dict(config["model"])
    model_config.update(
        {
            "d_model": 384,
            "d_abi": 128,
            "layers": 4,
            "local_decoder": "parallel_patch",
            "local_width": 384,
            "local_layers": 0,
            "routed_cake_experts": 2,
        }
    )
    model = _build_model(model_config, device)
    model.set_cake_route(0)
    return model


def _build_layercake_routed_depth4_parallel_next_byte(
    device: torch.device,
) -> torch.nn.Module:
    model = _build_layercake_routed_depth4_parallel(device)
    dormant_prefixes = ("patch_generator.", "to_abi.", "from_abi.")
    for name, parameter in model.named_parameters():
        if name == "bos_context" or name.startswith(dormant_prefixes):
            parameter.requires_grad_(False)
    return model


def _build_layercake_routed_depth2_parallel(
    device: torch.device,
) -> torch.nn.Module:
    """Four two-layer cakes; one route is active for a domain batch."""
    config = _load_config_with_extends(LAYERCAKE_CONFIG)
    model_config = dict(config["model"])
    model_config.update(
        {
            "d_model": 384,
            "d_abi": 128,
            "layers": 2,
            "local_decoder": "parallel_patch",
            "local_width": 384,
            "local_layers": 0,
            "routed_cake_experts": 4,
        }
    )
    model = _build_model(model_config, device)
    model.set_cake_route(0)
    return model


def _build_layercake_routed_depth2_parallel_next_byte(
    device: torch.device,
) -> torch.nn.Module:
    model = _build_layercake_routed_depth2_parallel(device)
    dormant_prefixes = ("patch_generator.", "to_abi.", "from_abi.")
    for name, parameter in model.named_parameters():
        if name == "bos_context" or name.startswith(dormant_prefixes):
            parameter.requires_grad_(False)
    return model


def _build_layercake_shared2_routed_tail(
    device: torch.device,
) -> torch.nn.Module:
    """Frozen two-layer foundation plus a trainable two-layer domain cake."""
    config = _load_config_with_extends(LAYERCAKE_CONFIG)
    model_config = dict(config["model"])
    model_config.update(
        {
            "d_model": 384,
            "d_abi": 128,
            "layers": 4,
            "shared_cake_layers": 2,
            "local_decoder": "parallel_patch",
            "local_width": 384,
            "local_layers": 0,
            "routed_cake_experts": 3,
        }
    )
    model = _build_model(model_config, device)
    model.set_cake_route(0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.core[2:]:
        for parameter in block.active_expert_parameters(0):
            parameter.requires_grad_(True)
    return model


def _build_layercake_shared3_routed_tail(
    device: torch.device,
) -> torch.nn.Module:
    """Frozen three-layer foundation plus one trainable domain cake layer."""
    config = _load_config_with_extends(ROUTED_LAYERCAKE_CONFIG)
    model_config = dict(config["model"])
    model_config.update(
        {
            "d_model": 384,
            "d_abi": 128,
            "layers": 4,
            "shared_cake_layers": 3,
            "local_decoder": "routed_window_transformer",
            "local_width": 384,
            "local_layers": 0,
            "routed_cake_experts": 5,
        }
    )
    model = _build_model(model_config, device)
    model.set_cake_route(0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.core[3].active_expert_parameters(0):
        parameter.requires_grad_(True)
    return model


def _build_layercake_shared3_routed_tail_int8_foundation(
    device: torch.device,
) -> torch.nn.Module:
    if device.type != "cpu":
        return _build_layercake_shared3_routed_tail(device)
    model = _build_layercake_shared3_routed_tail(device)
    model._logical_total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    for index in range(3):
        torch.ao.quantization.quantize_dynamic(
            model.core[index],
            {torch.nn.Linear},
            dtype=torch.qint8,
            inplace=True,
        )
    return model


def _build_transformer(device: torch.device) -> torch.nn.Module:
    config = _load_config_with_extends(TRANSFORMER_CONFIG)
    model = config["model"]
    tokenizer = config["tokenizer"]
    training = config["training"]
    return BPETokenTransformerLM(
        vocab_size=int(tokenizer["vocab_size"]),
        d_model=int(model["d_model"]),
        layers=int(model["layers"]),
        heads=int(model["heads"]),
        max_len=int(training["seq_len"]),
        ff_mult=int(model.get("ff_mult", 4)),
        dropout=float(model.get("dropout", 0.0)),
    ).to(device)


def _benchmark_model(
    *,
    name: str,
    build: Callable[[torch.device], torch.nn.Module],
    loss_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    shape: tuple[int, int],
    logical_bytes_per_step: float,
    token_high: int,
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
    repeat: int,
    sparse_route: int | None = None,
) -> dict[str, Any]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(24680 + repeat)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(24680 + repeat)
    initialization_started = time.perf_counter()
    model = build(device)
    model.train()
    optimizer = _optimizer(model, device, sparse_route=sparse_route)
    initialization_seconds = time.perf_counter() - initialization_started
    total_parameters, trainable_parameters = _count_parameters(model)
    optimizer_parameters = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    generator = torch.Generator(device=device).manual_seed(97531 + repeat)
    rows = torch.randint(
        0,
        token_high,
        shape,
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    warmup_seconds, _, _ = _run_steps(
        model=model,
        optimizer=optimizer,
        rows=rows,
        loss_fn=loss_fn,
        device=device,
        steps=warmup_steps,
        scaler=scaler,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed, final_loss, gradient_parameters = _run_steps(
        model=model,
        optimizer=optimizer,
        rows=rows,
        loss_fn=loss_fn,
        device=device,
        steps=measured_steps,
        scaler=scaler,
    )
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    state_bytes = _tensor_state_bytes(model, optimizer)
    result = {
        "name": name,
        "repeat": repeat,
        "shape_including_target": list(shape),
        "logical_bytes_per_step": logical_bytes_per_step,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "optimizer_parameters": optimizer_parameters,
        "gradient_parameters": gradient_parameters,
        "gradient_parameter_fraction": gradient_parameters
        / max(trainable_parameters, 1),
        "gradient_optimizer_parameter_fraction": gradient_parameters
        / max(optimizer_parameters, 1),
        "initialization_seconds_excluded": initialization_seconds,
        "warmup_steps": warmup_steps,
        "warmup_seconds_excluded": warmup_seconds,
        "measured_steps": measured_steps,
        "measured_seconds": elapsed,
        "steps_per_second": measured_steps / elapsed,
        "logical_bytes_per_second": logical_bytes_per_step
        * measured_steps
        / elapsed,
        "final_loss": final_loss,
        "finite_loss": math.isfinite(final_loss),
        "tensor_state_bytes": state_bytes,
        "cuda_peak_memory_allocated_bytes": peak_memory,
    }
    del rows, optimizer, model, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    throughput = [float(row["logical_bytes_per_second"]) for row in rows]
    seconds = [float(row["measured_seconds"]) for row in rows]
    return {
        "repeats": len(rows),
        "logical_bytes_per_second": {
            "values": throughput,
            "median": statistics.median(throughput),
            "minimum": min(throughput),
            "maximum": max(throughput),
        },
        "measured_seconds": {
            "values": seconds,
            "median": statistics.median(seconds),
        },
        "total_parameters": rows[0]["total_parameters"],
        "trainable_parameters": rows[0]["trainable_parameters"],
        "optimizer_parameters": rows[0]["optimizer_parameters"],
        "minimum_gradient_parameter_fraction": min(
            float(row["gradient_parameter_fraction"]) for row in rows
        ),
        "minimum_gradient_optimizer_parameter_fraction": min(
            float(row["gradient_optimizer_parameter_fraction"])
            for row in rows
        ),
        "maximum_tensor_state_bytes": max(
            int(row["tensor_state_bytes"]["total"]) for row in rows
        ),
        "maximum_cuda_peak_memory_allocated_bytes": (
            max(
                int(row["cuda_peak_memory_allocated_bytes"])
                for row in rows
            )
            if rows[0]["cuda_peak_memory_allocated_bytes"] is not None
            else None
        ),
        "all_losses_finite": all(bool(row["finite_loss"]) for row in rows),
    }


def _bytes_per_token() -> float:
    document = json.loads(TRANSFORMER_METRICS.read_text(encoding="utf-8"))
    latest = document["latest"]
    config = document["training_config"]
    token_positions = (
        int(config["micro_batch_size"])
        * int(config.get("grad_accum_steps", 1))
        * int(config["seq_len"])
    )
    return float(latest["bytes_per_step"]) / token_positions


def _device_result(
    *,
    device_name: str,
    cpu_threads: int,
    cpu_batch_size: int,
    gpu_batch_size: int,
    raw_sequence_bytes: int,
    warmup_steps: int,
    measured_steps: int,
    repeats: int,
    layercake_mode: str,
) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    batch_size = gpu_batch_size if device.type == "cuda" else cpu_batch_size
    bytes_per_token = _bytes_per_token()
    transformer_tokens = max(2, round(raw_sequence_bytes / bytes_per_token))
    transformer_config = _load_config_with_extends(TRANSFORMER_CONFIG)
    max_tokens = int(transformer_config["training"]["seq_len"])
    if transformer_tokens > max_tokens:
        raise ValueError(
            f"matched transformer length {transformer_tokens} exceeds max {max_tokens}"
        )
    workload = Workload(
        raw_sequence_bytes=raw_sequence_bytes,
        batch_size=batch_size,
        transformer_tokens=transformer_tokens,
        transformer_bytes_per_token=bytes_per_token,
    )
    layercake_rows: list[dict[str, Any]] = []
    transformer_rows: list[dict[str, Any]] = []
    if layercake_mode == "recipe":
        layercake_build = _build_layercake
        layercake_loss = _layercake_loss
        sparse_route = None
    elif layercake_mode == "next_byte_only":
        layercake_build = _build_layercake_next_byte_only
        layercake_loss = _layercake_next_byte_loss
        sparse_route = None
    elif layercake_mode == "routed_top1":
        layercake_build = _build_layercake_routed_top1
        layercake_loss = _layercake_loss
        sparse_route = 0
    elif layercake_mode == "routed_top1_next_byte":
        layercake_build = _build_layercake_routed_top1_next_byte
        layercake_loss = _layercake_next_byte_loss
        sparse_route = 0
    elif layercake_mode == "routed_parallel_top1":
        layercake_build = _build_layercake_routed_parallel
        layercake_loss = _layercake_loss
        sparse_route = 0
    elif layercake_mode == "routed_parallel_top1_next_byte":
        layercake_build = _build_layercake_routed_parallel_next_byte
        layercake_loss = _layercake_next_byte_loss
        sparse_route = 0
    elif layercake_mode == "routed_depth4_parallel":
        layercake_build = _build_layercake_routed_depth4_parallel
        layercake_loss = _layercake_loss
        sparse_route = 0
    elif layercake_mode == "routed_depth4_parallel_next_byte":
        layercake_build = _build_layercake_routed_depth4_parallel_next_byte
        layercake_loss = _layercake_next_byte_loss
        sparse_route = 0
    elif layercake_mode == "routed_depth2_parallel":
        layercake_build = _build_layercake_routed_depth2_parallel
        layercake_loss = _layercake_loss
        sparse_route = 0
    elif layercake_mode == "routed_depth2_parallel_next_byte":
        layercake_build = _build_layercake_routed_depth2_parallel_next_byte
        layercake_loss = _layercake_next_byte_loss
        sparse_route = 0
    elif layercake_mode == "shared2_routed_tail":
        layercake_build = _build_layercake_shared2_routed_tail
        layercake_loss = _layercake_next_byte_loss
        sparse_route = 0
    elif layercake_mode == "shared3_routed_tail":
        layercake_build = _build_layercake_shared3_routed_tail
        layercake_loss = _layercake_domain_cake_loss
        sparse_route = 0
    elif layercake_mode == "shared3_routed_tail_int8_foundation":
        layercake_build = _build_layercake_shared3_routed_tail_int8_foundation
        layercake_loss = _layercake_domain_cake_loss
        sparse_route = 0
    else:
        raise ValueError(f"unsupported LayerCake mode: {layercake_mode}")
    for repeat in range(repeats):
        specifications = [
            (
                "layercake",
                layercake_build,
                layercake_loss,
                (batch_size, raw_sequence_bytes + 1),
                workload.layercake_bytes_per_step,
                256,
            ),
            (
                "transformer",
                _build_transformer,
                _transformer_loss,
                (batch_size, transformer_tokens + 1),
                workload.transformer_bytes_per_step,
                4096,
            ),
        ]
        if repeat % 2:
            specifications.reverse()
        for name, build, loss_fn, shape, logical_bytes, token_high in specifications:
            row = _benchmark_model(
                name=name,
                build=build,
                loss_fn=loss_fn,
                shape=shape,
                logical_bytes_per_step=logical_bytes,
                token_high=token_high,
                device=device,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
                repeat=repeat,
                sparse_route=sparse_route if name == "layercake" else None,
            )
            (layercake_rows if name == "layercake" else transformer_rows).append(row)

    layercake = _summary(layercake_rows)
    transformer = _summary(transformer_rows)
    repeat_ratios = [
        layercake_rows[index]["logical_bytes_per_second"]
        / transformer_rows[index]["logical_bytes_per_second"]
        for index in range(repeats)
    ]
    median_ratio = statistics.median(repeat_ratios)
    minimum_ratio = min(repeat_ratios)
    parameter_ratio = (
        layercake["total_parameters"] / transformer["total_parameters"]
    )
    logical_batch_ratio = (
        workload.layercake_bytes_per_step / workload.transformer_bytes_per_step
    )
    gates = {
        "parameter_count_within_5_percent": 0.95 <= parameter_ratio <= 1.05,
        "logical_bytes_per_step_within_1_percent": 0.99
        <= logical_batch_ratio
        <= 1.01,
        "all_losses_finite": layercake["all_losses_finite"]
        and transformer["all_losses_finite"],
        "all_optimizer_layercake_parameters_receive_gradients": layercake[
            "minimum_gradient_optimizer_parameter_fraction"
        ]
        >= 0.99,
        "all_trainable_transformer_parameters_receive_gradients": transformer[
            "minimum_gradient_parameter_fraction"
        ]
        == 1.0,
        "minimum_repeat_training_throughput_at_least_5x": minimum_ratio >= 5.0,
    }
    return {
        "device": device.type,
        "environment": _execution_environment(device),
        "timing_scope": (
            "steady-state complete optimizer steps: zero_grad + forward + "
            "loss + backward + gradient clipping + AdamW update"
        ),
        "layercake_training_mode": layercake_mode,
        "excluded_from_timing": [
            "model initialization",
            "tokenizer training/encoding",
            "checkpoint I/O",
            "evaluation",
            "warmup",
        ],
        "workload": {
            "raw_sequence_bytes": workload.raw_sequence_bytes,
            "batch_size": workload.batch_size,
            "layercake_bytes_per_step": workload.layercake_bytes_per_step,
            "transformer_tokens_per_sequence": workload.transformer_tokens,
            "transformer_bytes_per_token": workload.transformer_bytes_per_token,
            "transformer_bytes_per_step": workload.transformer_bytes_per_step,
            "logical_batch_bytes_ratio_layercake_over_transformer": logical_batch_ratio,
        },
        "layercake": layercake,
        "transformer": transformer,
        "repeat_details": {
            "layercake": layercake_rows,
            "transformer": transformer_rows,
        },
        "ratios": {
            "parameter_count_layercake_over_transformer": parameter_ratio,
            "training_throughput_layercake_over_transformer_per_repeat": repeat_ratios,
            "median_training_throughput_layercake_over_transformer": median_ratio,
            "minimum_training_throughput_layercake_over_transformer": minimum_ratio,
            "tensor_state_bytes_transformer_over_layercake": transformer[
                "maximum_tensor_state_bytes"
            ]
            / layercake["maximum_tensor_state_bytes"],
            "cuda_peak_memory_transformer_over_layercake": (
                transformer["maximum_cuda_peak_memory_allocated_bytes"]
                / layercake["maximum_cuda_peak_memory_allocated_bytes"]
                if device.type == "cuda"
                else None
            ),
        },
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure matched 15M LayerCake/transformer training throughput"
    )
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--cpu-batch-size", type=int, default=1)
    parser.add_argument("--gpu-batch-size", type=int, default=16)
    parser.add_argument("--raw-sequence-bytes", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--layercake-mode",
        choices=(
            "recipe",
            "next_byte_only",
            "routed_top1",
            "routed_top1_next_byte",
            "routed_parallel_top1",
            "routed_parallel_top1_next_byte",
            "routed_depth4_parallel",
            "routed_depth4_parallel_next_byte",
            "routed_depth2_parallel",
            "routed_depth2_parallel_next_byte",
            "shared2_routed_tail",
            "shared3_routed_tail",
            "shared3_routed_tail_int8_foundation",
        ),
        default="recipe",
        help=(
            "recipe includes the configured auxiliary patch loss; "
            "next_byte_only is a favorable lower-bound core benchmark; "
            "routed_top1 modes measure the six-expert sparse architecture"
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.cpu_batch_size < 1 or args.gpu_batch_size < 1:
        raise ValueError("thread and batch counts must be positive")
    if args.raw_sequence_bytes < 4 or args.raw_sequence_bytes % 4:
        raise ValueError("raw sequence bytes must be positive and patch-4 aligned")
    if args.warmup_steps < 1 or args.measured_steps < 1 or args.repeats < 2:
        raise ValueError("warmup/measured steps must be positive and repeats >= 2")

    requested = [item.strip() for item in args.devices.split(",") if item.strip()]
    results: dict[str, Any] = {}
    for device_name in requested:
        result = _device_result(
            device_name=device_name,
            cpu_threads=args.cpu_threads,
            cpu_batch_size=args.cpu_batch_size,
            gpu_batch_size=args.gpu_batch_size,
            raw_sequence_bytes=args.raw_sequence_bytes,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            repeats=args.repeats,
            layercake_mode=args.layercake_mode,
        )
        results[device_name] = result
        print(
            f"{device_name}: median={result['ratios']['median_training_throughput_layercake_over_transformer']:.3f}x "
            f"minimum={result['ratios']['minimum_training_throughput_layercake_over_transformer']:.3f}x "
            f"status={result['status']}"
        )

    required_devices = {"cpu", "cuda"}
    complete_platform_matrix = required_devices.issubset(results)
    all_device_gates = complete_platform_matrix and all(
        results[name]["status"] == "PASS" for name in required_devices
    )
    routed_domain_mode = args.layercake_mode.startswith("shared3_routed_tail")
    artifact = {
        "schema_version": 2,
        "benchmark": (
            "northstar_v23_domain_cake_training_speed"
            if routed_domain_mode
            else "northstar_v22_full_core_training_speed"
        ),
        "claim_scope": (
            "Matched-capacity steady-state selected-domain-cake fine-tuning versus "
            "full tokenizer-transformer training on the recorded CPU and GPU. The "
            "LayerCake foundation and portable decoder are frozen. This is not a "
            "full-foundation pretraining or time-to-quality claim."
            if routed_domain_mode
            else (
                "Matched-parameter full-core steady-state training throughput on the "
                "recorded CPU and GPU; this does not measure convergence or time-to-quality."
            )
        ),
        "architecture_inputs": {
            "layercake_config": str(
                (ROUTED_LAYERCAKE_CONFIG if routed_domain_mode else LAYERCAKE_CONFIG)
                .relative_to(ROOT)
            ).replace("\\", "/"),
            "transformer_config": str(TRANSFORMER_CONFIG.relative_to(ROOT)).replace("\\", "/"),
            "transformer_bytes_per_token_source": str(
                TRANSFORMER_METRICS.relative_to(ROOT)
            ).replace("\\", "/"),
        },
        "protocol": {
            "cpu_threads": args.cpu_threads,
            "cpu_batch_size": args.cpu_batch_size,
            "gpu_batch_size": args.gpu_batch_size,
            "raw_sequence_bytes": args.raw_sequence_bytes,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "repeats": args.repeats,
            "layercake_training_mode": args.layercake_mode,
            "repeat_order": "alternating LayerCake-first / transformer-first",
            "precision": {
                "cpu": (
                    "dynamic INT8 frozen foundation + float32 active tail/decoder"
                    if args.layercake_mode == "shared3_routed_tail_int8_foundation"
                    else "float32"
                ),
                "cuda": "AMP float16",
            },
            "optimizer": "AdamW; fused on CUDA",
            "parameter_count_scope": (
                "pre-quantization logical architecture capacity"
                if args.layercake_mode == "shared3_routed_tail_int8_foundation"
                else "live torch parameters"
            ),
        },
        "devices": results,
        "required_gates": {
            "cpu_and_cuda_measured": complete_platform_matrix,
            "all_cpu_and_cuda_gates_pass": all_device_gates,
        },
        "failed_required": [
            name
            for name, passed in {
                "cpu_and_cuda_measured": complete_platform_matrix,
                "all_cpu_and_cuda_gates_pass": all_device_gates,
            }.items()
            if not passed
        ],
        "status": "PASS" if all_device_gates else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": artifact["status"]}))


if __name__ == "__main__":
    main()
