"""Train and evaluate the budgeted hierarchical CountCake candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import (  # noqa: E402
    DEFAULT_BACKOFF_STRENGTHS,
    DEFAULT_ONLINE_CACHE_SPECS,
    HierarchicalCountCakeLM,
    PrunedBackoffByteCake,
    assert_parameter_budget,
    save_count_cake_bundle,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _empty_count_cake(
    device: torch.device,
    *,
    max_order: int,
) -> PrunedBackoffByteCake:
    empty_keys = torch.empty(0, device=device, dtype=torch.int64)
    empty_counts = torch.empty(0, device=device, dtype=torch.float32)
    return PrunedBackoffByteCake(
        unigram_counts=torch.ones(256, device=device),
        order_tables=[(empty_keys, empty_counts)] * int(max_order),
        backoff_strengths=DEFAULT_BACKOFF_STRENGTHS,
        state_budget=256,
        corpus_bytes=1,
    )


def _build_host(
    cake: PrunedBackoffByteCake,
    *,
    patch_size: int,
    chunking_mode: str,
    d_byte: int,
    d_model: int,
    d_abi: int,
    patch_layers: int,
    patch_core_type: str,
    patch_selective_rank: int,
    patch_attention_heads: int,
    scratchpad_stride: int,
    dynamic_hash_buckets: int,
    dynamic_hash_width: int,
    dynamic_hash_tables: int,
    dynamic_hash_sparse: bool,
    neural_context_buckets: int,
    neural_context_order: int,
    neural_context_sparse: bool,
    local_width: int,
    local_recurrent: bool,
    local_continuous: bool,
    local_decoder: str | None,
    local_layers: int,
    local_dilation_growth: int,
    local_gru_layers: int,
    local_rank: int,
    byte_head: str,
    online_cache: bool,
    initial_neural_fraction: float,
    prediction_start: int,
    confidence_gate: bool,
    expert_confidence_gate: bool,
    count_distribution_gate: bool,
    count_order_routing: bool,
    count_order_router_hidden: int,
    gate_hidden_width: int,
) -> HierarchicalCountCakeLM:
    return HierarchicalCountCakeLM(
        cake,
        patch_size=patch_size,
        chunking_mode=chunking_mode,
        d_byte=d_byte,
        d_model=d_model,
        d_abi=d_abi,
        patch_layers=patch_layers,
        patch_core_type=patch_core_type,
        patch_selective_rank=patch_selective_rank,
        patch_attention_heads=patch_attention_heads,
        scratchpad_stride=scratchpad_stride,
        dynamic_hash_buckets=dynamic_hash_buckets,
        dynamic_hash_width=dynamic_hash_width,
        dynamic_hash_tables=dynamic_hash_tables,
        dynamic_hash_sparse=dynamic_hash_sparse,
        neural_context_buckets=neural_context_buckets,
        neural_context_order=neural_context_order,
        neural_context_sparse=neural_context_sparse,
        local_width=local_width,
        local_recurrent=local_recurrent,
        local_continuous=local_continuous,
        local_decoder=local_decoder,
        local_layers=local_layers,
        local_dilation_growth=local_dilation_growth,
        local_gru_layers=local_gru_layers,
        local_rank=local_rank,
        byte_head=byte_head,
        online_cache_specs=(DEFAULT_ONLINE_CACHE_SPECS if online_cache else ()),
        initial_neural_fraction=initial_neural_fraction,
        prediction_start=prediction_start,
        confidence_gate=confidence_gate,
        expert_confidence_gate=expert_confidence_gate,
        count_distribution_gate=count_distribution_gate,
        count_order_routing=count_order_routing,
        count_order_router_hidden=count_order_router_hidden,
        gate_hidden_width=gate_hidden_width,
    )


@torch.no_grad()
def _evaluate(
    model: HierarchicalCountCakeLM,
    payload: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    row_count = payload.numel() // seq_len
    if row_count == 0:
        raise ValueError("evaluation payload is shorter than one sequence")
    rows = payload[: row_count * seq_len].reshape(row_count, seq_len)
    total_nll = 0.0
    total_bytes = 0
    for offset in range(0, row_count, batch_size):
        batch = rows[offset : offset + batch_size].to(
            device=device,
            dtype=torch.long,
        )
        loss = model.loss(batch, neural_auxiliary_weight=0.0)
        predicted = batch.shape[0] * (seq_len - model.prediction_start)
        total_nll += float(loss) * predicted
        total_bytes += predicted
    nll = total_nll / total_bytes
    return {
        "bytes": total_bytes,
        "nll": nll,
        "bpb": nll / math.log(2.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default="runs_experiment/production_v24_corpus/train.bin",
    )
    parser.add_argument(
        "--neural-train",
        help=(
            "optional broader corpus used only for sampled neural updates; "
            "the --train corpus remains the deterministic count pass"
        ),
    )
    parser.add_argument(
        "--eval",
        default="runs_experiment/production_v24_corpus/eval.bin",
    )
    parser.add_argument(
        "--out-dir",
        default="runs_experiment/production_v24_1m_count_cake",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target-parameters", type=int, default=989_276)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument(
        "--backoff-mode", choices=("fixed", "distinct", "discount"), default="fixed"
    )
    parser.add_argument("--discount", type=float, default=0.75)
    parser.add_argument(
        "--backoff-strengths",
        help="comma-separated strengths (default: architecture recipe)",
    )
    parser.add_argument(
        "--count-budget-mode",
        choices=(
            "sequential",
            "balanced",
            "hybrid",
            "hybrid3",
            "hybrid2",
            "information",
        ),
        default="sequential",
    )
    parser.add_argument(
        "--count-chunk-bytes",
        type=int,
        default=0,
        help="bounded-memory streaming count training when positive",
    )
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument(
        "--chunking-mode", choices=("fixed", "delimiter"), default="fixed"
    )
    parser.add_argument(
        "--prediction-start",
        type=int,
        default=None,
        help="unscored byte warm-up (default: one patch)",
    )
    parser.add_argument("--d-byte", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-abi", type=int, default=64)
    parser.add_argument("--patch-layers", type=int, default=1)
    parser.add_argument(
        "--patch-core-type",
        choices=(
            "gru",
            "selective_scan",
            "low_rank_selective_scan",
            "attention",
        ),
        default="gru",
    )
    parser.add_argument("--patch-selective-rank", type=int, default=128)
    parser.add_argument("--patch-attention-heads", type=int, default=8)
    parser.add_argument("--scratchpad-stride", type=int, default=0)
    parser.add_argument("--dynamic-hash-buckets", type=int, default=0)
    parser.add_argument("--dynamic-hash-width", type=int, default=64)
    parser.add_argument("--dynamic-hash-tables", type=int, default=1)
    parser.add_argument(
        "--sparse-dynamic-hash",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="update only delimiter-hash rows touched by the current batch",
    )
    parser.add_argument("--neural-context-buckets", type=int, default=0)
    parser.add_argument("--neural-context-order", type=int, default=3)
    parser.add_argument(
        "--sparse-neural-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="update only causal context rows touched by the current batch",
    )
    parser.add_argument("--local-width", type=int, default=32)
    parser.add_argument("--local-recurrent", action="store_true")
    parser.add_argument("--local-continuous", action="store_true")
    parser.add_argument(
        "--local-decoder",
        choices=("position", "gru", "lstm", "scan", "dilated_conv"),
    )
    parser.add_argument("--local-layers", type=int, default=5)
    parser.add_argument("--local-dilation-growth", type=int, default=2)
    parser.add_argument("--local-gru-layers", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=64)
    parser.add_argument("--byte-head", choices=("radix", "direct"), default="radix")
    parser.add_argument("--confidence-gate", action="store_true")
    parser.add_argument("--expert-confidence-gate", action="store_true")
    parser.add_argument("--count-distribution-gate", action="store_true")
    parser.add_argument("--count-order-routing", action="store_true")
    parser.add_argument("--count-order-router-hidden", type=int, default=0)
    parser.add_argument(
        "--confidence-gate-lr-scale",
        type=float,
        default=1.0,
        help="learning-rate multiplier for target-independent confidence scalars",
    )
    parser.add_argument("--gate-hidden-width", type=int, default=0)
    parser.add_argument(
        "--online-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable the frozen causal document-cache recipe (default: recurrent hosts)",
    )
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="evaluation-only batch size (default: --batch-size)",
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--training-source-byte-budget",
        type=int,
        default=0,
        help=(
            "when positive, include the one-pass count corpus and choose full "
            "sequences plus a partial final batch to stay at or below this "
            "total source-byte exposure"
        ),
    )
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--min-lr", type=float, default=0.0001)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--neural-auxiliary-weight", type=float, default=0.1)
    parser.add_argument(
        "--mixture-calibration-steps",
        type=int,
        default=0,
        help=(
            "train the neural expert without frozen count lookups until the "
            "final N steps, then calibrate the complete mixture"
        ),
    )
    parser.add_argument("--initial-neural-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=24001)
    args = parser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.eval_batch_size <= 0:
        raise ValueError("eval-batch-size must be positive")
    if args.mixture_calibration_steps < 0:
        raise ValueError("mixture-calibration-steps cannot be negative")
    if args.confidence_gate_lr_scale <= 0:
        raise ValueError("confidence-gate-lr-scale must be positive")
    if args.online_cache is None:
        args.online_cache = args.local_recurrent or args.local_decoder in {
            "scan",
            "dilated_conv",
        }
    if args.prediction_start is None:
        args.prediction_start = args.patch_size
    backoff_strengths = (
        DEFAULT_BACKOFF_STRENGTHS
        if args.backoff_strengths is None
        else tuple(float(value) for value in args.backoff_strengths.split(","))
    )
    if len(backoff_strengths) < args.max_order:
        raise ValueError("backoff strengths are shorter than max-order")

    if args.seq_len % args.patch_size:
        raise ValueError("seq-len must be divisible by patch-size")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    total_started = time.perf_counter()
    train_bytes = Path(args.train).read_bytes()
    neural_train_path = args.train if args.neural_train is None else args.neural_train
    neural_train_bytes = Path(neural_train_path).read_bytes()
    eval_bytes = Path(args.eval).read_bytes()
    loaded_seconds = time.perf_counter() - total_started
    train_cpu = torch.frombuffer(bytearray(train_bytes), dtype=torch.uint8)
    neural_train_cpu = torch.frombuffer(
        bytearray(neural_train_bytes), dtype=torch.uint8
    )
    eval_cpu = torch.frombuffer(bytearray(eval_bytes), dtype=torch.uint8).to(
        dtype=torch.long
    )
    requested_steps = args.steps
    optimization_batch_sizes = [args.batch_size] * args.steps
    if args.training_source_byte_budget > 0:
        neural_source_budget = args.training_source_byte_budget - len(train_bytes)
        if neural_source_budget < args.seq_len:
            raise ValueError(
                "training-source-byte-budget must cover the count pass and at "
                "least one neural sequence"
            )
        neural_sequences = neural_source_budget // args.seq_len
        full_batches, final_batch = divmod(neural_sequences, args.batch_size)
        optimization_batch_sizes = [args.batch_size] * full_batches
        if final_batch:
            optimization_batch_sizes.append(final_batch)
        args.steps = len(optimization_batch_sizes)

    empty = _empty_count_cake(device, max_order=args.max_order)
    host_probe = _build_host(
        empty,
        patch_size=args.patch_size,
        chunking_mode=args.chunking_mode,
        d_byte=args.d_byte,
        d_model=args.d_model,
        d_abi=args.d_abi,
        patch_layers=args.patch_layers,
        patch_core_type=args.patch_core_type,
        patch_selective_rank=args.patch_selective_rank,
        patch_attention_heads=args.patch_attention_heads,
        scratchpad_stride=args.scratchpad_stride,
        dynamic_hash_buckets=args.dynamic_hash_buckets,
        dynamic_hash_width=args.dynamic_hash_width,
        dynamic_hash_tables=args.dynamic_hash_tables,
        dynamic_hash_sparse=args.sparse_dynamic_hash,
        neural_context_buckets=args.neural_context_buckets,
        neural_context_order=args.neural_context_order,
        neural_context_sparse=args.sparse_neural_context,
        local_width=args.local_width,
        local_recurrent=args.local_recurrent,
        local_continuous=args.local_continuous,
        local_decoder=args.local_decoder,
        local_layers=args.local_layers,
        local_dilation_growth=args.local_dilation_growth,
        local_gru_layers=args.local_gru_layers,
        local_rank=args.local_rank,
        byte_head=args.byte_head,
        online_cache=args.online_cache,
        initial_neural_fraction=args.initial_neural_fraction,
        prediction_start=args.prediction_start,
        confidence_gate=args.confidence_gate,
        expert_confidence_gate=args.expert_confidence_gate,
        count_distribution_gate=args.count_distribution_gate,
        count_order_routing=args.count_order_routing,
        count_order_router_hidden=args.count_order_router_hidden,
        gate_hidden_width=args.gate_hidden_width,
    )
    neural_parameters = host_probe.neural_parameters
    del host_probe, empty
    state_budget = args.target_parameters - neural_parameters
    if state_budget < 256:
        raise ValueError("target parameter budget is too small for this host")

    count_started = time.perf_counter()
    if args.count_chunk_bytes > 0:
        cake = PrunedBackoffByteCake.train_streaming_from_bytes(
            train_cpu,
            device=device,
            state_budget=state_budget,
            max_order=args.max_order,
            chunk_bytes=args.count_chunk_bytes,
            backoff_mode=args.backoff_mode,
            discount=args.discount,
            budget_mode=args.count_budget_mode,
            backoff_strengths=backoff_strengths,
        )
    else:
        train_device = train_cpu.to(device=device, dtype=torch.long)
        cake = PrunedBackoffByteCake.train_from_bytes(
            train_device,
            state_budget=state_budget,
            max_order=args.max_order,
            backoff_mode=args.backoff_mode,
            discount=args.discount,
            budget_mode=args.count_budget_mode,
            backoff_strengths=backoff_strengths,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    count_seconds = time.perf_counter() - count_started
    model = _build_host(
        cake,
        patch_size=args.patch_size,
        chunking_mode=args.chunking_mode,
        d_byte=args.d_byte,
        d_model=args.d_model,
        d_abi=args.d_abi,
        patch_layers=args.patch_layers,
        patch_core_type=args.patch_core_type,
        patch_selective_rank=args.patch_selective_rank,
        patch_attention_heads=args.patch_attention_heads,
        scratchpad_stride=args.scratchpad_stride,
        dynamic_hash_buckets=args.dynamic_hash_buckets,
        dynamic_hash_width=args.dynamic_hash_width,
        dynamic_hash_tables=args.dynamic_hash_tables,
        dynamic_hash_sparse=args.sparse_dynamic_hash,
        neural_context_buckets=args.neural_context_buckets,
        neural_context_order=args.neural_context_order,
        neural_context_sparse=args.sparse_neural_context,
        local_width=args.local_width,
        local_recurrent=args.local_recurrent,
        local_continuous=args.local_continuous,
        local_decoder=args.local_decoder,
        local_layers=args.local_layers,
        local_dilation_growth=args.local_dilation_growth,
        local_gru_layers=args.local_gru_layers,
        local_rank=args.local_rank,
        byte_head=args.byte_head,
        online_cache=args.online_cache,
        initial_neural_fraction=args.initial_neural_fraction,
        prediction_start=args.prediction_start,
        confidence_gate=args.confidence_gate,
        expert_confidence_gate=args.expert_confidence_gate,
        count_distribution_gate=args.count_distribution_gate,
        count_order_routing=args.count_order_routing,
        count_order_router_hidden=args.count_order_router_hidden,
        gate_hidden_width=args.gate_hidden_width,
    ).to(device)
    assert_parameter_budget(model, target=args.target_parameters, relative_tolerance=0.0)

    initial_eval_started = time.perf_counter()
    initial_eval = _evaluate(
        model,
        eval_cpu,
        seq_len=args.seq_len,
        batch_size=args.eval_batch_size,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    initial_eval_seconds = time.perf_counter() - initial_eval_started

    optimizer_kwargs = {
        "lr": args.lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.01,
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    confidence_parameters = []
    base_parameters = []
    sparse_parameters = []
    for name, parameter in model.named_parameters():
        if (
            model.dynamic_hash_sparse
            and (
                name == "dynamic_hash_embedding.weight"
                or name.startswith("dynamic_hash_embeddings.")
            )
        ):
            sparse_parameters.append(parameter)
        elif (
            model.neural_context_sparse
            and name == "neural_context_embedding.weight"
        ):
            sparse_parameters.append(parameter)
        elif name.startswith(
            (
                "confidence_gate.",
                "expert_confidence_gate.",
                "count_distribution_gate.",
            )
        ):
            confidence_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    parameter_groups = [{"params": base_parameters, "lr_scale": 1.0}]
    if confidence_parameters:
        parameter_groups.append(
            {
                "params": confidence_parameters,
                "lr_scale": float(args.confidence_gate_lr_scale),
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
    sparse_optimizer = (
        torch.optim.SparseAdam(
            sparse_parameters,
            lr=args.lr,
            betas=(0.9, 0.95),
        )
        if sparse_parameters
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_generator = torch.Generator().manual_seed(args.seed + 17)
    offsets = torch.arange(args.seq_len, dtype=torch.long)
    max_start = neural_train_cpu.numel() - args.seq_len
    if max_start < 0:
        raise ValueError("neural training corpus is shorter than seq-len")
    model.train()
    optimization_started = time.perf_counter()
    final_loss = float("nan")
    for step, step_batch_size in enumerate(optimization_batch_sizes, start=1):
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            learning_rate = args.lr * step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(
                args.steps - args.warmup_steps,
                1,
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            learning_rate = args.min_lr + (args.lr - args.min_lr) * cosine
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * float(group["lr_scale"])
        if sparse_optimizer is not None:
            for group in sparse_optimizer.param_groups:
                group["lr"] = learning_rate
        starts = torch.randint(
            max_start + 1,
            (step_batch_size,),
            generator=start_generator,
        )
        rows = neural_train_cpu[starts[:, None] + offsets].to(
            device=device,
            dtype=torch.long,
        )
        optimizer.zero_grad(set_to_none=True)
        if sparse_optimizer is not None:
            sparse_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            calibration_start = max(
                1, args.steps - args.mixture_calibration_steps + 1
            )
            if (
                args.mixture_calibration_steps > 0
                and step < calibration_start
            ):
                loss = model.neural_loss(rows)
            else:
                loss = model.loss(
                    rows,
                    neural_auxiliary_weight=args.neural_auxiliary_weight,
                )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if sparse_optimizer is not None:
            scaler.unscale_(sparse_optimizer)
        torch.nn.utils.clip_grad_norm_(
            base_parameters + confidence_parameters,
            1.0,
        )
        scaler.step(optimizer)
        if sparse_optimizer is not None:
            scaler.step(sparse_optimizer)
        scaler.update()
        final_loss = float(loss.detach())
        if step == 1 or step % 100 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "steps": args.steps,
                        "loss": final_loss,
                        "lr": learning_rate,
                        "elapsed_seconds": time.perf_counter()
                        - optimization_started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    optimization_seconds = time.perf_counter() - optimization_started

    final_eval_started = time.perf_counter()
    final_eval = _evaluate(
        model,
        eval_cpu,
        seq_len=args.seq_len,
        batch_size=args.eval_batch_size,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    final_eval_seconds = time.perf_counter() - final_eval_started
    gradient_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    total_seconds = time.perf_counter() - total_started

    result = {
        "format": "layercake-count-cake-training/1",
        "status": "COMPLETE",
        "device": str(device),
        "corpus": {
            "train_bytes": len(train_bytes),
            "train_sha256": _sha256(train_bytes),
            "count_train_path": args.train,
            "neural_train_path": neural_train_path,
            "neural_sampling_corpus_bytes": len(neural_train_bytes),
            "neural_sampling_corpus_sha256": _sha256(neural_train_bytes),
            "eval_bytes": len(eval_bytes),
            "eval_sha256": _sha256(eval_bytes),
        },
        "parameters": {
            "target": args.target_parameters,
            "logical_total": model.logical_total_parameters,
            "neural": model.neural_parameters,
            "count_state_budget": state_budget,
            "count_state_entries": cake.state_entries,
            "count_order_entries": list(cake.order_entries),
            "gradient_parameters": gradient_parameters,
            "gradient_neural_fraction": gradient_parameters
            / model.neural_parameters,
        },
        "training": {
            "steps": args.steps,
            "requested_steps": requested_steps,
            "batch_size": args.batch_size,
            "final_batch_size": (
                optimization_batch_sizes[-1] if optimization_batch_sizes else 0
            ),
            "seq_len": args.seq_len,
            "predicted_bytes_per_sequence": args.seq_len - args.prediction_start,
            "neural_training_bytes": sum(optimization_batch_sizes)
            * (args.seq_len - args.prediction_start),
            "neural_training_source_bytes": sum(optimization_batch_sizes)
            * args.seq_len,
            "count_training_source_bytes": len(train_bytes),
            "total_training_source_bytes": len(train_bytes)
            + sum(optimization_batch_sizes) * args.seq_len,
            "training_source_byte_budget": args.training_source_byte_budget,
            "training_source_byte_budget_delta": (
                0
                if args.training_source_byte_budget <= 0
                else args.training_source_byte_budget
                - len(train_bytes)
                - sum(optimization_batch_sizes) * args.seq_len
            ),
            "count_corpus_passes": 1.0,
            "final_objective": final_loss,
            "neural_only_steps": max(
                0, args.steps - min(args.steps, args.mixture_calibration_steps)
            ) if args.mixture_calibration_steps > 0 else 0,
            "mixture_calibration_steps": min(
                args.steps, args.mixture_calibration_steps
            ),
        },
        "quality": {
            "initial": initial_eval,
            "final": final_eval,
        },
        "timing": {
            "load_seconds": loaded_seconds,
            "count_training_seconds": count_seconds,
            "neural_optimization_seconds": optimization_seconds,
            "initial_evaluation_seconds": initial_eval_seconds,
            "final_evaluation_seconds": final_eval_seconds,
            "training_seconds": loaded_seconds + count_seconds + optimization_seconds,
            "end_to_end_seconds": total_seconds,
        },
        "config": vars(args),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "model.npz"
    bundle_manifest = save_count_cake_bundle(
        model,
        bundle_path,
        metadata={
            "corpus": result["corpus"],
            "training": result["training"],
            "config": result["config"],
        },
    )
    bundle_payload = bundle_path.read_bytes()
    result["artifact"] = {
        "path": str(bundle_path),
        "format": bundle_manifest["format"],
        "bytes": len(bundle_payload),
        "sha256": _sha256(bundle_payload),
    }
    (out_dir / "training_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
