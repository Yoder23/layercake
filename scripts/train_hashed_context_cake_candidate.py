"""Train a byte-native sparse multi-order context LayerCake.

This architecture has no tokenizer and no recurrent backpropagation.  Each
prediction combines learned embeddings of deterministic byte-context hashes at
multiple causal orders with an empirical/trainable byte-transition layer.
Sparse optimizer updates touch only context rows observed in the batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class HashedContextCake(nn.Module):
    def __init__(
        self,
        *,
        target_parameters: int,
        width: int,
        hidden: int,
        orders: tuple[int, ...],
        transition_log_probability: torch.Tensor,
    ) -> None:
        super().__init__()
        self.orders = orders
        self.prediction_start = max(orders)
        self.width = int(width)
        # Dense modules are created first so every remaining parameter can be
        # assigned to learned context rows under an exact hard budget.
        self.order_scale = nn.Parameter(torch.ones(len(orders)))
        self.norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_gate = nn.Linear(width, hidden, bias=False)
        self.ffn_value = nn.Linear(width, hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, width, bias=False)
        self.head = nn.Linear(width, 256)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)
        self.transition = nn.Embedding(256, 256, sparse=True)
        with torch.no_grad():
            self.transition.weight.copy_(transition_log_probability)

        dense_and_transition = sum(p.numel() for p in self.parameters())
        context_capacity = target_parameters - dense_and_transition
        if context_capacity < width * len(orders):
            raise ValueError("parameter target is too small for context tables")
        total_rows, remainder = divmod(context_capacity, width)
        # Give short contexts enough rows to be nearly collision-free and
        # distribute the rest toward long contexts.  The final table absorbs
        # the exact residual row capacity.
        desired = {
            1: 256,
            2: 32768,
            3: 65536,
            4: 65536,
        }
        sizes = []
        remaining_rows = total_rows
        remaining_tables = len(orders)
        for order in orders[:-1]:
            proposed = desired.get(order, 32768)
            size = min(proposed, remaining_rows - (remaining_tables - 1))
            sizes.append(size)
            remaining_rows -= size
            remaining_tables -= 1
        sizes.append(remaining_rows)
        self.context_sizes = tuple(int(value) for value in sizes)
        self.context_embeddings = nn.ModuleList(
            nn.Embedding(size, width, sparse=True) for size in self.context_sizes
        )
        for embedding in self.context_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        # At most width-1 values remain after assigning complete embedding rows.
        self.exact_budget_tail = nn.Parameter(torch.zeros(remainder))
        self.logical_parameters = sum(p.numel() for p in self.parameters())
        if self.logical_parameters != target_parameters:
            raise AssertionError(
                f"exact parameter budget failed: {self.logical_parameters} != {target_parameters}"
            )

    def _context_indices(self, rows: torch.Tensor) -> list[torch.Tensor]:
        start = self.prediction_start
        length = rows.shape[1] - start
        rolling = torch.zeros(
            rows.shape[0], length, device=rows.device, dtype=torch.int64
        )
        result = []
        order_index = 0
        for lag in range(start):
            previous = rows[:, start - 1 - lag : rows.shape[1] - 1 - lag]
            rolling = (rolling * 257 + previous.to(torch.int64) + 1).bitwise_and(
                0x7FFFFFFF
            )
            order = lag + 1
            if order == self.orders[order_index]:
                result.append(rolling.remainder(self.context_sizes[order_index]))
                order_index += 1
                if order_index == len(self.orders):
                    break
        return result

    def logits(self, rows: torch.Tensor) -> torch.Tensor:
        rows = rows.to(torch.int64)
        indices = self._context_indices(rows)
        hidden = None
        for index, (embedding, context_index) in enumerate(
            zip(self.context_embeddings, indices)
        ):
            contribution = embedding(context_index) * self.order_scale[index]
            hidden = contribution if hidden is None else hidden + contribution
        assert hidden is not None
        hidden = hidden / math.sqrt(len(indices))
        hidden = self.norm(hidden)
        normalized = self.ffn_norm(hidden)
        hidden = hidden + self.ffn_out(
            F.silu(self.ffn_gate(normalized)) * self.ffn_value(normalized)
        )
        previous = rows[:, self.prediction_start - 1 : -1]
        return self.head(hidden) + self.transition(previous)

    def loss(self, rows: torch.Tensor) -> torch.Tensor:
        targets = rows[:, self.prediction_start :]
        return F.cross_entropy(
            self.logits(rows).reshape(-1, 256), targets.reshape(-1)
        )


def _empirical_transition(payload: bytes, chunk_bytes: int) -> torch.Tensor:
    counts = np.full((256, 256), 0.5, dtype=np.float64)
    view = np.frombuffer(payload, dtype=np.uint8)
    for start in range(0, max(0, view.size - 1), chunk_bytes):
        stop = min(view.size - 1, start + chunk_bytes)
        joint = view[start:stop].astype(np.int64) * 256 + view[start + 1 : stop + 1]
        counts += np.bincount(joint, minlength=65536).reshape(256, 256)
    probability = counts / counts.sum(axis=1, keepdims=True)
    return torch.from_numpy(np.log(probability).astype(np.float32))


@torch.inference_mode()
def _evaluate(
    model: HashedContextCake,
    payload: bytes,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    row_count = len(payload) // seq_len
    rows = torch.from_numpy(
        np.frombuffer(payload[: row_count * seq_len], dtype=np.uint8)
        .reshape(row_count, seq_len)
        .copy()
    )
    model.eval()
    total_nll = 0.0
    scored = 0
    for start in range(0, row_count, batch_size):
        batch = rows[start : start + batch_size].to(device=device, dtype=torch.long)
        loss = model.loss(batch)
        count = batch.shape[0] * (seq_len - model.prediction_start)
        total_nll += float(loss) * count
        scored += count
    nll = total_nll / scored
    return {"bytes": scored, "nll": nll, "bpb": nll / math.log(2.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--target-parameters", type=int, default=24_935_904)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--orders", default="1,2,3,4,6,8,12,16,24,32")
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--training-source-byte-budget", type=int, required=True)
    parser.add_argument("--sparse-lr", type=float, default=0.03)
    parser.add_argument("--dense-lr", type=float, default=0.003)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=24050)
    parser.add_argument("--count-chunk-bytes", type=int, default=24_000_000)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    total_started = time.perf_counter()
    train_payload = Path(args.train).read_bytes()
    eval_payload = Path(args.eval).read_bytes()
    load_seconds = time.perf_counter() - total_started
    count_started = time.perf_counter()
    transition = _empirical_transition(train_payload, args.count_chunk_bytes)
    count_seconds = time.perf_counter() - count_started
    orders = tuple(int(value) for value in args.orders.split(","))
    if tuple(sorted(set(orders))) != orders:
        raise ValueError("orders must be unique and increasing")
    model = HashedContextCake(
        target_parameters=args.target_parameters,
        width=args.width,
        hidden=args.hidden,
        orders=orders,
        transition_log_probability=transition,
    ).to(device)
    initial_eval_started = time.perf_counter()
    initial_eval = _evaluate(
        model,
        eval_payload,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    initial_eval_seconds = time.perf_counter() - initial_eval_started

    sparse_parameters = [
        model.transition.weight,
        *(embedding.weight for embedding in model.context_embeddings),
    ]
    sparse_ids = {id(parameter) for parameter in sparse_parameters}
    dense_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in sparse_ids and parameter.requires_grad
    ]
    sparse_optimizer = torch.optim.SparseAdam(
        sparse_parameters, lr=args.sparse_lr
    )
    dense_optimizer = torch.optim.AdamW(
        dense_parameters,
        lr=args.dense_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    source_budget = args.training_source_byte_budget - len(train_payload)
    sequences = source_budget // args.seq_len
    full_steps, final_batch = divmod(sequences, args.batch_size)
    batch_sizes = [args.batch_size] * full_steps
    if final_batch:
        batch_sizes.append(final_batch)
    if len(batch_sizes) > args.steps:
        batch_sizes = batch_sizes[: args.steps]
    train_cpu = torch.frombuffer(bytearray(train_payload), dtype=torch.uint8)
    offsets = torch.arange(args.seq_len, dtype=torch.long)
    max_start = train_cpu.numel() - args.seq_len
    generator = torch.Generator().manual_seed(args.seed + 17)
    optimization_started = time.perf_counter()
    history = []
    model.train()
    final_loss = float("nan")
    for step, step_batch_size in enumerate(batch_sizes, start=1):
        if step <= args.warmup_steps:
            scale = step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(
                len(batch_sizes) - args.warmup_steps, 1
            )
            scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * min(progress, 1.0))
            )
        sparse_optimizer.param_groups[0]["lr"] = args.sparse_lr * scale
        dense_optimizer.param_groups[0]["lr"] = args.dense_lr * scale
        starts = torch.randint(
            max_start + 1, (step_batch_size,), generator=generator
        )
        rows = train_cpu[starts[:, None] + offsets].to(
            device=device, dtype=torch.long
        )
        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = model.loss(rows)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense_parameters, 1.0)
        sparse_optimizer.step()
        dense_optimizer.step()
        final_loss = float(loss.detach())
        if step == 1 or step % 50 == 0:
            item = {
                "step": step,
                "steps": len(batch_sizes),
                "loss": final_loss,
                "bpb": final_loss / math.log(2.0),
                "elapsed_seconds": time.perf_counter() - optimization_started,
            }
            history.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    optimization_seconds = time.perf_counter() - optimization_started
    final_eval_started = time.perf_counter()
    final_eval = _evaluate(
        model,
        eval_payload,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    final_eval_seconds = time.perf_counter() - final_eval_started
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
    }
    arrays["manifest_json"] = np.frombuffer(
        json.dumps(
            {
                "format": "layercake-hashed-context-cake/1",
                "target_parameters": args.target_parameters,
                "width": args.width,
                "hidden": args.hidden,
                "orders": orders,
                "context_sizes": model.context_sizes,
                "prediction_start": model.prediction_start,
            },
            sort_keys=True,
        ).encode("utf-8"),
        dtype=np.uint8,
    ).copy()
    artifact_path = out_dir / "model.npz"
    np.savez_compressed(artifact_path, **arrays)
    total_seconds = time.perf_counter() - total_started
    neural_source_bytes = sum(batch_sizes) * args.seq_len
    report = {
        "format": "layercake-hashed-context-training/1",
        "status": "COMPLETE",
        "device": str(device),
        "architecture": {
            "type": "sparse_multi_order_hashed_byte_context",
            "tokenizer": None,
            "orders": orders,
            "context_sizes": model.context_sizes,
            "width": args.width,
            "hidden": args.hidden,
        },
        "parameters": {
            "target": args.target_parameters,
            "logical_total": model.logical_parameters,
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "corpus": {
            "train_path": args.train,
            "train_bytes": len(train_payload),
            "train_sha256": _sha256(train_payload),
            "eval_path": args.eval,
            "eval_bytes": len(eval_payload),
            "eval_sha256": _sha256(eval_payload),
        },
        "training": {
            "steps": len(batch_sizes),
            "batch_size": args.batch_size,
            "final_batch_size": batch_sizes[-1],
            "seq_len": args.seq_len,
            "neural_training_source_bytes": neural_source_bytes,
            "count_training_source_bytes": len(train_payload),
            "total_training_source_bytes": len(train_payload) + neural_source_bytes,
            "training_source_byte_budget": args.training_source_byte_budget,
            "final_loss": final_loss,
            "history": history,
        },
        "quality": {"initial": initial_eval, "final": final_eval},
        "timing": {
            "load_seconds": load_seconds,
            "transition_training_seconds": count_seconds,
            "neural_optimization_seconds": optimization_seconds,
            "initial_evaluation_seconds": initial_eval_seconds,
            "final_evaluation_seconds": final_eval_seconds,
            "training_seconds": load_seconds + count_seconds + optimization_seconds,
            "end_to_end_seconds": total_seconds,
        },
        "artifact": {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path.read_bytes()),
        },
        "config": vars(args),
    }
    (out_dir / "training_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
