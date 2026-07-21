"""Probe exact nonlinear candidate-wise fusion of all CountCake stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from probe_count_conditioned_product import _count_distribution  # noqa: E402


class NonlinearOrderFusion(nn.Module):
    def __init__(
        self,
        stages: int,
        hidden: int,
        candidate_width: int,
        *,
        base_index: int | None = None,
        context_input_width: int = 0,
        context_width: int = 0,
    ) -> None:
        super().__init__()
        self.stages = int(stages)
        self.base_index = self.stages - 1 if base_index is None else int(base_index)
        if not 0 <= self.base_index < self.stages:
            raise ValueError("base_index must select one fusion stage")
        self.candidate_embedding = nn.Embedding(256, candidate_width)
        self.candidate_bias = nn.Parameter(torch.zeros(256))
        self.context_width = int(context_width)
        if self.context_width:
            if context_input_width <= 0:
                raise ValueError("context_input_width must be positive")
            self.context_projection = nn.Linear(
                int(context_input_width), self.context_width
            )
        self.raw_base_temperature = nn.Parameter(torch.log(torch.expm1(torch.tensor(0.5))))
        self.network = nn.Sequential(
            nn.Linear(
                stages * 2 - 1 + candidate_width + self.context_width,
                hidden,
            ),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.zero_()

    def forward(
        self,
        stage_probability: torch.Tensor,
        context_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logs = stage_probability.float().clamp_min(1e-30).log()
        candidate = self.candidate_embedding(
            torch.arange(256, device=logs.device)
        )
        candidate = candidate.view(
            *((1,) * (logs.ndim - 2)), 256, candidate.shape[-1]
        ).expand(*logs.shape[:-2], 256, candidate.shape[-1])
        # Candidate axis precedes the stage-feature axis for the shared MLP.
        by_candidate = logs.transpose(-2, -1)
        features = [
            by_candidate / 8.0,
            (by_candidate[..., 1:] - by_candidate[..., :-1]) / 8.0,
            candidate,
        ]
        if self.context_width:
            if context_hidden is None:
                raise ValueError("context_hidden is required by this fusion head")
            context = self.context_projection(context_hidden)
            context = context.unsqueeze(-2).expand(
                *context.shape[:-1], 256, context.shape[-1]
            )
            features.append(context)
        features = torch.cat(features, dim=-1)
        residual = self.network(features).squeeze(-1)
        temperature = F.softplus(self.raw_base_temperature)
        return (
            temperature * logs[..., self.base_index, :]
            + residual
            + self.candidate_bias
        )


def _stage_inputs(model, rows, *, start: int, include_neural: bool):
    count_stages = _count_distribution(
        model.count_cake, rows, start, return_stages=True
    )
    if not include_neural:
        return count_stages, None
    if model.chunking_mode != "fixed":
        raise ValueError("neural fusion currently requires fixed CountCake chunks")
    context = model._patch_context(rows)
    targets = rows[:, start:].reshape(rows.shape[0], -1, model.patch_size)
    _, neural_hidden = model._neural_log_probs(context, targets)
    neural = model._neural_probabilities(neural_hidden).reshape(
        rows.shape[0], -1, 256
    )
    return (
        torch.cat([count_stages, neural.unsqueeze(-2)], dim=-2),
        neural_hidden.reshape(rows.shape[0], -1, neural_hidden.shape[-1]),
    )


@torch.inference_mode()
def _evaluate(
    head,
    model,
    rows,
    *,
    start: int,
    batch_size: int,
    include_neural: bool,
) -> dict:
    total_nll = 0.0
    count_nll = 0.0
    neural_nll = 0.0
    oracle_nll = 0.0
    scored = 0
    for offset in range(0, rows.shape[0], batch_size):
        batch = rows[offset : offset + batch_size]
        stages, neural_hidden = _stage_inputs(
            model, batch, start=start, include_neural=include_neural
        )
        targets = batch[:, start:]
        logits = head(stages, neural_hidden)
        total_nll += float(
            F.cross_entropy(
                logits.reshape(-1, 256), targets.reshape(-1), reduction="sum"
            )
        )
        observed = stages.gather(
            -1,
            targets[..., None, None].expand(-1, -1, head.stages, 1),
        ).squeeze(-1).clamp_min(1e-30)
        count_nll += float(
            -observed[..., model.count_cake.max_order].log().sum()
        )
        if include_neural:
            neural_nll += float(-observed[..., -1].log().sum())
        oracle_nll += float(-observed.max(dim=-1).values.log().sum())
        scored += targets.numel()
    return {
        "scored_bytes": scored,
        "fusion_bpb": total_nll / scored / math.log(2.0),
        "final_count_stage_bpb": count_nll / scored / math.log(2.0),
        "neural_bpb": (
            neural_nll / scored / math.log(2.0) if include_neural else None
        ),
        "target_aware_oracle_bpb": oracle_nll / scored / math.log(2.0),
        "base_temperature": float(F.softplus(head.raw_base_temperature)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-rows", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--candidate-width", type=int, default=8)
    parser.add_argument("--include-neural", action="store_true")
    parser.add_argument("--context-width", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=24081)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device="cuda")
    model.eval()
    cake = model.count_cake
    train_payload = Path(args.train).read_bytes()
    train_cpu = torch.frombuffer(bytearray(train_payload), dtype=torch.uint8)
    offsets = torch.arange(args.seq_len, dtype=torch.long)
    max_start = train_cpu.numel() - args.seq_len
    head = NonlinearOrderFusion(
        cake.max_order + 1 + int(args.include_neural),
        args.hidden,
        args.candidate_width,
        base_index=cake.max_order,
        context_input_width=(
            model.mixture_gate.in_features if args.include_neural else 0
        ),
        context_width=args.context_width,
    ).cuda()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.001
    )
    generator = torch.Generator().manual_seed(args.seed + 17)
    trace = []
    train_started = time.perf_counter()
    for step in range(1, args.steps + 1):
        starts = torch.randint(
            max_start + 1,
            (args.train_batch_size,),
            generator=generator,
        )
        rows = train_cpu[starts[:, None] + offsets].to(
            device="cuda", dtype=torch.long
        )
        with torch.no_grad():
            stages, neural_hidden = _stage_inputs(
                model,
                rows,
                start=model.prediction_start,
                include_neural=args.include_neural,
            )
            targets = rows[:, model.prediction_start :]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            head(stages, neural_hidden).reshape(-1, 256), targets.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            item = {
                "step": step,
                "train_bpb": float(loss.detach()) / math.log(2.0),
                "elapsed_seconds": time.perf_counter() - train_started,
            }
            trace.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - train_started
    reports = []
    for path_string in args.eval:
        payload = Path(path_string).read_bytes()
        available = len(payload) // args.seq_len
        selected = min(args.eval_rows, available)
        indices = np.linspace(0, available - 1, num=selected, dtype=np.int64)
        array = np.frombuffer(payload, dtype=np.uint8)[
            : available * args.seq_len
        ].reshape(available, args.seq_len)
        rows = torch.from_numpy(array[indices].copy()).to(
            device="cuda", dtype=torch.long
        )
        report = _evaluate(
            head,
            model,
            rows,
            start=model.prediction_start,
            batch_size=args.eval_batch_size,
            include_neural=args.include_neural,
        )
        report.update(
            {
                "path": path_string,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "selected_rows": selected,
                "sampling": "evenly_spaced_rows",
            }
        )
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    head_path = Path(args.head)
    head_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        head_path,
        **{
            name: tensor.detach().cpu().numpy()
            for name, tensor in head.state_dict().items()
        },
    )
    report = {
        "format": "layercake-nonlinear-order-fusion-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection probe; not final evidence",
        "bundle": args.bundle,
        "base_logical_parameters": manifest["parameters"]["logical_total"],
        "head": {
            "path": str(head_path),
            "parameters": sum(p.numel() for p in head.parameters()),
            "hidden": args.hidden,
            "candidate_width": args.candidate_width,
            "base_stage": "final_count",
            "include_neural": args.include_neural,
            "context_width": args.context_width,
        },
        "training": {
            "path": args.train,
            "source_bytes_per_step": args.train_batch_size * args.seq_len,
            "total_source_bytes": args.steps
            * args.train_batch_size
            * args.seq_len,
            "steps": args.steps,
            "seconds": training_seconds,
            "trace": trace,
        },
        "evaluation": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
