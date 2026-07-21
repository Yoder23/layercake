"""Train a causal normalized context mixer over complete CountCake stages.

This architecture-selection probe learns only from the training corpus.  For
each byte context it predicts positive exponents over the unigram and every
backoff-order distribution, combines all 256 candidates, and renormalizes
exactly.  Evaluation rows are never used for fitting.
"""

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


class ContextualOrderProduct(nn.Module):
    def __init__(
        self,
        *,
        stages: int,
        max_order: int,
        embedding_width: int,
        hidden_width: int,
        neural_width: int = 0,
        base_index: int | None = None,
    ) -> None:
        super().__init__()
        self.stages = int(stages)
        self.max_order = int(max_order)
        self.neural_width = int(neural_width)
        self.base_index = self.stages - 1 if base_index is None else int(base_index)
        self.byte_embedding = nn.Embedding(256, embedding_width)
        self.candidate_embedding = nn.Embedding(256, embedding_width)
        feature_width = (
            stages * 3
            + stages * embedding_width
            + max_order * embedding_width
            + self.neural_width
        )
        self.network = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, stages),
        )
        initial = torch.full((stages,), 0.01)
        initial[self.base_index] = 0.5
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.copy_(torch.log(torch.expm1(initial)))

    def exponents(
        self,
        stage_probability: torch.Tensor,
        byte_context: torch.Tensor,
        neural_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probability = stage_probability.float().clamp_min(1e-30)
        log_probability = probability.log()
        entropy = -(probability * log_probability).sum(dim=-1) / math.log(256.0)
        top_probability, top_candidate = probability.max(dim=-1)
        top_two = probability.topk(2, dim=-1).values
        margin = top_two[..., 0] - top_two[..., 1]
        candidate = self.candidate_embedding(top_candidate).flatten(-2)
        context = self.byte_embedding(byte_context).flatten(-2)
        features = [entropy, top_probability, margin, candidate, context]
        if self.neural_width:
            if neural_hidden is None or neural_hidden.shape[-1] != self.neural_width:
                raise ValueError("neural_hidden does not match configured neural_width")
            features.append(neural_hidden)
        features = torch.cat(features, dim=-1)
        return F.softplus(self.network(features))

    def logits(
        self,
        stage_probability: torch.Tensor,
        byte_context: torch.Tensor,
        neural_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        exponents = self.exponents(
            stage_probability, byte_context, neural_hidden
        )
        return torch.einsum(
            "btsv,bts->btv",
            stage_probability.float().clamp_min(1e-30).log(),
            exponents,
        )


def _byte_context(rows: torch.Tensor, start: int, order: int) -> torch.Tensor:
    return torch.stack(
        [
            rows[:, start - order + offset : rows.shape[1] - order + offset]
            for offset in range(order)
        ],
        dim=-1,
    )


def _fusion_inputs(model, rows, *, start: int, include_neural: bool):
    stages = _count_distribution(
        model.count_cake, rows, start, return_stages=True
    )
    byte_context = _byte_context(rows, start, model.count_cake.max_order)
    if not include_neural:
        return stages, byte_context, None
    if model.chunking_mode != "fixed":
        raise ValueError("neural fusion currently requires fixed CountCake chunks")
    patch_context = model._patch_context(rows)
    targets = rows[:, start:].reshape(rows.shape[0], -1, model.patch_size)
    _, neural_hidden = model._neural_log_probs(patch_context, targets)
    neural_hidden = neural_hidden.reshape(rows.shape[0], -1, neural_hidden.shape[-1])
    neural = model._neural_probabilities(neural_hidden)
    stages = torch.cat([stages, neural.unsqueeze(-2)], dim=-2)
    return stages, byte_context, neural_hidden


@torch.inference_mode()
def _evaluate(
    head: ContextualOrderProduct,
    model,
    rows: torch.Tensor,
    *,
    start: int,
    batch_size: int,
    include_neural: bool,
) -> dict:
    total_nll = 0.0
    final_count_nll = 0.0
    neural_nll = 0.0
    oracle_nll = 0.0
    scored = 0
    exponent_sum = torch.zeros(head.stages, device=rows.device)
    for offset in range(0, rows.shape[0], batch_size):
        batch = rows[offset : offset + batch_size]
        stages, context, neural_hidden = _fusion_inputs(
            model, batch, start=start, include_neural=include_neural
        )
        targets = batch[:, start:]
        logits = head.logits(stages, context, neural_hidden)
        total_nll += float(
            F.cross_entropy(
                logits.reshape(-1, 256), targets.reshape(-1), reduction="sum"
            )
        )
        observed = stages.gather(
            -1,
            targets[..., None, None].expand(-1, -1, head.stages, 1),
        ).squeeze(-1).clamp_min(1e-30)
        final_count_nll += float(
            -observed[..., model.count_cake.max_order].log().sum()
        )
        if include_neural:
            neural_nll += float(-observed[..., -1].log().sum())
        oracle_nll += float(-observed.max(dim=-1).values.log().sum())
        exponent_sum += head.exponents(
            stages, context, neural_hidden
        ).sum(dim=(0, 1))
        scored += targets.numel()
    return {
        "scored_bytes": scored,
        "product_bpb": total_nll / scored / math.log(2.0),
        "final_count_stage_bpb": final_count_nll / scored / math.log(2.0),
        "neural_bpb": (
            neural_nll / scored / math.log(2.0) if include_neural else None
        ),
        "target_aware_oracle_bpb": oracle_nll / scored / math.log(2.0),
        "mean_exponents": (exponent_sum / scored).cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-rows", type=int, default=96)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--embedding-width", type=int, default=8)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--include-neural", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=24071)
    parser.add_argument("--router", required=True)
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
    if max_start < 0:
        raise ValueError("training corpus is shorter than one row")
    head = ContextualOrderProduct(
        stages=cake.max_order + 1 + int(args.include_neural),
        max_order=cake.max_order,
        embedding_width=args.embedding_width,
        hidden_width=args.hidden_width,
        neural_width=(
            model.mixture_gate.in_features if args.include_neural else 0
        ),
        base_index=cake.max_order,
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
            stages, context, neural_hidden = _fusion_inputs(
                model,
                rows,
                start=model.prediction_start,
                include_neural=args.include_neural,
            )
            targets = rows[:, model.prediction_start :]
        optimizer.zero_grad(set_to_none=True)
        logits = head.logits(stages, context, neural_hidden)
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0:
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
    router_path = Path(args.router)
    router_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        router_path,
        **{
            name: tensor.detach().cpu().numpy()
            for name, tensor in head.state_dict().items()
        },
    )
    report = {
        "format": "layercake-contextual-order-product-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection probe; not final evidence",
        "bundle": args.bundle,
        "base_logical_parameters": manifest["parameters"]["logical_total"],
        "router": {
            "path": str(router_path),
            "parameters": sum(p.numel() for p in head.parameters()),
            "embedding_width": args.embedding_width,
            "hidden_width": args.hidden_width,
            "include_neural": args.include_neural,
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
