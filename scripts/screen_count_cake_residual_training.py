"""Train a normalized byte-wise residual head as an architectural screen."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402


def _hidden(model, rows: torch.Tensor) -> torch.Tensor:
    if model.chunking_mode == "delimiter":
        _, hidden = model._dynamic_neural_log_probs(rows)
        return hidden.reshape(rows.shape[0], -1, hidden.shape[-1])
    context = model._patch_context(rows)
    targets = rows[:, model.prediction_start :].reshape(
        rows.shape[0], -1, model.patch_size
    )
    _, hidden = model._neural_log_probs(context, targets, rows=rows)
    return hidden.reshape(rows.shape[0], -1, hidden.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--finetune-host", action="store_true")
    parser.add_argument("--seed", type=int, default=24437)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.finetune_host:
        for name, parameter in model.named_parameters():
            if not name.startswith("count_cake."):
                parameter.requires_grad_(True)
        model.train()
    residual = nn.Linear(model.mixture_gate.in_features, 256).to(device)
    with torch.no_grad():
        residual.weight.zero_()
        residual.bias.zero_()
    optimizer_kwargs = {"lr": args.lr, "betas": (0.9, 0.95)}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimized_parameters = list(residual.parameters()) + [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(optimized_parameters, **optimizer_kwargs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    train_bytes = Path(args.train).read_bytes()
    train = torch.frombuffer(bytearray(train_bytes), dtype=torch.uint8)
    offsets = torch.arange(args.seq_len)
    generator = torch.Generator().manual_seed(args.seed)
    started = time.perf_counter()
    final_loss = float("nan")
    for _ in range(args.steps):
        starts = torch.randint(
            train.numel() - args.seq_len + 1,
            (args.batch_size,),
            generator=generator,
        )
        rows = train[starts[:, None] + offsets].to(device, torch.long)
        if args.finetune_host:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                hidden = _hidden(model, rows)
        else:
            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                hidden = _hidden(model, rows)
        with torch.no_grad():
            count = model.count_cake.all_probabilities(
                rows, start=model.prediction_start
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = count.clamp_min(1e-30).log() + residual(hidden)
            targets = rows[:, model.prediction_start :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256), targets.reshape(-1)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        final_loss = float(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started

    eval_bytes = Path(args.eval).read_bytes()
    evaluation = torch.frombuffer(bytearray(eval_bytes), dtype=torch.uint8).to(
        torch.long
    )
    row_count = evaluation.numel() // args.seq_len
    evaluation = evaluation[: row_count * args.seq_len].reshape(
        row_count, args.seq_len
    )
    total_nll = 0.0
    total_bytes = 0
    eval_started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, row_count, args.eval_batch_size):
            rows = evaluation[offset : offset + args.eval_batch_size].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                hidden = _hidden(model, rows)
                count = model.count_cake.all_probabilities(
                    rows, start=model.prediction_start
                )
                log_probability = torch.log_softmax(
                    count.clamp_min(1e-30).log() + residual(hidden), dim=-1
                )
            targets = rows[:, model.prediction_start :]
            total_nll -= float(
                log_probability.gather(-1, targets.unsqueeze(-1)).sum()
            )
            total_bytes += targets.numel()
    evaluation_seconds = time.perf_counter() - eval_started
    nll = total_nll / total_bytes
    rendered = json.dumps(
            {
                "format": "layercake-residual-training-screen/1",
                "status": "COMPLETE",
                "warning": "architecture-selection screen; not release evidence",
                "source_bundle": args.bundle,
                "source_logical_parameters": manifest["parameters"][
                    "logical_total"
                ],
                "residual_parameters": sum(
                    parameter.numel() for parameter in residual.parameters()
                ),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "source_bytes": args.steps * args.batch_size * args.seq_len,
                "finetune_host": args.finetune_host,
                "final_training_loss": final_loss,
                "training_seconds": training_seconds,
                "evaluated_bytes": total_bytes,
                "evaluation_seconds": evaluation_seconds,
                "nll": nll,
                "bpb": nll / math.log(2.0),
            },
            indent=2,
        ) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
