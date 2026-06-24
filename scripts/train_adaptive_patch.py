from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from _common import emit
from layercake.causal_byte_models import CausalAdaptiveBytePatchLM
from run_paired_byte_experiment import (
    batch,
    load_jsonl_bytes,
    load_python_bytes,
)


@torch.no_grad()
def evaluate(model, stream, seq, batch_size, batches, device):
    model.eval()
    generator = torch.Generator().manual_seed(991)
    losses, ratios = [], []
    for _ in range(batches):
        x, y = batch(stream, seq, batch_size, generator, device)
        logits, _, metadata = model(x)
        losses.append(
            F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
        )
        counts = metadata["valid_patches"].sum(dim=1).float()
        ratios.append((seq / counts).mean().item())
    loss = sum(losses) / len(losses)
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "bpb": loss / math.log(2),
        "mean_bytes_per_patch": sum(ratios) / len(ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--general-bytes", type=int, default=20_000_000)
    parser.add_argument("--domain-bytes", type=int, default=2_000_000)
    parser.add_argument("--d-byte", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=376)
    parser.add_argument("--d-abi", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--local-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--local-window", type=int, default=16)
    parser.add_argument("--difficulty-fraction", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    general = load_jsonl_bytes(
        root.parent
        / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        args.general_bytes,
    )
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder", args.domain_bytes
    )
    general_train, general_eval = general[:-200_000], general[-200_000:]
    domain_eval = domain[-100_000:]
    boundary_table = torch.zeros(65536, dtype=torch.bool)
    if args.difficulty_fraction:
        transitions = general_train[:-1] * 256 + general_train[1:]
        counts = torch.bincount(transitions, minlength=65536).double()
        matrix = counts.reshape(256, 256)
        probabilities = (matrix + 0.1) / (
            matrix.sum(dim=1, keepdim=True) + 25.6
        )
        order = torch.argsort(
            -probabilities.log().flatten(), descending=True
        )
        cumulative = counts[order].cumsum(0)
        selected = order[
            cumulative <= counts.sum() * args.difficulty_fraction
        ]
        boundary_table[selected] = True
    model = CausalAdaptiveBytePatchLM(
        d_byte=args.d_byte,
        d_model=args.d_model,
        d_abi=args.d_abi,
        layers=args.layers,
        local_layers=args.local_layers,
        heads=args.heads,
        max_patches=args.seq // 2,
        local_window=args.local_window,
        transition_boundary_table=boundary_table,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        x, y = batch(
            general_train, args.seq, args.batch, generator, device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, _, metadata = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 500 == 0:
            ratio = (
                args.seq
                / metadata["valid_patches"].sum(dim=1).float()
            ).mean()
            item = {
                "step": step,
                "lm_loss": loss.item(),
                "mean_bytes_per_patch": ratio.item(),
            }
            history.append(item)
            print(item, flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "layercake-adaptive-patch/1",
            "args": vars(args),
            "model": model.state_dict(),
        },
        artifact_path,
    )
    emit(
        {
            "status": "TRAINED",
            "device": str(device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "steps": args.steps,
            "elapsed_seconds": elapsed,
            "estimated_total_training_bytes": (
                args.steps * args.batch * args.seq
            ),
            "history": history,
            "general": evaluate(
                model, general_eval, args.seq, args.batch, 30, device
            ),
            "python_domain": evaluate(
                model, domain_eval, args.seq, args.batch, 30, device
            ),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
