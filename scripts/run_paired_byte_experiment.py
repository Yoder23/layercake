from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from _common import emit
from layercake.abi import ABISpec
from layercake.causal_byte_models import CausalByteLM, CausalBytePatchLM
from layercake.canonical_anchors import causal_byte_anchors, patch_context_anchors
from layercake.domain_bricks import LowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec


def load_jsonl_bytes(path: Path, limit: int) -> torch.Tensor:
    data = bytearray()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            data.extend(text.encode("utf-8", errors="replace"))
            data.append(10)
            if len(data) >= limit:
                break
    return torch.tensor(list(data[:limit]), dtype=torch.long)


def load_python_bytes(root: Path, limit: int) -> torch.Tensor:
    files = sorted(root.rglob("*.py"))
    data = bytearray()
    for path in files:
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        data.extend(path.read_bytes())
        data.extend(b"\n")
        if len(data) >= limit:
            break
    return torch.tensor(list(data[:limit]), dtype=torch.long)


def batch(stream, seq, size, generator, device):
    starts = torch.randint(0, len(stream) - seq - 1, (size,), generator=generator)
    rows = torch.stack([stream[i : i + seq + 1] for i in starts])
    return rows[:, :-1].to(device), rows[:, 1:].to(device)


@torch.no_grad()
def evaluate(model, stream, seq, batch_size, batches, device, brick=None):
    model.eval()
    generator = torch.Generator().manual_seed(991)
    losses = []
    for _ in range(batches):
        x, y = batch(stream, seq, batch_size, generator, device)
        logits, _ = model(x, brick=brick)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y[:, : logits.shape[1]].flatten()).item())
    loss = sum(losses) / len(losses)
    return {"loss": loss, "ppl": math.exp(loss), "bpb": loss / math.log(2)}


def train_pair(
    byte_model, patch_model, stream, steps, seq, batch_size, device,
    align_weight, anchor_weight
):
    params = list(byte_model.parameters()) + list(patch_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(42)
    history = []
    for step in range(1, steps + 1):
        x, y = batch(stream, seq, batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            byte_logits, byte_abi = byte_model(x)
            patch_logits, patch_abi = patch_model(x)
            byte_boundaries = byte_model.boundary_abi(byte_abi, patch_model.patch_size)
            # Byte state after patch N is the context used by patch N+1.
            align = F.mse_loss(byte_boundaries[:, :-1], patch_abi[:, 1:])
            byte_anchor = causal_byte_anchors(x, byte_abi.shape[-1])
            patch_anchor = patch_context_anchors(
                x, patch_abi.shape[-1], patch_model.patch_size
            )
            anchor = F.mse_loss(byte_abi, byte_anchor) + F.mse_loss(
                patch_abi, patch_anchor
            )
            byte_loss = F.cross_entropy(byte_logits.flatten(0, 1), y.flatten())
            patch_loss = F.cross_entropy(patch_logits.flatten(0, 1), y[:, : patch_logits.shape[1]].flatten())
            loss = byte_loss + patch_loss + align_weight * align + anchor_weight * anchor
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0:
            history.append({
                "step": step, "byte_loss": byte_loss.item(),
                "patch_loss": patch_loss.item(), "align": align.item(),
                "anchor": anchor.item(),
            })
            print(history[-1], flush=True)
    return history


def train_brick(model, brick, domain_stream, general_stream, steps, seq, batch_size, device, preserve_weight=2.0):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train()
    brick.train()
    optimizer = torch.optim.AdamW(brick.parameters(), lr=1e-3)
    domain_generator = torch.Generator().manual_seed(314)
    general_generator = torch.Generator().manual_seed(2718)
    for _ in range(steps):
        x, y = batch(domain_stream, seq, batch_size, domain_generator, device)
        logits, _ = model(x, brick=brick)
        domain_loss = F.cross_entropy(logits.flatten(0, 1), y[:, : logits.shape[1]].flatten())
        gx, _ = batch(general_stream, seq, batch_size, general_generator, device)
        with torch.no_grad():
            base_logits, _ = model(gx)
        adapted_logits, _ = model(gx, brick=brick)
        preserve = F.kl_div(
            F.log_softmax(adapted_logits, dim=-1),
            F.softmax(base_logits, dim=-1),
            reduction="batchmean",
        ) / adapted_logits.shape[1]
        loss = domain_loss + preserve_weight * preserve
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--brick-steps", type=int, default=300)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--general-bytes", type=int, default=8_000_000)
    parser.add_argument("--domain-bytes", type=int, default=2_000_000)
    parser.add_argument("--align-weight", type=float, default=0.1)
    parser.add_argument("--preserve-weight", type=float, default=2.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-byte", type=int, default=48)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--patch-d-model", type=int)
    parser.add_argument("--patch-layers", type=int)
    parser.add_argument("--patch-heads", type=int)
    parser.add_argument("--d-abi", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--continuous-local", action="store_true")
    parser.add_argument("--direct-global-context", action="store_true")
    parser.add_argument("--ngram-buckets", type=int, default=0)
    parser.add_argument("--local-decoder", choices=["gru", "conv"], default="gru")
    parser.add_argument("--conv-layers", type=int, default=4)
    parser.add_argument("--mtp-depth", type=int, default=0)
    parser.add_argument("--output", default="results/paired_byte_experiment.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    general_path = root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl"
    general = load_jsonl_bytes(general_path, args.general_bytes)
    domain = load_python_bytes(root.parent / "layercakeogwithdecoder", args.domain_bytes)
    general_train, general_eval = general[:-200_000], general[-200_000:]
    domain_train, domain_eval = domain[:-100_000], domain[-100_000:]
    byte_model = CausalByteLM(
        d_model=args.d_model, d_abi=args.d_abi, layers=args.layers,
        heads=args.heads, max_len=args.seq
    ).to(device)
    patch_model = CausalBytePatchLM(
        patch_size=args.patch_size,
        d_byte=args.d_byte, d_model=args.patch_d_model or args.d_model,
        d_abi=args.d_abi, layers=args.patch_layers or args.layers,
        heads=args.patch_heads or args.heads,
        max_patches=args.seq // args.patch_size,
        continuous_local=args.continuous_local,
        direct_global_context=args.direct_global_context,
        ngram_buckets=args.ngram_buckets,
        local_decoder=args.local_decoder,
        conv_layers=args.conv_layers,
        mtp_depth=args.mtp_depth,
    ).to(device)
    started = time.time()
    history = train_pair(
        byte_model, patch_model, general_train, args.steps, args.seq, args.batch,
        device, args.align_weight, args.anchor_weight
    )
    base = {
        "byte_general": evaluate(byte_model, general_eval, args.seq, args.batch, 20, device),
        "patch_general": evaluate(patch_model, general_eval, args.seq, args.batch, 20, device),
        "byte_domain": evaluate(byte_model, domain_eval, args.seq, args.batch, 20, device),
        "patch_domain": evaluate(patch_model, domain_eval, args.seq, args.batch, 20, device),
    }
    spec = ABISpec(version="lc-abi/2", d_abi=args.d_abi, input_interface=InputInterfaceSpec(mode="byte_patch", patching=f"fixed:{args.patch_size}", max_patch_size=args.patch_size))
    brick = LowRankDomainOperator(spec, rank=16, alpha_init=0.01).to(device)
    train_brick(
        patch_model, brick, domain_train, general_train, args.brick_steps,
        args.seq, args.batch, device, args.preserve_weight
    )
    transferred = {
        "patch_domain": evaluate(patch_model, domain_eval, args.seq, args.batch, 20, device, brick),
        "patch_general": evaluate(patch_model, general_eval, args.seq, args.batch, 20, device, brick),
        "byte_domain": evaluate(byte_model, domain_eval, args.seq, args.batch, 20, device, brick),
        "byte_general": evaluate(byte_model, general_eval, args.seq, args.batch, 20, device, brick),
    }
    payload = {
        "device": str(device), "seed": args.seed, "steps": args.steps, "brick_steps": args.brick_steps,
        "align_weight": args.align_weight, "anchor_weight": args.anchor_weight,
        "preserve_weight": args.preserve_weight,
        "elapsed_seconds": time.time() - started, "history": history, "base": base,
        "transferred": transferred,
        "cross_interface_domain_ppl_ratio": transferred["byte_domain"]["ppl"] / transferred["patch_domain"]["ppl"],
        "byte_general_regression_ratio": transferred["byte_general"]["ppl"] / base["byte_general"]["ppl"],
        "status": "PASS" if (
            transferred["byte_domain"]["ppl"] < base["byte_domain"]["ppl"]
            and transferred["byte_general"]["ppl"] / base["byte_general"]["ppl"] <= 1.05
        ) else "FAIL",
    }
    if args.artifact:
        artifact = Path(args.artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "seed": args.seed,
                "byte_model": byte_model.state_dict(),
                "patch_model": patch_model.state_dict(),
                "brick": brick.state_dict(),
                "args": vars(args),
            },
            artifact,
        )
    emit(payload, args.output)


if __name__ == "__main__":
    main()
