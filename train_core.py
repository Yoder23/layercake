#!/usr/bin/env python3
"""
train_core.py — Train a LayerCake core language model from scratch.

Usage:
    python train_core.py \
        --config   configs/48M.json \
        --train_data data/tokens/c4_train.npy \
        --eval_data  data/tokens/c4_val.npy \
        --steps    20000 \
        --batch    32 \
        --lr       3e-4 \
        --seed     42 \
        --out_dir  runs/48M_core

After training, runs/<name>/best.pt contains the checkpoint.
Use paste_domain.py to add domain modules to the trained core.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import LayerCakeLMFixedABI
from data import load_tokens, LM1DDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_causal_lm_batch(tokens: torch.Tensor, seq_len: int, batch_size: int,
                          rng: torch.Generator, device: torch.device):
    n = len(tokens) - seq_len - 1
    idxs = torch.randint(0, n, (batch_size,), generator=rng)
    batch = torch.stack([
        torch.from_numpy(tokens[i : i + seq_len + 1].numpy().copy()).long()
        for i in idxs
    ])
    x = batch[:, :-1].to(device)
    y = batch[:, 1:].to(device)
    return x, y


@torch.no_grad()
def evaluate_ppl(model: nn.Module, tokens: torch.Tensor, seq_len: int,
                  batch_size: int, n_eval_tokens: int, device: torch.device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(12345)
    total_loss = 0.0
    total_count = 0
    evaluated = 0
    while evaluated < n_eval_tokens:
        x, y = make_causal_lm_batch(tokens, seq_len, batch_size, rng, device)
        logits, _ = model(x, domain_mask=None)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        n = y.numel()
        total_loss += loss.item() * n
        total_count += n
        evaluated += n
    model.train()
    return math.exp(total_loss / total_count)


def build_model(cfg: dict, vocab_size: int, seq_len: int,
                device: torch.device) -> LayerCakeLMFixedABI:
    core = cfg["core"]
    return LayerCakeLMFixedABI(
        vocab_size=vocab_size,
        d_model=int(core["d_model"]),
        d_abi=int(core.get("d_abi", 512)),
        n_core_layers=int(core["n_layers"]),
        n_heads=int(core["n_heads"]),
        d_ff=int(core["d_ff"]),
        domain_names=[],        # core-only training — no domains
        max_seq_len=seq_len,
        use_router=False,
    ).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train LayerCake core LM")
    parser.add_argument("--config",     required=True, help="ABI config JSON (e.g. configs/48M.json)")
    parser.add_argument("--train_data", required=True, help="Pre-tokenized training tokens (.npy)")
    parser.add_argument("--eval_data",  default=None,  help="Pre-tokenized eval tokens (.npy)")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--seq_len",    type=int, default=256)
    parser.add_argument("--batch",      type=int, default=32)
    parser.add_argument("--steps",      type=int, default=20000)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--wd",         type=float, default=0.01)
    parser.add_argument("--warmup",     type=int, default=500)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--n_eval_tokens", type=int, default=200_000)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print(f"[train_core] Config: {args.config}")
    print(f"[train_core] Steps: {args.steps}, Batch: {args.batch}, LR: {args.lr}")
    print(f"[train_core] Device: {device}, Seed: {args.seed}")

    # Load data
    train_tokens = load_tokens(args.train_data)
    eval_tokens  = load_tokens(args.eval_data) if args.eval_data else train_tokens[:100_000]
    print(f"[train_core] Train tokens: {len(train_tokens):,}")
    print(f"[train_core] Eval tokens:  {len(eval_tokens):,}")

    # Build model
    model = build_model(cfg, args.vocab_size, args.seq_len, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_core] Model params: {n_params / 1e6:.2f}M")

    # Optimizer + scheduler (cosine with warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    def get_lr(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    # Training loop
    model.train()
    best_eval_ppl = float("inf")
    best_ckpt_path = out_dir / "best.pt"
    log_loss = 0.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # Update LR
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step - 1)

        x, y = make_causal_lm_batch(train_tokens, args.seq_len, args.batch, rng, device)
        logits, _ = model(x, domain_mask=None)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        log_loss += loss.item()

        # Logging
        if step % 100 == 0:
            avg_loss = log_loss / 100
            ppl = math.exp(avg_loss)
            elapsed = time.time() - t0
            print(f"  step {step:6d}/{args.steps}  loss={avg_loss:.4f}  ppl={ppl:.2f}"
                  f"  lr={get_lr(step - 1):.2e}  {elapsed:.0f}s")
            log_loss = 0.0

        # Eval
        if step % args.eval_every == 0:
            eval_ppl = evaluate_ppl(model, eval_tokens, args.seq_len,
                                    args.batch, args.n_eval_tokens, device)
            print(f"  [EVAL] step {step}  eval_ppl={eval_ppl:.4f}")
            if eval_ppl < best_eval_ppl:
                best_eval_ppl = eval_ppl
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "eval_ppl": eval_ppl,
                    "config": cfg,
                    "vocab_size": args.vocab_size,
                    "seq_len": args.seq_len,
                    "d_model": cfg["core"]["d_model"],
                    "d_abi": cfg["core"].get("d_abi", 512),
                }, best_ckpt_path)
                print(f"  [SAVE] New best: {best_eval_ppl:.4f} → {best_ckpt_path}")

        # Periodic checkpoint
        if step % args.save_every == 0:
            ckpt_path = out_dir / f"step_{step}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "config": cfg,
                "vocab_size": args.vocab_size,
                "seq_len": args.seq_len,
            }, ckpt_path)

    print(f"\n[train_core] Done. Best eval PPL: {best_eval_ppl:.4f}")
    print(f"[train_core] Checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
