#!/usr/bin/env python3
"""
Train a 250M parameter LayerCake core model on English corpus.
Optimized for fluent general English with game domain augmentation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from layercake.causal_byte_models import CausalBytePatchLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnglishCorpusDataset(IterableDataset):
    """Stream English corpus from JSONL with byte-level tokenization."""

    def __init__(self, jsonl_path: str | Path, seq_len: int = 2048):
        self.jsonl_path = Path(jsonl_path)
        self.seq_len = seq_len
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.jsonl_path}")
        logger.info("Streaming from: %s", self.jsonl_path)

    def __iter__(self):
        with self.jsonl_path.open("r", encoding="utf-8", errors="replace") as handle:
            buffer = b""
            for line in handle:
                try:
                    doc = json.loads(line)
                    if not isinstance(doc, dict):
                        continue
                    text = doc.get("text", "") or doc.get("content", "")
                    if not isinstance(text, str):
                        continue
                    text_bytes = text.encode("utf-8", errors="replace")
                    buffer += text_bytes

                    while len(buffer) >= self.seq_len:
                        chunk = buffer[: self.seq_len]
                        buffer = buffer[self.seq_len :]
                        yield torch.tensor(list(chunk), dtype=torch.long)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue


def build_250m_model(device: torch.device) -> CausalBytePatchLM:
    """Build 250M parameter LayerCake model."""
    model = CausalBytePatchLM(
        patch_size=2,
        d_byte=64,
        d_model=768,
        d_abi=256,
        layers=33,
        heads=16,
        max_patches=1024,  # 2048 bytes / 2 bytes per patch
        continuous_local=False,
        direct_global_context=True,
        local_decoder="window_transformer",
        local_layers=4,
        local_width=512,
        modern_blocks=True,
        fused_attention=True,
        local_window=64,
        patch_unit_buckets=0,
        dropout=0.1,
        qk_norm=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model initialized: %.1fM parameters", param_count / 1e6)
    return model


def train_step(
    model: CausalBytePatchLM,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    """Single training step."""
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        x, y = batch[:, :-1], batch[:, 1:]
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            y.flatten(),
            reduction="mean",
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


@torch.inference_mode()
def evaluate_bpb(
    model: CausalBytePatchLM,
    dataloader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> float:
    """Evaluate next-byte bits per byte on a bounded held-out stream."""

    model.eval()
    loss_sum = 0.0
    target_count = 0
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= max_batches:
            break
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        logits, _ = model(x)
        loss_sum += float(
            F.cross_entropy(
                logits.flatten(0, 1),
                y.flatten(),
                reduction="sum",
            ).item()
        )
        target_count += y.numel()
    model.train()
    if target_count == 0:
        raise RuntimeError("evaluation corpus did not produce a complete batch")
    return loss_sum / target_count / math.log(2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train 250M LayerCake core")
    parser.add_argument(
        "--corpus",
        default="../layercakeogwithdecoder/data/v6/redpajama_english_train.jsonl",
    )
    parser.add_argument(
        "--eval-corpus",
        default="../layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
    )
    parser.add_argument("--output", default="runs_experiment/layercake_250m_english_core.pt")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=250)
    args = parser.parse_args()

    if args.steps <= 0 or args.eval_steps <= 0:
        parser.error("--steps and --eval-steps must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build model
    model = build_250m_model(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Setup data
    dataset = EnglishCorpusDataset(
        args.corpus,
        seq_len=args.seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        EnglishCorpusDataset(args.eval_corpus, seq_len=args.seq_len),
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
    )

    # Setup mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Training loop
    model.train()
    losses = []
    start_time = time.perf_counter()
    completed_steps = 0

    try:
        for step, batch in enumerate(dataloader):
            if step >= args.steps:
                break

            loss = train_step(model, batch, optimizer, scaler, device)
            losses.append(loss)
            completed_steps = step + 1

            if (step + 1) % 100 == 0:
                mean_loss = sum(losses[-100:]) / len(losses[-100:])
                elapsed = time.perf_counter() - start_time
                rate = (step + 1) / elapsed
                logger.info(
                    f"Step {step+1}/{args.steps} | Loss: {mean_loss:.4f} | "
                    f"Rate: {rate:.1f} steps/sec | Elapsed: {elapsed/3600:.1f}h"
                )

            if device.type == "cuda" and (step + 1) % 500 == 0:
                torch.cuda.synchronize()
    except KeyboardInterrupt:
        logger.info("Training interrupted")

    if not losses:
        raise RuntimeError("training corpus did not produce a complete batch")
    eval_bpb = evaluate_bpb(
        model,
        eval_dataloader,
        device=device,
        max_batches=args.eval_steps,
    )

    # Save checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "args": vars(args),
            "model": model.state_dict(),
            "model_config": {
                "patch_size": 2,
                "d_byte": 64,
                "d_model": 768,
                "d_abi": 256,
                "layers": 33,
                "heads": 16,
                "local_layers": 4,
                "local_width": 512,
                "local_window": 64,
            },
            "step": completed_steps,
            "final_loss": losses[-1],
            "mean_loss": sum(losses) / len(losses),
            "eval_bpb": eval_bpb,
        },
        output_path,
    )

    logger.info("Model saved to: %s", output_path)
    logger.info(
        "Training complete: %d steps in %.1fh",
        completed_steps,
        (time.perf_counter() - start_time) / 3600,
    )
    logger.info("Final loss: %.4f | eval BPB: %.4f", losses[-1], eval_bpb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
