from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile
import time

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

from _common import emit
from layercake.causal_byte_models import causal_mask
from run_paired_byte_experiment import batch, load_jsonl_bytes, load_python_bytes


class BPETokenLM(nn.Module):
    def __init__(self, vocab_size: int, d_model=96, layers=2, heads=4, max_len=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, batch_first=True, norm_first=True
        )
        self.core = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.emb(x) + self.pos(positions)[None]
        h = self.core(h, mask=causal_mask(x.shape[1], x.device))
        return self.head(self.norm(h))


@torch.no_grad()
def evaluate(model, tokens, original_bytes, seq, batch_size, batches, device):
    model.eval()
    generator = torch.Generator().manual_seed(991)
    losses = []
    for _ in range(batches):
        x, y = batch(tokens, seq, batch_size, generator, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    nll_per_token = sum(losses) / len(losses)
    bytes_per_token = original_bytes / max(tokens.numel(), 1)
    return {
        "token_ppl": math.exp(nll_per_token),
        "nll_per_token": nll_per_token,
        "bytes_per_token": bytes_per_token,
        "bpb": nll_per_token / bytes_per_token / math.log(2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--general-bytes", type=int, default=8_000_000)
    parser.add_argument("--domain-bytes", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    general = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        args.general_bytes,
    )
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder", args.domain_bytes
    )
    general_train, general_eval = general[:-200_000], general[-200_000:]
    domain_eval = domain[-100_000:]
    work = root / "runs_experiment/bpe_baseline"
    work.mkdir(parents=True, exist_ok=True)
    corpus = work / "corpus.txt"
    corpus.write_text(
        bytes(general_train.tolist()).decode("utf-8", errors="replace"),
        encoding="utf-8",
    )
    prefix = work / f"spm_{args.vocab}_{args.general_bytes}"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=args.vocab,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        byte_fallback=True,
        minloglevel=2,
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")

    def encode(raw):
        text = bytes(raw.tolist()).decode("utf-8", errors="replace")
        return torch.tensor(tokenizer.encode(text, out_type=int), dtype=torch.long)

    train_tokens = encode(general_train)
    eval_tokens = encode(general_eval)
    domain_tokens = encode(domain_eval)
    model = BPETokenLM(
        tokenizer.vocab_size(),
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        max_len=args.seq,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    started = time.time()
    for step in range(1, args.steps + 1):
        x, y = batch(train_tokens, args.seq, args.batch, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 1000 == 0:
            print({"step": step, "loss": loss.item()}, flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started
    result = {
        "model": "BPE-token-transformer",
        "vocab_size": tokenizer.vocab_size(),
        "d_model": args.d_model,
        "layers": args.layers,
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": args.steps,
        "seed": args.seed,
        "train_tokens": train_tokens.numel(),
        "train_bytes": general_train.numel(),
        "estimated_bytes_per_update": (
            args.batch * args.seq * general_train.numel() / train_tokens.numel()
        ),
        "estimated_total_training_bytes": (
            args.steps
            * args.batch
            * args.seq
            * general_train.numel()
            / train_tokens.numel()
        ),
        "elapsed_seconds": elapsed,
        "general": evaluate(
            model, eval_tokens, general_eval.numel(), args.seq, args.batch, 30, device
        ),
        "python_domain": evaluate(
            model, domain_tokens, domain_eval.numel(), args.seq, args.batch, 30, device
        ),
    }
    if args.artifact:
        artifact = Path(args.artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "layercake-bpe-baseline/1",
                "args": vars(args),
                "model": model.state_dict(),
                "vocab_size": tokenizer.vocab_size(),
                "tokenizer_model": Path(str(prefix) + ".model").read_bytes(),
            },
            artifact,
        )
    emit(result, args.output)


if __name__ == "__main__":
    main()
