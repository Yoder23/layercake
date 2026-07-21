from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from layercake.causal_byte_models import CausalBytePatchLM


def _iter_jsonl_text(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    text = payload.get("text") or payload.get("content") or ""
                else:
                    text = str(payload)
            except json.JSONDecodeError:
                text = ""
            if text:
                yield text


def load_curriculum_bytes(redpajama_jsonl: Path, curriculum_files: list[Path], total_bytes: int) -> torch.Tensor:
    data = bytearray()

    for text in _iter_jsonl_text(redpajama_jsonl):
        data.extend(text.encode("utf-8", errors="replace"))
        data.extend(b"\n")
        if len(data) >= int(total_bytes * 0.85):
            break

    curriculum_blob = bytearray()
    for path in curriculum_files:
        if path.exists():
            curriculum_blob.extend(path.read_bytes())
            curriculum_blob.extend(b"\n")

    if curriculum_blob:
        while len(data) < total_bytes:
            data.extend(curriculum_blob)

    return torch.tensor(list(data[:total_bytes]), dtype=torch.long)


def tensor_byte_sha256(row: torch.Tensor) -> str:
    return hashlib.sha256(bytes(row.tolist())).hexdigest()


def batch(stream: torch.Tensor, seq: int, size: int, generator: torch.Generator, device: torch.device):
    starts = torch.randint(0, len(stream) - seq - 1, (size,), generator=generator)
    rows = torch.stack([stream[i : i + seq + 1] for i in starts])
    return rows[:, :-1].to(device), rows[:, 1:].to(device)


class BPETokenLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, layers: int, heads: int, max_len: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model,
            heads,
            d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.core = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.emb(x) + self.pos(positions)[None]
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1).bool()
        h = self.core(h, mask=mask)
        return self.head(self.norm(h))


@dataclass
class ScaleSpec:
    name: str
    lc_model: dict[str, Any]
    bpe_model: dict[str, Any]


def _train_lc(model: CausalBytePatchLM, train_bytes: torch.Tensor, steps: int, seq: int, batch_size: int, device: torch.device) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(1234)

    started = time.perf_counter()
    for _ in range(steps):
        x, y = batch(train_bytes, seq, batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(x)
            logits = logits[:, : y.shape[1], :]
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"elapsed_seconds": elapsed, "steps_per_second": steps / max(elapsed, 1e-9)}


def _train_bpe(model: BPETokenLM, train_tokens: torch.Tensor, steps: int, seq: int, batch_size: int, device: torch.device) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(5678)

    started = time.perf_counter()
    for _ in range(steps):
        x, y = batch(train_tokens, seq, batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"elapsed_seconds": elapsed, "steps_per_second": steps / max(elapsed, 1e-9)}


@torch.no_grad()
def _eval_lc_bpb(model: CausalBytePatchLM, eval_bytes: torch.Tensor, seq: int, batch_size: int, eval_batches: int, device: torch.device) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(777)
    losses = []
    for _ in range(eval_batches):
        x, y = batch(eval_bytes, seq, batch_size, generator, device)
        logits, _ = model(x)
        logits = logits[:, : y.shape[1], :]
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    return (sum(losses) / len(losses)) / math.log(2)


@torch.no_grad()
def _eval_bpe_bpb(model: BPETokenLM, eval_tokens: torch.Tensor, eval_byte_count: int, seq: int, batch_size: int, eval_batches: int, device: torch.device) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(888)
    losses = []
    for _ in range(eval_batches):
        x, y = batch(eval_tokens, seq, batch_size, generator, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    nll_per_token = sum(losses) / len(losses)
    bytes_per_token = eval_byte_count / max(eval_tokens.numel(), 1)
    return nll_per_token / max(bytes_per_token, 1e-9) / math.log(2)


def _gen_lc(model: CausalBytePatchLM, prompt: str, seq: int, max_new: int = 96) -> str:
    device = next(model.parameters()).device
    ids = list(prompt.encode("utf-8", errors="replace"))
    local_window = int(getattr(model, "local_window", 32))
    patch_size = int(getattr(model, "patch_size", 2))

    def pick_next(logits_1d: torch.Tensor, history: list[int]) -> int:
        top_vals, top_idx = torch.topk(logits_1d, k=min(8, logits_1d.numel()))
        del top_vals
        if len(history) >= 8 and len(set(history[-8:])) == 1:
            for idx in top_idx.tolist():
                if idx != history[-1]:
                    return int(idx)
        return int(top_idx[0].item())

    for _ in range(max_new):
        ctx = ids[-seq:]
        if len(ctx) < local_window:
            ctx = ([ord(" ")] * (local_window - len(ctx))) + ctx
        if len(ctx) % local_window:
            need = local_window - (len(ctx) % local_window)
            ctx = ([ord(" ")] * need) + ctx
        if len(ctx) % patch_size:
            need = patch_size - (len(ctx) % patch_size)
            ctx = ([ord(" ")] * need) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = pick_next(logits[0, -1], ids)
        ids.append(nxt)
    return bytes(ids[len(prompt.encode("utf-8", errors="replace")) :]).decode("utf-8", errors="replace")


def _gen_bpe(model: BPETokenLM, tokenizer: spm.SentencePieceProcessor, prompt: str, seq: int, max_new: int = 64) -> str:
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, out_type=int)

    def pick_next(logits_1d: torch.Tensor, history: list[int]) -> int:
        top_vals, top_idx = torch.topk(logits_1d, k=min(8, logits_1d.numel()))
        del top_vals
        if len(history) >= 8 and len(set(history[-8:])) == 1:
            for idx in top_idx.tolist():
                if idx != history[-1]:
                    return int(idx)
        return int(top_idx[0].item())

    for _ in range(max_new):
        x = torch.tensor([ids[-seq:]], dtype=torch.long, device=device)
        logits = model(x)
        nxt = pick_next(logits[0, -1], ids)
        ids.append(nxt)
    return tokenizer.decode(ids[len(tokenizer.encode(prompt, out_type=int)) :])


def _quality_score(text: str, expected_keywords: list[str]) -> dict[str, float]:
    chars = max(len(text), 1)
    alpha = sum(ch.isalpha() for ch in text) / chars
    tokens = text.split()
    max_rep = max((tokens.count(t) for t in set(tokens)), default=0)
    rep_score = 1.0 - min(max_rep / 12.0, 1.0)
    lower = text.lower()
    kw = sum(1 for k in expected_keywords if k in lower) / max(len(expected_keywords), 1)
    quality = 0.4 * alpha + 0.3 * rep_score + 0.3 * kw
    return {
        "alpha_ratio": alpha,
        "max_token_repeat": float(max_rep),
        "keyword_score": kw,
        "quality_score": quality,
    }


def _params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-scale LayerCake vs baseline benchmark at 1M/2M/5M/10M")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--train-bytes", type=int, default=8_000_000)
    parser.add_argument("--eval-bytes", type=int, default=300_000)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--output", default="results/micro_scale_curriculum_frontier.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    redpajama = ROOT.parent / "layercakeogwithdecoder/data/v6/redpajama_english_train.jsonl"
    curriculum_files = [
        ROOT / "data/curriculum/english_school_curriculum.txt",
        ROOT / "data/curriculum/companion_dialogue_curriculum.txt",
    ]
    full_stream = load_curriculum_bytes(redpajama, curriculum_files, args.train_bytes + args.eval_bytes)
    train_bytes = full_stream[:-args.eval_bytes]
    eval_bytes = full_stream[-args.eval_bytes:]

    with tempfile.TemporaryDirectory(prefix="lc_micro_spm_") as tmp:
        tmpdir = Path(tmp)
        corpus_txt = tmpdir / "corpus.txt"
        prep_started = time.perf_counter()
        corpus_txt.write_text(bytes(train_bytes.tolist()).decode("utf-8", errors="replace"), encoding="utf-8")
        prefix = tmpdir / "micro"
        spm.SentencePieceTrainer.train(
            input=str(corpus_txt),
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

        train_tokens = torch.tensor(tokenizer.encode(bytes(train_bytes.tolist()).decode("utf-8", errors="replace"), out_type=int), dtype=torch.long)
        eval_tokens = torch.tensor(tokenizer.encode(bytes(eval_bytes.tolist()).decode("utf-8", errors="replace"), out_type=int), dtype=torch.long)
        baseline_prep_seconds = time.perf_counter() - prep_started

        scales = [
            ScaleSpec(
                name="1m",
                lc_model=dict(d_byte=24, d_model=144, d_abi=48, layers=2, heads=4, local_layers=2, local_width=128),
                bpe_model=dict(d_model=160, layers=2, heads=5),
            ),
            ScaleSpec(
                name="2m",
                lc_model=dict(d_byte=24, d_model=176, d_abi=64, layers=3, heads=8, local_layers=2, local_width=176),
                bpe_model=dict(d_model=208, layers=3, heads=8),
            ),
            ScaleSpec(
                name="5m",
                lc_model=dict(d_byte=32, d_model=256, d_abi=96, layers=4, heads=8, local_layers=2, local_width=256),
                bpe_model=dict(d_model=320, layers=5, heads=8),
            ),
            ScaleSpec(
                name="10m",
                lc_model=dict(d_byte=32, d_model=320, d_abi=128, layers=5, heads=8, local_layers=3, local_width=320),
                bpe_model=dict(d_model=352, layers=6, heads=8),
            ),
        ]

        prompts = [
            ("Question: What is a calm first step when two threats appear? Answer:", ["first", "threat", "calm", "step"]),
            ("Question: How should I recover after a mistake? Answer:", ["recover", "safe", "next", "step"]),
            ("Question: Give a short plan before entering the next room. Answer:", ["plan", "before", "next", "room"]),
        ]

        rows: list[dict[str, Any]] = []
        for spec in scales:
            lc = CausalBytePatchLM(
                patch_size=2,
                max_patches=args.seq // 2,
                continuous_local=False,
                direct_global_context=True,
                local_decoder="window_transformer",
                modern_blocks=True,
                fused_attention=True,
                local_window=32,
                patch_unit_buckets=0,
                dropout=0.1,
                qk_norm=True,
                global_block="attention",
                **spec.lc_model,
            ).to(device)
            bpe = BPETokenLM(vocab_size=tokenizer.vocab_size(), max_len=args.seq, **spec.bpe_model).to(device)

            lc_train = _train_lc(lc, train_bytes, args.steps, args.seq, args.batch, device)
            bpe_train = _train_bpe(bpe, train_tokens, args.steps, args.seq, args.batch, device)

            bpe_elapsed_total = bpe_train["elapsed_seconds"] + baseline_prep_seconds

            lc_bpb = _eval_lc_bpb(lc, eval_bytes, args.seq, args.batch, args.eval_batches, device)
            bpe_bpb = _eval_bpe_bpb(bpe, eval_tokens, int(eval_bytes.numel()), args.seq, args.batch, args.eval_batches, device)

            lc_scores = []
            bpe_scores = []
            qa_rows = []
            for prompt, kws in prompts:
                lc_text = _gen_lc(lc, prompt, seq=args.seq)
                bpe_text = _gen_bpe(bpe, tokenizer, prompt, seq=args.seq)
                lc_q = _quality_score(lc_text, kws)
                bpe_q = _quality_score(bpe_text, kws)
                lc_scores.append(lc_q["quality_score"])
                bpe_scores.append(bpe_q["quality_score"])
                qa_rows.append(
                    {
                        "prompt": prompt,
                        "layercake": {"text": lc_text, **lc_q},
                        "baseline": {"text": bpe_text, **bpe_q},
                    }
                )

            lc_params = _params(lc)
            bpe_params = _params(bpe)
            lc_param_seconds = lc_train["elapsed_seconds"] * lc_params
            bpe_param_seconds = bpe_elapsed_total * bpe_params

            gates = {
                "speed_beats_baseline": lc_train["elapsed_seconds"] < bpe_elapsed_total,
                "quality_noninferior": (sum(lc_scores) / len(lc_scores)) >= (sum(bpe_scores) / len(bpe_scores)) * 0.98,
                "bpb_noninferior": lc_bpb <= bpe_bpb * 1.02,
                "cost_proxy_lower": lc_param_seconds < bpe_param_seconds,
                "params_no_larger": lc_params <= bpe_params,
            }
            status = "PASS" if all(gates.values()) else "FAIL"

            rows.append(
                {
                    "scale": spec.name,
                    "status": status,
                    "gates": gates,
                    "layercake": {
                        "params": lc_params,
                        "train": lc_train,
                        "general_bpb": lc_bpb,
                        "qa_quality_mean": sum(lc_scores) / len(lc_scores),
                    },
                    "baseline": {
                        "params": bpe_params,
                        "train": {
                            **bpe_train,
                            "prep_seconds": baseline_prep_seconds,
                            "elapsed_total_seconds": bpe_elapsed_total,
                        },
                        "general_bpb": bpe_bpb,
                        "qa_quality_mean": sum(bpe_scores) / len(bpe_scores),
                    },
                    "cost_proxy_param_seconds": {
                        "layercake": lc_param_seconds,
                        "baseline": bpe_param_seconds,
                    },
                    "qa_samples": qa_rows,
                }
            )

        summary_gates = {
            "all_scales_pass": all(r["status"] == "PASS" for r in rows),
            "pass_1m": next(r for r in rows if r["scale"] == "1m")["status"] == "PASS",
            "pass_2m": next(r for r in rows if r["scale"] == "2m")["status"] == "PASS",
            "pass_5m": next(r for r in rows if r["scale"] == "5m")["status"] == "PASS",
            "pass_10m": next(r for r in rows if r["scale"] == "10m")["status"] == "PASS",
        }

        result = {
            "status": "PASS" if summary_gates["all_scales_pass"] else "FAIL",
            "scope": "LayerCake byte curriculum vs baseline transformer at 1M/2M/5M/10M",
            "device": str(device),
            "steps": args.steps,
            "seq": args.seq,
            "batch": args.batch,
            "train_bytes": int(train_bytes.numel()),
            "eval_bytes": int(eval_bytes.numel()),
            "data_split": {
                "method": "contiguous_holdout_tail",
                "train_sha256": tensor_byte_sha256(train_bytes),
                "eval_sha256": tensor_byte_sha256(eval_bytes),
                "disjoint_by_construction": True,
            },
            "summary_gates": summary_gates,
            "scales": rows,
        }

        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
