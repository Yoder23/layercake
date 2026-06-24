from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import sentencepiece as spm
import torch
from torch import nn
import torch.nn.functional as F

import _common
from benchmark_bpe_baseline import BPETokenLM, evaluate
from layercake.causal_byte_models import causal_mask
from run_paired_byte_experiment import batch, load_jsonl_bytes, load_python_bytes


def load_domain_stream(args, root: Path) -> torch.Tensor:
    if args.domain_file:
        payload = bytearray()
        for item in args.domain_file:
            path = Path(item)
            if not path.is_absolute():
                path = root / path
            payload.extend(path.read_bytes())
            payload.extend(b"\n")
        if len(payload) < args.seq * 4:
            raise ValueError(
                "domain files are too small for the requested sequence length"
            )
        return torch.tensor(list(payload[: args.domain_bytes]), dtype=torch.long)
    return load_python_bytes(
        root.parent / "layercakeogwithdecoder", args.domain_bytes
    )


class ResidualAdapter(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.01))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.scale * self.up(F.silu(self.down(self.norm(h))))


class AdaptedBPE(nn.Module):
    def __init__(self, base: BPETokenLM, rank: int):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        width = base.emb.embedding_dim
        self.adapters = nn.ModuleList(
            ResidualAdapter(width, rank) for _ in self.base.core.layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.base.emb(x) + self.base.pos(positions)[None]
        mask = causal_mask(x.shape[1], x.device)
        for layer, adapter in zip(self.base.core.layers, self.adapters):
            h = adapter(layer(h, src_mask=mask))
        if self.base.core.norm is not None:
            h = self.base.core.norm(h)
        return self.base.head(self.base.norm(h))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=6262)
    parser.add_argument("--domain-bytes", type=int, default=2_000_000)
    parser.add_argument(
        "--domain-file",
        action="append",
        help="Text/JSONL file to use as adapter domain data.",
    )
    parser.add_argument("--general-bytes", type=int, default=20_000_000)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--output-artifact")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.artifact, map_location="cpu")
    config = artifact["args"]
    base = BPETokenLM(
        artifact["vocab_size"],
        d_model=config["d_model"],
        layers=config["layers"],
        heads=config["heads"],
        max_len=max(config["seq"], args.seq),
    ).to(device)
    base.load_state_dict(artifact["model"])
    model = AdaptedBPE(base, args.rank).to(device)
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
        handle.write(artifact["tokenizer_model"])
        tokenizer_path = Path(handle.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer_path.unlink(missing_ok=True)

    root = Path(__file__).resolve().parents[1]
    domain = load_domain_stream(args, root)
    general = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        args.general_bytes,
    )
    eval_bytes = min(100_000, max(args.seq * args.batch * 2, domain.numel() // 10))
    if domain.numel() <= eval_bytes + args.seq + 1:
        raise ValueError("domain stream is too small after reserving eval bytes")
    domain_train, domain_eval = domain[:-eval_bytes], domain[-eval_bytes:]
    general_eval = general[-min(200_000, general.numel()):]

    def encode(raw: torch.Tensor) -> torch.Tensor:
        text = bytes(raw.tolist()).decode("utf-8", errors="replace")
        return torch.tensor(tokenizer.encode(text, out_type=int), dtype=torch.long)

    domain_train_tokens = encode(domain_train)
    domain_eval_tokens = encode(domain_eval)
    general_eval_tokens = encode(general_eval)
    eval_batch = min(args.batch, 32)
    before = {
        "domain": evaluate(
            base, domain_eval_tokens, domain_eval.numel(), args.seq,
            eval_batch, args.eval_batches, device
        ),
        "general": evaluate(
            base, general_eval_tokens, general_eval.numel(), args.seq,
            eval_batch, args.eval_batches, device
        ),
    }
    optimizer = torch.optim.AdamW(model.adapters.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = batch(
            domain_train_tokens, args.seq, args.batch, generator, device
        )
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.adapters.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 250 == 0:
            item = {"step": step, "loss": loss.item()}
            history.append(item)
            print(item, flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started
    after = {
        "domain": evaluate(
            model, domain_eval_tokens, domain_eval.numel(), args.seq,
            eval_batch, args.eval_batches, device
        ),
        "general": evaluate(
            model, general_eval_tokens, general_eval.numel(), args.seq,
            eval_batch, args.eval_batches, device
        ),
    }
    adapter_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.adapters.state_dict().items()
    }
    if args.output_artifact:
        torch.save(
            {
                "format": "layercake-bpe-adapter/1",
                "base_artifact": args.artifact,
                "rank": args.rank,
                "state_dict": adapter_state,
            },
            args.output_artifact,
        )
    bytes_per_token = domain_train.numel() / domain_train_tokens.numel()
    result = {
        "status": "TRAINED",
        "rank": args.rank,
        "seed": args.seed,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.adapters.parameters()
        ),
        "artifact_bytes_fp32": sum(
            tensor.numel() * tensor.element_size()
            for tensor in adapter_state.values()
        ),
        "estimated_total_training_bytes": (
            args.steps * args.batch * args.seq * bytes_per_token
        ),
        "domain_files": args.domain_file or [],
        "train_bytes": int(domain_train.numel()),
        "eval_bytes": int(domain_eval.numel()),
        "before": before,
        "after": after,
        "general_bpb_regression": (
            after["general"]["bpb"] - before["general"]["bpb"]
        ),
        "history": history,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
