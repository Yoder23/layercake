from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import sentencepiece as spm
import torch

import _common
from benchmark_bpe_baseline import BPETokenLM
from train_bpe_adapter import AdaptedBPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    base_artifact = torch.load(args.base, map_location="cpu")
    adapter_artifact = torch.load(args.adapter, map_location="cpu")
    config = base_artifact["args"]
    base = BPETokenLM(
        base_artifact["vocab_size"],
        d_model=config["d_model"],
        layers=config["layers"],
        heads=config["heads"],
        max_len=max(config["seq"], args.seq),
    ).to(device)
    base.load_state_dict(base_artifact["model"])
    model = AdaptedBPE(base, adapter_artifact["rank"]).to(device)
    model.adapters.load_state_dict(adapter_artifact["state_dict"])
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
        handle.write(base_artifact["tokenizer_model"])
        tokenizer_path = Path(handle.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer_path.unlink(missing_ok=True)
    sample = (
        "def portable_domain(payload, runtime):\n"
        "    return runtime.install(payload)\n"
    ) * 128
    token_ids = tokenizer.encode(sample, out_type=int)
    bytes_per_token = len(sample.encode("utf-8")) / len(token_ids)
    x = torch.randint(
        0,
        base_artifact["vocab_size"],
        (args.batch, args.seq),
        device=device,
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000)
    median_ms = statistics.median(samples)
    tokens_per_second = args.batch * args.seq / (median_ms / 1000)
    result = {
        "status": "PASS",
        "device": str(device),
        "threads": args.threads,
        "batch": args.batch,
        "sequence_tokens": args.seq,
        "estimated_context_bytes": args.seq * bytes_per_token,
        "median_ms": median_ms,
        "tokens_per_second": tokens_per_second,
        "estimated_bytes_per_second": tokens_per_second * bytes_per_token,
        "base_parameters": sum(p.numel() for p in base.parameters()),
        "adapter_parameters": sum(
            p.numel() for p in model.adapters.parameters()
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
