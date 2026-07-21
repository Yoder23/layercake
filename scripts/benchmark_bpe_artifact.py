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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    config = artifact["args"]
    model = BPETokenLM(
        artifact["vocab_size"],
        d_model=config["d_model"],
        layers=config["layers"],
        heads=config["heads"],
        max_len=config["seq"],
    ).to(device)
    model.load_state_dict(artifact["model"])
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
        handle.write(artifact["tokenizer_model"])
        tokenizer_path = Path(handle.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer_path.unlink(missing_ok=True)
    sample = (
        "LayerCake measures tokenizer transformer throughput over the same "
        "underlying UTF-8 byte budget. "
    ).encode("utf-8")
    repeated = (sample * (4096 // len(sample) + 1))[:4096]
    token_ids = tokenizer.encode(repeated.decode("utf-8"), out_type=int)
    bytes_per_token = len(repeated) / len(token_ids)
    x = torch.randint(
        0,
        artifact["vocab_size"],
        (args.batch, config["seq"]),
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
    tokens_per_second = args.batch * config["seq"] / (median_ms / 1000)
    result = {
        "status": "PASS",
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "batch": args.batch,
        "sequence_tokens": config["seq"],
        "estimated_bytes_per_token": bytes_per_token,
        "median_ms": median_ms,
        "p95_ms": percentile(samples, 0.95),
        "tokens_per_second": tokens_per_second,
        "estimated_bytes_per_second": tokens_per_second * bytes_per_token,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
