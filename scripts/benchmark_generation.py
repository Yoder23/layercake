from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import sentencepiece as spm
import torch

import _common
from artifact_utils import build_models
from benchmark_bpe_baseline import BPETokenLM
from run_paired_byte_experiment import load_jsonl_bytes


@torch.no_grad()
def generate_layercake(
    model, prompt: torch.Tensor, new_bytes: int, mode: str
):
    generated = prompt.clone()
    state = (
        model.begin_cached_generation(prompt)
        if mode == "stateful_cached"
        else None
    )
    started = time.perf_counter()
    while generated.shape[1] - prompt.shape[1] < new_bytes:
        context = generated[:, -model.patch_pos.num_embeddings * model.patch_size :]
        if mode == "stateful_cached":
            next_patch = model.cached_generation_step(state)
        elif mode == "cached":
            next_patch = model.generate_cached_patch(context)
        elif mode == "verified":
            next_patch = model.generate_verified_patch(context)
        else:
            next_patch = model.generate_next_patch(context)
        generated = torch.cat([generated, next_patch], dim=1)
    if prompt.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    continuation = generated[:, prompt.shape[1] : prompt.shape[1] + new_bytes]
    return continuation, elapsed


@torch.no_grad()
def generate_bpe(model, tokenizer, prompt_text: str, new_bytes: int, device):
    token_ids = tokenizer.encode(prompt_text, out_type=int)
    generated = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    original_text = tokenizer.decode(token_ids)
    started = time.perf_counter()
    continuation = ""
    while len(continuation.encode("utf-8")) < new_bytes:
        context = generated[:, -model.pos.num_embeddings :]
        next_token = model(context)[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        decoded = tokenizer.decode(generated[0].tolist())
        continuation = decoded[len(original_text) :]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return continuation.encode("utf-8")[:new_bytes], elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake", required=True)
    parser.add_argument("--bpe", required=True)
    parser.add_argument("--new-bytes", type=int, default=64)
    parser.add_argument(
        "--layercake-mode",
        choices=["draft", "verified", "cached", "stateful_cached"],
        default="draft",
    )
    parser.add_argument("--prompt-bytes", type=int, default=128)
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda"
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    layercake_artifact = torch.load(args.layercake, map_location="cpu")
    _, layercake = build_models(layercake_artifact, device)
    layercake.eval()
    if not layercake.patch_prediction:
        raise ValueError("LayerCake artifact requires patch-prediction heads")

    bpe_artifact = torch.load(args.bpe, map_location="cpu")
    config = bpe_artifact["args"]
    bpe = BPETokenLM(
        bpe_artifact["vocab_size"],
        d_model=config["d_model"],
        layers=config["layers"],
        heads=config["heads"],
        max_len=config["seq"],
    ).to(device)
    bpe.load_state_dict(bpe_artifact["model"])
    bpe.eval()
    with tempfile.NamedTemporaryFile(suffix=".model", delete=False) as handle:
        handle.write(bpe_artifact["tokenizer_model"])
        tokenizer_path = Path(handle.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer_path.unlink(missing_ok=True)

    root = Path(__file__).resolve().parents[1]
    stream = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        20_000_000,
    )[-200_000:]
    prompt_bytes = bytes(
        stream[4096 : 4096 + args.prompt_bytes].tolist()
    )
    prompt_text = prompt_bytes.decode("utf-8", errors="replace")
    prompt = torch.tensor(
        list(prompt_bytes), dtype=torch.long, device=device
    ).unsqueeze(0)

    # Warm both paths.
    generate_layercake(layercake, prompt, 4, args.layercake_mode)
    generate_bpe(bpe, tokenizer, prompt_text, 4, device)
    layercake_bytes, layercake_seconds = generate_layercake(
        layercake, prompt, args.new_bytes, args.layercake_mode
    )
    bpe_bytes, bpe_seconds = generate_bpe(
        bpe, tokenizer, prompt_text, args.new_bytes, device
    )
    layercake_raw = bytes(layercake_bytes[0].cpu().tolist())
    result = {
        "status": "PASS",
        "new_bytes": args.new_bytes,
        "device": str(device),
        "cpu_threads": (
            args.cpu_threads if device.type == "cpu" else None
        ),
        "layercake_mode": args.layercake_mode,
        "layercake": {
            "seconds": layercake_seconds,
            "bytes_per_second": args.new_bytes / layercake_seconds,
            "utf8": layercake_raw.decode("utf-8", errors="replace"),
            "hex": layercake_raw.hex(),
        },
        "bpe": {
            "seconds": bpe_seconds,
            "bytes_per_second": args.new_bytes / bpe_seconds,
            "utf8": bpe_bytes.decode("utf-8", errors="replace"),
            "hex": bpe_bytes.hex(),
        },
        "speed_ratio": (
            (args.new_bytes / layercake_seconds)
            / (args.new_bytes / bpe_seconds)
        ),
        "scope": (
            "Greedy batch-1 generation without KV caches. LayerCake emits a "
            f"{args.layercake_mode} {layercake.patch_size}-byte patch per step; "
            "BPE emits one token per step."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
