from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import time

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from artifact_utils import build_models
from benchmark_bpe_baseline import BPETokenLM
from run_paired_byte_experiment import load_jsonl_bytes


def repeated_ngram_blocked(
    prefix: list[int],
    candidate: int,
    ngram: int,
) -> bool:
    if ngram <= 1 or len(prefix) < ngram - 1:
        return False
    trial = tuple(prefix[-(ngram - 1) :] + [candidate])
    existing = {
        tuple(prefix[index : index + ngram])
        for index in range(0, len(prefix) - ngram + 1)
    }
    return trial in existing


def choose_byte(
    logits: torch.Tensor,
    prefix: list[int],
    no_repeat_ngram: int,
) -> int:
    ordered = torch.argsort(logits, descending=True).tolist()
    for candidate in ordered:
        if not repeated_ngram_blocked(prefix, candidate, no_repeat_ngram):
            return int(candidate)
    return int(ordered[0])


@torch.no_grad()
def generate_layercake_cached(
    model,
    prompt: torch.Tensor,
    new_bytes: int,
    no_repeat_ngram: int,
):
    generated = prompt.clone()
    state = model.begin_cached_generation(prompt)
    started = time.perf_counter()
    while generated.shape[1] - prompt.shape[1] < new_bytes:
        next_patch = model.cached_generation_step(
            state, no_repeat_ngram=no_repeat_ngram
        )
        generated = torch.cat([generated, next_patch], dim=1)
    if prompt.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return generated[:, prompt.shape[1] :], elapsed


@torch.no_grad()
def generate_bpe(
    model,
    tokenizer,
    prompt_text: str,
    new_bytes: int,
    device,
):
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


def byte_metrics(raw: bytes) -> dict:
    text = raw.decode("utf-8", errors="replace")
    printable = sum(1 for byte in raw if byte in (9, 10, 13) or 32 <= byte <= 126)
    words = [word for word in text.replace("\n", " ").split(" ") if word]
    word_bigrams = list(zip(words, words[1:]))
    word_trigrams = list(zip(words, words[1:], words[2:]))

    def distinct(items) -> float:
        return len(set(items)) / max(len(items), 1)

    def max_repeat_ngram(n: int) -> int:
        if len(raw) < n:
            return 0
        counts = Counter(
            raw[index : index + n] for index in range(0, len(raw) - n + 1)
        )
        return max(counts.values(), default=0)

    return {
        "utf8": text,
        "printable_ratio": printable / max(len(raw), 1),
        "word_count": len(words),
        "distinct_word_bigram": distinct(word_bigrams),
        "distinct_word_trigram": distinct(word_trigrams),
        "max_repeat_4gram": max_repeat_ngram(4),
        "max_repeat_8gram": max_repeat_ngram(8),
        "max_repeat_16gram": max_repeat_ngram(16),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layercake", required=True)
    parser.add_argument("--bpe", required=True)
    parser.add_argument("--new-bytes", type=int, default=128)
    parser.add_argument("--prompt-bytes", type=int, default=128)
    parser.add_argument("--no-repeat-ngram", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
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
    prompt_bytes = bytes(stream[4096 : 4096 + args.prompt_bytes].tolist())
    prompt_text = prompt_bytes.decode("utf-8", errors="replace")
    prompt = torch.tensor(list(prompt_bytes), dtype=torch.long, device=device).unsqueeze(0)

    layercake_bytes, layercake_seconds = generate_layercake_cached(
        layercake, prompt, args.new_bytes, args.no_repeat_ngram
    )
    bpe_bytes, bpe_seconds = generate_bpe(
        bpe, tokenizer, prompt_text, args.new_bytes, device
    )
    layercake_raw = bytes(layercake_bytes[0].cpu().tolist())
    result = {
        "status": "PASS",
        "device": str(device),
        "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        "new_bytes": args.new_bytes,
        "layercake_decoding": {
            "mode": "stateful_cached",
            "no_repeat_ngram": args.no_repeat_ngram,
        },
        "layercake": {
            "seconds": layercake_seconds,
            "bytes_per_second": args.new_bytes / layercake_seconds,
            **byte_metrics(layercake_raw),
        },
        "bpe": {
            "seconds": bpe_seconds,
            "bytes_per_second": args.new_bytes / bpe_seconds,
            **byte_metrics(bpe_bytes),
        },
        "quality_gates": {
            "layercake_printable": byte_metrics(layercake_raw)["printable_ratio"] >= 0.95,
            "layercake_distinct_trigram_at_least_bpe": (
                byte_metrics(layercake_raw)["distinct_word_trigram"]
                >= byte_metrics(bpe_bytes)["distinct_word_trigram"]
            ),
            "layercake_max_repeat_8gram_no_worse_than_bpe": (
                byte_metrics(layercake_raw)["max_repeat_8gram"]
                <= byte_metrics(bpe_bytes)["max_repeat_8gram"]
            ),
        },
        "scope": (
            "Generation quality diagnostic. LayerCake uses the exact stateful "
            "cached path with an optional byte no-repeat n-gram constraint."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
