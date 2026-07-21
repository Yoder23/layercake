"""Deterministically evaluate every complete token block in a BPE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_bpe_transformer_from_config import BPETokenTransformerLM  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefix-bytes", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    payload = Path(args.data).read_bytes()
    started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    max_len = int(checkpoint["model"]["pos.weight"].shape[0])
    model = BPETokenTransformerLM(
        vocab_size=int(checkpoint["model"]["emb.weight"].shape[0]),
        d_model=int(config["d_model"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        max_len=max_len,
        ff_mult=int(config["ff_mult"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = spm.SentencePieceProcessor(
        model_proto=checkpoint["tokenizer_model"]
    )
    decoded_payload = payload.decode("utf-8", errors="replace")
    encoded = tokenizer.encode(decoded_payload, out_type="immutable_proto")
    pieces = encoded.pieces
    tokens = torch.tensor([int(piece.id) for piece in pieces], dtype=torch.long)
    row_count = (tokens.numel() - 1) // max_len
    if row_count == 0:
        raise ValueError("data contains fewer than one complete token block")
    usable = row_count * max_len
    x = tokens[:usable].reshape(row_count, max_len)
    y = tokens[1 : usable + 1].reshape(row_count, max_len)
    bytes_per_token = len(payload) / int(tokens.numel())
    prefix_tokens = max(1, math.ceil(args.prefix_bytes / bytes_per_token))
    if prefix_tokens >= max_len:
        raise ValueError("prefix consumes the complete token block")
    first_scored_position = prefix_tokens - 1
    # SentencePiece exposes character offsets for every normalized token.  Use
    # those exact spans rather than corpus-average bytes/token, which can bias a
    # close BPB comparison when prefix tokens differ in length.
    utf8_prefix = [0]
    for character in encoded.text:
        utf8_prefix.append(utf8_prefix[-1] + len(character.encode("utf-8")))
    total_scored_bytes = 0
    for row in range(row_count):
        first_target = row * max_len + prefix_tokens
        last_target = row * max_len + max_len
        begin = int(pieces[first_target].begin)
        end = int(pieces[last_target].end)
        total_scored_bytes += utf8_prefix[end] - utf8_prefix[begin]
    if total_scored_bytes <= 0:
        raise ValueError("token blocks contain no scored UTF-8 bytes")
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for offset in range(0, row_count, args.batch_size):
            batch_x = x[offset : offset + args.batch_size].to(device)
            batch_y = y[offset : offset + args.batch_size].to(device)
            logits = model(batch_x)
            total_nll += float(
                F.cross_entropy(
                    logits[:, first_scored_position:].flatten(0, 1),
                    batch_y[:, first_scored_position:].flatten(),
                    reduction="sum",
                )
            )
            total_tokens += batch_y[:, first_scored_position:].numel()
    nll_per_token = total_nll / total_tokens
    bpb = total_nll / total_scored_bytes / math.log(2.0)
    legacy_estimated_bpb = nll_per_token / bytes_per_token / math.log(2.0)
    report = {
        "format": "layercake-transformer-quality/1",
        "status": "COMPLETE",
        "checkpoint": {
            "path": args.checkpoint,
            "parameters": int(checkpoint["trainable_params"]),
        },
        "data": {
            "path": args.data,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "replacement_decode": True,
            "tokens": int(tokens.numel()),
            "scored_tokens": total_tokens,
            "bytes_per_token": bytes_per_token,
            "normalized_utf8_bytes": utf8_prefix[-1],
            "exact_scored_utf8_bytes": total_scored_bytes,
            "complete_blocks": row_count,
            "token_block_size": max_len,
            "requested_prefix_bytes": args.prefix_bytes,
            "unscored_prefix_tokens_per_block": first_scored_position,
        },
        "quality": {
            "nll_per_token": nll_per_token,
            "bpb": bpb,
            "bpb_denominator": "exact scored SentencePiece UTF-8 spans",
            "legacy_corpus_average_estimated_bpb": legacy_estimated_bpb,
        },
        "device": args.device,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
