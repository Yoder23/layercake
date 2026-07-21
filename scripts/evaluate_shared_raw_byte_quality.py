"""Evaluate CountCake and a token Transformer on identical raw-byte spans.

The ordinary model-specific evaluators choose different block boundaries because
one model consumes bytes and the other consumes SentencePiece tokens.  This
utility instead predeclares UTF-8-safe raw-byte rows, resets both models at the
same row boundaries, and scores only the exact byte suffix covered by complete
Transformer target tokens after the shared prefix.  The CountCake mask is built
from those same token spans, so both BPB denominators are byte-for-byte equal.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import (  # noqa: E402
    apply_causal_online_cache_to_observed,
    load_count_cake_bundle,
)
from scripts.train_bpe_transformer_from_config import (  # noqa: E402
    BPETokenTransformerLM,
)
from scripts.train_byte_core_from_config import _build_model  # noqa: E402


def _utf8_rows(payload: bytes, row_bytes: int) -> list[bytes]:
    """Split near ``row_bytes`` without cutting a UTF-8 code point."""
    rows: list[bytes] = []
    start = 0
    while start < len(payload):
        end = min(start + row_bytes, len(payload))
        if end < len(payload):
            while end > start and payload[end] & 0xC0 == 0x80:
                end -= 1
        if end == start:
            raise ValueError("row_bytes is too small to contain one UTF-8 code point")
        row = payload[start:end]
        row.decode("utf-8", errors="strict")
        rows.append(row)
        start = end
    return rows


def _piece_byte_offsets(encoded) -> tuple[list[int], list[tuple[int, int]]]:
    utf8_prefix = [0]
    for character in encoded.text:
        utf8_prefix.append(utf8_prefix[-1] + len(character.encode("utf-8")))
    spans = [
        (utf8_prefix[int(piece.begin)], utf8_prefix[int(piece.end)])
        for piece in encoded.pieces
    ]
    return utf8_prefix, spans


def _load_transformer(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
    return model, tokenizer, checkpoint, max_len


def _span_length_bucket(length: int) -> str:
    if length == 1:
        return "1"
    if length == 2:
        return "2"
    if length <= 4:
        return "3-4"
    if length <= 8:
        return "5-8"
    return "9+"


def _content_bucket(payload: bytes) -> str:
    if any(byte >= 128 for byte in payload):
        return "non_ascii"
    text = payload.decode("ascii")
    if text.isspace():
        return "whitespace"
    if text.isalpha():
        return "alpha"
    if text.isdigit():
        return "digit"
    if all(character.isalnum() or character == "_" for character in text):
        return "word_mixed"
    if all(not character.isalnum() and not character.isspace() for character in text):
        return "punctuation"
    return "mixed"


def _render_buckets(accumulator: dict) -> dict:
    rendered = {}
    for name, values in sorted(accumulator.items()):
        byte_count = int(values["bytes"])
        transformer_bpb = values["transformer_nll"] / byte_count / math.log(2.0)
        base_bpb = values["base_nll"] / byte_count / math.log(2.0)
        cached_bpb = values["cached_nll"] / byte_count / math.log(2.0)
        rendered[name] = {
            "spans": int(values["spans"]),
            "bytes": byte_count,
            "transformer_bpb": transformer_bpb,
            "layercake_base_bpb": base_bpb,
            "layercake_cached_bpb": cached_bpb,
            "base_delta": base_bpb - transformer_bpb,
            "cached_delta": cached_bpb - transformer_bpb,
        }
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    layercake_group = parser.add_mutually_exclusive_group(required=True)
    layercake_group.add_argument("--count-bundle")
    layercake_group.add_argument("--layercake-checkpoint")
    parser.add_argument("--transformer-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--row-bytes", type=int, default=288)
    parser.add_argument("--prefix-bytes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.row_bytes <= args.prefix_bytes:
        raise ValueError("row-bytes must exceed prefix-bytes")
    if args.row_bytes % 32:
        raise ValueError("row-bytes must be divisible by 32 for CountCake")
    device = torch.device(args.device)
    payload = Path(args.data).read_bytes()
    payload.decode("utf-8", errors="strict")
    rows = _utf8_rows(payload, args.row_bytes)
    started = time.perf_counter()

    transformer, tokenizer, checkpoint, token_limit = _load_transformer(
        args.transformer_checkpoint, device
    )
    count_model = None
    count_manifest = None
    byte_model = None
    byte_checkpoint = None
    if args.count_bundle:
        count_model, count_manifest = load_count_cake_bundle(
            args.count_bundle, device=device
        )
        count_model.eval()
        layercake_start = count_model.prediction_start
        layercake_artifact = args.count_bundle
        layercake_parameters = count_manifest["parameters"]["logical_total"]
        layercake_format = "count_cake"
    else:
        byte_checkpoint = torch.load(
            args.layercake_checkpoint, map_location="cpu", weights_only=False
        )
        byte_model = _build_model(byte_checkpoint["model_config"], device)
        byte_model.load_state_dict(byte_checkpoint["model"])
        byte_model.eval()
        layercake_start = int(byte_checkpoint["model_config"]["patch_size"])
        layercake_artifact = args.layercake_checkpoint
        layercake_parameters = int(byte_checkpoint["trainable_params"])
        layercake_format = "recurrent_patch_cake"
    if layercake_start > args.prefix_bytes:
        raise ValueError(
            "shared prefix is shorter than CountCake prediction_start: "
            f"{args.prefix_bytes} < {layercake_start}"
        )

    token_rows: list[list[int]] = []
    first_targets: list[int] = []
    scored_starts: list[int] = []
    scored_ends: list[int] = []
    scored_spans: list[list[tuple[int, int]]] = []
    normalization_equal = True
    maximum_tokens = 0
    for row_index, row in enumerate(rows):
        text = row.decode("utf-8", errors="strict")
        encoded = tokenizer.encode(text, out_type="immutable_proto")
        normalized = encoded.text.encode("utf-8")
        if normalized != row:
            normalization_equal = False
            raise ValueError(
                "SentencePiece normalization changed raw bytes in row "
                f"{row_index}; a shared-byte comparison is impossible"
            )
        tokens = [int(piece.id) for piece in encoded.pieces]
        if len(tokens) < 2:
            raise ValueError(f"row {row_index} contains fewer than two tokens")
        maximum_tokens = max(maximum_tokens, len(tokens))
        if len(tokens) - 1 > token_limit:
            raise ValueError(
                f"row {row_index} needs {len(tokens) - 1} Transformer positions, "
                f"exceeding checkpoint limit {token_limit}"
            )
        _, spans = _piece_byte_offsets(encoded)
        first_target = next(
            (index for index, (begin, _) in enumerate(spans) if begin >= args.prefix_bytes),
            None,
        )
        if first_target is None or first_target == 0:
            raise ValueError(f"row {row_index} has no scoreable post-prefix token")
        scored_start = spans[first_target][0]
        scored_end = spans[-1][1]
        target_spans = list(spans[first_target:])
        if (
            scored_start < args.prefix_bytes
            or scored_end > len(row)
            or not target_spans
            or any(end < begin for begin, end in target_spans)
        ):
            empty_spans = [
                {
                    "index": first_target + index,
                    "begin": begin,
                    "end": end,
                    "id": int(encoded.pieces[first_target + index].id),
                    "piece": encoded.pieces[first_target + index].piece,
                    "surface": encoded.pieces[first_target + index].surface,
                }
                for index, (begin, end) in enumerate(target_spans)
                if end <= begin
            ]
            raise RuntimeError(
                "invalid shared scored span in row "
                f"{row_index}: start={scored_start}, end={scored_end}, "
                f"row_bytes={len(row)}, first_target={first_target}, "
                f"pieces={len(spans)}, empty_spans={empty_spans[:4]}"
            )
        token_rows.append(tokens)
        first_targets.append(first_target)
        scored_starts.append(scored_start)
        scored_ends.append(scored_end)
        scored_spans.append(target_spans)

    transformer_nll = 0.0
    transformer_tokens = 0
    transformer_token_nlls: list[list[float]] = []
    transformer_started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch_tokens = token_rows[offset : offset + args.batch_size]
            maximum = max(len(tokens) - 1 for tokens in batch_tokens)
            inputs = torch.zeros(
                len(batch_tokens), maximum, device=device, dtype=torch.long
            )
            for local_index, tokens in enumerate(batch_tokens):
                inputs[local_index, : len(tokens) - 1] = torch.tensor(
                    tokens[:-1], device=device, dtype=torch.long
                )
            logits = transformer(inputs)
            for local_index, tokens in enumerate(batch_tokens):
                global_index = offset + local_index
                first = first_targets[global_index]
                targets = torch.tensor(
                    tokens[first:], device=device, dtype=torch.long
                )
                selected = logits[local_index, first - 1 : len(tokens) - 1]
                losses = F.cross_entropy(selected, targets, reduction="none")
                transformer_nll += float(losses.sum())
                transformer_tokens += targets.numel()
                transformer_token_nlls.append(losses.float().cpu().tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    transformer_seconds = time.perf_counter() - transformer_started

    padded = np.zeros((len(rows), args.row_bytes), dtype=np.uint8)
    byte_mask = np.zeros(
        (len(rows), args.row_bytes - layercake_start), dtype=np.bool_
    )
    for index, row in enumerate(rows):
        padded[index, : len(row)] = np.frombuffer(row, dtype=np.uint8)
        for absolute_begin, absolute_end in scored_spans[index]:
            begin = absolute_begin - layercake_start
            end = absolute_end - layercake_start
            byte_mask[index, begin:end] = True

    base_parts: list[np.ndarray] = []
    count_started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch = torch.from_numpy(
                padded[offset : offset + args.batch_size]
            ).to(device=device, dtype=torch.long)
            if count_model is not None:
                observed = count_model.target_log_probs(batch)
            else:
                predictions, targets = byte_model.domain_cake_patch_predictions(batch)
                logits = torch.stack(predictions, dim=2)
                observed = F.log_softmax(logits.float(), dim=-1).gather(
                    -1, targets.unsqueeze(-1)
                ).squeeze(-1).reshape(batch.shape[0], -1)
                observed = observed[:, : args.row_bytes - layercake_start]
            base_parts.append(observed.exp().cpu().numpy())
    base = np.concatenate(base_parts, axis=0).astype(np.float64)
    if count_model is None or args.no_cache or not (
        count_model.online_cache_specs
        or count_model.recent_cache_specs
        or count_model.normalized_cache_specs
    ):
        cached = base
        cache_enabled = False
    else:
        cached = apply_causal_online_cache_to_observed(
            base,
            padded,
            start=layercake_start,
            specs=count_model.online_cache_specs,
            reset_each_row=True,
            window=count_model.online_cache_window,
            recent_specs=count_model.recent_cache_specs,
            normalized_specs=count_model.normalized_cache_specs,
            normalization=count_model.cache_normalization,
        )
        cache_enabled = True
    if device.type == "cuda":
        torch.cuda.synchronize()
    count_seconds = time.perf_counter() - count_started

    scored_bytes = int(byte_mask.sum())
    expected_bytes = sum(
        end - begin
        for row_spans in scored_spans
        for begin, end in row_spans
    )
    if scored_bytes != expected_bytes:
        raise RuntimeError("LayerCake byte mask does not match Transformer spans")
    base_nll = float(-np.log(np.clip(base[byte_mask], 1e-30, None)).sum())
    cached_nll = float(-np.log(np.clip(cached[byte_mask], 1e-30, None)).sum())
    diagnostics = {
        "span_bytes": defaultdict(
            lambda: {
                "spans": 0,
                "bytes": 0,
                "transformer_nll": 0.0,
                "base_nll": 0.0,
                "cached_nll": 0.0,
            }
        ),
        "content": defaultdict(
            lambda: {
                "spans": 0,
                "bytes": 0,
                "transformer_nll": 0.0,
                "base_nll": 0.0,
                "cached_nll": 0.0,
            }
        ),
        "boundary": defaultdict(
            lambda: {
                "spans": 0,
                "bytes": 0,
                "transformer_nll": 0.0,
                "base_nll": 0.0,
                "cached_nll": 0.0,
            }
        ),
        "row_quartile": defaultdict(
            lambda: {
                "spans": 0,
                "bytes": 0,
                "transformer_nll": 0.0,
                "base_nll": 0.0,
                "cached_nll": 0.0,
            }
        ),
    }
    for row_index, (row, row_spans, token_losses) in enumerate(
        zip(rows, scored_spans, transformer_token_nlls)
    ):
        if len(row_spans) != len(token_losses):
            raise RuntimeError("Transformer losses do not align with scored spans")
        pending_token_nll = 0.0
        last_bucket_items = []
        for (begin, end), token_nll in zip(row_spans, token_losses):
            pending_token_nll += float(token_nll)
            if end == begin:
                continue
            relative_begin = begin - layercake_start
            relative_end = end - layercake_start
            span_base_nll = float(
                -np.log(
                    np.clip(
                        base[row_index, relative_begin:relative_end],
                        1e-30,
                        None,
                    )
                ).sum()
            )
            span_cached_nll = float(
                -np.log(
                    np.clip(
                        cached[row_index, relative_begin:relative_end],
                        1e-30,
                        None,
                    )
                ).sum()
            )
            span = row[begin:end]
            preceding = row[begin - 1] if begin else None
            boundary = (
                "row_start"
                if preceding is None
                else "after_space"
                if chr(preceding).isspace()
                else "after_punctuation"
                if preceding < 128 and not chr(preceding).isalnum()
                else "continuation"
            )
            quartile = str(min(3, begin * 4 // max(len(row), 1)))
            labels = {
                "span_bytes": _span_length_bucket(len(span)),
                "content": _content_bucket(span),
                "boundary": boundary,
                "row_quartile": quartile,
            }
            for family, label in labels.items():
                item = diagnostics[family][label]
                item["spans"] += 1
                item["bytes"] += len(span)
                item["transformer_nll"] += pending_token_nll
                item["base_nll"] += span_base_nll
                item["cached_nll"] += span_cached_nll
                last_bucket_items.append(item)
            pending_token_nll = 0.0
        if pending_token_nll:
            if not last_bucket_items:
                raise RuntimeError("row contains only zero-width target tokens")
            for item in last_bucket_items[-4:]:
                item["transformer_nll"] += pending_token_nll
    report = {
        "format": "layercake-shared-raw-byte-quality/1",
        "status": "COMPLETE",
        "contract": {
            "row_boundaries": "identical UTF-8-safe raw-byte rows",
            "reset_points_identical": True,
            "scored_byte_spans_identical": True,
            "bpb_denominator_bytes_identical": True,
            "sentencepiece_normalization_byte_exact": normalization_equal,
            "row_bytes_maximum": args.row_bytes,
            "prefix_bytes_minimum": args.prefix_bytes,
            "partial_final_row_included": True,
        },
        "data": {
            "path": args.data,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(rows),
            "scored_bytes_each": scored_bytes,
        },
        "transformer": {
            "checkpoint": args.transformer_checkpoint,
            "parameters": int(checkpoint["trainable_params"]),
            "artifact_bytes": Path(args.transformer_checkpoint).stat().st_size,
            "maximum_tokens_per_row": maximum_tokens,
            "position_limit": token_limit,
            "scored_tokens": transformer_tokens,
            "nll": transformer_nll / scored_bytes,
            "bpb": transformer_nll / scored_bytes / math.log(2.0),
            "evaluation_seconds": transformer_seconds,
        },
        "layercake": {
            "format": layercake_format,
            "artifact": layercake_artifact,
            "parameters": layercake_parameters,
            "artifact_bytes": Path(layercake_artifact).stat().st_size,
            "base_nll": base_nll / scored_bytes,
            "base_bpb": base_nll / scored_bytes / math.log(2.0),
            "cached_nll": cached_nll / scored_bytes,
            "cached_bpb": cached_nll / scored_bytes / math.log(2.0),
            "causal_cache_enabled": cache_enabled,
            "causal_cache_reset_each_row": cache_enabled,
            "evaluation_seconds": count_seconds,
        },
        "comparison": {
            "base_bpb_delta_layercake_minus_transformer": (
                base_nll - transformer_nll
            )
            / scored_bytes
            / math.log(2.0),
            "cached_bpb_delta_layercake_minus_transformer": (
                cached_nll - transformer_nll
            )
            / scored_bytes
            / math.log(2.0),
        },
        "diagnostics": {
            family: _render_buckets(values)
            for family, values in diagnostics.items()
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
