"""Screen a tokenizer-free variable-length delimiter-trie byte expert.

This is an architecture-selection probe, not a release benchmark.  It learns
normalized next-byte counts for raw ASCII word prefixes, tunes only a scalar
Bayesian backoff strength on a training-corpus suffix, and evaluates a separate
development file.  No text is converted to token IDs: the runtime state is an
incremental hash of the raw bytes since the latest delimiter.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402


_WORD = re.compile(rb"[A-Za-z]{2,64}")
_HASH_SEED = 0x14650FB0739D0383
_HASH_MULTIPLIER = 0x100000001B3
_HASH_MASK = (1 << 64) - 1


def _is_alpha(value: int) -> bool:
    return 65 <= value <= 90 or 97 <= value <= 122


def _lower(value: int) -> int:
    return value + 32 if 65 <= value <= 90 else value


def _advance(prefix_hash: int, value: int) -> int:
    return (
        (prefix_hash ^ (_lower(value) + 1)) * _HASH_MULTIPLIER
    ) & _HASH_MASK


def _word_counts(payload: bytes, chunk_bytes: int) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    carry = b""
    for offset in range(0, len(payload), chunk_bytes):
        combined = carry + payload[offset : offset + chunk_bytes]
        trailing = re.search(rb"[A-Za-z]+$", combined)
        carry = trailing.group(0) if trailing else b""
        stable = combined[: len(combined) - len(carry)] if carry else combined
        counts.update(_WORD.findall(stable))
    if 2 <= len(carry) <= 64:
        counts[carry] += 1
    return counts


def _build_trie(
    payload: bytes,
    *,
    max_words: int,
    max_states: int,
    chunk_bytes: int,
) -> tuple[dict[int, int], dict[int, int], dict]:
    started = time.perf_counter()
    words = _word_counts(payload, chunk_bytes)
    selected_words = words.most_common(max_words)
    joint: Counter[int] = Counter()
    for word, frequency in selected_words:
        prefix_hash = _HASH_SEED
        for offset in range(len(word) - 1):
            prefix_hash = _advance(prefix_hash, word[offset])
            key = (prefix_hash << 8) | int(word[offset + 1])
            joint[key] += int(frequency)
    retained = joint.most_common(max_states)
    joint_table = {int(key): int(count) for key, count in retained}
    totals: defaultdict[int, int] = defaultdict(int)
    for key, count in retained:
        totals[int(key) >> 8] += int(count)
    return joint_table, dict(totals), {
        "source_bytes": len(payload),
        "unique_words": len(words),
        "selected_words": len(selected_words),
        "unpruned_joint_states": len(joint),
        "retained_joint_states": len(joint_table),
        "contexts": len(totals),
        "seconds": time.perf_counter() - started,
    }


def _rows(payload: bytes, seq_len: int) -> np.ndarray:
    row_count = len(payload) // seq_len
    if row_count == 0:
        raise ValueError("payload is shorter than one row")
    return np.frombuffer(
        payload[: row_count * seq_len], dtype=np.uint8
    ).reshape(row_count, seq_len).copy()


def _base_probability(model, rows: np.ndarray, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    chunks = []
    with torch.inference_mode():
        for offset in range(0, rows.shape[0], batch_size):
            batch = torch.from_numpy(rows[offset : offset + batch_size]).to(
                device=device, dtype=torch.long
            )
            chunks.append(model.target_log_probs(batch).exp().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)


def _trie_statistics(
    rows: np.ndarray,
    *,
    start: int,
    joint: dict[int, int],
    totals: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = rows.shape[1] - start
    observed = np.zeros((rows.shape[0], width), dtype=np.float64)
    total = np.zeros_like(observed)
    lengths = np.zeros((rows.shape[0], width), dtype=np.uint8)
    for row_index, row in enumerate(rows):
        prefix_hash = _HASH_SEED
        prefix_length = 0
        for offset, byte_value in enumerate(row.tolist()):
            value = int(byte_value)
            if offset >= start and prefix_length:
                observed[row_index, offset - start] = joint.get(
                    (prefix_hash << 8) | value, 0
                )
                total[row_index, offset - start] = totals.get(prefix_hash, 0)
                lengths[row_index, offset - start] = min(prefix_length, 255)
            if _is_alpha(value):
                prefix_hash = _advance(prefix_hash, value)
                prefix_length += 1
            else:
                prefix_hash = _HASH_SEED
                prefix_length = 0
    return observed, total, lengths


def _length_calibrated_score(
    base: np.ndarray,
    observed: np.ndarray,
    total: np.ndarray,
    lengths: np.ndarray,
    weights: dict[int, float],
) -> dict:
    empirical = observed / np.maximum(total, 1.0)
    selected_weight = np.zeros_like(base)
    for length_bucket, weight in weights.items():
        mask = np.minimum(lengths, 8) == length_bucket
        selected_weight[mask] = weight
    probability = (
        (1.0 - selected_weight) * base
        + selected_weight * empirical
    )
    nll = float(-np.log(probability.clip(1e-30)).mean())
    return {
        "nll": nll,
        "bpb": nll / math.log(2.0),
        "matched_fraction": float((total > 0).mean()),
    }


def _score(
    base: np.ndarray,
    observed: np.ndarray,
    total: np.ndarray,
    strength: float,
) -> dict:
    probability = (observed + strength * base) / (total + strength)
    nll = float(-np.log(probability.clip(1e-30)).mean())
    matched = total > 0
    return {
        "strength": float(strength),
        "nll": nll,
        "bpb": nll / math.log(2.0),
        "matched_fraction": float(matched.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seq-len", type=int, default=544)
    parser.add_argument("--fit-bytes", type=int, default=8_000_000)
    parser.add_argument("--max-words", type=int, default=1_000_000)
    parser.add_argument("--max-states", type=int, default=4_000_000)
    parser.add_argument("--chunk-bytes", type=int, default=24_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--strengths",
        default="1,2,4,8,16,32,64,128,256,512,1024",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    train = Path(args.train).read_bytes()
    if args.fit_bytes <= 0 or args.fit_bytes >= len(train):
        raise ValueError("fit-bytes must reserve a proper training suffix")
    trie_train = train[: -args.fit_bytes]
    fit_payload = train[-args.fit_bytes :]
    eval_payload = Path(args.eval).read_bytes()
    started = time.perf_counter()
    joint, totals, trie_summary = _build_trie(
        trie_train,
        max_words=args.max_words,
        max_states=args.max_states,
        chunk_bytes=args.chunk_bytes,
    )
    model, manifest = load_count_cake_bundle(
        args.bundle, device=torch.device(args.device)
    )
    model.eval()
    fit_rows = _rows(fit_payload, args.seq_len)
    fit_base = _base_probability(model, fit_rows, args.batch_size)
    fit_observed, fit_totals, fit_lengths = _trie_statistics(
        fit_rows,
        start=model.prediction_start,
        joint=joint,
        totals=totals,
    )
    strengths = tuple(float(value) for value in args.strengths.split(","))
    fit_scores = [
        _score(fit_base, fit_observed, fit_totals, strength)
        for strength in strengths
    ]
    selected = min(fit_scores, key=lambda item: item["bpb"])
    weight_grid = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
    length_weights = {}
    length_fit_bpb = {}
    fit_empirical = fit_observed / np.maximum(fit_totals, 1.0)
    for length_bucket in range(1, 9):
        mask = (np.minimum(fit_lengths, 8) == length_bucket) & (fit_totals > 0)
        if not bool(mask.any()):
            length_weights[length_bucket] = 0.0
            length_fit_bpb[length_bucket] = None
            continue
        candidates = []
        for weight in weight_grid:
            probability = (
                (1.0 - weight) * fit_base[mask]
                + weight * fit_empirical[mask]
            )
            candidates.append(
                (
                    float(-np.log(probability.clip(1e-30)).mean()),
                    weight,
                )
            )
        best_nll, best_weight = min(candidates)
        length_weights[length_bucket] = best_weight
        length_fit_bpb[length_bucket] = best_nll / math.log(2.0)
    eval_rows = _rows(eval_payload, args.seq_len)
    eval_base = _base_probability(model, eval_rows, args.batch_size)
    eval_observed, eval_totals, eval_lengths = _trie_statistics(
        eval_rows,
        start=model.prediction_start,
        joint=joint,
        totals=totals,
    )
    base_nll = float(-np.log(eval_base.clip(1e-30)).mean())
    evaluation = _score(
        eval_base,
        eval_observed,
        eval_totals,
        selected["strength"],
    )
    calibrated_evaluation = _length_calibrated_score(
        eval_base,
        eval_observed,
        eval_totals,
        eval_lengths,
        length_weights,
    )
    report = {
        "format": "layercake-delimiter-trie-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection probe; trie states are additive and not yet parameter-reallocated release evidence",
        "bundle": {
            "path": args.bundle,
            "logical_parameters": manifest["parameters"]["logical_total"],
        },
        "trie": trie_summary,
        "logical_parameters_with_additive_probe": int(
            manifest["parameters"]["logical_total"] + len(joint)
        ),
        "fit_protocol": {
            "trie_train_bytes": len(trie_train),
            "strength_fit_bytes": len(fit_payload),
            "evaluation_used_for_selection": False,
            "scores": fit_scores,
            "selected_strength": selected["strength"],
            "length_weights": length_weights,
            "length_fit_bpb": length_fit_bpb,
        },
        "evaluation": {
            "path": args.eval,
            "sha256": hashlib.sha256(eval_payload).hexdigest(),
            "bytes": int(eval_base.size),
            "base_bpb": base_nll / math.log(2.0),
            "trie_bpb": evaluation["bpb"],
            "delta_bpb": evaluation["bpb"] - base_nll / math.log(2.0),
            "matched_fraction": evaluation["matched_fraction"],
            "length_calibrated_bpb": calibrated_evaluation["bpb"],
            "length_calibrated_delta_bpb": calibrated_evaluation["bpb"]
            - base_nll / math.log(2.0),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
