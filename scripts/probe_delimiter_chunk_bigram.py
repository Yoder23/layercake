"""Bounded byte-span phrase-cake quality estimator.

The probe segments bytes only at deterministic character-class boundaries,
stores frequent spans losslessly, and counts span transitions.  It estimates
whether a production byte-trie/phrase cake is worth implementing; it is not a
release benchmark or final evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CHUNK_PATTERN = re.compile(
    rb"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[ \t\r\n]+|[^A-Za-z0-9 \t\r\n]+"
)


def _chunks(payload: bytes):
    for match in CHUNK_PATTERN.finditer(payload):
        yield match.group(0)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evaluate(
    payload: bytes,
    *,
    vocabulary: dict[bytes, int],
    unigram_probability: np.ndarray,
    bigram_counts: dict[int, int],
    bigram_totals: np.ndarray,
    trigram_counts: dict[int, int],
    trigram_totals: dict[int, int],
    fourgram_counts: dict[int, int],
    fourgram_totals: dict[int, int],
    unknown_byte_probability: np.ndarray,
    strengths: tuple[float, ...],
) -> list[dict]:
    reports = [
        {"strength": strength, "nll": 0.0, "chunks": 0, "unknown_chunks": 0}
        for strength in strengths
    ]
    previous3 = 0
    previous2 = 0
    previous = 0
    for chunk in _chunks(payload):
        target = vocabulary.get(chunk, 0)
        key = previous * unigram_probability.size + target
        count = bigram_counts.get(key, 0)
        total = float(bigram_totals[previous])
        trigram_context = previous2 * unigram_probability.size + previous
        trigram_key = trigram_context * unigram_probability.size + target
        trigram_count = trigram_counts.get(trigram_key, 0)
        trigram_total = float(trigram_totals.get(trigram_context, 0))
        fourgram_context = (
            (previous3 * unigram_probability.size + previous2)
            * unigram_probability.size
            + previous
        )
        fourgram_key = fourgram_context * unigram_probability.size + target
        fourgram_count = fourgram_counts.get(fourgram_key, 0)
        fourgram_total = float(fourgram_totals.get(fourgram_context, 0))
        unknown_nll = 0.0
        if target == 0:
            values = np.frombuffer(chunk, dtype=np.uint8)
            unknown_nll = float(-np.log(unknown_byte_probability[values]).sum())
            unknown_nll -= math.log(float(unknown_byte_probability[256]))
        for report in reports:
            strength = report["strength"]
            bigram_probability = (
                count + strength * float(unigram_probability[target])
            ) / (total + strength)
            trigram_probability = (
                trigram_count + strength * bigram_probability
            ) / (trigram_total + strength)
            probability = (
                fourgram_count + strength * trigram_probability
            ) / (fourgram_total + strength)
            report["nll"] += -math.log(max(probability, 1e-30)) + unknown_nll
            report["chunks"] += 1
            report["unknown_chunks"] += int(target == 0)
        previous3, previous2, previous = previous2, previous, target
    for report in reports:
        report["bytes"] = len(payload)
        report["bpb"] = report.pop("nll") / len(payload) / math.log(2.0)
        report["unknown_chunk_fraction"] = (
            report["unknown_chunks"] / max(report["chunks"], 1)
        )
    return reports


def _phrase_events(
    payload: bytes,
    *,
    vocabulary: dict[bytes, int],
    unigram_probability: np.ndarray,
    bigram_counts: dict[int, int],
    bigram_totals: np.ndarray,
    trigram_counts: dict[int, int],
    trigram_totals: dict[int, int],
    fourgram_counts: dict[int, int],
    fourgram_totals: dict[int, int],
    unknown_byte_probability: np.ndarray,
    strength: float,
) -> list[tuple[int, int, float]]:
    events = []
    previous3 = previous2 = previous = 0
    size = unigram_probability.size
    for match in CHUNK_PATTERN.finditer(payload):
        chunk = match.group(0)
        target = vocabulary.get(chunk, 0)
        bigram_key = previous * size + target
        bigram_probability = (
            bigram_counts.get(bigram_key, 0)
            + strength * float(unigram_probability[target])
        ) / (float(bigram_totals[previous]) + strength)
        trigram_context = previous2 * size + previous
        trigram_probability = (
            trigram_counts.get(trigram_context * size + target, 0)
            + strength * bigram_probability
        ) / (float(trigram_totals.get(trigram_context, 0)) + strength)
        fourgram_context = (previous3 * size + previous2) * size + previous
        probability = (
            fourgram_counts.get(fourgram_context * size + target, 0)
            + strength * trigram_probability
        ) / (float(fourgram_totals.get(fourgram_context, 0)) + strength)
        nll = -math.log(max(probability, 1e-30))
        if target == 0:
            values = np.frombuffer(chunk, dtype=np.uint8)
            nll -= float(np.log(unknown_byte_probability[values]).sum())
            nll -= math.log(float(unknown_byte_probability[256]))
        events.append((match.start(), match.end(), nll))
        previous3, previous2, previous = previous2, previous, target
    return events


def _count_phrase_oracle(
    payload: bytes,
    events: list[tuple[int, int, float]],
    *,
    bundle: str,
    seq_len: int,
) -> dict:
    import torch
    from layercake.count_cake import load_count_cake_bundle

    model, _ = load_count_cake_bundle(bundle, device="cuda")
    model.eval()
    row_count = len(payload) // seq_len
    scored_nll = np.full(row_count * seq_len, np.nan, dtype=np.float32)
    rows = torch.from_numpy(
        np.frombuffer(payload[: row_count * seq_len], dtype=np.uint8)
        .reshape(row_count, seq_len)
        .copy()
    )
    with torch.inference_mode():
        for row_start in range(0, row_count, 32):
            batch = rows[row_start : row_start + 32].to(
                device="cuda", dtype=torch.long
            )
            nll = -model.target_log_probs(batch).cpu().numpy()
            for local_index in range(batch.shape[0]):
                absolute = (row_start + local_index) * seq_len
                scored_nll[
                    absolute + model.prediction_start : absolute + seq_len
                ] = nll[local_index]
    count_total = phrase_total = oracle_total = 0.0
    scored_bytes = 0
    phrase_wins = 0
    events_scored = 0
    for start, stop, phrase_nll in events:
        if stop > scored_nll.size:
            break
        values = scored_nll[start:stop]
        if values.size == 0 or not np.isfinite(values).all():
            continue
        count_nll = float(values.sum())
        count_total += count_nll
        phrase_total += phrase_nll
        oracle_total += min(count_nll, phrase_nll)
        phrase_wins += int(phrase_nll < count_nll)
        scored_bytes += stop - start
        events_scored += 1
    scale = max(scored_bytes, 1) * math.log(2.0)
    return {
        "bundle": bundle,
        "seq_len": seq_len,
        "scored_bytes": scored_bytes,
        "scored_chunks": events_scored,
        "countcake_bpb": count_total / scale,
        "phrase_bpb": phrase_total / scale,
        "target_aware_chunk_oracle_bpb": oracle_total / scale,
        "target_aware_phrase_win_fraction": phrase_wins / max(events_scored, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--max-train-bytes", type=int, default=100_000_000)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--target-state", type=int, default=24_935_904)
    parser.add_argument("--max-order", type=int, choices=(3, 4), default=4)
    parser.add_argument("--count-bundle")
    parser.add_argument("--count-seq-len", type=int, default=1056)
    parser.add_argument("--oracle-strength", type=float, default=300.0)
    parser.add_argument("--strengths", default="10,30,100,300,1000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    train = Path(args.train).read_bytes()[: args.max_train_bytes]
    eval_payloads = [(path, Path(path).read_bytes()) for path in args.eval]

    unigram = Counter(_chunks(train))
    retained = unigram.most_common(max(args.vocab_size - 1, 1))
    vocabulary = {chunk: index for index, (chunk, _) in enumerate(retained, 1)}
    vocabulary_size = len(vocabulary) + 1
    token_counts = np.full(vocabulary_size, 0.5, dtype=np.float64)
    unknown_bytes = np.full(257, 0.5, dtype=np.float64)
    raw_bigram = Counter()
    raw_trigram = Counter()
    raw_fourgram = Counter()
    previous3 = 0
    previous2 = 0
    previous = 0
    for chunk in _chunks(train):
        target = vocabulary.get(chunk, 0)
        token_counts[target] += 1.0
        raw_bigram[previous * vocabulary_size + target] += 1
        trigram_context = previous2 * vocabulary_size + previous
        raw_trigram[trigram_context * vocabulary_size + target] += 1
        fourgram_context = (
            (previous3 * vocabulary_size + previous2) * vocabulary_size
            + previous
        )
        if args.max_order >= 4:
            raw_fourgram[fourgram_context * vocabulary_size + target] += 1
        if target == 0:
            unknown_bytes[:256] += np.bincount(
                np.frombuffer(chunk, dtype=np.uint8), minlength=256
            )
            unknown_bytes[256] += 1.0
        previous3, previous2, previous = previous2, previous, target

    lexicon_bytes = sum(len(chunk) for chunk in vocabulary)
    fixed_state = lexicon_bytes + vocabulary_size + 257
    # Bigram state is compact and forms the normalized backoff base.  Give the
    # remaining exact budget to two-span contexts.
    retained_bigrams = raw_bigram.most_common(
        max(args.target_state - fixed_state, 0)
    )
    bigram_counts = dict(retained_bigrams)
    bigram_totals = np.zeros(vocabulary_size, dtype=np.float64)
    for key, count in retained_bigrams:
        bigram_totals[key // vocabulary_size] += count
    trigram_budget = max(
        args.target_state - fixed_state - len(retained_bigrams), 0
    )
    retained_trigrams = raw_trigram.most_common(trigram_budget)
    trigram_counts = dict(retained_trigrams)
    trigram_totals: dict[int, int] = {}
    for key, count in retained_trigrams:
        context = key // vocabulary_size
        trigram_totals[context] = trigram_totals.get(context, 0) + count
    fourgram_budget = max(
        args.target_state
        - fixed_state
        - len(retained_bigrams)
        - len(retained_trigrams),
        0,
    )
    retained_fourgrams = raw_fourgram.most_common(fourgram_budget)
    fourgram_counts = dict(retained_fourgrams)
    fourgram_totals: dict[int, int] = {}
    for key, count in retained_fourgrams:
        context = key // vocabulary_size
        fourgram_totals[context] = fourgram_totals.get(context, 0) + count
    unigram_probability = token_counts / token_counts.sum()
    unknown_byte_probability = unknown_bytes / unknown_bytes.sum()
    strengths = tuple(float(value) for value in args.strengths.split(","))
    evaluations = []
    for path, payload in eval_payloads:
        evaluation = {
                "path": path,
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "strengths": _evaluate(
                    payload,
                    vocabulary=vocabulary,
                    unigram_probability=unigram_probability,
                    bigram_counts=bigram_counts,
                    bigram_totals=bigram_totals,
                    trigram_counts=trigram_counts,
                    trigram_totals=trigram_totals,
                    fourgram_counts=fourgram_counts,
                    fourgram_totals=fourgram_totals,
                    unknown_byte_probability=unknown_byte_probability,
                    strengths=strengths,
                ),
            }
        if args.count_bundle:
            events = _phrase_events(
                payload,
                vocabulary=vocabulary,
                unigram_probability=unigram_probability,
                bigram_counts=bigram_counts,
                bigram_totals=bigram_totals,
                trigram_counts=trigram_counts,
                trigram_totals=trigram_totals,
                fourgram_counts=fourgram_counts,
                fourgram_totals=fourgram_totals,
                unknown_byte_probability=unknown_byte_probability,
                strength=args.oracle_strength,
            )
            evaluation["count_phrase_oracle"] = _count_phrase_oracle(
                payload,
                events,
                bundle=args.count_bundle,
                seq_len=args.count_seq_len,
            )
        evaluations.append(evaluation)
    report = {
        "format": "layercake-delimiter-chunk-bigram-probe/1",
        "status": "COMPLETE",
        "warning": (
            "architecture-selection estimator; unknown-span boundary coding "
            "must be made exactly byte-normalized before production promotion"
        ),
        "train": {
            "path": args.train,
            "bytes": len(train),
            "sha256": _sha256(train),
        },
        "state": {
            "target": args.target_state,
            "logical_used": (
                fixed_state
                + len(retained_bigrams)
                + len(retained_trigrams)
                + len(retained_fourgrams)
            ),
            "lexicon_bytes": lexicon_bytes,
            "vocabulary_entries": vocabulary_size,
            "observed_bigram_entries": len(raw_bigram),
            "retained_bigram_entries": len(retained_bigrams),
            "observed_trigram_entries": len(raw_trigram),
            "retained_trigram_entries": len(retained_trigrams),
            "observed_fourgram_entries": len(raw_fourgram),
            "retained_fourgram_entries": len(retained_fourgrams),
        },
        "evaluation": evaluations,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
