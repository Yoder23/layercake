"""Screen a deterministic byte-lexeme sparse cake.

ASCII letter runs and digit runs are maximal lexemes; every other raw byte is
its own lexeme.  This segmentation is fixed and vocabulary-independent.  The
model remains lossless through an explicit unknown-lexeme escape followed by a
normalized geometric length code and uniform raw-byte payload.  This probe is
for architecture selection only and does not implement generation kernels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time


LEXEME = re.compile(rb"[A-Za-z]+|[0-9]+|[^A-Za-z0-9]")
UNKNOWN = b"\x00<layercake-raw>\x00"


def _iter_lexemes(payload: bytes):
    for match in LEXEME.finditer(payload):
        yield match.group(0)


def _raw_escape_bits(lexeme: bytes) -> float:
    # P(length=n)=2^-n for n>=1, followed by n uniform bytes.
    return 9.0 * len(lexeme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--vocabulary", type=int, default=100_000)
    parser.add_argument("--transition-budget", type=int, default=24_000_000)
    parser.add_argument("--max-context-order", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-train-bytes", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    train_payload = Path(args.train).read_bytes()
    if args.max_train_bytes > 0:
        train_payload = train_payload[: args.max_train_bytes]
    eval_payload = Path(args.eval).read_bytes()

    unigram_full = Counter(_iter_lexemes(train_payload))
    vocabulary = {
        token
        for token, _ in unigram_full.most_common(max(1, args.vocabulary - 1))
    }
    vocabulary.add(UNKNOWN)
    unigram = Counter()
    bigrams = Counter()
    trigrams = Counter()
    previous2 = None
    previous = None
    for token in _iter_lexemes(train_payload):
        mapped = token if token in vocabulary else UNKNOWN
        unigram[mapped] += 1
        if previous is not None:
            bigrams[(previous, mapped)] += 1
        if args.max_context_order >= 2 and previous2 is not None:
            trigrams[(previous2, previous, mapped)] += 1
        previous2 = previous
        previous = mapped
    if args.max_context_order >= 2:
        bigram_budget = min(len(bigrams), max(1, args.transition_budget // 4))
        trigram_budget = max(0, args.transition_budget - bigram_budget)
    else:
        bigram_budget = args.transition_budget
        trigram_budget = 0
    if len(bigrams) > bigram_budget:
        bigrams = Counter(dict(bigrams.most_common(bigram_budget)))
    if len(trigrams) > trigram_budget:
        trigrams = Counter(dict(trigrams.most_common(trigram_budget)))
    bigram_totals = Counter()
    for (context, _), count in bigrams.items():
        bigram_totals[context] += count
    trigram_totals = Counter()
    for (context2, context1, _), count in trigrams.items():
        trigram_totals[(context2, context1)] += count
    unigram_total = sum(unigram.values())
    unigram_probability = {
        token: count / unigram_total for token, count in unigram.items()
    }

    reports = []
    strengths = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
    recipes = (
        [(bigram_strength, None) for bigram_strength in strengths]
        if args.max_context_order == 1
        else [
            (bigram_strength, trigram_strength)
            for bigram_strength in strengths
            for trigram_strength in strengths
        ]
    )
    for bigram_strength, trigram_strength in recipes:
        bits = 0.0
        lexemes = 0
        unknown_lexemes = 0
        unknown_bytes = 0
        previous2 = None
        previous = None
        for token in _iter_lexemes(eval_payload):
            mapped = token if token in vocabulary else UNKNOWN
            base = unigram_probability.get(mapped, 1e-30)
            if previous is None:
                probability = base
            else:
                probability = (
                    bigrams.get((previous, mapped), 0) + bigram_strength * base
                ) / (bigram_totals.get(previous, 0) + bigram_strength)
            if trigram_strength is not None and previous2 is not None:
                probability = (
                    trigrams.get((previous2, previous, mapped), 0)
                    + trigram_strength * probability
                ) / (
                    trigram_totals.get((previous2, previous), 0)
                    + trigram_strength
                )
            bits -= math.log2(max(probability, 1e-30))
            if mapped == UNKNOWN:
                bits += _raw_escape_bits(token)
                unknown_lexemes += 1
                unknown_bytes += len(token)
            lexemes += 1
            previous2 = previous
            previous = mapped
        reports.append(
            {
                "bigram_strength": bigram_strength,
                "trigram_strength": trigram_strength,
                "bpb": bits / len(eval_payload),
                "total_bits": bits,
                "lexemes": lexemes,
                "unknown_lexemes": unknown_lexemes,
                "unknown_bytes": unknown_bytes,
                "unknown_byte_fraction": unknown_bytes / len(eval_payload),
            }
        )
    reports.sort(key=lambda item: item["bpb"])
    report = {
        "format": "layercake-byte-lexeme-cake-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection probe; recipe selected on validation",
        "segmentation": (
            "maximal ASCII letter runs, maximal digit runs, and singleton other bytes"
        ),
        "lossless_unknown_escape": {
            "length_probability": "P(n)=2^-n for n>=1",
            "payload_probability": "uniform 8-bit bytes",
            "charged_bits_per_unknown_byte": 9.0,
        },
        "parameters": {
            "vocabulary_states": len(vocabulary),
            "retained_bigram_states": len(bigrams),
            "retained_trigram_states": len(trigrams),
            "retained_transition_states": len(bigrams) + len(trigrams),
            "logical_total": len(vocabulary) + len(bigrams) + len(trigrams),
        },
        "train": {
            "path": args.train,
            "bytes": len(train_payload),
            "sha256": hashlib.sha256(train_payload).hexdigest(),
        },
        "eval": {
            "path": args.eval,
            "bytes": len(eval_payload),
            "sha256": hashlib.sha256(eval_payload).hexdigest(),
        },
        "best": reports[0],
        "results": reports,
        "timing": {"end_to_end_seconds": time.perf_counter() - started},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
