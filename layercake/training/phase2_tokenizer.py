"""Build a compact word-preserving extension of the locked Phase 1 BPE.

The first 384 token IDs remain bit-for-bit identical to the locked reference tokenizer,
so trained LayerCake blocks can be transferred.  Additional hierarchical merges encode
frequent complete English word forms and reduce both fragment corruption and decode work.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from layercake.models.baseline_transformer import BytePairTokenizer
from layercake.models.phase2_english_planner import realize_english
from layercake.training.data import sha256_file


ROOT = Path(__file__).resolve().parents[2]


def build_extended_tokenizer(
    root: Path, *, base_path: Path, corpus_path: Path, output_path: Path,
    total_merges: int, training_bytes: int,
) -> dict:
    base_path = (root / base_path).resolve()
    corpus_path = (root / corpus_path).resolve()
    output_path = (root / output_path).resolve()
    output_path.relative_to(root.resolve())
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise RuntimeError(f"extended tokenizer artifact is immutable: {output_path}")
    base_document = json.loads(base_path.read_text(encoding="utf-8"))
    base_merges = [tuple(pair) for pair in base_document["merges"]]
    if total_merges <= len(base_merges):
        raise ValueError("extended tokenizer must add merges beyond the locked base")
    corpus = corpus_path.read_bytes()[:training_bytes]
    forms = re.findall(rb" [A-Za-z]{2,}|[A-Za-z]{3,}", corpus)
    counts = Counter(forms)
    candidates = sorted(counts, key=lambda value: (-counts[value], value))
    merges = list(base_merges)
    pair_ids = {pair: index for index, pair in enumerate(merges, start=256)}
    fully_encoded = 0
    considered = 0
    for word in candidates:
        if len(merges) >= total_merges:
            break
        considered += 1
        tokenizer = BytePairTokenizer(merges)
        sequence = tokenizer.encode(word)
        while len(sequence) > 1 and len(merges) < total_merges:
            pair = (sequence[0], sequence[1])
            new_id = pair_ids.get(pair)
            if new_id is None:
                new_id = 256 + len(merges)
                merges.append(pair)
                pair_ids[pair] = new_id
            sequence = [new_id, *sequence[2:]]
        if len(sequence) == 1:
            fully_encoded += 1
    tokenizer = BytePairTokenizer(merges)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tokenizer.canonical_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sample = corpus[: min(len(corpus), 200_000)]
    encoded = tokenizer.encode(sample)
    if tokenizer.decode(encoded) != sample:
        raise RuntimeError("extended tokenizer failed exact byte round-trip")
    base = BytePairTokenizer(base_merges)
    manifest = {
        "format": "layercake-phase2-word-preserving-bpe/1",
        "status": "PASS",
        "tokenizer_path": output_path.relative_to(root).as_posix(),
        "tokenizer_sha256": sha256_file(output_path),
        "base_tokenizer_path": base_path.relative_to(root).as_posix(),
        "base_tokenizer_sha256": sha256_file(base_path),
        "base_merges_preserved_as_exact_prefix": merges[:len(base_merges)] == base_merges,
        "base_vocab_size": base.vocab_size,
        "vocab_size": tokenizer.vocab_size,
        "merges": len(merges),
        "training_corpus_path": corpus_path.relative_to(root).as_posix(),
        "training_corpus_sha256": sha256_file(corpus_path),
        "training_bytes": len(corpus),
        "candidate_word_forms_considered": considered,
        "candidate_word_forms_fully_encoded": fully_encoded,
        "sample_raw_bytes": len(sample),
        "sample_tokens": len(encoded),
        "sample_bytes_per_token": len(sample) / len(encoded),
        "sample_sha256": hashlib.sha256(sample).hexdigest(),
        "round_trip_exact": True,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_planner_tokenizer(
    root: Path, *, base_path: Path, curriculum_path: Path, output_path: Path,
    extension_merges: int,
) -> dict:
    """Extend an immutable BPE prefix with generic grammatical phrase pieces."""

    base_path = (root / base_path).resolve()
    curriculum_path = (root / curriculum_path).resolve()
    output_path = (root / output_path).resolve()
    output_path.relative_to(root.resolve())
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise RuntimeError(f"planner tokenizer artifact is immutable: {output_path}")
    base_document = json.loads(base_path.read_text(encoding="utf-8"))
    base_merges = [tuple(pair) for pair in base_document["merges"]]
    rows = [json.loads(line) for line in curriculum_path.read_text(encoding="utf-8").splitlines()]
    topics = []
    for row in rows:
        topic = str(row["topic"])
        if topic != "disjoint recall" and topic not in topics:
            topics.append(topic)
        if len(topics) == 2:
            break
    prompts = [row["prompt"] for row in rows if row["topic"] in topics]
    planner_payloads = [
        realize_english(prompt, variant=variant, sustained=sustained).encode("utf-8")
        for prompt in prompts
        for variant in range(4)
        for sustained in (False, True)
    ]
    tokenizer = BytePairTokenizer(base_merges)
    sequences = [tokenizer.encode(payload) for payload in planner_payloads]
    merges = list(base_merges)
    for _ in range(extension_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for sequence in sequences:
            counts.update(zip(sequence, sequence[1:]))
        if not counts:
            break
        pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
        if frequency < 2:
            break
        new_id = 256 + len(merges)
        merges.append(pair)
        replaced_sequences = []
        for sequence in sequences:
            replaced = []
            index = 0
            while index < len(sequence):
                if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                    replaced.append(new_id)
                    index += 2
                else:
                    replaced.append(sequence[index])
                    index += 1
            replaced_sequences.append(replaced)
        sequences = replaced_sequences
    tokenizer = BytePairTokenizer(merges)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tokenizer.canonical_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sample = b"\n".join(planner_payloads)
    encoded = tokenizer.encode(sample)
    if tokenizer.decode(encoded) != sample:
        raise RuntimeError("planner tokenizer failed exact byte round-trip")
    manifest = {
        "format": "layercake-phase2-planner-preserving-bpe/1",
        "status": "PASS",
        "tokenizer_path": output_path.relative_to(root).as_posix(),
        "tokenizer_sha256": sha256_file(output_path),
        "base_tokenizer_path": base_path.relative_to(root).as_posix(),
        "base_tokenizer_sha256": sha256_file(base_path),
        "base_merges_preserved_as_exact_prefix": merges[:len(base_merges)] == base_merges,
        "base_vocab_size": 256 + len(base_merges),
        "vocab_size": tokenizer.vocab_size,
        "extension_merges_requested": extension_merges,
        "extension_merges_learned": len(merges) - len(base_merges),
        "curriculum_path": curriculum_path.relative_to(root).as_posix(),
        "curriculum_sha256": sha256_file(curriculum_path),
        "curriculum_topics_used": topics,
        "frozen_evaluation_content": False,
        "planner_payloads": len(planner_payloads),
        "sample_raw_bytes": len(sample),
        "sample_tokens": len(encoded),
        "sample_bytes_per_token": len(sample) / len(encoded),
        "sample_sha256": hashlib.sha256(sample).hexdigest(),
        "round_trip_exact": True,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m layercake.training.phase2_tokenizer")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--base", type=Path,
        default=Path("artifacts/final/medium-transformers/seed-9801/tokenizer.json"),
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("data/moonshot/v2/wikitext103/train_medium.bin"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/moonshot/phase2/word_preserving_bpe_2304.json"),
    )
    parser.add_argument("--merges", type=int, default=2048)
    parser.add_argument("--training-bytes", type=int, default=10_000_000)
    parser.add_argument("--planner-extension-merges", type=int, default=0)
    parser.add_argument(
        "--curriculum", type=Path,
        default=Path("data/moonshot/phase2/instruction_curriculum_clean.jsonl"),
    )
    args = parser.parse_args()
    if args.planner_extension_merges:
        result = build_planner_tokenizer(
            args.root.resolve(), base_path=args.base, curriculum_path=args.curriculum,
            output_path=args.output, extension_merges=args.planner_extension_merges,
        )
    else:
        result = build_extended_tokenizer(
            args.root.resolve(), base_path=args.base, corpus_path=args.corpus,
            output_path=args.output, total_merges=args.merges,
            training_bytes=args.training_bytes,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
