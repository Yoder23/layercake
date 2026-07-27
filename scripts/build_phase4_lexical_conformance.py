from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from transformers import GPT2TokenizerFast

import _common
from layercake.training.data import sha256_file


ROOT = Path(__file__).resolve().parents[1]
_LEADING = re.compile(r" [A-Za-z][A-Za-z0-9_]*")
_INTERIOR = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TEMPLATES = (
    "Return only Python code. Define {name}(value) and return value unchanged.",
    "Write a Python identity function {name}(value). Output code only.",
    "Create {name}(value) in Python; the result must equal value.",
    "No prose. Implement {name}(value) as an identity function.",
    "Produce valid Python for {name}(value), returning its argument.",
    "Define the callable {name}(value) so it returns value unchanged.",
    "Code only: make {name}(value) return the supplied value.",
    "Implement a Python function {name}(value) that returns value.",
)


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _eligible_units(
    tokenizer: GPT2TokenizerFast,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    leading = []
    interior = []
    for token_id in range(tokenizer.vocab_size):
        text = tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        )
        if len(text) <= 20 and _LEADING.fullmatch(text):
            leading.append((token_id, text[1:]))
        if len(text) <= 20 and _INTERIOR.fullmatch(text):
            interior.append((token_id, text))
    return leading, interior


def _build_rows(
    tokenizer: GPT2TokenizerFast,
    *,
    split: str,
    count: int,
    leading: list[tuple[int, str]],
    interior: list[tuple[int, str]],
    used_target_ids: set[int],
    row_offset: int,
) -> list[dict[str, object]]:
    rows = []
    pair_index = 0
    attempts = 0
    while len(rows) < count:
        if attempts >= len(leading) * 3:
            raise RuntimeError(
                f"could not construct {count} lexical {split} rows"
            )
        leading_id, leading_text = leading[
            attempts % len(leading)
        ]
        interior_id, interior_text = interior[
            (attempts * 7919 + 17) % len(interior)
        ]
        attempts += 1
        if (
            leading_id in used_target_ids
            or interior_id in used_target_ids
            or leading_id == interior_id
        ):
            continue
        nonce = row_offset + pair_index
        name = f"{leading_text}_{interior_text}_{nonce:05d}"
        pair_index += 1
        if len(name) > 64 or _IDENTIFIER.fullmatch(name) is None:
            continue
        encoded = tokenizer.encode(" " + name)
        if leading_id not in encoded or interior_id not in encoded:
            continue
        used_target_ids.update((leading_id, interior_id))
        prompt = _TEMPLATES[nonce % len(_TEMPLATES)].format(name=name)
        rows.append(
            {
                "id": f"lexical-{split}-{len(rows):05d}",
                "split": split,
                "family": "lexical_identity",
                "prompt": prompt,
                "response": (
                    f"def {name}(value):\n"
                    "    return value\n"
                ),
                "function_name": name,
                "target_token_ids": [leading_id, interior_id],
                "tests": (
                    []
                    if split == "train"
                    else [
                        {"args": [7], "expected": 7},
                        {"args": ["cake"], "expected": "cake"},
                    ]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--probe-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9444)
    args = parser.parse_args()

    checkpoint = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else ROOT / args.checkpoint
    )
    output = (
        args.output if args.output.is_absolute() else ROOT / args.output
    )
    if output.exists():
        raise RuntimeError(f"dataset artifact is immutable: {output}")
    tokenizer = GPT2TokenizerFast.from_pretrained(str(checkpoint))
    leading, interior = _eligible_units(tokenizer)
    rng = random.Random(args.seed)
    rng.shuffle(leading)
    rng.shuffle(interior)
    train_target_ids: set[int] = set()
    train = _build_rows(
        tokenizer,
        split="train",
        count=args.train_rows,
        leading=leading,
        interior=interior,
        used_target_ids=train_target_ids,
        row_offset=0,
    )
    probe_target_ids: set[int] = set(train_target_ids)
    validation = _build_rows(
        tokenizer,
        split="validation",
        count=args.probe_rows,
        leading=leading,
        interior=interior,
        used_target_ids=probe_target_ids,
        row_offset=args.train_rows,
    )
    validation_ids = probe_target_ids - train_target_ids
    rows = train + validation
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "format": "layercake-phase4-lexical-conformance-dataset/1",
        "status": "SYNTHETIC_DIAGNOSTIC_NO_PROMOTION_CREDIT",
        "source": "frozen declared GPT-2 tokenizer vocabulary",
        "seed": args.seed,
        "dataset": output.relative_to(ROOT).as_posix(),
        "dataset_sha256": sha256_file(output),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "tokenizer_json_sha256": sha256_file(
            checkpoint / "tokenizer.json"
        ),
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": 0,
        },
        "prompt_templates": len(_TEMPLATES),
        "eligible_leading_token_units": len(leading),
        "eligible_interior_token_units": len(interior),
        "train_target_token_units": len(train_target_ids),
        "validation_target_token_units": len(validation_ids),
        "target_token_id_sets_disjoint": not bool(
            train_target_ids & validation_ids
        ),
        "train_target_token_ids_sha256": _canonical_sha(
            sorted(train_target_ids)
        ),
        "validation_target_token_ids_sha256": _canonical_sha(
            sorted(validation_ids)
        ),
        "functional_validation_rows_used": 0,
        "functional_test_rows_used": 0,
        "promotion_credit": 0,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
