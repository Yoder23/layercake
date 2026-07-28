from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys

from transformers import GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.training.data import sha256_file
from scripts.build_phase4_lexical_conformance import (
    _TEMPLATES,
    _eligible_units,
)
from scripts.train_phase4_semantic_action_plan import CHECKPOINT


ROOT = Path(__file__).resolve().parents[1]
PARENT_DATASET = (
    ROOT / "data/moonshot/phase4/lexical_conformance_v1.jsonl"
)
PARENT_DATASET_SHA256 = (
    "d6a7c054c1104c38007c97263031576c10c2a4dc7f78cf494a4f848a362e38bb"
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_fresh_rows(
    tokenizer: GPT2TokenizerFast,
    *,
    count: int,
    leading: list[tuple[int, str]],
    interior: list[tuple[int, str]],
    excluded_target_ids: set[int],
    row_offset: int,
) -> tuple[list[dict[str, object]], set[int]]:
    rows: list[dict[str, object]] = []
    fresh_ids: set[int] = set()
    attempts = 0
    while len(rows) < count:
        if attempts >= max(len(leading), len(interior)) * 8:
            raise RuntimeError(
                f"could not construct {count} fresh lexical rows"
            )
        leading_id, leading_text = leading[attempts % len(leading)]
        interior_id, interior_text = interior[
            (attempts * 7919 + 17) % len(interior)
        ]
        attempts += 1
        if (
            leading_id in excluded_target_ids
            or interior_id in excluded_target_ids
            or leading_id in fresh_ids
            or interior_id in fresh_ids
            or leading_id == interior_id
        ):
            continue
        nonce = row_offset + len(rows)
        name = f"{leading_text}_{interior_text}_{nonce:05d}"
        if len(name) > 64 or _IDENTIFIER.fullmatch(name) is None:
            continue
        encoded = tokenizer.encode(" " + name)
        if leading_id not in encoded or interior_id not in encoded:
            continue
        fresh_ids.update((leading_id, interior_id))
        prompt = _TEMPLATES[nonce % len(_TEMPLATES)].format(name=name)
        rows.append(
            {
                "id": f"lexical-fresh-probe-{len(rows):05d}",
                "split": "fresh_probe",
                "family": "lexical_identity",
                "prompt": prompt,
                "response": (
                    f"def {name}(value):\n"
                    "    return value\n"
                ),
                "function_name": name,
                "target_token_ids": [leading_id, interior_id],
                "tests": [
                    {"args": [7], "expected": 7},
                    {"args": ["cake"], "expected": "cake"},
                ],
            }
        )
    return rows, fresh_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=10940)
    parser.add_argument("--row-offset", type=int, default=20000)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise RuntimeError(f"fresh probe is immutable: {output}")
    if sha256_file(PARENT_DATASET) != PARENT_DATASET_SHA256:
        raise RuntimeError("parent lexical dataset identity changed")
    tokenizer = GPT2TokenizerFast.from_pretrained(str(CHECKPOINT))
    leading, interior = _eligible_units(tokenizer)
    rng = random.Random(args.seed)
    rng.shuffle(leading)
    rng.shuffle(interior)
    parent_rows = _load_rows(PARENT_DATASET)
    excluded = {
        int(token_id)
        for row in parent_rows
        for token_id in row["target_token_ids"]
    }
    rows, fresh_ids = _build_fresh_rows(
        tokenizer,
        count=args.rows,
        leading=leading,
        interior=interior,
        excluded_target_ids=excluded,
        row_offset=args.row_offset,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "format": "layercake-phase4-lexical-fresh-probe/1",
        "status": "LOCKED_FRESH_PROMOTION_SCREEN_NOT_ACCESSED",
        "source": "frozen declared GPT-2 tokenizer vocabulary",
        "seed": args.seed,
        "dataset": output.relative_to(ROOT).as_posix(),
        "dataset_sha256": sha256_file(output),
        "rows": len(rows),
        "split": "fresh_probe",
        "prompt_templates": len(_TEMPLATES),
        "row_offset": args.row_offset,
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
        "tokenizer_json_sha256": sha256_file(
            CHECKPOINT / "tokenizer.json"
        ),
        "parent_dataset": PARENT_DATASET.relative_to(ROOT).as_posix(),
        "parent_dataset_sha256": PARENT_DATASET_SHA256,
        "parent_target_token_ids": len(excluded),
        "fresh_target_token_ids": len(fresh_ids),
        "fresh_target_ids_unique": len(fresh_ids) == 2 * len(rows),
        "fresh_target_ids_disjoint_from_parent": not bool(
            fresh_ids & excluded
        ),
        "parent_target_token_ids_sha256": _canonical_sha(
            sorted(excluded)
        ),
        "fresh_target_token_ids_sha256": _canonical_sha(
            sorted(fresh_ids)
        ),
        "calibration_rows_used_to_construct_probe": 0,
        "model_outputs_accessed": 0,
        "promotion_result_accessed": False,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
