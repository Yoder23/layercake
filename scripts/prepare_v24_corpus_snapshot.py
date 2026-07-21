"""Materialize byte-identical train/eval snapshots for fair model comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_bpe_transformer_from_config import _load_bytes


def _snapshot(
    *,
    repo_root: Path,
    source: str,
    size: int,
    offset: int,
    destination: Path,
) -> dict:
    if size <= 0 or offset < 0:
        raise ValueError("snapshot size must be positive and offset non-negative")
    payload = _load_bytes(
        root=repo_root,
        data_roots=[source],
        include_suffixes={".jsonl", ".json", ".txt", ".md", ".csv", ".bin"},
        max_bytes=offset + size,
        read_block_bytes=1 << 20,
    )
    payload = payload[offset : offset + size]
    if len(payload) != size:
        raise ValueError(
            f"source produced {len(payload)} bytes after offset {offset}; "
            f"expected {size}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "path": str(destination.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": source,
        "source_payload_offset": offset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--eval-source", required=True)
    parser.add_argument("--test-source", default=None)
    parser.add_argument("--train-bytes", type=int, default=24_000_000)
    parser.add_argument("--eval-bytes", type=int, default=1_000_000)
    parser.add_argument("--test-bytes", type=int, default=1_000_000)
    parser.add_argument("--test-offset", type=int, default=12_000_000)
    parser.add_argument("--test2-offset", type=int, default=20_000_000)
    parser.add_argument("--test3-offset", type=int, default=23_000_000)
    parser.add_argument(
        "--out-dir",
        default="runs_experiment/production_v24_corpus",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    train = _snapshot(
        repo_root=repo_root,
        source=args.train_source,
        size=args.train_bytes,
        offset=0,
        destination=out_dir / "train.bin",
    )
    evaluation = _snapshot(
        repo_root=repo_root,
        source=args.eval_source,
        size=args.eval_bytes,
        offset=0,
        destination=out_dir / "eval.bin",
    )
    test = _snapshot(
        repo_root=repo_root,
        source=args.test_source or args.eval_source,
        size=args.test_bytes,
        offset=args.test_offset,
        destination=out_dir / "test.bin",
    )
    test2 = _snapshot(
        repo_root=repo_root,
        source=args.test_source or args.eval_source,
        size=args.test_bytes,
        offset=args.test2_offset,
        destination=out_dir / "test2.bin",
    )
    test3 = _snapshot(
        repo_root=repo_root,
        source=args.test_source or args.eval_source,
        size=args.test_bytes,
        offset=args.test3_offset,
        destination=out_dir / "test3.bin",
    )
    manifest = {
        "format": "layercake-frozen-corpus/1",
        "train": train,
        "evaluation": evaluation,
        "test": test,
        "test2": test2,
        "test3": test3,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
