"""Materialize and hash one predeclared, byte-exact sealed corpus split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_bpe_transformer_from_config import _load_bytes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--bytes", type=int, default=1_000_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--label", default="sealed")
    args = parser.parse_args()
    if args.offset < 0 or args.bytes <= 0:
        raise ValueError("offset must be non-negative and bytes must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    payload = _load_bytes(
        root=repo_root,
        data_roots=[args.source],
        include_suffixes={".jsonl", ".json", ".txt", ".md", ".csv", ".bin"},
        max_bytes=args.offset + args.bytes,
        read_block_bytes=1 << 20,
    )[args.offset : args.offset + args.bytes]
    if len(payload) != args.bytes:
        raise ValueError(
            f"source produced {len(payload)} bytes; expected {args.bytes}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    report = {
        "format": "layercake-sealed-split/1",
        "label": args.label,
        "state": "MATERIALIZED_UNOPENED",
        "source": args.source,
        "source_payload_offset": args.offset,
        "path": str(output),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
