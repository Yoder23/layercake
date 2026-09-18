"""Build the preregistered knowledge-light Phase 2 English substrate.

The output is deliberately a byte corpus rather than a token cache.  This
keeps raw-data exposure auditable and lets every representation account for
its own model-visible units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
TINY_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--roneneldan--TinyStories"
    / "snapshots"
    / "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    / "data"
    / "train-00000-of-00004-2d5a1467fff1081b.parquet"
)
WIKITEXT = ROOT / "data" / "moonshot" / "v2" / "wikitext103" / "train_medium.bin"
DEFAULT_OUTPUT = (
    ROOT / "data" / "moonshot" / "phase2" / "english_substrate_200m.bin"
)
TINY_BYTES = 150_000_000
WIKI_BYTES = 50_000_000
EXPECTED_TINY_SHA256 = (
    "77cf780cebe52b6e83e3a2ac84bc56d8059363113e41d17a023f1d8b2ed0fc0b"
)
EXPECTED_WIKI_SHA256 = (
    "ec54bd8fa09c2cf1a6d442538a98c62ce8e62de14378a19556310836891d23b6"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_utf8_prefix(data: bytes, limit: int) -> bytes:
    """Return the longest valid UTF-8 prefix no longer than ``limit``."""

    if len(data) <= limit:
        return data
    return data[:limit].decode("utf-8", errors="ignore").encode("utf-8")


def write_tiny_stories(handle, source: Path, budget: int) -> dict:
    written = 0
    records = 0
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(columns=["text"], batch_size=1024):
        for value in batch.column(0).to_pylist():
            record = (str(value).strip() + "\n\n").encode("utf-8")
            remaining = budget - written
            if remaining <= 0:
                break
            piece = valid_utf8_prefix(record, remaining)
            handle.write(piece)
            written += len(piece)
            records += 1
            if written == budget:
                break
        if written == budget:
            break
    if written < budget:
        # At most three bytes can remain after UTF-8-safe truncation.  ASCII
        # separators preserve validity and make the byte budget exact.
        padding = b"\n" * (budget - written)
        handle.write(padding)
        written += len(padding)
    if written != budget:
        raise RuntimeError(f"TinyStories byte budget mismatch: {written} != {budget}")
    return {"bytes": written, "records_touched": records}


def write_wikitext(handle, source: Path, budget: int) -> dict:
    data = source.read_bytes()
    if len(data) < budget:
        raise RuntimeError("WikiText source is smaller than the preregistered budget")
    piece = valid_utf8_prefix(data, budget)
    handle.write(piece)
    if len(piece) < budget:
        handle.write(b"\n" * (budget - len(piece)))
    return {"bytes": budget, "source_prefix_bytes": budget}


def build(output: Path) -> dict:
    output = output.resolve()
    output.relative_to(ROOT.resolve())
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise RuntimeError(f"English substrate is immutable: {output}")
    if sha256_file(TINY_SNAPSHOT) != EXPECTED_TINY_SHA256:
        raise RuntimeError("TinyStories source hash differs from preregistration")
    if sha256_file(WIKITEXT) != EXPECTED_WIKI_SHA256:
        raise RuntimeError("WikiText source hash differs from preregistration")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        tiny = write_tiny_stories(handle, TINY_SNAPSHOT, TINY_BYTES)
        wiki = write_wikitext(handle, WIKITEXT, WIKI_BYTES)
    if output.stat().st_size != TINY_BYTES + WIKI_BYTES:
        raise RuntimeError("combined substrate byte budget mismatch")
    manifest = {
        "format": "layercake-phase2-english-substrate/1",
        "status": "PASS",
        "output": output.relative_to(ROOT).as_posix(),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "components": {
            "tiny_stories": {
                **tiny,
                "source": str(TINY_SNAPSHOT),
                "source_bytes": TINY_SNAPSHOT.stat().st_size,
                "source_sha256": EXPECTED_TINY_SHA256,
                "purpose": "knowledge-light narrative English substrate",
            },
            "wikitext": {
                **wiki,
                "source": WIKITEXT.relative_to(ROOT).as_posix(),
                "source_bytes": WIKITEXT.stat().st_size,
                "source_sha256": EXPECTED_WIKI_SHA256,
                "purpose": "preserve locked held-out distribution coverage",
            },
        },
        "ordering": (
            "TinyStories parquet row order with double-newline separators, "
            "then the WikiText byte prefix; UTF-8-safe boundary padding only"
        ),
        "specialist_domain_corpus_used": False,
        "test_split_accessed": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
