"""Leakage-locked byte corpora and reproducible WikiText preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import unicodedata
import importlib.metadata
import sys

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(character for character in value if character == "\n" or character == "\t" or ord(character) >= 32)
    return value.rstrip() + "\n" if value.strip() else ""


def _looks_english(text: str) -> bool:
    letters = sum(character.isalpha() for character in text)
    ascii_letters = sum(("a" <= character.lower() <= "z") for character in text)
    return letters < 20 or ascii_letters / max(letters, 1) >= 0.70


def _normalized_unique_rows(rows) -> tuple[list[bytes], dict]:
    accepted: list[bytes] = []
    exact = set()
    rejected_language = 0
    duplicates = 0
    for row in rows:
        normalized = _normalize(str(row))
        if not normalized:
            continue
        if not _looks_english(normalized):
            rejected_language += 1
            continue
        encoded = normalized.encode("utf-8")
        digest = hashlib.sha256(encoded).digest()
        if digest in exact:
            duplicates += 1
            continue
        exact.add(digest)
        accepted.append(encoded)
    return accepted, {
        "accepted_records": len(accepted),
        "exact_duplicates_removed": duplicates,
        "language_filter_rejections": rejected_language,
    }


def _write_prefix(path: Path, records: list[bytes], limit: int | None = None) -> int:
    written = 0
    with path.open("wb") as handle:
        for record in records:
            if limit is not None and written + len(record) > limit:
                remaining = limit - written
                if remaining > 0:
                    handle.write(record[:remaining])
                    written += remaining
                break
            handle.write(record)
            written += len(record)
    return written


def _ngram_sample(path: Path, width: int = 64, stride: int = 4096) -> set[bytes]:
    raw = path.read_bytes()
    return {
        hashlib.sha256(raw[index:index + width]).digest()
        for index in range(0, max(0, len(raw) - width + 1), stride)
    }


def prepare_wikitext103(
    output_dir: str | Path,
    *,
    cache_dir: str | Path = ".datasets_cache",
    development_bytes: int = 10_000_000,
    medium_bytes: int = 100_000_000,
) -> dict:
    """Download, normalize, deduplicate, split, hash, and document WikiText-103."""

    from datasets import load_dataset

    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=str(cache_dir)
    )
    train, train_stats = _normalized_unique_rows(dataset["train"]["text"])
    validation_rows, validation_stats = _normalized_unique_rows(dataset["validation"]["text"])
    test, test_stats = _normalized_unique_rows(dataset["test"]["text"])

    # Architecture selection and validation are disjoint normalized records,
    # assigned before any training based on a stable content hash.
    architecture_selection: list[bytes] = []
    validation: list[bytes] = []
    for row in validation_rows:
        target = architecture_selection if hashlib.sha256(row).digest()[0] % 2 == 0 else validation
        target.append(row)

    paths = {
        "development_train": output / "train_development.bin",
        "medium_train": output / "train_medium.bin",
        "architecture_selection": output / "architecture_selection.bin",
        "validation": output / "validation.bin",
        "test": output / "test.bin",
    }
    sizes = {
        "development_train": _write_prefix(paths["development_train"], train, development_bytes),
        "medium_train": _write_prefix(paths["medium_train"], train, medium_bytes),
        "architecture_selection": _write_prefix(paths["architecture_selection"], architecture_selection),
        "validation": _write_prefix(paths["validation"], validation),
        "test": _write_prefix(paths["test"], test),
    }
    if sizes["development_train"] < development_bytes or sizes["medium_train"] < medium_bytes:
        raise RuntimeError("WikiText-103 did not satisfy configured corpus tiers")
    samples = {name: _ngram_sample(path) for name, path in paths.items()}
    overlaps = {}
    names = list(samples)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            if left.endswith("train") and right.endswith("train"):
                continue
            overlaps[f"{left}:{right}"] = len(samples[left] & samples[right])
    manifest = {
        "format": "layercake-corpus-manifest/2",
        "dataset": "Salesforce/wikitext",
        "dataset_configuration": "wikitext-103-raw-v1",
        "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
        "license": "CC BY-SA 3.0",
        "preparation": {
            "normalization": "Unicode NFKC; CRLF/CR to LF; control-character removal",
            "language_identification": "WikiText English source plus >=70% ASCII-letter heuristic for records with >=20 letters",
            "deduplication": "exact SHA-256 normalized-record deduplication per official split",
            "partitioning": "official train/test; official validation content-hash partitioned before training",
            "near_duplicate_check": "SHA-256 of 64-byte windows sampled every 4096 bytes",
            "download_cache": str(Path(cache_dir).resolve()),
        },
        "tier_requirements": {
            "development_minimum_bytes": 10_000_000,
            "medium_minimum_bytes": 100_000_000,
            "proof_candidate_minimum_bytes": 1_000_000_000,
            "proof_candidate_available": False,
        },
        "records": {
            "train": train_stats,
            "original_validation": validation_stats,
            "test": test_stats,
        },
        "files": {
            name: {"path": str(path.resolve()), "bytes": sizes[name], "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "sampled_cross_split_64byte_overlaps": overlaps,
        "preparation_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


class ByteCorpus:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.data = np.memmap(self.path, mode="r", dtype=np.uint8)
        if len(self.data) < 2:
            raise ValueError("byte corpus is too small")

    def __len__(self) -> int:
        return int(len(self.data))

    def batches(
        self,
        *,
        batch_size: int,
        sequence_bytes: int,
        seed: int,
        steps: int,
        device: torch.device | str,
    ):
        if sequence_bytes < 2 or len(self.data) <= sequence_bytes:
            raise ValueError("corpus is smaller than the requested training sequence")
        generator = np.random.default_rng(seed)
        for _ in range(steps):
            offsets = generator.integers(0, len(self.data) - sequence_bytes - 1, size=batch_size)
            rows = np.stack([
                np.asarray(self.data[offset:offset + sequence_bytes + 1], dtype=np.int64)
                for offset in offsets
            ])
            yield torch.from_numpy(rows).to(device=device, dtype=torch.long)

    def fixed_batches(
        self,
        *,
        batch_size: int,
        sequence_bytes: int,
        batches: int,
        device: torch.device | str,
    ):
        maximum = min(len(self.data) - sequence_bytes - 1, batch_size * batches * sequence_bytes)
        offsets = np.linspace(0, maximum, num=batch_size * batches, dtype=np.int64)
        for start in range(0, len(offsets), batch_size):
            rows = np.stack([
                np.asarray(self.data[offset:offset + sequence_bytes + 1], dtype=np.int64)
                for offset in offsets[start:start + batch_size]
            ])
            yield torch.from_numpy(rows).to(device=device, dtype=torch.long)


def prepare_python_distribution_corpus(output_dir: str | Path) -> dict:
    """Build a permissively licensed Python corpus with distribution-level splits."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assignments = {
        "train": {
            "torch": "BSD-3-Clause",
            "numpy": "BSD-3-Clause",
            "transformers": "Apache-2.0",
            "datasets": "Apache-2.0",
        },
        "validation": {"huggingface-hub": "Apache-2.0"},
        "test": {"scikit-learn": "BSD-3-Clause"},
    }
    records: dict[str, list[bytes]] = {split: [] for split in assignments}
    provenance = {}
    seen_hashes = set()
    for split, distributions in assignments.items():
        for distribution_name, declared_license in distributions.items():
            distribution = importlib.metadata.distribution(distribution_name)
            files = []
            byte_count = 0
            for relative in distribution.files or ():
                value = str(relative).replace("\\", "/")
                if not value.endswith(".py") or "/tests/" in f"/{value}/" or "/test/" in f"/{value}/":
                    continue
                path = Path(distribution.locate_file(relative))
                try:
                    raw = path.read_bytes()
                except OSError:
                    continue
                digest = hashlib.sha256(raw).digest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                header = f"\n# source-distribution: {distribution_name}; path: {value}\n".encode()
                records[split].append(header + raw + b"\n")
                files.append(value)
                byte_count += len(raw)
            provenance[distribution_name] = {
                "split": split,
                "version": distribution.version,
                "declared_license": declared_license,
                "metadata_license": distribution.metadata.get("License"),
                "source_homepage": distribution.metadata.get("Home-page"),
                "files": len(files),
                "bytes": byte_count,
            }
    paths = {split: output / f"python_{split}.bin" for split in assignments}
    for split, path in paths.items():
        _write_prefix(path, records[split])
    manifest = {
        "format": "layercake-python-corpus-manifest/1",
        "description": "Installed public Python source from explicitly allow-listed permissive distributions",
        "separation": "entire package distributions assigned to exactly one split",
        "licenses": sorted({license_name for values in assignments.values() for license_name in values.values()}),
        "python": sys.version,
        "provenance": provenance,
        "files": {
            split: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for split, path in paths.items()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
