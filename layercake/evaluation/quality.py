from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


def bits_per_byte(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.shape[:-1] != targets.shape or logits.shape[-1] != 256:
        raise ValueError("logits/targets must describe 256-way byte predictions")
    loss = F.cross_entropy(logits.reshape(-1, 256).float(), targets.reshape(-1), reduction="mean")
    return float(loss.item() / math.log(2.0))


def error_rate(predictions: Iterable[object], targets: Iterable[object]) -> float:
    predictions = list(predictions)
    targets = list(targets)
    if not targets or len(predictions) != len(targets):
        raise ValueError("predictions and non-empty targets must have equal length")
    return sum(prediction != target for prediction, target in zip(predictions, targets)) / len(targets)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dataset_integrity(splits: dict[str, bytes | str | Path], *, leakage_ngram: int = 32) -> dict:
    materialized: dict[str, bytes] = {}
    for name, value in splits.items():
        if isinstance(value, Path):
            materialized[name] = value.read_bytes()
        elif isinstance(value, str):
            materialized[name] = value.encode("utf-8")
        else:
            materialized[name] = value
    required = {"train", "validation", "test", "architecture_selection"}
    if set(materialized) != required:
        raise ValueError(f"dataset splits must be exactly {sorted(required)}")
    ngrams = {
        name: {data[index : index + leakage_ngram] for index in range(max(0, len(data) - leakage_ngram + 1))}
        for name, data in materialized.items()
    }
    overlaps = {}
    names = sorted(materialized)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}:{right}"] = len(ngrams[left] & ngrams[right])
    exact_hashes = {name: _hash_bytes(data) for name, data in materialized.items()}
    unique_hashes = len(set(exact_hashes.values())) == len(exact_hashes)
    return {
        "status": "PASS" if unique_hashes and not any(overlaps.values()) else "FAIL",
        "sha256": exact_hashes,
        "bytes": {name: len(data) for name, data in materialized.items()},
        "leakage_ngram": leakage_ngram,
        "cross_split_ngram_overlaps": overlaps,
        "exact_split_hashes_unique": unique_hashes,
    }
