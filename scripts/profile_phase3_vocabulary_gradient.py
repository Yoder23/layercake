"""Measure tied-vocabulary gradient scale for the next bounded Phase 3 branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import psutil
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from layercake.models.shallow_sparse_english import ShallowSparseEnglishCore
from layercake.training.phase3_cpu import (
    PHASE2_TOKENIZER,
    ROOT,
    TrainingOnlyHorizonHeads,
    WIKI,
    _chunked_exact_next_token_loss,
    _foundation_batch,
    _hidden,
    _sampled_multihorizon_loss,
    _weight,
)


PARENT = (
    ROOT
    / "artifacts"
    / "moonshot"
    / "phase3_cpu_training"
    / "layercake-seed9824-units15p5m-exact"
)
OUTPUT = (
    ROOT
    / "results"
    / "moonshot"
    / "phase3"
    / "profiles"
    / "vocabulary_gradient_profile.json"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "minimum": float(values.min()),
        "p10": float(torch.quantile(values, 0.10)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "p90": float(torch.quantile(values, 0.90)),
        "maximum": float(values.max()),
    }


def _model_and_heads() -> tuple[
    ShallowSparseEnglishCore, TrainingOnlyHorizonHeads
]:
    model = ShallowSparseEnglishCore()
    model.load_state_dict(
        load_file(str(PARENT / "model.safetensors")),
        strict=True,
    )
    heads = TrainingOnlyHorizonHeads(768, (1, 2, 4))
    heads.load_state_dict(
        load_file(
            str(PARENT / "training_only_horizon_heads.safetensors")
        ),
        strict=True,
    )
    model.train()
    heads.train()
    return model, heads


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"gradient profile is immutable: {OUTPUT}")
    torch.set_num_threads(14)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    tokenizer = AutoTokenizer.from_pretrained(
        PHASE2_TOKENIZER,
        local_files_only=True,
    )
    tokens = torch.tensor(
        tokenizer.encode(WIKI.read_bytes().decode("utf-8", errors="replace")),
        dtype=torch.long,
    )
    batch = _foundation_batch(
        tokens,
        tokenizer=tokenizer,
        batch_size=1,
        sequence=1020,
        horizons=(1, 2, 4),
        rng=random.Random(9824),
    )
    inputs, targets, routes, attention, raw_bytes = batch
    process = psutil.Process()
    started = time.perf_counter()

    model, heads = _model_and_heads()
    hidden, _ = _hidden(
        model,
        inputs,
        routes,
        None,
        attention,
        sparse_vocabulary_gradient=True,
    )
    sampled_loss, sampled_metrics = _sampled_multihorizon_loss(
        hidden,
        targets,
        _weight(model),
        heads,
        negatives=2048,
        generator=torch.Generator().manual_seed(9824 ^ 0xBAD5EED),
        sparse_vocabulary_gradient=True,
    )
    sampled_loss.backward()
    sampled_gradient = _weight(model).grad
    if sampled_gradient is None or not sampled_gradient.is_sparse:
        raise RuntimeError("sampled vocabulary gradient is not sparse")
    sampled_gradient = sampled_gradient.coalesce()
    sampled_row_rms = sampled_gradient.values().square().mean(dim=1).sqrt()
    sampled_rows = torch.unique(sampled_gradient.indices()[0])
    current_sgd_lr = 1.0e-2
    suggested_rowwise_lr = (
        current_sgd_lr * float(sampled_row_rms.median())
    )
    sampled = {
        "loss": float(sampled_loss.detach()),
        "candidate_vocabulary_rows": sampled_metrics[
            "candidate_vocabulary_rows"
        ],
        "gradient_rows": int(sampled_rows.numel()),
        "row_gradient_rms": _summary(sampled_row_rms),
        "current_stateless_sgd_learning_rate": current_sgd_lr,
        "current_update_row_rms": _summary(
            sampled_row_rms * current_sgd_lr
        ),
        "suggested_rowwise_adagrad_learning_rate": suggested_rowwise_lr,
        "derivation": (
            "current SGD learning rate multiplied by observed median per-row "
            "gradient RMS; this matches the median first-use row update RMS"
        ),
    }
    del model, heads, hidden, sampled_gradient

    model, heads = _model_and_heads()
    hidden, _ = _hidden(
        model,
        inputs,
        routes,
        None,
        attention,
        sparse_vocabulary_gradient=False,
    )
    sampled_loss, _ = _sampled_multihorizon_loss(
        hidden,
        targets,
        _weight(model),
        heads,
        negatives=2048,
        generator=torch.Generator().manual_seed(9824 ^ 0xBAD5EED),
        sparse_vocabulary_gradient=False,
    )
    exact_loss, exact_metrics = _chunked_exact_next_token_loss(
        hidden,
        targets[1],
        _weight(model),
        maximum_positions=64,
        vocabulary_chunk_rows=4096,
    )
    combined = sampled_loss + exact_loss
    combined.backward()
    exact_gradient = _weight(model).grad
    if exact_gradient is None or exact_gradient.is_sparse:
        raise RuntimeError("exact vocabulary gradient is not dense")
    exact_row_rms = exact_gradient.square().mean(dim=1).sqrt()
    exact = {
        "combined_loss": float(combined.detach()),
        "exact_loss": exact_metrics["exact_softmax_loss"],
        "gradient_rows": int(exact_gradient.shape[0]),
        "row_gradient_rms": _summary(exact_row_rms),
        "stateless_dense_sgd_retained": True,
    }
    result = {
        "format": "layercake-phase3-vocabulary-gradient-profile/1",
        "status": "PASS",
        "promotion_eligible": False,
        "device": "cpu",
        "parent_checkpoint": {
            "path": PARENT.relative_to(ROOT).as_posix(),
            "model_sha256": _sha(PARENT / "model.safetensors"),
            "metadata_sha256": _sha(PARENT / "metadata.json"),
        },
        "configuration": {
            "seed": 9824,
            "sequence_units": 1020,
            "raw_utf8_bytes": raw_bytes,
            "negative_vocabulary_rows": 2048,
            "prediction_horizons": [1, 2, 4],
            "exact_softmax_positions": 64,
            "exact_softmax_chunk_rows": 4096,
        },
        "sampled_step": sampled,
        "exact_step": exact,
        "wall_time_seconds": time.perf_counter() - started,
        "process_resident_memory_bytes": int(process.memory_info().rss),
        "cuda_memory_allocated": torch.cuda.memory_allocated(),
        "inference_architecture_change": False,
    }
    result["evidence_sha256"] = _canonical_sha(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
