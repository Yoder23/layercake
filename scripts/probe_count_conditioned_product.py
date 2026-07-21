"""Probe normalized product fusion of frozen neural and CountCake experts.

Unlike the existing arithmetic gate, this head applies byte-specific evidence
from both complete 256-way distributions and renormalizes exactly.  Parameters
are fit only on a development prefix; suffix and additional files are reported
separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402


@torch.inference_mode()
def _count_distribution(
    cake,
    rows: torch.Tensor,
    start: int,
    *,
    return_stages: bool = False,
) -> torch.Tensor:
    """Return exact normalized p(next byte | context) for every scored byte."""
    rows = rows.to(torch.int64)
    targets_shape = rows[:, start:].shape
    candidates = torch.arange(256, device=rows.device, dtype=torch.int64)
    smoothed = cake.unigram_counts + 0.5
    probability = (smoothed / smoothed.sum()).view(1, 1, 256).expand(
        *targets_shape, 256
    )
    stages = [probability]
    for order in range(1, cake.max_order + 1):
        context = torch.zeros(targets_shape, device=rows.device, dtype=torch.int64)
        if cake.order_encodings[order - 1] == "packed":
            for lag in range(order):
                context.add_(
                    rows[:, start - 1 - lag : rows.shape[1] - 1 - lag]
                    << (8 * lag)
                )
            joint = context.unsqueeze(-1).bitwise_left_shift(8) + candidates
            counts = cake._lookup(
                getattr(cake, f"keys_{order}"),
                getattr(cake, f"counts_{order}"),
                joint,
            )
            totals = cake._lookup(
                getattr(cake, f"context_keys_{order}"),
                getattr(cake, f"context_totals_{order}"),
                context,
            )
            distinct = cake._lookup(
                getattr(cake, f"context_keys_{order}"),
                getattr(cake, f"context_distinct_{order}"),
                context,
            )
        else:
            mask = (1 << cake.context_hash_bits[order - 1]) - 1
            for offset in range(order):
                context.mul_(257).add_(
                    rows[
                        :,
                        start - order + offset : rows.shape[1] - order + offset,
                    ]
                    + 1
                ).bitwise_and_(mask)
            context_keys = getattr(cake, f"context_keys_{order}")
            positions = torch.searchsorted(context_keys, context)
            safe = positions.clamp(max=context_keys.numel() - 1)
            found = (positions < context_keys.numel()) & (
                context_keys[safe] == context
            )
            joint = safe.unsqueeze(-1) * 256 + candidates
            counts = cake._lookup(
                getattr(cake, f"keys_{order}"),
                getattr(cake, f"counts_{order}"),
                joint,
            )
            counts = torch.where(found.unsqueeze(-1), counts, torch.zeros_like(counts))
            totals = torch.where(
                found,
                getattr(cake, f"context_totals_{order}")[safe],
                torch.zeros_like(context, dtype=torch.float32),
            )
            distinct = torch.where(
                found,
                getattr(cake, f"context_distinct_{order}")[safe],
                torch.zeros_like(context, dtype=torch.float32),
            )
        if cake.backoff_mode == "discount":
            discounted = (counts - cake.discount).clamp_min(0.0)
            escape = cake.discount * distinct
            updated = (
                discounted + escape.unsqueeze(-1) * probability
            ) / totals.clamp_min(1.0).unsqueeze(-1)
            probability = torch.where(
                (totals > 0).unsqueeze(-1), updated, probability
            )
        elif cake.backoff_mode == "distinct":
            updated = (
                counts + distinct.unsqueeze(-1) * probability
            ) / (totals + distinct).clamp_min(1.0).unsqueeze(-1)
            probability = torch.where(
                (totals > 0).unsqueeze(-1), updated, probability
            )
        else:
            strength = cake.backoff_strengths[order - 1]
            probability = (
                counts + strength * probability
            ) / (totals.unsqueeze(-1) + strength)
        stages.append(probability)
    if return_stages:
        return torch.stack(
            [stage / stage.sum(dim=-1, keepdim=True) for stage in stages],
            dim=-2,
        )
    return probability / probability.sum(dim=-1, keepdim=True)


@torch.inference_mode()
def _prepare(
    model,
    path: str,
    *,
    seq_len: int,
    batch_size: int,
    rows_per_split: int,
) -> dict:
    payload = Path(path).read_bytes()
    available_rows = len(payload) // seq_len
    selected_rows = min(available_rows, rows_per_split)
    # Evenly spaced, chronological rows make the bounded probe cover the full
    # file without retaining every 256-way distribution in host memory.
    row_indices = np.linspace(
        0, available_rows - 1, num=selected_rows, dtype=np.int64
    )
    payload_view = np.frombuffer(payload, dtype=np.uint8).reshape(-1)[: available_rows * seq_len]
    rows = torch.from_numpy(
        payload_view.reshape(available_rows, seq_len)[row_indices].copy()
    ).to(device="cuda", dtype=torch.long)
    target_logs = []
    observed_targets = []
    feature_chunks = []
    for batch_start in range(0, selected_rows, batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        if model.chunking_mode != "fixed":
            raise ValueError("the product probe currently requires fixed chunks")
        context = model._patch_context(batch)
        targets = batch[:, model.prediction_start :].reshape(
            batch.shape[0], -1, model.patch_size
        )
        _, neural_hidden = model._neural_log_probs(context, targets)
        neural = model._neural_probabilities(neural_hidden).reshape(
            batch.shape[0], -1, 256
        )
        count = _count_distribution(
            model.count_cake, batch, model.prediction_start
        )
        observed = batch[:, model.prediction_start :]
        neural_observed = neural.gather(-1, observed.unsqueeze(-1)).squeeze(-1)
        count_observed = count.gather(-1, observed.unsqueeze(-1)).squeeze(-1)
        # Store log normalizers for arbitrary two-expert exponents. Complete
        # distributions remain on CPU only for this architecture-selection run.
        target_logs.append(
            torch.stack(
                [count_observed.clamp_min(1e-30).log(), neural_observed.clamp_min(1e-30).log()],
                dim=-1,
            ).cpu().numpy().astype(np.float32)
        )
        observed_targets.append(observed.cpu().numpy().astype(np.uint8))
        feature_chunks.append(
            torch.stack(
                [count.clamp_min(1e-30).log(), neural.clamp_min(1e-30).log()],
                dim=-2,
            ).to(torch.float16).cpu().numpy()
        )
        print(
            json.dumps(
                {
                    "prepared": path,
                    "selected_rows": min(batch_start + batch_size, selected_rows),
                    "available_rows": available_rows,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "available_rows": available_rows,
        "selected_rows": selected_rows,
        "row_indices": row_indices.tolist(),
        "target_log": np.concatenate(target_logs, axis=0).reshape(-1, 2),
        "targets": np.concatenate(observed_targets, axis=0).reshape(-1),
        "expert_log": np.concatenate(feature_chunks, axis=0).reshape(-1, 2, 256),
    }


def _bpb(split: dict, parameters: np.ndarray, indices: slice) -> float:
    weights = np.exp(parameters).astype(np.float32)
    target = split["target_log"][indices] @ weights
    expert = split["expert_log"][indices].astype(np.float32)
    logits = expert[:, 0] * weights[0] + expert[:, 1] * weights[1]
    maximum = logits.max(axis=-1)
    log_normalizer = maximum + np.log(
        np.exp(logits - maximum[:, None]).sum(axis=-1)
    )
    return float(-(target - log_normalizer).mean() / math.log(2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rows-per-split", type=int, default=256)
    parser.add_argument("--fit-rows", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the product-fusion probe")
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device="cuda")
    model.eval()
    splits = [
        _prepare(
            model,
            path,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            rows_per_split=args.rows_per_split,
        )
        for path in args.data
    ]
    fit_count = min(
        splits[0]["target_log"].shape[0],
        args.fit_rows * (args.seq_len - model.prediction_start),
    )
    if fit_count >= splits[0]["target_log"].shape[0]:
        raise ValueError("fit rows must leave a development suffix")

    def objective(parameters: np.ndarray) -> float:
        return _bpb(splits[0], parameters, slice(0, fit_count))

    result = minimize(
        objective,
        np.log(np.asarray([1.0, 1.0], dtype=np.float64)),
        method="Powell",
        bounds=[(math.log(0.01), math.log(4.0))] * 2,
        options={"maxiter": 30, "xtol": 0.002, "ftol": 1e-10},
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    reports = []
    for index, split in enumerate(splits):
        evaluation_slice = (
            slice(fit_count, None) if index == 0 else slice(None)
        )
        reports.append(
            {
                "path": split["path"],
                "bytes": split["bytes"],
                "sha256": split["sha256"],
                "evaluation": "heldout_suffix" if index == 0 else "untouched_full_split",
                "scored_bytes": int(split["target_log"][evaluation_slice].shape[0]),
                "count_bpb": _bpb(split, np.log([1.0, 0.01]), evaluation_slice),
                "neural_bpb": _bpb(split, np.log([0.01, 1.0]), evaluation_slice),
                "product_bpb": _bpb(split, parameters, evaluation_slice),
            }
        )
    report = {
        "format": "layercake-count-conditioned-product-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection probe; fusion exponents fit only on the first split prefix",
        "bundle": args.bundle,
        "logical_parameters": manifest["parameters"]["logical_total"],
        "fit_scored_bytes": fit_count,
        "row_sampling": "evenly_spaced_chronological_rows",
        "rows_per_split": args.rows_per_split,
        "fitted_exponents": np.exp(parameters).tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "splits": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
