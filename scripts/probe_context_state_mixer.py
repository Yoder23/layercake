"""Screen a frozen causal context-state mixer over CountCake experts.

The mixer is fit only on training bytes.  A bucket is addressed by a rolling
hash of preceding raw bytes and stores historical expert log scores.  At
inference the resulting target-independent weights mix complete normalized
expert distributions, so the mechanism is autoregressive and tokenizer-free.
This script evaluates observed probabilities only and is an architecture
screen, not a release certificate.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from scripts.optimize_count_cake_backoff_fast import _count_statistics  # noqa: E402


def _sample_payload(path: Path, maximum_bytes: int, seq_len: int) -> bytes:
    payload = path.read_bytes()
    if len(payload) <= maximum_bytes:
        return payload[: len(payload) // seq_len * seq_len]
    segment_count = 16
    segment_bytes = maximum_bytes // segment_count // seq_len * seq_len
    starts = np.linspace(
        0, len(payload) - segment_bytes, num=segment_count, dtype=np.int64
    )
    return b"".join(
        payload[int(start) : int(start) + segment_bytes] for start in starts
    )


@torch.inference_mode()
def _expert_probabilities(model, rows: torch.Tensor) -> np.ndarray:
    statistics = _count_statistics(model.count_cake, rows, model.prediction_start)
    probability = statistics["unigram"]
    stages = [probability.astype(np.float32)]
    for count, total, strength in zip(
        statistics["counts"],
        statistics["totals"],
        model.count_cake.backoff_strengths,
    ):
        probability = (count + strength * probability) / (total + strength)
        stages.append(probability.astype(np.float32))
    if model.chunking_mode == "delimiter":
        neural_log_probability, _ = model._dynamic_neural_log_probs(rows)
    else:
        context = model._patch_context(rows)
        targets = rows[:, model.prediction_start :].reshape(
            rows.shape[0], -1, model.patch_size
        )
        neural_log_probability, _ = model._neural_log_probs(
            context, targets, rows=rows
        )
    neural = neural_log_probability.reshape(-1).exp().cpu().numpy().astype(np.float32)
    return np.concatenate(
        [
            np.stack(stages, axis=-1).reshape(-1, len(stages)),
            neural[:, None],
        ],
        axis=-1,
    )


def _contexts(
    rows: torch.Tensor,
    *,
    start: int,
    order: int,
    buckets: int,
) -> np.ndarray:
    targets = rows[:, start:]
    context = torch.zeros_like(targets, dtype=torch.int64)
    for offset in range(order):
        source = rows[
            :,
            start - order + offset : rows.shape[1] - order + offset,
        ]
        context.mul_(257).add_(source + 1).bitwise_and_(buckets - 1)
    return context.reshape(-1).cpu().numpy()


def _rows(payload: bytes, seq_len: int, device: torch.device) -> torch.Tensor:
    count = len(payload) // seq_len
    return torch.frombuffer(
        bytearray(payload[: count * seq_len]), dtype=torch.uint8
    ).reshape(count, seq_len).to(device=device, dtype=torch.long)


def _evaluate_grid(
    probabilities: np.ndarray,
    contexts: np.ndarray,
    score_sums: np.ndarray,
    bucket_counts: np.ndarray,
    global_mean: np.ndarray,
    prior: float,
    temperature: float,
    batch_size: int = 262_144,
) -> float:
    total_nll = 0.0
    total = 0
    for start in range(0, contexts.size, batch_size):
        ids = contexts[start : start + batch_size]
        count = bucket_counts[ids, None]
        scores = (score_sums[ids] + prior * global_mean) / (count + prior)
        logits = temperature * (scores - scores.max(axis=-1, keepdims=True))
        weights = np.exp(logits)
        weights /= weights.sum(axis=-1, keepdims=True)
        mixture = np.sum(
            weights * probabilities[start : start + batch_size], axis=-1
        )
        total_nll -= float(np.log(np.clip(mixture, 1e-30, None)).sum())
        total += mixture.size
    return total_nll / total / math.log(2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--fit-data", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--max-fit-bytes", type=int, default=16_000_000)
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-order", type=int, default=8)
    parser.add_argument("--buckets", type=int, default=1_048_576)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.buckets <= 0 or args.buckets & (args.buckets - 1):
        raise ValueError("buckets must be a power of two")
    device = torch.device("cuda")
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    if args.context_order > model.prediction_start:
        raise ValueError("context order exceeds the unscored prefix")

    fit_payload = _sample_payload(
        Path(args.fit_data), args.max_fit_bytes, args.seq_len
    )
    fit_rows = _rows(fit_payload, args.seq_len, device)
    expert_count = model.count_cake.max_order + 2
    score_sums = np.zeros((args.buckets, expert_count), dtype=np.float32)
    bucket_counts = np.zeros(args.buckets, dtype=np.float32)
    global_sums = np.zeros(expert_count, dtype=np.float64)
    global_count = 0
    fit_started = time.perf_counter()
    for offset in range(0, fit_rows.shape[0], args.batch_size):
        batch = fit_rows[offset : offset + args.batch_size]
        probability = _expert_probabilities(model, batch)
        context = _contexts(
            batch,
            start=model.prediction_start,
            order=args.context_order,
            buckets=args.buckets,
        )
        log_probability = np.log(np.clip(probability, 1e-30, None))
        np.add.at(bucket_counts, context, 1.0)
        for expert in range(expert_count):
            np.add.at(score_sums[:, expert], context, log_probability[:, expert])
        global_sums += log_probability.sum(axis=0, dtype=np.float64)
        global_count += probability.shape[0]
    global_mean = (global_sums / global_count).astype(np.float32)
    fit_seconds = time.perf_counter() - fit_started

    eval_payload = Path(args.eval).read_bytes()
    eval_rows = _rows(eval_payload, args.seq_len, device)
    probability_parts = []
    context_parts = []
    for offset in range(0, eval_rows.shape[0], args.batch_size):
        batch = eval_rows[offset : offset + args.batch_size]
        probability_parts.append(_expert_probabilities(model, batch))
        context_parts.append(
            _contexts(
                batch,
                start=model.prediction_start,
                order=args.context_order,
                buckets=args.buckets,
            )
        )
    probabilities = np.concatenate(probability_parts, axis=0)
    contexts = np.concatenate(context_parts, axis=0)
    results = []
    for prior in (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0):
        for temperature in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            results.append(
                {
                    "prior": prior,
                    "temperature": temperature,
                    "bpb": _evaluate_grid(
                        probabilities,
                        contexts,
                        score_sums,
                        bucket_counts,
                        global_mean,
                        prior,
                        temperature,
                    ),
                }
            )
    results.sort(key=lambda item: item["bpb"])
    final_count_bpb = float(
        -np.log(np.clip(probabilities[:, -2], 1e-30, None)).mean()
        / math.log(2.0)
    )
    neural_bpb = float(
        -np.log(np.clip(probabilities[:, -1], 1e-30, None)).mean()
        / math.log(2.0)
    )
    oracle_bpb = float(
        -np.log(np.clip(probabilities.max(axis=-1), 1e-30, None)).mean()
        / math.log(2.0)
    )
    report = {
        "format": "layercake-context-state-mixer-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection probe with unmatched extra state",
        "causality": {
            "weights_fit_on_training_bytes_only": True,
            "inference_key": f"preceding {args.context_order} raw bytes",
            "current_target_used_by_router": False,
            "normalized_expert_mixture": True,
            "generation_valid": True,
        },
        "bundle": {
            "path": args.bundle,
            "logical_parameters": manifest["parameters"]["logical_total"],
        },
        "mixer": {
            "context_order": args.context_order,
            "buckets": args.buckets,
            "experts": expert_count,
            "unmatched_state_values": args.buckets * expert_count + args.buckets,
            "occupied_buckets": int(np.count_nonzero(bucket_counts)),
        },
        "fit": {
            "path": args.fit_data,
            "sampled_bytes": len(fit_payload),
            "scored_bytes": global_count,
            "seconds": fit_seconds,
        },
        "evaluation": {
            "path": args.eval,
            "bytes": len(eval_payload),
            "sha256": hashlib.sha256(eval_payload).hexdigest(),
            "scored_bytes": int(probabilities.shape[0]),
            "final_count_bpb": final_count_bpb,
            "neural_bpb": neural_bpb,
            "target_aware_oracle_bpb": oracle_bpb,
            "best": results[0],
            "top": results[:20],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
