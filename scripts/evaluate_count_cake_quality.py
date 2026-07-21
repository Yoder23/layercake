"""Evaluate a portable CountCake bundle with its frozen causal cache recipe."""

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

from layercake.count_cake import (  # noqa: E402
    apply_causal_online_cache_to_observed,
    load_count_cake_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="ablate the neural host while retaining the identical count state",
    )
    parser.add_argument(
        "--backoff-strengths",
        help="comma-separated inference ablation; omitted uses the bundle recipe",
    )
    parser.add_argument(
        "--backoff-mode",
        choices=("fixed", "distinct", "discount"),
    )
    parser.add_argument("--discount", type=float)
    parser.add_argument(
        "--online-cache-specs",
        help="comma-separated order:strength validation ablation",
    )
    parser.add_argument(
        "--cache-scope",
        choices=("row", "stream"),
    )
    parser.add_argument("--cache-window", type=int)
    parser.add_argument(
        "--recent-cache-specs",
        help="comma-separated order:strength last-continuation stages",
    )
    parser.add_argument("--normalized-cache-specs")
    parser.add_argument(
        "--cache-normalization",
        choices=("casefold", "classes"),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="report base quality without running any causal cache stages",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    payload = Path(args.data).read_bytes()
    device = torch.device(args.device)
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    if args.backoff_strengths:
        strengths = tuple(
            float(value) for value in args.backoff_strengths.split(",")
        )
        if len(strengths) < model.count_cake.max_order:
            raise ValueError("backoff override is shorter than the trained orders")
        model.count_cake.backoff_strengths = strengths
    if args.backoff_mode:
        model.count_cake.backoff_mode = args.backoff_mode
    if args.discount is not None:
        if not 0.0 < args.discount < 1.0:
            raise ValueError("discount must be strictly between zero and one")
        model.count_cake.discount = args.discount
    model.eval()
    row_count = len(payload) // args.seq_len
    if row_count == 0:
        raise ValueError("data is shorter than one evaluation row")
    rows = np.frombuffer(
        payload[: row_count * args.seq_len], dtype=np.uint8
    ).reshape(row_count, args.seq_len).copy()
    chunks: list[np.ndarray] = []
    neural_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, row_count, args.batch_size):
            batch = torch.from_numpy(rows[offset : offset + args.batch_size]).to(
                device=device, dtype=torch.long
            )
            if args.count_only:
                log_probability = model.count_cake.target_log_probs(
                    batch,
                    start=model.prediction_start,
                )
            else:
                log_probability, neural_log_probability = model.target_log_probs(
                    batch,
                    return_neural=True,
                )
                count_log_probability = model.count_cake.target_log_probs(
                    batch,
                    start=model.prediction_start,
                )
                neural_chunks.append(neural_log_probability.cpu().numpy())
                count_chunks.append(count_log_probability.cpu().numpy())
            chunks.append(log_probability.exp().cpu().numpy())
    base = np.concatenate(chunks, axis=0).astype(np.float64)
    recipe_source = "bundle"
    cache_specs = model.online_cache_specs
    if args.online_cache_specs is not None:
        recipe_source = "command_line_override"
        cache_specs = tuple(
            (int(item.split(":", 1)[0]), float(item.split(":", 1)[1]))
            for item in args.online_cache_specs.split(",")
        )
    recent_specs = model.recent_cache_specs
    if args.recent_cache_specs is not None:
        recipe_source = "command_line_override"
        recent_specs = tuple(
            (int(item.split(":", 1)[0]), float(item.split(":", 1)[1]))
            for item in args.recent_cache_specs.split(",")
        )
    normalized_specs = model.normalized_cache_specs
    if args.normalized_cache_specs is not None:
        recipe_source = "command_line_override"
        normalized_specs = tuple(
            (int(item.split(":", 1)[0]), float(item.split(":", 1)[1]))
            for item in args.normalized_cache_specs.split(",")
        )
    cache_window = (
        model.online_cache_window
        if args.cache_window is None
        else args.cache_window
    )
    cache_normalization = args.cache_normalization or model.cache_normalization
    cache_scope = args.cache_scope or "stream"
    if args.no_cache:
        recipe_source = "disabled_by_command_line"
        cache_specs = ()
        recent_specs = ()
        normalized_specs = ()
    cache_enabled = bool(cache_specs or recent_specs or normalized_specs)
    if cache_enabled:
        cached = apply_causal_online_cache_to_observed(
            base,
            rows,
            start=model.prediction_start,
            specs=cache_specs,
            reset_each_row=cache_scope == "row",
            window=cache_window,
            recent_specs=recent_specs,
            normalized_specs=normalized_specs,
            normalization=cache_normalization,
        )
    else:
        cached = base.copy()
    base_nll = float(-np.log(base).mean())
    cached_nll = float(-np.log(cached).mean())
    components = None
    if neural_chunks:
        neural_log = np.concatenate(neural_chunks, axis=0).astype(np.float64)
        count_log = np.concatenate(count_chunks, axis=0).astype(np.float64)
        neural_nll = float(-neural_log.mean())
        count_nll = float(-count_log.mean())
        # This target-aware selector is deliberately unattainable at inference
        # time.  It is a diagnostic lower bound: if it cannot clear a quality
        # gate, no causal mixture of these two frozen experts can clear it.
        oracle_log = np.maximum(neural_log, count_log)
        neural_probability = np.exp(neural_log)
        count_probability = np.exp(count_log)

        def scalar_mixture_nll(weight: float) -> float:
            return float(
                -np.log(
                    (1.0 - weight) * count_probability
                    + weight * neural_probability
                ).mean()
            )

        # Convex one-dimensional fit without introducing an optimizer
        # dependency into the standalone evaluator.
        lower, upper = 0.0, 1.0
        for _ in range(40):
            left = lower + (upper - lower) / 3.0
            right = upper - (upper - lower) / 3.0
            if scalar_mixture_nll(left) <= scalar_mixture_nll(right):
                upper = right
            else:
                lower = left
        best_scalar_weight = (lower + upper) / 2.0
        best_scalar_nll = scalar_mixture_nll(best_scalar_weight)
        components = {
            "neural_bpb": neural_nll / math.log(2.0),
            "count_bpb": count_nll / math.log(2.0),
            "target_aware_oracle_bpb": float(-oracle_log.mean() / math.log(2.0)),
            "target_aware_oracle_neural_fraction": float(
                np.mean(neural_log > count_log)
            ),
            "best_scalar_mixture_bpb": float(
                best_scalar_nll / math.log(2.0)
            ),
            "best_scalar_neural_weight": best_scalar_weight,
        }
    report = {
        "format": "layercake-count-cake-quality/1",
        "status": "COMPLETE",
        "bundle": {
            "path": args.bundle,
            "logical_parameters": manifest["parameters"]["logical_total"],
        },
        "data": {
            "path": args.data,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "evaluated_bytes": int(base.size),
            "rows": row_count,
            "row_bytes": args.seq_len,
            "unscored_prefix_bytes_per_row": model.prediction_start,
        },
        "causal_online_cache": {
            "enabled": cache_enabled,
            "recipe_source": recipe_source,
            "reset": (
                "each evaluation row"
                if cache_scope == "row"
                else "once at evaluation stream start"
            ),
            "prefill": "row prefix only",
            "update": "after observed target is scored",
            "specs": [
                {"order": order, "strength": strength}
                for order, strength in cache_specs
            ],
            "window": cache_window,
            "recent_specs": [
                {"order": order, "strength": strength}
                for order, strength in recent_specs
            ],
            "normalized_specs": [
                {"order": order, "strength": strength}
                for order, strength in normalized_specs
            ],
            "normalization": cache_normalization,
        },
        "quality": {
            "base_nll": base_nll,
            "base_bpb": base_nll / math.log(2.0),
            "cached_nll": cached_nll,
            "cached_bpb": cached_nll / math.log(2.0),
            "components": components,
        },
        "device": args.device,
        "count_only_ablation": args.count_only,
        "backoff_strengths": list(model.count_cake.backoff_strengths),
        "backoff_mode": model.count_cake.backoff_mode,
        "discount": model.count_cake.discount,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
