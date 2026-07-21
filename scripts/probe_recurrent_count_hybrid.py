"""Screen a normalized recurrent-LayerCake/CountCake expert mixture.

This validation-only utility combines complete normalized byte distributions;
the scalar gate is therefore generation-valid.  The target-aware oracle is
reported only as an explicit unattainable upper bound for architecture
selection.
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
from scipy.optimize import minimize_scalar
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import (  # noqa: E402
    apply_causal_online_cache_to_observed,
    load_count_cake_bundle,
)
from scripts.train_byte_core_from_config import _build_model  # noqa: E402


def _bpb(probability: np.ndarray) -> float:
    return float(-np.log(np.clip(probability, 1e-30, None)).mean() / math.log(2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recurrent-checkpoint", required=True)
    parser.add_argument("--count-bundle", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq-len", type=int, default=7968)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.perf_counter()
    checkpoint = torch.load(args.recurrent_checkpoint, map_location="cpu")
    recurrent = _build_model(checkpoint["model_config"], torch.device("cuda"))
    recurrent.load_state_dict(checkpoint["model"])
    recurrent.eval()
    count_model, count_manifest = load_count_cake_bundle(
        args.count_bundle, device="cuda"
    )
    count_model.eval()
    payload = Path(args.data).read_bytes()
    row_count = len(payload) // args.seq_len
    rows_np = np.frombuffer(
        payload[: row_count * args.seq_len], dtype=np.uint8
    ).reshape(row_count, args.seq_len).copy()
    rows = torch.from_numpy(rows_np).to(device="cuda", dtype=torch.long)
    neural_parts = []
    count_parts = []
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for offset in range(0, row_count, args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            predictions, targets = recurrent.domain_cake_patch_predictions(batch)
            logits = torch.stack(predictions, dim=2)
            observed = F.log_softmax(logits.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            neural_parts.append(observed.reshape(batch.shape[0], -1).exp().cpu())
            count_parts.append(
                count_model.count_cake.target_log_probs(
                    batch, start=count_model.prediction_start
                ).exp().cpu()
            )
    neural = torch.cat(neural_parts).numpy().astype(np.float64)
    count = torch.cat(count_parts).numpy().astype(np.float64)
    if neural.shape[0] == count.shape[0] and neural.shape[1] > count.shape[1]:
        # Patch heads return one padded final target patch; the production
        # loss masks positions beyond the source row.  CountCake already
        # returns only the valid post-prefix targets.
        neural = neural[:, : count.shape[1]]
    if neural.shape != count.shape:
        raise RuntimeError(f"expert score shapes differ: {neural.shape} vs {count.shape}")

    def objective(weight: float) -> float:
        return _bpb((1.0 - weight) * count + weight * neural)

    fitted = minimize_scalar(
        objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6}
    )
    weight = float(fitted.x)
    mixture = (1.0 - weight) * count + weight * neural
    cached = apply_causal_online_cache_to_observed(
        mixture,
        rows_np,
        start=count_model.prediction_start,
        specs=((8, 7.129015), (6, 7.7779), (4, 74.2035), (2, 267.882)),
        window=1344,
        recent_specs=((24, 0.81267), (16, 2.55238), (12, 5.87549), (10, 4.76027)),
        normalized_specs=((5, 25.7671), (3, 41.8524)),
        normalization="classes",
        reset_each_row=False,
    )
    report = {
        "format": "layercake-recurrent-count-hybrid-probe/1",
        "status": "COMPLETE",
        "warning": "validation-selected optimistic architecture probe, not a release result",
        "parameters": {
            "recurrent": sum(parameter.numel() for parameter in recurrent.parameters()),
            "count_logical": count_manifest["parameters"]["logical_total"],
            "combined_unmatched": (
                sum(parameter.numel() for parameter in recurrent.parameters())
                + count_manifest["parameters"]["logical_total"]
            ),
        },
        "data": {
            "path": args.data,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": row_count,
            "scored_bytes": int(neural.size),
        },
        "quality": {
            "recurrent_bpb": _bpb(neural),
            "count_bpb": _bpb(count),
            "target_aware_oracle_bpb": _bpb(np.maximum(neural, count)),
            "best_scalar_neural_weight": weight,
            "best_scalar_bpb": _bpb(mixture),
            "cached_scalar_bpb": _bpb(cached),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
