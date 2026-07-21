"""Bounded probe for a normalized nonlinear count/neural fusion head.

The head scores all 256 candidates and applies an exact softmax.  It is fit on
an early development sample, selected on a disjoint middle sample, and reported
on a later untouched sample.  This is an architecture-selection probe, not a
publishable final measurement.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402
from probe_count_conditioned_product import _prepare  # noqa: E402


class NonlinearProductHead(nn.Module):
    """Small candidate-aware residual on top of a normalized product."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        initial = torch.tensor([0.40, 0.31])
        self.raw_weights = nn.Parameter(torch.log(torch.expm1(initial)))
        self.byte_count_scale = nn.Parameter(torch.zeros(256))
        self.byte_neural_scale = nn.Parameter(torch.zeros(256))
        self.byte_bias = nn.Parameter(torch.zeros(256))
        self.shared = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.shared[-1].weight.zero_()
            self.shared[-1].bias.zero_()

    def forward(self, expert_log: torch.Tensor) -> torch.Tensor:
        logs = expert_log.float().clamp_min(-30.0)
        count_log = logs[:, 0]
        neural_log = logs[:, 1]
        weights = F.softplus(self.raw_weights)
        logits = weights[0] * count_log + weights[1] * neural_log
        logits = logits + 0.25 * torch.tanh(self.byte_count_scale) * count_log
        logits = logits + 0.25 * torch.tanh(self.byte_neural_scale) * neural_log
        logits = logits + self.byte_bias
        features = torch.stack(
            [
                count_log / 8.0,
                neural_log / 8.0,
                (count_log - neural_log) / 8.0,
                (count_log + neural_log) / 16.0,
            ],
            dim=-1,
        )
        return logits + 0.5 * self.shared(features).squeeze(-1)


@torch.inference_mode()
def _bpb(
    head: nn.Module,
    expert_log: torch.Tensor,
    targets: torch.Tensor,
    start: int,
    stop: int,
    *,
    batch_size: int = 2048,
) -> float:
    total = 0.0
    count = 0
    for offset in range(start, stop, batch_size):
        end = min(stop, offset + batch_size)
        loss = F.cross_entropy(
            head(expert_log[offset:end]), targets[offset:end], reduction="sum"
        )
        total += float(loss)
        count += end - offset
    return total / count / math.log(2.0)


@torch.inference_mode()
def _expert_bpb(
    expert_log: torch.Tensor,
    targets: torch.Tensor,
    expert: int,
    start: int,
    stop: int,
) -> float:
    selected = expert_log[start:stop, expert].gather(
        1, targets[start:stop, None]
    )
    return float(-selected.float().mean() / math.log(2.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--prepare-batch-size", type=int, default=4)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--fit-rows", type=int, default=64)
    parser.add_argument("--validation-rows", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device="cuda")
    model.eval()
    split = _prepare(
        model,
        args.data,
        seq_len=args.seq_len,
        batch_size=args.prepare_batch_size,
        rows_per_split=args.rows,
    )
    del model
    torch.cuda.empty_cache()
    expert_log = torch.from_numpy(split["expert_log"]).to("cuda")
    targets = torch.from_numpy(split["targets"].astype(np.int64)).to("cuda")
    scored_per_row = args.seq_len - int(manifest["model"]["prediction_start"])
    fit_stop = args.fit_rows * scored_per_row
    validation_stop = fit_stop + args.validation_rows * scored_per_row
    if validation_stop >= targets.numel():
        raise ValueError("fit and validation rows must leave an untouched suffix")

    head = NonlinearProductHead(args.hidden).cuda()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    best_validation = float("inf")
    best_step = 0
    best_state = None
    trace = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            fit_stop,
            (args.train_batch_size,),
            device="cuda",
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = head(expert_log[indices])
        loss = F.cross_entropy(logits, targets[indices])
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0:
            validation_bpb = _bpb(
                head, expert_log, targets, fit_stop, validation_stop
            )
            item = {
                "step": step,
                "fit_minibatch_bpb": float(loss.detach()) / math.log(2.0),
                "validation_bpb": validation_bpb,
            }
            trace.append(item)
            print(json.dumps(item, sort_keys=True), flush=True)
            if validation_bpb < best_validation:
                best_validation = validation_bpb
                best_step = step
                best_state = copy.deepcopy(head.state_dict())
    assert best_state is not None
    head.load_state_dict(best_state)
    report = {
        "format": "layercake-nonlinear-product-head-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection probe; not final evidence",
        "bundle": args.bundle,
        "logical_parameters_before_head": manifest["parameters"]["logical_total"],
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "data": {
            "path": split["path"],
            "bytes": split["bytes"],
            "sha256": split["sha256"],
            "sampling": "evenly_spaced_chronological_rows",
            "selected_rows": split["selected_rows"],
        },
        "fit_scored_bytes": fit_stop,
        "validation_scored_bytes": validation_stop - fit_stop,
        "untouched_scored_bytes": targets.numel() - validation_stop,
        "best_step": best_step,
        "best_validation_bpb": best_validation,
        "untouched": {
            "count_bpb": _expert_bpb(
                expert_log, targets, 0, validation_stop, targets.numel()
            ),
            "neural_bpb": _expert_bpb(
                expert_log, targets, 1, validation_stop, targets.numel()
            ),
            "nonlinear_product_bpb": _bpb(
                head, expert_log, targets, validation_stop, targets.numel()
            ),
        },
        "trace": trace,
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
