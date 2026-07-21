"""Probe an exactly normalized product across CountCake order stages."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from probe_count_conditioned_product import _count_distribution  # noqa: E402


class OrderProduct(nn.Module):
    def __init__(self, stages: int) -> None:
        super().__init__()
        initial = torch.full((stages,), 0.01)
        initial[-1] = 0.5
        self.raw_exponents = nn.Parameter(torch.log(torch.expm1(initial)))

    def forward(self, log_stages: torch.Tensor) -> torch.Tensor:
        exponents = F.softplus(self.raw_exponents)
        return torch.einsum("nkv,k->nv", log_stages.float(), exponents)


@torch.inference_mode()
def _bpb(
    head: nn.Module,
    logs: torch.Tensor,
    targets: torch.Tensor,
    start: int,
    stop: int,
    batch_size: int = 2048,
) -> float:
    total = 0.0
    count = 0
    for offset in range(start, stop, batch_size):
        end = min(stop, offset + batch_size)
        total += float(
            F.cross_entropy(
                head(logs[offset:end]), targets[offset:end], reduction="sum"
            )
        )
        count += end - offset
    return total / count / math.log(2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--fit-rows", type=int, default=32)
    parser.add_argument("--validation-rows", type=int, default=32)
    parser.add_argument("--prepare-batch-size", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=24061)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    model, manifest = load_count_cake_bundle(args.bundle, device="cuda")
    model.eval()
    payload = Path(args.data).read_bytes()
    available_rows = len(payload) // args.seq_len
    selected_rows = min(args.rows, available_rows)
    row_indices = np.linspace(
        0, available_rows - 1, num=selected_rows, dtype=np.int64
    )
    payload_array = np.frombuffer(payload, dtype=np.uint8)[
        : available_rows * args.seq_len
    ].reshape(available_rows, args.seq_len)
    rows = torch.from_numpy(payload_array[row_indices].copy()).to(
        device="cuda", dtype=torch.long
    )
    predicted = args.seq_len - model.prediction_start
    stage_count = model.count_cake.max_order + 1
    stage_logs = np.empty(
        (selected_rows * predicted, stage_count, 256), dtype=np.float16
    )
    targets_np = np.empty(selected_rows * predicted, dtype=np.uint8)
    cursor = 0
    with torch.inference_mode():
        for start in range(0, selected_rows, args.prepare_batch_size):
            batch = rows[start : start + args.prepare_batch_size]
            distribution = _count_distribution(
                model.count_cake,
                batch,
                model.prediction_start,
                return_stages=True,
            )
            count = batch.shape[0] * predicted
            stage_logs[cursor : cursor + count] = (
                distribution.clamp_min(1e-30)
                .log()
                .reshape(count, stage_count, 256)
                .to(torch.float16)
                .cpu()
                .numpy()
            )
            targets_np[cursor : cursor + count] = (
                batch[:, model.prediction_start :]
                .reshape(-1)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            cursor += count
            print(
                json.dumps(
                    {"prepared_rows": min(start + args.prepare_batch_size, selected_rows)}
                ),
                flush=True,
            )
    del model, rows
    torch.cuda.empty_cache()
    logs = torch.from_numpy(stage_logs).to("cuda")
    targets = torch.from_numpy(targets_np.astype(np.int64)).to("cuda")
    fit_stop = args.fit_rows * predicted
    validation_stop = fit_stop + args.validation_rows * predicted
    if validation_stop >= targets.numel():
        raise ValueError("fit and validation rows must leave a test suffix")
    head = OrderProduct(stage_count).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 17)
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
        loss = F.cross_entropy(head(logs[indices]), targets[indices])
        loss.backward()
        optimizer.step()
        if step == 1 or step % 25 == 0:
            validation_bpb = _bpb(
                head, logs, targets, fit_stop, validation_stop
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
    test_start = validation_stop
    with torch.inference_mode():
        target_stage_logs = logs[test_start:].gather(
            2,
            targets[test_start:, None, None]
            .expand(-1, stage_count, 1),
        ).squeeze(-1).float()
        final_stage_bpb = float(
            -target_stage_logs[:, -1].mean() / math.log(2.0)
        )
        oracle_bpb = float(
            -target_stage_logs.max(dim=-1).values.mean() / math.log(2.0)
        )
    report = {
        "format": "layercake-count-order-product-probe/1",
        "status": "COMPLETE",
        "warning": "architecture-selection probe; not final evidence",
        "bundle": args.bundle,
        "logical_parameters_before_head": manifest["parameters"]["logical_total"],
        "head_parameters": stage_count,
        "data": {
            "path": args.data,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sampling": "evenly_spaced_chronological_rows",
            "selected_rows": selected_rows,
            "fit_rows": args.fit_rows,
            "validation_rows": args.validation_rows,
            "test_rows": selected_rows - args.fit_rows - args.validation_rows,
        },
        "best_step": best_step,
        "best_validation_bpb": best_validation,
        "fitted_exponents": F.softplus(head.raw_exponents).detach().cpu().tolist(),
        "untouched": {
            "final_stage_bpb": final_stage_bpb,
            "target_aware_oracle_bpb": oracle_bpb,
            "product_bpb": _bpb(
                head, logs, targets, test_start, targets.numel()
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
