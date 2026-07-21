"""Screen normalized geometric CountCake/neural composition on open validation data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.count_cake import load_count_cake_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seq-len", type=int, default=1056)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-bytes", type=int, default=200_000)
    parser.add_argument(
        "--scales", default="0,0.0625,0.125,0.25,0.5,0.75,1,1.5,2"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, manifest = load_count_cake_bundle(args.bundle, device=device)
    model.eval()
    payload = Path(args.data).read_bytes()[: args.max_bytes]
    data = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(torch.long)
    row_count = data.numel() // args.seq_len
    rows = data[: row_count * args.seq_len].reshape(row_count, args.seq_len)
    scales = tuple(float(value) for value in args.scales.split(","))
    total_nll = {scale: 0.0 for scale in scales}
    total_bytes = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, row_count, args.batch_size):
            batch = rows[offset : offset + args.batch_size].to(device)
            if model.chunking_mode == "delimiter":
                _, hidden = model._dynamic_neural_log_probs(batch)
            else:
                usable = batch.shape[1] // model.patch_size * model.patch_size
                batch = batch[:, :usable]
                context = model._patch_context(batch)
                targets = batch[:, model.prediction_start :].reshape(
                    batch.shape[0], -1, model.patch_size
                )
                _, hidden = model._neural_log_probs(
                    context, targets, rows=batch
                )
            neural = model._neural_probabilities(hidden).reshape(
                batch.shape[0], -1, 256
            )
            count = model.count_cake.all_probabilities(
                batch, start=model.prediction_start
            )
            observed = batch[:, model.prediction_start :]
            count_log = count.clamp_min(1e-30).log()
            neural_log = neural.clamp_min(1e-30).log()
            for scale in scales:
                log_probability = torch.log_softmax(
                    count_log + scale * neural_log, dim=-1
                )
                nll = -log_probability.gather(
                    -1, observed.unsqueeze(-1)
                ).sum()
                total_nll[scale] += float(nll)
            total_bytes += observed.numel()
    elapsed = time.perf_counter() - started
    results = [
        {
            "scale": scale,
            "nll": total_nll[scale] / total_bytes,
            "bpb": total_nll[scale] / total_bytes / math.log(2.0),
        }
        for scale in scales
    ]
    print(
        json.dumps(
            {
                "format": "layercake-product-screen/1",
                "bundle": args.bundle,
                "logical_parameters": manifest["parameters"]["logical_total"],
                "data": args.data,
                "evaluated_bytes": total_bytes,
                "scales": results,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
