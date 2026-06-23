from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifact_utils import build_models
from run_paired_byte_experiment import batch, load_jsonl_bytes


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.core, map_location="cpu")
    _, model = build_models(artifact, device)
    model.eval()
    if not model.patch_prediction:
        raise ValueError("artifact does not contain a patch generator")

    root = Path(__file__).resolve().parents[1]
    stream = load_jsonl_bytes(
        root.parent
        / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        20_000_000,
    )[-200_000:]
    seq = artifact["args"]["seq"]
    generator = torch.Generator().manual_seed(9917)
    loss_sum = 0.0
    byte_count = 0
    offset_loss_sums = [0.0 for _ in range(model.patch_size)]
    offset_counts = [0 for _ in range(model.patch_size)]
    for _ in range(args.batches):
        x, _ = batch(stream, seq, args.batch, generator, device)
        predictions = model(
            x, return_aux=True, return_patch_prediction=True
        )[3]
        targets = x.reshape(
            x.shape[0], -1, model.patch_size
        )[:, 1:]
        for offset, prediction in enumerate(predictions):
            prediction = prediction[:, : targets.shape[1]]
            offset_loss = F.cross_entropy(
                prediction.flatten(0, 1),
                targets[:, :, offset].flatten(),
                reduction="sum",
            ).item()
            offset_count = targets[:, :, offset].numel()
            loss_sum += offset_loss
            byte_count += offset_count
            offset_loss_sums[offset] += offset_loss
            offset_counts[offset] += offset_count
    loss = loss_sum / byte_count
    result = {
        "status": "PASS",
        "artifact": args.core,
        "device": str(device),
        "evaluation_bytes": byte_count,
        "loss": loss,
        "bpb": loss / math.log(2),
        "ppl": math.exp(loss),
        "offset_bpb": [
            (item / count) / math.log(2)
            for item, count in zip(offset_loss_sums, offset_counts)
        ],
        "scope": (
            "Teacher-forced heldout quality of the patch generator actually "
            "used by draft generation; this is distinct from core LM BPB."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
