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
from run_paired_byte_experiment import load_jsonl_bytes


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--segments", type=int, default=30)
    parser.add_argument("--prompt-bytes", type=int, default=128)
    parser.add_argument("--continuation-bytes", type=int, default=64)
    parser.add_argument("--baseline-bpb", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.continuation_bytes % 2:
        raise ValueError("continuation-bytes must be even")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.core, map_location="cpu")
    _, model = build_models(artifact, device)
    model.eval()
    root = Path(__file__).resolve().parents[1]
    stream = load_jsonl_bytes(
        root.parent
        / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        20_000_000,
    )[-200_000:]
    generator = torch.Generator().manual_seed(20260623)
    max_start = (
        stream.numel()
        - args.prompt_bytes
        - args.continuation_bytes
        - 1
    )
    starts = torch.randint(
        0, max_start, (args.segments,), generator=generator
    )
    total_loss = 0.0
    total_bytes = 0
    offset_losses = [0.0, 0.0]
    for start in starts.tolist():
        prompt = stream[
            start : start + args.prompt_bytes
        ].to(device).unsqueeze(0)
        continuation = stream[
            start
            + args.prompt_bytes : start
            + args.prompt_bytes
            + args.continuation_bytes
        ].to(device).reshape(1, -1, 2)
        state = model.begin_cached_generation(prompt)
        for target_patch in continuation.unbind(dim=1):
            _, logits = model.cached_generation_step(
                state,
                forced_patch=target_patch,
                return_logits=True,
            )
            for offset in range(2):
                loss = F.cross_entropy(
                    logits[:, offset], target_patch[:, offset],
                    reduction="sum",
                ).item()
                total_loss += loss
                offset_losses[offset] += loss
                total_bytes += target_patch.shape[0]
    loss = total_loss / total_bytes
    bpb = loss / math.log(2)
    result = {
        "status": "PASS" if bpb <= args.baseline_bpb else "FAIL",
        "artifact": args.core,
        "device": str(device),
        "segments": args.segments,
        "prompt_bytes": args.prompt_bytes,
        "evaluation_bytes": total_bytes,
        "loss": loss,
        "bpb": bpb,
        "ppl": math.exp(loss),
        "offset_bpb": [
            value / (total_bytes / 2) / math.log(2)
            for value in offset_losses
        ],
        "baseline_bpb": args.baseline_bpb,
        "bpb_margin": args.baseline_bpb - bpb,
        "scope": (
            "Teacher-forced quality of the exact stateful cached generation "
            "path on deterministic heldout byte continuations."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
