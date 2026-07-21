"""Validation-only sweep for a counted byte-transition residual.

This probe never reads the sealed test split. It combines the frozen v478
neural distribution with a bigram table counted from the training corpus and
reports held-out BPB for fixed residual scales.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_byte_core_from_config import (  # noqa: E402
    _build_model,
    _load_config_with_extends,
    _load_eval_byte_stream,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/production_v478_5m_longcontext_patch64_recurrent_hash_updates_layercake.json"
CHECKPOINT = ROOT / "runs_experiment/production_v478_5m_longcontext_patch64_recurrent_hash_updates_layercake/step_3000.pt"
TRAIN = ROOT / "runs_experiment/production_v24_corpus/train.bin"
EVAL = ROOT / "runs_experiment/production_v24_corpus/eval.bin"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _load_config_with_extends(CONFIG)
    model = _build_model(config["model"], device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    train = torch.frombuffer(bytearray(TRAIN.read_bytes()), dtype=torch.uint8).long()
    pair_ids = train[:-1] * 256 + train[1:]
    counts = torch.bincount(pair_ids, minlength=256 * 256).float().reshape(256, 256)
    counts.add_(0.1)
    prior = counts.log()
    prior = prior - torch.logsumexp(prior, dim=-1, keepdim=True)
    prior = prior.to(device)

    eval_stream = _load_eval_byte_stream(
        [EVAL], max_bytes=1_000_000, read_block_bytes=1_048_576
    )
    seq_len = int(config["training"]["seq_len"])
    batch_size = int(config["training"]["micro_batch_size"])
    generator = torch.Generator().manual_seed(int(config["training"]["eval_seed"]))
    scales = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
    totals = {scale: [0.0, 0] for scale in scales}
    max_start = eval_stream.numel() - seq_len

    with torch.inference_mode():
        for _ in range(int(config["training"]["eval_batches"])):
            starts = torch.randint(0, max_start, (batch_size,), generator=generator)
            rows = torch.stack(
                [eval_stream[start : start + seq_len] for start in starts]
            ).to(device)
            predictions, targets = model.domain_cake_patch_predictions(rows)
            neural = torch.stack(predictions, dim=2)
            usable = rows.shape[1] // model.patch_size * model.patch_size
            source_last = rows[:, :usable].reshape(
                rows.shape[0], -1, model.patch_size
            )[..., -1]
            previous = torch.cat([source_last.unsqueeze(-1), targets[..., :-1]], dim=-1)
            counted = prior[previous]
            offsets = torch.arange(targets.shape[-1], device=device)
            contexts = torch.arange(targets.shape[1], device=device)
            valid = (
                (contexts[:, None] + 1) * model.patch_size + offsets[None]
                < rows.shape[1]
            ).unsqueeze(0).expand_as(targets)
            for scale in scales:
                combined = F.log_softmax(neural + scale * counted, dim=-1)
                nll = -combined.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                totals[scale][0] += float(nll[valid].sum().item())
                totals[scale][1] += int(valid.sum().item())

    print(
        json.dumps(
            {
                "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
                "sealed_test_read": False,
                "results": {
                    str(scale): total / count / math.log(2.0)
                    for scale, (total, count) in totals.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
