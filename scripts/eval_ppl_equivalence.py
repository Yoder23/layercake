from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from _common import emit
from artifact_utils import build_brick, build_models
from run_paired_byte_experiment import evaluate, load_python_bytes


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Strict unchanged-brick source/target PPL equivalence gate"
    )
    parser.add_argument("--brick", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brick_artifact = torch.load(args.brick, map_location="cpu")
    source = torch.load(brick_artifact["source_core"], map_location="cpu")
    target = torch.load(args.target, map_location="cpu")
    _, source_patch = build_models(source, device)
    _, target_patch = build_models(target, device)
    brick = build_brick(brick_artifact["brick_config"], device)
    brick.load_state_dict(brick_artifact["brick"])

    source_seq = source["args"].get("seq", 128)
    target_seq = target["args"].get("seq", 128)
    if source_seq != target_seq:
        raise ValueError(
            f"strict PPL equivalence requires equal context lengths: "
            f"{source_seq} != {target_seq}"
        )
    source_patch_size = source["args"].get("patch_size", 4)
    target_patch_size = target["args"].get("patch_size", 4)
    if source_patch_size != target_patch_size:
        raise ValueError("strict PPL equivalence requires equal patch contracts")
    source_d_abi = source["args"].get("d_abi", 64)
    target_d_abi = target["args"].get("d_abi", 64)
    if source_d_abi != target_d_abi:
        raise ValueError("strict PPL equivalence requires equal d_abi")

    root = Path(__file__).resolve().parents[1]
    domain_limit = max(
        source["args"].get("domain_bytes", 2_000_000),
        target["args"].get("domain_bytes", 2_000_000),
    )
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder", domain_limit
    )[-args.eval_bytes :]
    eval_hash = tensor_hash(domain)
    batch_size = min(
        source["args"].get("batch", 24),
        target["args"].get("batch", 24),
        24,
    )
    source_base = evaluate(
        source_patch, domain, source_seq, batch_size, args.batches, device
    )
    source_brick = evaluate(
        source_patch, domain, source_seq, batch_size, args.batches, device, brick
    )
    target_base = evaluate(
        target_patch, domain, target_seq, batch_size, args.batches, device
    )
    target_brick = evaluate(
        target_patch, domain, target_seq, batch_size, args.batches, device, brick
    )

    ppl_ratio = target_brick["ppl"] / source_brick["ppl"]
    symmetric_ppl_ratio = max(ppl_ratio, 1.0 / ppl_ratio)
    loss_gap = abs(target_brick["loss"] - source_brick["loss"])
    bpb_gap = abs(target_brick["bpb"] - source_brick["bpb"])
    source_gain = source_base["loss"] - source_brick["loss"]
    target_gain = target_base["loss"] - target_brick["loss"]
    gain_ratio = (
        target_gain / source_gain if abs(source_gain) > 1e-12 else None
    )
    passed = symmetric_ppl_ratio <= 1.0 + args.tolerance

    emit(
        {
            "status": "PASS" if passed else "FAIL",
            "contract": {
                "unchanged_brick_payload": True,
                "same_eval_bytes": True,
                "eval_sha256": eval_hash,
                "eval_bytes": domain.numel(),
                "context_bytes": source_seq,
                "patch_size": source_patch_size,
                "d_abi": source_d_abi,
                "ppl_tolerance": args.tolerance,
            },
            "source": {
                "seed": source["seed"],
                "d_model": source["args"].get("patch_d_model")
                or source["args"].get("d_model", 128),
                "base": source_base,
                "brick": source_brick,
                "nll_gain": source_gain,
            },
            "target": {
                "seed": target["seed"],
                "d_model": target["args"].get("patch_d_model")
                or target["args"].get("d_model", 128),
                "base": target_base,
                "brick": target_brick,
                "nll_gain": target_gain,
            },
            "equivalence": {
                "target_over_source_ppl": ppl_ratio,
                "symmetric_ppl_ratio": symmetric_ppl_ratio,
                "absolute_loss_gap": loss_gap,
                "absolute_bpb_gap": bpb_gap,
                "target_over_source_nll_gain": gain_ratio,
            },
        },
        args.output,
    )


if __name__ == "__main__":
    main()
