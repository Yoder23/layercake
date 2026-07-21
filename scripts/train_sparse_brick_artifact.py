from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from _common import emit
from artifact_utils import build_brick, build_models
from run_paired_byte_experiment import (
    evaluate,
    load_jsonl_bytes,
    load_python_bytes,
    train_brick,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--preserve-weight", type=float, default=6.0)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = torch.load(args.core, map_location="cpu", weights_only=True)
    byte, patch = build_models(core, device)
    config = {
        "type": "sparse_low_rank", "d_abi": core["args"].get("d_abi", 64),
        "rank": args.rank, "num_experts": args.experts, "top_k": args.top_k,
        "alpha_init": 0.01, "patch_size": core["args"].get("patch_size", 4),
    }
    brick = build_brick(config, device)
    root = Path(__file__).resolve().parents[1]
    core_args = core["args"]
    general = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        core_args.get("general_bytes", 8_000_000),
    )
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder",
        core_args.get("domain_bytes", 2_000_000),
    )
    general_train, general_eval = general[:-200_000], general[-200_000:]
    domain_train, domain_eval = domain[:-100_000], domain[-100_000:]
    seq, batch_size = core_args.get("seq", 128), min(core_args.get("batch", 24), 24)
    started = time.time()
    train_brick(
        patch, brick, domain_train, general_train, args.steps, seq,
        batch_size, device, args.preserve_weight
    )
    base_domain = evaluate(byte, domain_eval, seq, batch_size, 20, device)
    target_domain = evaluate(byte, domain_eval, seq, batch_size, 20, device, brick)
    base_general = evaluate(byte, general_eval, seq, batch_size, 20, device)
    target_general = evaluate(byte, general_eval, seq, batch_size, 20, device, brick)
    patch_base_domain = evaluate(patch, domain_eval, seq, batch_size, 20, device)
    patch_target_domain = evaluate(
        patch, domain_eval, seq, batch_size, 20, device, brick
    )
    patch_base_general = evaluate(patch, general_eval, seq, batch_size, 20, device)
    patch_target_general = evaluate(
        patch, general_eval, seq, batch_size, 20, device, brick
    )
    artifact = {
        "source_core": args.core, "source_seed": core["seed"],
        "brick_config": config, "brick": brick.state_dict(),
    }
    torch.save(artifact, args.artifact)
    emit(
        {
            "source_seed": core["seed"], "brick_config": config,
            "parameters": brick.parameter_count(),
            "elapsed_seconds": time.time() - started,
            "base_domain": base_domain, "target_domain": target_domain,
            "base_general": base_general, "target_general": target_general,
            "patch_base_domain": patch_base_domain,
            "patch_target_domain": patch_target_domain,
            "patch_base_general": patch_base_general,
            "patch_target_general": patch_target_general,
            "domain_ratio": target_domain["ppl"] / base_domain["ppl"],
            "general_ratio": target_general["ppl"] / base_general["ppl"],
            "status": "PASS" if target_domain["ppl"] < base_domain["ppl"]
            and target_general["ppl"] / base_general["ppl"] <= 1.05 else "FAIL",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
