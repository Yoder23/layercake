from __future__ import annotations

import argparse
from pathlib import Path
import torch

from _common import emit
from run_paired_byte_experiment import evaluate, load_jsonl_bytes, load_python_bytes
from layercake.abi import ABISpec
from layercake.causal_byte_models import CausalByteLM, CausalBytePatchLM
from layercake.domain_bricks import LowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.load(args.source, map_location="cpu")
    target = torch.load(args.target, map_location="cpu")
    target_args = target["args"]
    byte_model = CausalByteLM(
        d_model=target_args.get("d_model", 128),
        d_abi=target_args.get("d_abi", 64),
        layers=target_args.get("layers", 3),
        heads=target_args.get("heads", 4),
        max_len=target_args.get("seq", 128),
    ).to(device)
    patch_model = CausalBytePatchLM(
        patch_size=target_args.get("patch_size", 4),
        d_byte=target_args.get("d_byte", 48),
        d_model=target_args.get("patch_d_model") or target_args.get("d_model", 128),
        d_abi=target_args.get("d_abi", 64),
        layers=target_args.get("patch_layers") or target_args.get("layers", 3),
        heads=target_args.get("patch_heads") or target_args.get("heads", 4),
        max_patches=target_args.get("seq", 128) // target_args.get("patch_size", 4),
        continuous_local=target_args.get("continuous_local", False),
    ).to(device)
    byte_model.load_state_dict(target["byte_model"])
    patch_model.load_state_dict(target["patch_model"])
    source_args = source["args"]
    if source_args.get("d_abi", 64) != target_args.get("d_abi", 64):
        raise ValueError("cross-size transfer requires the same d_abi")
    source_patch_size = source_args.get("patch_size", 4)
    spec = ABISpec(version="lc-abi/2", d_abi=source_args.get("d_abi", 64), input_interface=InputInterfaceSpec(mode="byte_patch", patching=f"fixed:{source_patch_size}", max_patch_size=source_patch_size))
    brick = LowRankDomainOperator(spec, rank=16, alpha_init=0.01).to(device)
    brick.load_state_dict(source["brick"])
    root = Path(__file__).resolve().parents[1]
    general = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        target_args.get("general_bytes", 8_000_000),
    )[-200_000:]
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder",
        target_args.get("domain_bytes", 2_000_000),
    )[-100_000:]
    seq = target_args.get("seq", 128)
    eval_batch = min(target_args.get("batch", 24), 24)
    base_domain = evaluate(byte_model, domain, seq, eval_batch, 20, device)
    transfer_domain = evaluate(byte_model, domain, seq, eval_batch, 20, device, brick)
    base_general = evaluate(byte_model, general, seq, eval_batch, 20, device)
    transfer_general = evaluate(byte_model, general, seq, eval_batch, 20, device, brick)
    emit(
        {
            "source_seed": source["seed"], "target_seed": target["seed"],
            "source_d_model": source_args.get("d_model", 128),
            "target_d_model": target_args.get("d_model", 128),
            "base_domain": base_domain, "transfer_domain": transfer_domain,
            "base_general": base_general, "transfer_general": transfer_general,
            "domain_ratio": transfer_domain["ppl"] / base_domain["ppl"],
            "general_ratio": transfer_general["ppl"] / base_general["ppl"],
            "status": "PASS" if transfer_domain["ppl"] < base_domain["ppl"] and transfer_general["ppl"] / base_general["ppl"] <= 1.05 else "FAIL",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
