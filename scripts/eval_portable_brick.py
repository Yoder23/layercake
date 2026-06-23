from __future__ import annotations

import argparse
from pathlib import Path
import torch

from _common import emit
from artifact_utils import build_brick, build_models
from run_paired_byte_experiment import evaluate, load_jsonl_bytes, load_python_bytes


def quantize_state_int8(state: dict[str, torch.Tensor]):
    result, max_diff = {}, 0.0
    for key, value in state.items():
        if not value.is_floating_point() or value.numel() <= 1:
            result[key] = value.clone()
            continue
        scale = value.abs().max().clamp_min(1e-8) / 127
        restored = (value / scale).round().clamp(-127, 127) * scale
        max_diff = max(max_diff, (value - restored).abs().max().item())
        result[key] = restored
    return result, max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brick", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--quantize-int8", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brick_artifact = torch.load(args.brick, map_location="cpu")
    target = torch.load(args.target, map_location="cpu")
    byte, _ = build_models(target, device)
    brick = build_brick(brick_artifact["brick_config"], device)
    state, quant_diff = brick_artifact["brick"], 0.0
    if args.quantize_int8:
        state, quant_diff = quantize_state_int8(state)
    brick.load_state_dict(state)
    root = Path(__file__).resolve().parents[1]
    target_args = target["args"]
    general = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        target_args.get("general_bytes", 8_000_000),
    )[-200_000:]
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder",
        target_args.get("domain_bytes", 2_000_000),
    )[-100_000:]
    seq, batch_size = target_args.get("seq", 128), min(target_args.get("batch", 24), 24)
    base_domain = evaluate(byte, domain, seq, batch_size, 20, device)
    transfer_domain = evaluate(byte, domain, seq, batch_size, 20, device, brick)
    base_general = evaluate(byte, general, seq, batch_size, 20, device)
    transfer_general = evaluate(byte, general, seq, batch_size, 20, device, brick)
    emit(
        {
            "source_seed": brick_artifact["source_seed"], "target_seed": target["seed"],
            "source_core": brick_artifact["source_core"],
            "target_d_model": target["args"].get("d_model", 128),
            "brick_config": brick_artifact["brick_config"],
            "quantized_int8": args.quantize_int8, "quantization_max_diff": quant_diff,
            "base_domain": base_domain, "transfer_domain": transfer_domain,
            "base_general": base_general, "transfer_general": transfer_general,
            "domain_ratio": transfer_domain["ppl"] / base_domain["ppl"],
            "general_ratio": transfer_general["ppl"] / base_general["ppl"],
            "status": "PASS" if transfer_domain["ppl"] < base_domain["ppl"]
            and transfer_general["ppl"] / base_general["ppl"] <= 1.05 else "FAIL",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
