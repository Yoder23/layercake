from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from _common import emit
from artifact_utils import build_models
from run_paired_byte_experiment import (
    evaluate,
    load_jsonl_bytes,
    load_python_bytes,
    train_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.artifact, map_location="cpu")
    byte, patch = build_models(artifact, device)
    config = artifact["args"]
    root = Path(__file__).resolve().parents[1]
    general = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        config.get("general_bytes", 8_000_000),
    )
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder",
        config.get("domain_bytes", 2_000_000),
    )
    general_train, general_eval = general[:-200_000], general[-200_000:]
    domain_eval = domain[-100_000:]
    seq = config.get("seq", 128)
    batch_size = config.get("batch", 24)
    started = time.time()
    history = train_pair(
        byte,
        patch,
        general_train,
        args.steps,
        seq,
        batch_size,
        device,
        config.get("align_weight", 1.0),
        config.get("anchor_weight", 1.0),
    )
    artifact["byte_model"] = byte.state_dict()
    artifact["patch_model"] = patch.state_dict()
    previous = artifact.get("total_paired_steps", config.get("steps", 0))
    artifact["total_paired_steps"] = previous + args.steps
    artifact["continued_paired_steps"] = (
        artifact.get("continued_paired_steps", 0) + args.steps
    )
    output_artifact = Path(args.output_artifact)
    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_artifact)

    result = {
        "device": str(device),
        "continued_steps": args.steps,
        "total_paired_steps": artifact["total_paired_steps"],
        "elapsed_seconds": time.time() - started,
        "history": history,
        "byte_parameters": sum(p.numel() for p in byte.parameters()),
        "patch_parameters": sum(p.numel() for p in patch.parameters()),
        "byte_general": evaluate(
            byte, general_eval, seq, batch_size, 30, device
        ),
        "patch_general": evaluate(
            patch, general_eval, seq, batch_size, 30, device
        ),
        "byte_domain": evaluate(byte, domain_eval, seq, batch_size, 30, device),
        "patch_domain": evaluate(
            patch, domain_eval, seq, batch_size, 30, device
        ),
    }
    result["general_bpb_gap"] = (
        result["patch_general"]["bpb"] - result["byte_general"]["bpb"]
    )
    emit(result, args.output)


if __name__ == "__main__":
    main()
