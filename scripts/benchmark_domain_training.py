from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from _common import emit
from artifact_utils import build_brick, build_models
from run_paired_byte_experiment import (
    batch,
    evaluate,
    load_jsonl_bytes,
    load_python_bytes,
)


def train_full(model, domain, general, steps, seq, batch_size, device, preserve_weight):
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    domain_generator = torch.Generator().manual_seed(314)
    general_generator = torch.Generator().manual_seed(2718)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        x, y = batch(domain, seq, batch_size, domain_generator, device)
        gx, _ = batch(general, seq, batch_size, general_generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(x)
            domain_loss = F.cross_entropy(
                logits.flatten(0, 1), y[:, : logits.shape[1]].flatten()
            )
            with torch.no_grad():
                base_logits, _ = reference(gx)
            general_logits, _ = model(gx)
            preserve = F.kl_div(
                F.log_softmax(general_logits, dim=-1),
                F.softmax(base_logits, dim=-1),
                reduction="batchmean",
            ) / general_logits.shape[1]
            loss = domain_loss + preserve_weight * preserve
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started, (
        torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--sparse-brick", required=True)
    parser.add_argument("--sparse-result", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--preserve-weight", type=float, default=6.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = torch.load(args.core, map_location="cpu")
    _, base_patch = build_models(core, device)
    full_patch = copy.deepcopy(base_patch)
    brick_artifact = torch.load(args.sparse_brick, map_location="cpu")
    brick = build_brick(brick_artifact["brick_config"], device)
    brick.load_state_dict(brick_artifact["brick"])
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
    base_domain = evaluate(base_patch, domain_eval, seq, batch_size, 20, device)
    base_general = evaluate(base_patch, general_eval, seq, batch_size, 20, device)
    sparse_domain = evaluate(base_patch, domain_eval, seq, batch_size, 20, device, brick)
    sparse_general = evaluate(base_patch, general_eval, seq, batch_size, 20, device, brick)
    full_seconds, full_peak = train_full(
        full_patch, domain_train, general_train, args.steps, seq, batch_size,
        device, args.preserve_weight
    )
    full_domain = evaluate(full_patch, domain_eval, seq, batch_size, 20, device)
    full_general = evaluate(full_patch, general_eval, seq, batch_size, 20, device)
    sparse_result = json.loads(Path(args.sparse_result).read_text(encoding="utf-8"))
    full_params = sum(p.numel() for p in full_patch.parameters() if p.requires_grad)
    sparse_params = brick.parameter_count()
    emit(
        {
            "steps": args.steps, "device": str(device),
            "base": {"domain": base_domain, "general": base_general},
            "sparse": {
                "trainable_parameters": sparse_params,
                "wall_seconds": sparse_result["elapsed_seconds"],
                "domain": sparse_domain, "general": sparse_general,
            },
            "full_finetune": {
                "trainable_parameters": full_params,
                "wall_seconds": full_seconds,
                "peak_memory_bytes": full_peak,
                "domain": full_domain, "general": full_general,
            },
            "parameter_reduction": full_params / sparse_params,
            "wall_time_ratio_sparse_to_full": sparse_result["elapsed_seconds"] / full_seconds,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
