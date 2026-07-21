from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from _common import emit
from artifact_utils import build_models
from layercake.canonical_anchors import patch_context_anchors
from run_paired_byte_experiment import batch, evaluate, load_jsonl_bytes, load_python_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    byte, patch = build_models(artifact, device)
    byte.eval()
    for parameter in byte.parameters():
        parameter.requires_grad_(False)
    patch.train()
    optimizer = torch.optim.AdamW(patch.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(777)
    root = Path(__file__).resolve().parents[1]
    artifact_args = artifact["args"]
    general = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        artifact_args.get("general_bytes", 8_000_000),
    )
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder",
        artifact_args.get("domain_bytes", 2_000_000),
    )
    general_train, general_eval = general[:-200_000], general[-200_000:]
    domain_eval = domain[-100_000:]
    seq = artifact_args.get("seq", 128)
    batch_size = min(artifact_args.get("batch", 24), 24)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        x, y = batch(general_train, seq, batch_size, generator, device)
        with torch.no_grad():
            _, byte_abi = byte(x)
            byte_boundaries = byte.boundary_abi(byte_abi, patch.patch_size)
        logits, patch_abi = patch(x)
        lm_loss = F.cross_entropy(
            logits.flatten(0, 1), y[:, : logits.shape[1]].flatten()
        )
        align = F.mse_loss(byte_boundaries[:, :-1], patch_abi[:, 1:])
        anchors = patch_context_anchors(x, patch_abi.shape[-1], patch.patch_size)
        anchor = F.mse_loss(patch_abi, anchors)
        loss = lm_loss + align + anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(patch.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 500 == 0:
            item = {
                "step": step, "lm_loss": lm_loss.item(),
                "align": align.item(), "anchor": anchor.item(),
            }
            history.append(item)
            print(item, flush=True)
    artifact["patch_model"] = patch.state_dict()
    artifact["continued_patch_steps"] = artifact.get("continued_patch_steps", 0) + args.steps
    torch.save(artifact, args.output_artifact)
    patch_general = evaluate(patch, general_eval, seq, batch_size, 30, device)
    byte_general = evaluate(byte, general_eval, seq, batch_size, 30, device)
    patch_domain = evaluate(patch, domain_eval, seq, batch_size, 30, device)
    byte_domain = evaluate(byte, domain_eval, seq, batch_size, 30, device)
    emit(
        {
            "steps": args.steps, "elapsed_seconds": time.time() - started,
            "history": history,
            "byte_general": byte_general, "patch_general": patch_general,
            "byte_domain": byte_domain, "patch_domain": patch_domain,
            "general_ppl_ratio": patch_general["ppl"] / byte_general["ppl"],
            "domain_ppl_ratio": patch_domain["ppl"] / byte_domain["ppl"],
            "patch_parameters": sum(p.numel() for p in patch.parameters()),
            "byte_parameters": sum(p.numel() for p in byte.parameters()),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
