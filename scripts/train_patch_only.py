from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from _common import emit
from layercake.canonical_anchors import patch_context_anchors
from layercake.causal_byte_models import CausalByteLM, CausalBytePatchLM
from run_paired_byte_experiment import (
    batch,
    evaluate,
    load_jsonl_bytes,
    load_python_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--general-bytes", type=int, default=20_000_000)
    parser.add_argument("--domain-bytes", type=int, default=2_000_000)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-byte", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--d-abi", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--ngram-buckets", type=int, default=0)
    parser.add_argument(
        "--local-decoder",
        choices=[
            "gru",
            "conv",
            "transformer",
            "patch_transformer",
            "window_transformer",
        ],
        default="gru",
    )
    parser.add_argument("--conv-layers", type=int, default=4)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--local-window", type=int, default=16)
    parser.add_argument("--coarse-patch-size", type=int, default=0)
    parser.add_argument("--coarse-layers", type=int, default=0)
    parser.add_argument("--mtp-depth", type=int, default=0)
    parser.add_argument("--mtp-weight", type=float, default=0.2)
    parser.add_argument("--empirical-transition-head", action="store_true")
    parser.add_argument("--patch-unit-buckets", type=int, default=0)
    parser.add_argument("--patch-prediction", action="store_true")
    parser.add_argument("--patch-prediction-weight", type=float, default=0.5)
    parser.add_argument("--patch-prediction-stride", type=int, default=1)
    parser.add_argument(
        "--patch-prediction-mode",
        choices=["factorized", "autoregressive"],
        default="factorized",
    )
    parser.add_argument("--patch-generation-width", type=int, default=96)
    parser.add_argument("--patch-generation-context", type=int, default=0)
    parser.add_argument(
        "--patch-prediction-detach-context", action="store_true"
    )
    parser.add_argument(
        "--patch-prediction-context",
        choices=["global", "local"],
        default="global",
    )
    parser.add_argument("--patch-distill-weight", type=float, default=0.0)
    parser.add_argument("--patch-distill-temperature", type=float, default=1.0)
    parser.add_argument("--tie-byte-embeddings", action="store_true")
    parser.add_argument("--context-buckets", type=int, default=0)
    parser.add_argument("--context-order", type=int, default=3)
    parser.add_argument("--empirical-context-head", action="store_true")
    parser.add_argument("--local-position-embeddings", action="store_true")
    parser.add_argument("--modern-blocks", action="store_true")
    parser.add_argument("--fused-attention", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine", "late_cosine"],
        default="constant",
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--decay-start-ratio", type=float, default=0.8)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    general = load_jsonl_bytes(
        root.parent / "layercakeogwithdecoder/data/v6/redpajama_english_eval.jsonl",
        args.general_bytes,
    )
    domain = load_python_bytes(
        root.parent / "layercakeogwithdecoder", args.domain_bytes
    )
    general_train, general_eval = general[:-200_000], general[-200_000:]
    domain_eval = domain[-100_000:]
    transition_logits = None
    if args.empirical_transition_head:
        transition_ids = general_train[:-1] * 256 + general_train[1:]
        counts = torch.bincount(transition_ids, minlength=65536).reshape(256, 256)
        probabilities = (counts.float() + 0.1) / (
            counts.sum(dim=1, keepdim=True).float() + 25.6
        )
        transition_logits = probabilities.log()
    context_logits = None
    if args.context_buckets and args.empirical_context_head:
        context_ids = torch.zeros_like(general_train[:-1])
        source = general_train[:-1].unsqueeze(0)
        for lag in range(args.context_order):
            shifted = F.pad(
                source[:, : source.shape[1] - lag], (lag, 0)
            ).squeeze(0)
            context_ids = (
                context_ids * 257 + shifted + 1
            ) % args.context_buckets
        combined = context_ids * 256 + general_train[1:]
        counts = torch.bincount(
            combined, minlength=args.context_buckets * 256
        ).reshape(args.context_buckets, 256)
        probabilities = (counts.float() + 0.1) / (
            counts.sum(dim=1, keepdim=True).float() + 25.6
        )
        context_logits = probabilities.log()
    patch = CausalBytePatchLM(
        patch_size=args.patch_size,
        d_byte=args.d_byte,
        d_model=args.d_model,
        d_abi=args.d_abi,
        layers=args.layers,
        heads=args.heads,
        max_patches=args.seq // args.patch_size,
        continuous_local=True,
        direct_global_context=True,
        ngram_buckets=args.ngram_buckets,
        local_decoder=args.local_decoder,
        conv_layers=args.conv_layers,
        mtp_depth=args.mtp_depth,
        transition_logits=transition_logits,
        patch_unit_buckets=args.patch_unit_buckets,
        local_layers=args.local_layers,
        patch_prediction=args.patch_prediction,
        patch_prediction_stride=args.patch_prediction_stride,
        patch_prediction_mode=args.patch_prediction_mode,
        patch_generation_width=args.patch_generation_width,
        patch_generation_context=args.patch_generation_context,
        patch_prediction_detach_context=(
            args.patch_prediction_detach_context
        ),
        patch_prediction_context=args.patch_prediction_context,
        tie_byte_embeddings=args.tie_byte_embeddings,
        context_buckets=args.context_buckets,
        context_order=args.context_order,
        context_logits=context_logits,
        local_position_embeddings=args.local_position_embeddings,
        modern_blocks=args.modern_blocks,
        fused_attention=args.fused_attention,
        local_window=args.local_window,
        coarse_patch_size=args.coarse_patch_size,
        coarse_layers=args.coarse_layers,
    ).to(device)
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu")
        patch.load_state_dict(resumed["patch_model"])
    optimizer = torch.optim.AdamW(
        patch.parameters(),
        lr=args.lr,
        betas=(0.9, args.beta2),
        weight_decay=args.weight_decay,
    )
    def lr_multiplier(step: int) -> float:
        if args.lr_schedule == "constant":
            return 1.0
        if args.lr_schedule == "late_cosine":
            decay_start = round(args.steps * args.decay_start_ratio)
            if step <= decay_start:
                return 1.0
            progress = (step - decay_start) / max(
                args.steps - decay_start, 1
            )
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine
        if step <= args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(
            args.steps - args.warmup_steps, 1
        )
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_multiplier
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    started = time.time()
    history = []
    for step in range(1, args.steps + 1):
        x, y = batch(general_train, args.seq, args.batch, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = patch(
                x,
                return_aux=True,
                return_patch_prediction=args.patch_prediction,
            )
            if args.patch_prediction:
                logits, abi, aux_logits, patch_predictions = output
            else:
                logits, abi, aux_logits = output
                patch_predictions = []
            lm_loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            auxiliary_loss = lm_loss.new_zeros(())
            for index, auxiliary in enumerate(aux_logits, start=1):
                auxiliary_loss = auxiliary_loss + F.cross_entropy(
                    auxiliary[:, :-index].flatten(0, 1),
                    y[:, index:].flatten(),
                )
            if aux_logits:
                auxiliary_loss = auxiliary_loss / len(aux_logits)
            patch_prediction_loss = lm_loss.new_zeros(())
            patch_distill_loss = lm_loss.new_zeros(())
            if patch_predictions:
                targets = x[:, : logits.shape[1]].reshape(
                    x.shape[0], -1, args.patch_size
                )[:, 1:]
                target_indices = torch.arange(
                    0,
                    targets.shape[1],
                    args.patch_prediction_stride,
                    device=x.device,
                )
                targets = targets[:, target_indices]
                for offset, prediction in enumerate(patch_predictions):
                    prediction = prediction[
                        :, : targets.shape[1]
                    ]
                    patch_prediction_loss = patch_prediction_loss + F.cross_entropy(
                        prediction.flatten(0, 1),
                        targets[:, :, offset].flatten(),
                    )
                    if args.patch_distill_weight:
                        teacher_positions = (
                            (target_indices + 1) * args.patch_size
                            + offset
                            - 1
                        )
                        teacher = logits[:, teacher_positions].detach()
                        temperature = args.patch_distill_temperature
                        patch_distill_loss = (
                            patch_distill_loss
                            + F.kl_div(
                                F.log_softmax(
                                    prediction / temperature, dim=-1
                                ),
                                F.softmax(
                                    teacher / temperature, dim=-1
                                ),
                                reduction="batchmean",
                            )
                            * temperature**2
                            / prediction.shape[1]
                        )
                patch_prediction_loss = (
                    patch_prediction_loss / len(patch_predictions)
                )
                patch_distill_loss = (
                    patch_distill_loss / len(patch_predictions)
                )
            anchors = patch_context_anchors(
                x, args.d_abi, args.patch_size
            )
            anchor_loss = F.mse_loss(abi, anchors)
            loss = (
                lm_loss
                + args.anchor_weight * anchor_loss
                + args.mtp_weight * auxiliary_loss
                + args.patch_prediction_weight * patch_prediction_loss
                + args.patch_distill_weight * patch_distill_loss
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(patch.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step == 1 or step % 500 == 0:
            item = {
                "step": step,
                "lm_loss": lm_loss.item(),
                "anchor_loss": anchor_loss.item(),
                "auxiliary_loss": auxiliary_loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "patch_prediction_loss": patch_prediction_loss.item(),
                "patch_distill_loss": patch_distill_loss.item(),
            }
            history.append(item)
            print(item, flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started

    # Preserve the common artifact schema; the byte model is an untrained comparator
    # placeholder and is not used for patch-only quality claims.
    byte = CausalByteLM(
        d_model=args.d_model,
        d_abi=args.d_abi,
        layers=args.layers,
        heads=args.heads,
        max_len=args.seq,
    )
    artifact_args = {
        **vars(args),
        "d_model": args.d_model,
        "patch_d_model": args.d_model,
        "patch_layers": args.layers,
        "patch_heads": args.heads,
        "continuous_local": True,
        "direct_global_context": True,
        "ngram_buckets": args.ngram_buckets,
        "local_decoder": args.local_decoder,
        "conv_layers": args.conv_layers,
        "mtp_depth": args.mtp_depth,
        "empirical_transition_head": args.empirical_transition_head,
        "patch_unit_buckets": args.patch_unit_buckets,
        "patch_prediction": args.patch_prediction,
        "patch_prediction_stride": args.patch_prediction_stride,
        "patch_prediction_mode": args.patch_prediction_mode,
        "patch_generation_width": args.patch_generation_width,
        "patch_generation_context": args.patch_generation_context,
        "patch_prediction_detach_context": (
            args.patch_prediction_detach_context
        ),
        "patch_prediction_context": args.patch_prediction_context,
        "tie_byte_embeddings": args.tie_byte_embeddings,
        "context_buckets": args.context_buckets,
        "context_order": args.context_order,
        "local_position_embeddings": args.local_position_embeddings,
        "modern_blocks": args.modern_blocks,
        "fused_attention": args.fused_attention,
        "local_window": args.local_window,
        "local_layers": args.local_layers,
        "coarse_patch_size": args.coarse_patch_size,
        "coarse_layers": args.coarse_layers,
    }
    artifact_path = Path(args.artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": args.seed,
            "args": artifact_args,
            "byte_model": byte.state_dict(),
            "patch_model": patch.state_dict(),
            "training_mode": "patch_only_direct_global",
            "resumed_from": args.resume,
        },
        artifact_path,
    )
    result = {
        "status": "TRAINED",
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "estimated_bytes_per_update": args.batch * args.seq,
        "estimated_total_training_bytes": args.steps * args.batch * args.seq,
        "parameters": sum(p.numel() for p in patch.parameters()),
        "history": history,
        "general": evaluate(
            patch, general_eval, args.seq, args.batch, 30, device
        ),
        "python_domain": evaluate(
            patch, domain_eval, args.seq, args.batch, 30, device
        ),
    }
    emit(result, args.output)


if __name__ == "__main__":
    main()
