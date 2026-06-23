from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import _common  # Ensures the repository root is importable when run as a script.
from artifact_utils import build_brick, build_models
from layercake.canonical_anchors import patch_context_anchors
from layercake.domain_bricks import SparseLowRankDomainOperator
from run_paired_byte_experiment import batch, load_python_bytes


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locate the source of strict unchanged-brick PPL divergence"
    )
    parser.add_argument("--brick", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brick_artifact = torch.load(args.brick, map_location="cpu")
    source_artifact = torch.load(brick_artifact["source_core"], map_location="cpu")
    target_artifact = torch.load(args.target, map_location="cpu")
    _, source = build_models(source_artifact, device)
    _, target = build_models(target_artifact, device)
    brick = build_brick(brick_artifact["brick_config"], device)
    brick.load_state_dict(brick_artifact["brick"])
    source.eval()
    target.eval()
    brick.eval()

    source_args = source_artifact["args"]
    target_args = target_artifact["args"]
    seq = source_args.get("seq", 128)
    patch_size = source_args.get("patch_size", 4)
    d_abi = source_args.get("d_abi", 64)
    if seq != target_args.get("seq", 128):
        raise ValueError("diagnosis requires equal context lengths")
    if patch_size != target_args.get("patch_size", 4):
        raise ValueError("diagnosis requires equal patch sizes")
    if d_abi != target_args.get("d_abi", 64):
        raise ValueError("diagnosis requires equal ABI widths")

    root = Path(__file__).resolve().parents[1]
    domain_limit = max(
        source_args.get("domain_bytes", 2_000_000),
        target_args.get("domain_bytes", 2_000_000),
    )
    stream = load_python_bytes(
        root.parent / "layercakeogwithdecoder", domain_limit
    )[-args.eval_bytes :]
    batch_size = min(
        source_args.get("batch", 24), target_args.get("batch", 24), 24
    )
    generator = torch.Generator().manual_seed(991)

    metrics: dict[str, list[float]] = {
        "abi_cosine": [],
        "abi_relative_l2": [],
        "source_anchor_mse": [],
        "target_anchor_mse": [],
        "delta_cosine": [],
        "delta_relative_l2": [],
        "correction_cosine": [],
        "correction_relative_l2": [],
        "base_logit_cosine": [],
        "base_logit_relative_l2": [],
        "source_base_loss": [],
        "target_base_loss": [],
        "source_learned_brick_loss": [],
        "target_learned_brick_loss": [],
        "source_anchor_brick_loss": [],
        "target_anchor_brick_loss": [],
    }
    top1_agreement: list[float] = []
    topk_agreement: list[float] = []

    with torch.no_grad():
        for _ in range(args.batches):
            x, y = batch(stream, seq, batch_size, generator, device)
            source_logits, source_abi = source(x)
            target_logits, target_abi = target(x)
            anchors = patch_context_anchors(x, d_abi, patch_size)

            if isinstance(brick, SparseLowRankDomainOperator):
                source_bricked, source_routing = brick(
                    source_abi, return_routing=True
                )
                target_bricked, target_routing = brick(
                    target_abi, return_routing=True
                )
                source_topk = source_routing.topk(brick.top_k, dim=-1).indices
                target_topk = target_routing.topk(brick.top_k, dim=-1).indices
                top1_agreement.append(
                    (source_topk[..., 0] == target_topk[..., 0])
                    .float()
                    .mean()
                    .item()
                )
                topk_agreement.append(
                    (
                        source_topk.sort(dim=-1).values
                        == target_topk.sort(dim=-1).values
                    )
                    .all(dim=-1)
                    .float()
                    .mean()
                    .item()
                )
            else:
                source_bricked = brick(source_abi)
                target_bricked = brick(target_abi)

            source_delta = source_bricked - source_abi
            target_delta = target_bricked - target_abi
            source_correction = source_delta @ source.canonical_head
            target_correction = target_delta @ target.canonical_head

            anchor_delta = brick(anchors) - anchors
            anchor_correction = anchor_delta @ source.canonical_head

            def expand(correction: torch.Tensor) -> torch.Tensor:
                return correction.unsqueeze(2).expand(
                    -1, -1, patch_size, -1
                ).reshape(correction.shape[0], -1, 256)

            source_adjusted = source_logits + expand(source_correction)
            target_adjusted = target_logits + expand(target_correction)
            source_anchor_adjusted = source_logits + expand(anchor_correction)
            target_anchor_adjusted = target_logits + expand(anchor_correction)
            labels = y[:, : source_logits.shape[1]]

            def loss(logits: torch.Tensor) -> float:
                return F.cross_entropy(
                    logits.flatten(0, 1), labels.flatten()
                ).item()

            metrics["abi_cosine"].append(cosine(source_abi, target_abi))
            metrics["abi_relative_l2"].append(
                relative_l2(source_abi, target_abi)
            )
            metrics["source_anchor_mse"].append(
                F.mse_loss(source_abi, anchors).item()
            )
            metrics["target_anchor_mse"].append(
                F.mse_loss(target_abi, anchors).item()
            )
            metrics["delta_cosine"].append(cosine(source_delta, target_delta))
            metrics["delta_relative_l2"].append(
                relative_l2(source_delta, target_delta)
            )
            metrics["correction_cosine"].append(
                cosine(source_correction, target_correction)
            )
            metrics["correction_relative_l2"].append(
                relative_l2(source_correction, target_correction)
            )
            metrics["base_logit_cosine"].append(
                cosine(source_logits, target_logits)
            )
            metrics["base_logit_relative_l2"].append(
                relative_l2(source_logits, target_logits)
            )
            metrics["source_base_loss"].append(loss(source_logits))
            metrics["target_base_loss"].append(loss(target_logits))
            metrics["source_learned_brick_loss"].append(loss(source_adjusted))
            metrics["target_learned_brick_loss"].append(loss(target_adjusted))
            metrics["source_anchor_brick_loss"].append(
                loss(source_anchor_adjusted)
            )
            metrics["target_anchor_brick_loss"].append(
                loss(target_anchor_adjusted)
            )

    averaged = {name: mean(values) for name, values in metrics.items()}
    learned_ppl_ratio = math.exp(
        averaged["target_learned_brick_loss"]
        - averaged["source_learned_brick_loss"]
    )
    anchor_ppl_ratio = math.exp(
        averaged["target_anchor_brick_loss"]
        - averaged["source_anchor_brick_loss"]
    )
    result = {
        "source_seed": source_artifact["seed"],
        "target_seed": target_artifact["seed"],
        "source_d_model": source_args.get("patch_d_model")
        or source_args.get("d_model", 128),
        "target_d_model": target_args.get("patch_d_model")
        or target_args.get("d_model", 128),
        "batches": args.batches,
        "metrics": averaged,
        "routing": {
            "top1_agreement": mean(top1_agreement)
            if top1_agreement
            else None,
            "exact_topk_set_agreement": mean(topk_agreement)
            if topk_agreement
            else None,
        },
        "ppl_ratios": {
            "learned_abi_target_over_source": learned_ppl_ratio,
            "shared_anchor_input_target_over_source": anchor_ppl_ratio,
        },
        "interpretation": {
            "shared_anchor_correction_is_identical": True,
            "remaining_anchor_mode_gap_is_entirely_base_model_dependent": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
