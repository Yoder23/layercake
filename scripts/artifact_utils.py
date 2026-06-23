from __future__ import annotations

import torch

from layercake.abi import ABISpec
from layercake.causal_byte_models import CausalByteLM, CausalBytePatchLM
from layercake.domain_bricks import LowRankDomainOperator, SparseLowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec


def build_models(artifact: dict, device: torch.device):
    args = artifact["args"]
    byte = CausalByteLM(
        d_model=args.get("d_model", 128), d_abi=args.get("d_abi", 64),
        layers=args.get("layers", 3), heads=args.get("heads", 4),
        max_len=args.get("seq", 128),
    ).to(device)
    patch = CausalBytePatchLM(
        patch_size=args.get("patch_size", 4),
        d_byte=args.get("d_byte", 48),
        d_model=args.get("patch_d_model") or args.get("d_model", 128),
        d_abi=args.get("d_abi", 64),
        layers=args.get("patch_layers") or args.get("layers", 3),
        heads=args.get("patch_heads") or args.get("heads", 4),
        max_patches=args.get("seq", 128) // args.get("patch_size", 4),
        continuous_local=args.get("continuous_local", False),
        direct_global_context=args.get("direct_global_context", False),
        ngram_buckets=args.get("ngram_buckets", 0),
        local_decoder=args.get("local_decoder", "gru"),
        conv_layers=args.get("conv_layers", 4),
        mtp_depth=args.get("mtp_depth", 0),
        patch_unit_buckets=args.get("patch_unit_buckets", 0),
        local_layers=args.get("local_layers", 2),
        patch_prediction=args.get("patch_prediction", False),
        patch_prediction_stride=args.get("patch_prediction_stride", 1),
        patch_prediction_mode=args.get(
            "patch_prediction_mode", "factorized"
        ),
        patch_generation_width=args.get("patch_generation_width", 96),
        patch_generation_context=args.get("patch_generation_context", 0),
        patch_prediction_detach_context=args.get(
            "patch_prediction_detach_context", False
        ),
        patch_prediction_context=args.get(
            "patch_prediction_context", "global"
        ),
        tie_byte_embeddings=args.get("tie_byte_embeddings", False),
        context_buckets=args.get("context_buckets", 0),
        context_order=args.get("context_order", 3),
        local_position_embeddings=args.get(
            "local_position_embeddings", False
        ),
        modern_blocks=args.get("modern_blocks", False),
        fused_attention=args.get("fused_attention", False),
        local_window=args.get("local_window", 16),
        coarse_patch_size=args.get("coarse_patch_size", 0),
        coarse_layers=args.get("coarse_layers", 0),
    ).to(device)
    byte.load_state_dict(artifact["byte_model"])
    patch.load_state_dict(artifact["patch_model"], strict=False)
    return byte, patch


def build_brick(config: dict, device: torch.device):
    spec = ABISpec(
        version="lc-abi/2",
        d_abi=config["d_abi"],
        input_interface=InputInterfaceSpec(
            mode="byte_patch",
            patching=f"fixed:{config.get('patch_size', 4)}",
            max_patch_size=config.get("patch_size", 4),
        ),
    )
    if config["type"] == "low_rank":
        return LowRankDomainOperator(
            spec, rank=config["rank"], alpha_init=config.get("alpha_init", 0.01)
        ).to(device)
    if config["type"] == "sparse_low_rank":
        return SparseLowRankDomainOperator(
            spec,
            rank=config["rank"],
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            alpha_init=config.get("alpha_init", 0.01),
        ).to(device)
    raise ValueError(f"unsupported brick type: {config['type']}")
