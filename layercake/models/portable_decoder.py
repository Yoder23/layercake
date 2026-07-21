"""Strict semantic construction of the two declared cake classes."""

from __future__ import annotations

from typing import Any

from torch import nn

from layercake.cake.package import CakePackage
from layercake.portable_domain import PortableDomainDecoder

from .routed_cakes import HostResidualCake


def portable_decoder_manifest_architecture(
    *, feature_width: int = 64, hidden_width: int = 256,
    architecture: str = "anchor_mlp", embedding_width: int = 64,
) -> dict[str, Any]:
    return {
        "name": "portable_domain_decoder",
        "feature_width": int(feature_width),
        "hidden_width": int(hidden_width),
        "decoder_architecture": architecture,
        "embedding_width": int(embedding_width),
        "anchor_version": "lc-causal-byte-anchor/1",
    }


def load_cake_module(package: CakePackage) -> nn.Module:
    manifest = package.manifest
    architecture = manifest.architecture
    if manifest.cake_type == "portable_decoder":
        if architecture.get("name") != "portable_domain_decoder":
            raise ValueError("portable decoder architecture name is invalid")
        allowed = {
            "name", "feature_width", "hidden_width", "decoder_architecture",
            "embedding_width", "anchor_version",
        }
        if set(architecture) != allowed:
            raise ValueError("portable decoder architecture metadata is incomplete or ambiguous")
        if architecture["anchor_version"] != "lc-causal-byte-anchor/1":
            raise ValueError("unsupported deterministic anchor contract")
        model = PortableDomainDecoder(
            feature_width=int(architecture["feature_width"]),
            hidden_width=int(architecture["hidden_width"]),
            architecture=str(architecture["decoder_architecture"]),
            embedding_width=int(architecture["embedding_width"]),
        )
    elif manifest.cake_type == "host_residual":
        if set(architecture) != {"name", "d_abi", "rank"} or architecture.get("name") != "host_residual":
            raise ValueError("host residual architecture metadata is incomplete or ambiguous")
        model = HostResidualCake(d_abi=int(architecture["d_abi"]), rank=int(architecture["rank"]))
    else:
        raise ValueError(f"unsupported cake type: {manifest.cake_type}")
    model.load_state_dict(package.tensors, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
