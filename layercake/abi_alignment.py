"""Reusable losses and diagnostics for canonical ABI alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def abi_anchor_loss(states: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    if states.shape != anchors.shape:
        raise ValueError("states and anchors must have equal shapes")
    return F.mse_loss(states, anchors)


def abi_pairwise_alignment_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError("aligned ABI tensors must have equal shapes")
    return F.mse_loss(a, b)


def abi_cross_interface_alignment_loss(
    tokenized: torch.Tensor, byte_patch: torch.Tensor
) -> torch.Tensor:
    return abi_pairwise_alignment_loss(tokenized, byte_patch)


def abi_whitening_loss(states: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    flat = states.reshape(-1, states.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(flat.shape[0] - 1, 1)
    identity = torch.eye(covariance.shape[0], device=states.device, dtype=states.dtype)
    return F.mse_loss(covariance, identity) + eps * centered.mean().square()


def abi_distribution_drift(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("ABI widths differ")
    a_flat, b_flat = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    mean_mse = F.mse_loss(a_flat.mean(0), b_flat.mean(0)).item()
    std_mse = F.mse_loss(a_flat.std(0, unbiased=False), b_flat.std(0, unbiased=False)).item()
    return {"mean_mse": mean_mse, "std_mse": std_mse, "total": mean_mse + std_mse}


def orthogonal_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Diagnostic map only; canonical alignment must be learned during training."""
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must be equal [samples, d_abi] matrices")
    u, _, vh = torch.linalg.svd(source.T @ target, full_matrices=False)
    return u @ vh
