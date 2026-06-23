import torch

from layercake.abi_alignment import (
    abi_anchor_loss,
    abi_distribution_drift,
    abi_pairwise_alignment_loss,
    abi_whitening_loss,
    orthogonal_procrustes,
)


def test_alignment_losses_zero_on_identity():
    z = torch.randn(4, 8, 16)
    assert abi_anchor_loss(z, z).item() == 0.0
    assert abi_pairwise_alignment_loss(z, z).item() == 0.0
    assert abi_distribution_drift(z, z)["total"] == 0.0
    assert abi_whitening_loss(z).isfinite()


def test_procrustes_diagnostic_shape():
    source = torch.randn(32, 8)
    target = torch.randn(32, 8)
    rotation = orthogonal_procrustes(source, target)
    assert rotation.shape == (8, 8)
    assert torch.allclose(rotation.T @ rotation, torch.eye(8), atol=1e-5)
