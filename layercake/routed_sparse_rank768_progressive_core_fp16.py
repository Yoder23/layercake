"""Precision-conformant runtime successor for routed sparse rank-768 artifacts."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveCore


class PrecisionConformantRoutedSparseRank768ProgressiveCore(
    RoutedSparseRank768ProgressiveCore
):
    """Execute v15-compatible tensors while keeping router reduction in fp32."""

    def _select_route(self, source_ids: torch.Tensor) -> int:
        if source_ids.ndim != 2 or source_ids.shape[0] != 1 or not source_ids.shape[1]:
            raise ValueError("routed sparse host requires one nonempty request")
        feature = self.token_embedding(source_ids).float().mean(dim=1)
        feature = feature / torch.linalg.vector_norm(
            feature, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        logits = F.linear(
            feature,
            self.router.weight.float(),
            None if self.router.bias is None else self.router.bias.float(),
        )
        return int(logits.argmax(dim=-1).item())
