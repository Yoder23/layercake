"""Exact explicit-route residual successor to the generic v17 host."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from layercake_extensions.route_isolated_shallow_sparse_core import (
    RouteIsolatedCoreError,
    RouteIsolatedShallowSparseCoreHost,
)


ROUTE_ISOLATED_CORE_V18_ABI_VERSION = "lc-direct-neural-core/18"
ROUTE_ISOLATED_CORE_V18_ABI_SHA256 = (
    "9c984b783d04c13d592d608610c6396bfbbfa16f4fd91bd8515e6df9aadbf370"
)
ARCHITECTURE_V18_FORMAT = "layercake-route-isolated-shallow-sparse-core/2-explicit-route-tensors"


class ExplicitRouteResidual(nn.Module):
    """One physically selected [rank,width]/[width,rank] expert per row."""

    def __init__(self, width: int, rank: int, routes: int) -> None:
        super().__init__()
        self.width = width
        self.rank = rank
        self.routes = routes
        self.norm = nn.LayerNorm(width)
        self.down = nn.Parameter(torch.empty(routes, rank, width))
        self.up = nn.Parameter(torch.empty(routes, width, rank))

    def delta(self, hidden: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden)
        outputs = torch.zeros_like(hidden)
        for route in routes.unique(sorted=True):
            route_id = int(route.item())
            if route_id < 0:
                continue
            if route_id >= self.routes:
                raise RouteIsolatedCoreError("residual route is outside the package contract")
            rows = torch.nonzero(routes == route_id, as_tuple=False).flatten()
            selected = normalized.index_select(0, rows)
            low = torch.einsum("bsw,rw->bsr", selected, self.down[route_id])
            delta = torch.einsum("bsr,wr->bsw", F.silu(low), self.up[route_id])
            outputs.index_copy_(0, rows, delta)
        return outputs


class ExactRouteIsolatedShallowSparseCoreHost(RouteIsolatedShallowSparseCoreHost):
    ABI_VERSION = ROUTE_ISOLATED_CORE_V18_ABI_VERSION
    ABI_SHA256 = ROUTE_ISOLATED_CORE_V18_ABI_SHA256
    ARCHITECTURE_FORMAT = ARCHITECTURE_V18_FORMAT
    RESIDUAL_TYPE = ExplicitRouteResidual
