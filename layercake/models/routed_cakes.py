"""Physically sparse foundation experts and host-conditioned residual cakes."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FoundationCakeExpert(nn.Module):
    def __init__(self, width: int, expansion: int = 4):
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(self.norm(hidden)).chunk(2, dim=-1)
        return hidden + self.down(F.silu(gate) * value)


class Top1RoutedFoundationCakes(nn.Module):
    """Execute one expert per row (or one per homogeneous microbatch).

    Unselected expert modules are never called, receive no gradients, and do not
    allocate Adam state. The straight-through router gate learns despite hard routing.
    """

    def __init__(self, width: int, experts: int = 8, expansion: int = 4):
        super().__init__()
        if experts < 2:
            raise ValueError("at least two routed foundation cakes are required")
        self.width = int(width)
        self.expert_count = int(experts)
        self.router = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, experts, bias=False))
        self.experts = nn.ModuleList(
            FoundationCakeExpert(width, expansion=expansion) for _ in range(experts)
        )
        self.route_override: int | None = None
        self.last_routes: torch.Tensor | None = None
        self.last_probabilities: torch.Tensor | None = None

    def set_route(self, route: int | None) -> None:
        if route is not None and not 0 <= int(route) < self.expert_count:
            raise ValueError("route index is out of range")
        self.route_override = None if route is None else int(route)

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = hidden.mean(dim=1)
        probabilities = torch.softmax(self.router(pooled), dim=-1)
        if self.route_override is None:
            routes = probabilities.argmax(dim=-1)
        else:
            routes = torch.full(
                (hidden.shape[0],), self.route_override, dtype=torch.long, device=hidden.device
            )
        return routes, probabilities

    def forward(
        self, hidden: torch.Tensor, *, return_aux_loss: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        routes, probabilities = self.route(hidden)
        self.last_routes = routes.detach()
        self.last_probabilities = probabilities.detach()
        output = torch.empty_like(hidden)
        for index, expert in enumerate(self.experts):
            rows = torch.nonzero(routes == index, as_tuple=False).flatten()
            if rows.numel():
                selected = hidden.index_select(0, rows)
                output.index_copy_(0, rows, expert(selected))
        selected_probability = probabilities.gather(1, routes[:, None]).squeeze(1)
        straight_through = 1 + selected_probability - selected_probability.detach()
        output = hidden + (output - hidden) * straight_through[:, None, None]
        # Switch-style importance/load balance without executing inactive experts.
        importance = probabilities.mean(dim=0)
        load = F.one_hot(routes, self.expert_count).float().mean(dim=0)
        balance = self.expert_count * torch.sum(importance * load)
        return (output, balance) if return_aux_loss else output

    def expert_parameters(self, route: int):
        if not 0 <= int(route) < self.expert_count:
            raise ValueError("route index is out of range")
        yield from self.experts[int(route)].parameters()


class HostResidualCake(nn.Module):
    """Portable weights in a canonical ABI; behavior remains host-conditioned."""

    def __init__(self, d_abi: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        if d_abi <= 0 or rank <= 0:
            raise ValueError("d_abi and rank must be positive")
        self.d_abi = int(d_abi)
        self.rank = int(rank)
        self.down = nn.Linear(d_abi, rank, bias=False)
        self.up = nn.Linear(rank, d_abi, bias=False)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, abi_state: torch.Tensor) -> torch.Tensor:
        if abi_state.shape[-1] != self.d_abi:
            raise ValueError("host ABI state width does not match cake contract")
        return abi_state + self.alpha * self.up(F.silu(self.down(abi_state)))
