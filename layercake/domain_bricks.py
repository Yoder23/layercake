"""Portable dense, low-rank, and sparse ABI-space domain operators."""

from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from .abi import ABISpec


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    choices = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu, "tanh": torch.tanh}
    try:
        return choices[name]
    except KeyError as exc:
        raise ValueError(f"unsupported activation: {name}") from exc


class DomainOperator(nn.Module):
    brick_type = "base"

    def __init__(self, abi_spec: ABISpec, enabled: bool = True):
        super().__init__()
        self.abi_spec = abi_spec
        self.enabled = enabled

    def validate_abi(self, abi_spec: ABISpec) -> None:
        self.abi_spec.assert_compatible(abi_spec, brick_type=self.brick_type)

    def parameter_count(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def copy_lossless(self) -> "DomainOperator":
        return copy.deepcopy(self)


class LowRankDomainOperator(DomainOperator):
    brick_type = "low_rank"

    def __init__(
        self,
        abi_spec: ABISpec,
        rank: int = 16,
        activation: str = "silu",
        alpha_init: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__(abi_spec, enabled)
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.down = nn.Linear(abi_spec.d_abi, rank, bias=False)
        self.up = nn.Linear(rank, abi_spec.d_abi, bias=False)
        self.activation_name = activation
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor, abi_spec: ABISpec | None = None) -> torch.Tensor:
        if abi_spec is not None:
            self.validate_abi(abi_spec)
        if h.shape[-1] != self.abi_spec.d_abi:
            raise ValueError("input width does not match d_abi")
        if not self.enabled:
            return h
        delta = self.up(_activation(self.activation_name)(self.down(h)))
        return h + self.alpha * delta

    def estimated_flops_per_position(self) -> int:
        return 4 * self.abi_spec.d_abi * self.rank


class SparseLowRankDomainOperator(DomainOperator):
    brick_type = "sparse_low_rank"

    def __init__(
        self,
        abi_spec: ABISpec,
        rank: int = 8,
        num_experts: int = 8,
        top_k: int = 2,
        activation: str = "silu",
        alpha_init: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__(abi_spec, enabled)
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.rank = rank
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation_name = activation
        self.router = nn.Linear(abi_spec.d_abi, num_experts, bias=False)
        self.down = nn.Parameter(torch.empty(num_experts, rank, abi_spec.d_abi))
        self.up = nn.Parameter(torch.zeros(num_experts, abi_spec.d_abi, rank))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.normal_(self.down, std=0.02)

    def route_top_k(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(h)
        values, indices = logits.topk(self.top_k, dim=-1)
        return indices, torch.softmax(values, dim=-1)

    def routing_weights(self, h: torch.Tensor) -> torch.Tensor:
        indices, selected = self.route_top_k(h)
        logits = self.router(h)
        weights = torch.zeros_like(logits)
        return weights.scatter(-1, indices, selected)

    def forward(
        self,
        h: torch.Tensor,
        abi_spec: ABISpec | None = None,
        return_routing: bool = False,
    ):
        if abi_spec is not None:
            self.validate_abi(abi_spec)
        if h.shape[-1] != self.abi_spec.d_abi:
            raise ValueError("input width does not match d_abi")
        if not self.enabled:
            return (h, torch.zeros(*h.shape[:-1], self.num_experts, device=h.device)) if return_routing else h
        indices, selected_weights = self.route_top_k(h)
        flat_h = h.reshape(-1, h.shape[-1])
        flat_indices = indices.reshape(-1, self.top_k)
        flat_weights = selected_weights.reshape(-1, self.top_k)
        selected_down = self.down[flat_indices]
        selected_up = self.up[flat_indices]
        hidden = torch.einsum("nd,nkrd->nkr", flat_h, selected_down)
        hidden = _activation(self.activation_name)(hidden)
        deltas = torch.einsum("nkr,nkdr->nkd", hidden, selected_up)
        delta = (deltas * flat_weights.unsqueeze(-1)).sum(dim=1).reshape_as(h)
        result = h + self.alpha * delta
        if not return_routing:
            return result
        weights = torch.zeros(
            *h.shape[:-1], self.num_experts, device=h.device, dtype=h.dtype
        ).scatter(-1, indices, selected_weights)
        return result, weights

    def estimated_flops_per_position(self) -> int:
        active = self.top_k
        return (
            2 * self.abi_spec.d_abi * self.num_experts
            + 4 * self.abi_spec.d_abi * self.rank * active
        )


class GatedSparseDomainOperator(SparseLowRankDomainOperator):
    brick_type = "gated_sparse"
