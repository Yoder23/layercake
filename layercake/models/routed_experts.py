"""Causal sparse foundation experts with measured utilization.

Routing operates on each already-computed patch state.  No decision for a patch
depends on a later patch, which makes the routed path valid for autoregressive
training and exactly reproducible by incremental decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class FoundationExpert(nn.Module):
    def __init__(self, width: int, expansion: int = 4):
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(self.norm(hidden)).chunk(2, dim=-1)
        return hidden + self.down(F.silu(gate) * value)


@dataclass(frozen=True)
class RoutingSnapshot:
    mode: str
    routed_tokens: int
    assignments: tuple[int, ...]
    utilization: tuple[float, ...]
    probability_entropy: float
    normalized_entropy: float
    maximum_load_fraction: float
    experts_used: int

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "routed_tokens": self.routed_tokens,
            "assignments": list(self.assignments),
            "utilization": list(self.utilization),
            "probability_entropy": self.probability_entropy,
            "normalized_entropy": self.normalized_entropy,
            "maximum_load_fraction": self.maximum_load_fraction,
            "experts_used": self.experts_used,
        }


class CausalRoutedFoundationExperts(nn.Module):
    """Dispatch patch tokens without evaluating inactive experts.

    The parameter names intentionally match the V2 router (``router`` and
    ``experts``), so committed V2 checkpoints remain loadable when the default
    ``fixed`` mode is used.
    """

    MODES = {
        "fixed",
        "learned_top1",
        "learned_top2",
        "expert_choice",
        "microbatch",
        "hierarchical",
        "shared_top1",
    }

    def __init__(
        self,
        width: int,
        experts: int = 8,
        expansion: int = 4,
        *,
        mode: str = "fixed",
        temperature: float = 1.0,
    ):
        super().__init__()
        if experts < 2:
            raise ValueError("at least two routed foundation experts are required")
        if mode not in self.MODES:
            raise ValueError(f"unsupported routing mode: {mode}")
        if temperature <= 0:
            raise ValueError("routing temperature must be positive")
        self.width = int(width)
        self.expert_count = int(experts)
        self.mode = str(mode)
        self.temperature = float(temperature)
        self.router = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, experts, bias=False))
        self.experts = nn.ModuleList(
            FoundationExpert(width, expansion=expansion) for _ in range(experts)
        )
        self.route_override: int | None = 0 if mode == "fixed" else None
        self.last_routes: torch.Tensor | None = None
        self.last_probabilities: torch.Tensor | None = None
        self.last_snapshot: RoutingSnapshot | None = None
        self.last_assignment_counts: torch.Tensor | None = None
        self.last_load: torch.Tensor | None = None
        self.last_entropy: torch.Tensor | None = None

    def set_route(self, route: int | None) -> None:
        if route is not None and not 0 <= int(route) < self.expert_count:
            raise ValueError("route index is out of range")
        self.route_override = None if route is None else int(route)

    def _selection(self, hidden: torch.Tensor, probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = hidden.shape
        if self.route_override is not None:
            indices = torch.full(
                (batch, tokens, 1), self.route_override, dtype=torch.long, device=hidden.device
            )
            weights = torch.ones(batch, tokens, 1, dtype=hidden.dtype, device=hidden.device)
            return indices, weights
        if self.mode == "fixed":
            raise ValueError("fixed routing requires an explicit route")
        if self.mode == "microbatch":
            first = probabilities[:, :1].argmax(dim=-1, keepdim=True)
            indices = first.expand(batch, tokens, 1)
            selected = probabilities.gather(-1, indices)
            return indices, 1 + selected - selected.detach()
        if self.mode == "hierarchical":
            groups = min(4, self.expert_count)
            if self.expert_count % groups:
                groups = 2 if self.expert_count % 2 == 0 else 1
            per_group = self.expert_count // groups
            grouped = probabilities.reshape(batch, tokens, groups, per_group)
            group = grouped.sum(dim=-1).argmax(dim=-1)
            within = grouped.gather(
                2, group[..., None, None].expand(batch, tokens, 1, per_group)
            ).squeeze(2).argmax(dim=-1)
            indices = (group * per_group + within)[..., None]
            selected = probabilities.gather(-1, indices)
            return indices, 1 + selected - selected.detach()
        if self.mode == "learned_top2":
            values, indices = probabilities.topk(2, dim=-1)
            return indices, values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.mode == "expert_choice":
            # Each expert independently bids for a token.  At most the two best
            # bids are accepted and a below-uniform second bid is rejected.  The
            # choice uses only this token's causal state, never other tokens.
            values, indices = probabilities.topk(2, dim=-1)
            accept_second = (values[..., 1:2] >= (1.0 / self.expert_count)).to(values.dtype)
            weights = torch.cat([values[..., :1], values[..., 1:2] * accept_second], dim=-1)
            return indices, weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        indices = probabilities.argmax(dim=-1, keepdim=True)
        selected = probabilities.gather(-1, indices)
        # A straight-through gate gives the hard top-1 router a gradient without
        # turning the expert computation into a dense mixture.
        return indices, 1 + selected - selected.detach()

    def forward(
        self, hidden: torch.Tensor, *, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | dict]]:
        if hidden.ndim != 3:
            raise ValueError("routed expert input must be [batch, tokens, width]")
        logits = self.router(hidden) / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
        indices, weights = self._selection(hidden, probabilities)
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_indices = indices.reshape(flat_hidden.shape[0], indices.shape[-1])
        flat_weights = weights.reshape(flat_hidden.shape[0], weights.shape[-1])
        output = torch.zeros_like(flat_hidden)
        assignment_counts = torch.zeros(self.expert_count, device=hidden.device, dtype=torch.long)
        for expert_index, expert in enumerate(self.experts):
            for slot in range(flat_indices.shape[1]):
                rows = torch.nonzero(flat_indices[:, slot] == expert_index, as_tuple=False).flatten()
                if not rows.numel():
                    continue
                selected = flat_hidden.index_select(0, rows)
                transformed = expert(selected[:, None])[:, 0]
                selected_weight = flat_weights.index_select(0, rows)[:, slot:slot + 1]
                contribution = transformed * selected_weight.to(transformed.dtype)
                output.index_add_(0, rows, contribution)
                assignment_counts[expert_index] += rows.numel()
        output = output.reshape_as(hidden)
        assignments_total = assignment_counts.sum().clamp_min(1)
        load = assignment_counts.to(probabilities.dtype) / assignments_total
        importance = probabilities.mean(dim=(0, 1))
        balance_loss = self.expert_count * torch.sum(importance * load)
        entropy = -(probabilities.clamp_min(1e-9).log() * probabilities).sum(dim=-1).mean()
        normalized_entropy = entropy / torch.log(probabilities.new_tensor(float(self.expert_count)))
        self.last_routes = indices.detach()
        self.last_probabilities = probabilities.detach()
        self.last_assignment_counts = assignment_counts.detach()
        self.last_load = load.detach()
        self.last_entropy = entropy.detach()
        if return_aux:
            return output, {
                "balance_loss": balance_loss,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "load": load,
                "importance": importance,
                "assignment_counts": assignment_counts,
            }
        return output

    def snapshot(self) -> RoutingSnapshot | None:
        if self.last_assignment_counts is None or self.last_load is None or self.last_entropy is None:
            return None
        counts = tuple(int(value) for value in self.last_assignment_counts.cpu().tolist())
        utilization = tuple(float(value) for value in self.last_load.cpu().tolist())
        normalized = self.last_entropy / torch.log(
            self.last_entropy.new_tensor(float(self.expert_count))
        )
        self.last_snapshot = RoutingSnapshot(
            mode=self.mode,
            routed_tokens=int(sum(counts)),
            assignments=counts,
            utilization=utilization,
            probability_entropy=float(self.last_entropy.cpu()),
            normalized_entropy=float(normalized.cpu()),
            maximum_load_fraction=max(utilization),
            experts_used=sum(value > 0 for value in counts),
        )
        return self.last_snapshot

    def expert_parameters(self, route: int):
        if not 0 <= int(route) < self.expert_count:
            raise ValueError("route index is out of range")
        yield from self.experts[int(route)].parameters()
