"""Progressive core with hard top-1 sparse residual experts."""
from __future__ import annotations

from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

from .dual_path_progressive_core import DualPathProgressiveCore, DualPathProgressiveLayer


class SparseResidualProgressiveLayer(DualPathProgressiveLayer):
    def __init__(self, *args, residual_experts: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if residual_experts < 2:
            raise ValueError("sparse residual host requires at least two experts")
        del self.mlp_input_projection, self.mlp_norm, self.gate_up_proj, self.down_proj
        self.residual_experts = int(residual_experts)
        self.residual_router = nn.Linear(self.full_width, self.residual_experts, bias=False)
        self.expert_coefficient_weights = nn.Parameter(
            torch.empty(self.residual_experts, self.bottleneck_width, self.full_width)
        )
        self.expert_output_bases = nn.Parameter(
            torch.empty(self.residual_experts, self.full_width, self.bottleneck_width)
        )
        self.expert_residual_means = nn.Parameter(
            torch.zeros(self.residual_experts, self.full_width)
        )
        del self.mlp_output_projection
        for expert in range(self.residual_experts):
            nn.init.kaiming_uniform_(self.expert_coefficient_weights[expert], a=sqrt(5))
            nn.init.kaiming_uniform_(self.expert_output_bases[expert], a=sqrt(5))
        self.last_active_expert_counts: tuple[int, ...] = ()

    def _mlp_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        features = self.post_attention_norm(hidden)
        selected = self.residual_router(features).argmax(dim=-1)
        flat_features = features.reshape(-1, self.full_width)
        flat_selected = selected.reshape(-1)
        flat_output = torch.zeros_like(flat_features)
        counts = []
        for expert in range(self.residual_experts):
            locations = torch.nonzero(flat_selected == expert, as_tuple=False).flatten()
            counts.append(int(locations.numel()))
            if locations.numel() == 0:
                continue
            values = flat_features.index_select(0, locations)
            coefficients = F.linear(values, self.expert_coefficient_weights[expert])
            residual = F.linear(coefficients, self.expert_output_bases[expert])
            residual = residual + self.expert_residual_means[expert]
            flat_output.index_copy_(0, locations, residual)
        self.last_active_expert_counts = tuple(counts)
        return flat_output.view_as(features)


class SparseResidualProgressiveCore(DualPathProgressiveCore):
    def __init__(
        self, *, fixed_vocab_size: int, full_width: int = 3072,
        bottleneck_width: int = 192, attention_heads: int = 2,
        replacement_layers: int = 32, intermediate_size: int = 768,
        residual_experts: int = 4, rms_epsilon: float = 1e-5,
        rope_theta: float = 10000.0, maximum_source_actions: int = 192,
        maximum_target_actions: int = 320, maximum_sequence_actions: int = 512,
    ) -> None:
        count = int(replacement_layers)
        super().__init__(
            fixed_vocab_size=fixed_vocab_size, full_width=full_width,
            bottleneck_width=bottleneck_width, attention_heads=attention_heads,
            replacement_layers=0, intermediate_size=intermediate_size,
            rms_epsilon=rms_epsilon, rope_theta=rope_theta,
            maximum_source_actions=maximum_source_actions,
            maximum_target_actions=maximum_target_actions,
            maximum_sequence_actions=maximum_sequence_actions,
        )
        self.replacement_layers = count
        self.residual_experts = int(residual_experts)
        self.layers = nn.ModuleList(
            SparseResidualProgressiveLayer(
                self.full_width, self.bottleneck_width, self.attention_heads,
                self.intermediate_size, residual_experts=self.residual_experts,
                rms_epsilon=self.rms_epsilon, rope_theta=self.rope_theta,
            ) for _ in range(count)
        )

    def canonical_config(self) -> dict:
        value = super().canonical_config()
        value["residual_experts"] = self.residual_experts
        return value

    @staticmethod
    def parameter_count_for_config(
        *, fixed_vocab_size: int, full_width: int, bottleneck_width: int,
        replacement_layers: int, intermediate_size: int, residual_experts: int,
    ) -> int:
        del intermediate_size
        copied = 2 * fixed_vocab_size * full_width + (2 * replacement_layers + 1) * full_width
        attention = (
            2 * full_width * bottleneck_width
            + 4 * bottleneck_width * bottleneck_width
            + bottleneck_width
        )
        sparse_mlp = (
            full_width * residual_experts
            + residual_experts * (2 * full_width * bottleneck_width + full_width)
        )
        return copied + replacement_layers * (attention + sparse_mlp)
