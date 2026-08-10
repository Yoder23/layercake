"""Dual-path progressive core with an explicit MLP residual mean and basis."""

from __future__ import annotations

import torch
from torch import nn

from .dual_path_progressive_core import DualPathProgressiveCore, DualPathProgressiveLayer


class BasisAlignedProgressiveLayer(DualPathProgressiveLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mlp_residual_mean = nn.Parameter(torch.zeros(self.full_width))

    def _mlp_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp_residual_mean + super()._mlp_delta(hidden)


class BasisAlignedProgressiveCore(DualPathProgressiveCore):
    def __init__(
        self,
        *,
        fixed_vocab_size: int,
        full_width: int = 3072,
        bottleneck_width: int = 192,
        attention_heads: int = 2,
        replacement_layers: int = 32,
        intermediate_size: int = 768,
        rms_epsilon: float = 1e-5,
        rope_theta: float = 10000.0,
        maximum_source_actions: int = 192,
        maximum_target_actions: int = 320,
        maximum_sequence_actions: int = 512,
    ) -> None:
        requested_layers = int(replacement_layers)
        super().__init__(
            fixed_vocab_size=fixed_vocab_size,
            full_width=full_width,
            bottleneck_width=bottleneck_width,
            attention_heads=attention_heads,
            replacement_layers=0,
            intermediate_size=intermediate_size,
            rms_epsilon=rms_epsilon,
            rope_theta=rope_theta,
            maximum_source_actions=maximum_source_actions,
            maximum_target_actions=maximum_target_actions,
            maximum_sequence_actions=maximum_sequence_actions,
        )
        self.replacement_layers = requested_layers
        self.layers = nn.ModuleList(
            BasisAlignedProgressiveLayer(
                self.full_width,
                self.bottleneck_width,
                self.attention_heads,
                self.intermediate_size,
                rms_epsilon=self.rms_epsilon,
                rope_theta=self.rope_theta,
            )
            for _ in range(self.replacement_layers)
        )

    @staticmethod
    def parameter_count_for_config(
        *,
        fixed_vocab_size: int,
        full_width: int,
        bottleneck_width: int,
        replacement_layers: int,
        intermediate_size: int,
    ) -> int:
        dual = DualPathProgressiveCore.parameter_count_for_config(
            fixed_vocab_size=fixed_vocab_size,
            full_width=full_width,
            bottleneck_width=bottleneck_width,
            replacement_layers=replacement_layers,
            intermediate_size=intermediate_size,
        )
        return dual + replacement_layers * full_width
