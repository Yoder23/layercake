"""Progressive core with a gated nonlinear rank-768 residual path."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .dual_path_progressive_core import DualPathProgressiveCore, DualPathProgressiveLayer


class NonlinearRank768ProgressiveLayer(DualPathProgressiveLayer):
    def __init__(self, *args, residual_rank: int = 768, nonlinear_hidden: int = 384, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        del self.mlp_input_projection, self.mlp_norm, self.gate_up_proj, self.down_proj
        self.residual_rank = int(residual_rank); self.nonlinear_hidden = int(nonlinear_hidden)
        self.mlp_gate_up_projection = nn.Linear(self.full_width, 2 * self.nonlinear_hidden, bias=False)
        self.mlp_coefficient_projection = nn.Linear(self.nonlinear_hidden, self.residual_rank, bias=False)
        self.mlp_output_projection = nn.Linear(self.residual_rank, self.full_width, bias=False)
        self.mlp_residual_mean = nn.Parameter(torch.zeros(self.full_width))

    def _mlp_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = self.mlp_gate_up_projection(self.post_attention_norm(hidden)).chunk(2, dim=-1)
        coefficients = self.mlp_coefficient_projection(F.silu(gate) * up)
        return self.mlp_residual_mean + self.mlp_output_projection(coefficients)


class NonlinearRank768ProgressiveCore(DualPathProgressiveCore):
    def __init__(self, *, fixed_vocab_size: int, full_width: int = 3072, bottleneck_width: int = 192, attention_heads: int = 2, replacement_layers: int = 32, intermediate_size: int = 768, residual_rank: int = 768, nonlinear_hidden: int = 384, rms_epsilon: float = 1e-5, rope_theta: float = 10000.0, maximum_source_actions: int = 192, maximum_target_actions: int = 320, maximum_sequence_actions: int = 512) -> None:
        count = int(replacement_layers)
        super().__init__(fixed_vocab_size=fixed_vocab_size, full_width=full_width, bottleneck_width=bottleneck_width, attention_heads=attention_heads, replacement_layers=0, intermediate_size=intermediate_size, rms_epsilon=rms_epsilon, rope_theta=rope_theta, maximum_source_actions=maximum_source_actions, maximum_target_actions=maximum_target_actions, maximum_sequence_actions=maximum_sequence_actions)
        self.replacement_layers = count; self.residual_rank = int(residual_rank); self.nonlinear_hidden = int(nonlinear_hidden)
        self.layers = nn.ModuleList(NonlinearRank768ProgressiveLayer(self.full_width, self.bottleneck_width, self.attention_heads, self.intermediate_size, residual_rank=self.residual_rank, nonlinear_hidden=self.nonlinear_hidden, rms_epsilon=self.rms_epsilon, rope_theta=self.rope_theta) for _ in range(count))

    def canonical_config(self) -> dict:
        value = super().canonical_config(); value.update({"residual_rank": self.residual_rank, "nonlinear_hidden": self.nonlinear_hidden}); return value

    @staticmethod
    def parameter_count_for_config(*, fixed_vocab_size: int, full_width: int, bottleneck_width: int, replacement_layers: int, intermediate_size: int, residual_rank: int, nonlinear_hidden: int) -> int:
        del intermediate_size
        copied = 2 * fixed_vocab_size * full_width + (2 * replacement_layers + 1) * full_width
        attention = 2 * full_width * bottleneck_width + 4 * bottleneck_width * bottleneck_width + bottleneck_width
        residual = 2 * full_width * nonlinear_hidden + nonlinear_hidden * residual_rank + full_width * residual_rank + full_width
        return copied + replacement_layers * (attention + residual)
