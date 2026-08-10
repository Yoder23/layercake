"""Dual-attention rank-768 core with request-routed sparse residual correction."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .dual_path_progressive_core import DualPathProgressiveCore, DualPathProgressiveLayer
from .progressive_replacement_core import ProgressiveReplacementCoreState


class RoutedSparseRank768ProgressiveLayer(DualPathProgressiveLayer):
    def __init__(
        self,
        *args,
        residual_rank: int = 768,
        sparse_width: int = 384,
        routes: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        del self.mlp_input_projection, self.mlp_norm, self.gate_up_proj, self.down_proj
        self.residual_rank = int(residual_rank)
        self.sparse_width = int(sparse_width)
        self.routes = int(routes)
        self.secondary_attention_input_projection = nn.Linear(self.full_width, self.bottleneck_width, bias=False)
        self.secondary_attention_norm = type(self.attention_norm)(self.bottleneck_width, self.attention_norm.epsilon)
        self.secondary_qkv_proj = nn.Linear(self.bottleneck_width, 3 * self.bottleneck_width, bias=False)
        self.secondary_o_proj = nn.Linear(self.bottleneck_width, self.bottleneck_width, bias=False)
        self.secondary_attention_output_projection = nn.Linear(self.bottleneck_width, self.full_width, bias=False)
        self.sparse_gate_up_projection = nn.Linear(self.full_width, 2 * self.sparse_width, bias=False)
        self.linear_coefficient_projection = nn.Linear(self.full_width, self.residual_rank, bias=False)
        self.route_coefficient_projections = nn.ModuleList(
            nn.Linear(self.sparse_width, self.residual_rank, bias=False)
            for _ in range(self.routes)
        )
        self.mlp_output_projection = nn.Linear(self.residual_rank, self.full_width, bias=False)
        self.mlp_residual_mean = nn.Parameter(torch.zeros(self.full_width))

    def _secondary_qkv(self, latent: torch.Tensor, positions: torch.Tensor):
        batch, length, _ = latent.shape
        qkv = self.secondary_qkv_proj(self.secondary_attention_norm(latent)).view(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        from .structural_causal_core import _rotary

        return (
            _rotary(query.transpose(1, 2), positions, self.rope_theta),
            _rotary(key.transpose(1, 2), positions, self.rope_theta),
            value.transpose(1, 2),
        )

    def _mlp_delta(self, hidden: torch.Tensor, route_index: int) -> torch.Tensor:
        if not 0 <= int(route_index) < self.routes:
            raise ValueError("routed sparse route index is invalid")
        feature = self.post_attention_norm(hidden)
        gate, up = self.sparse_gate_up_projection(feature).chunk(2, dim=-1)
        coefficients = self.linear_coefficient_projection(feature)
        coefficients = coefficients + self.route_coefficient_projections[int(route_index)](
            F.silu(gate) * up
        )
        return self.mlp_residual_mean + self.mlp_output_projection(coefficients)

    def forward_with_cache(self, hidden: torch.Tensor, positions: torch.Tensor, route_index: int):
        normalized = self.input_norm(hidden)
        primary_latent = self.attention_input_projection(normalized)
        primary_query, primary_key, primary_value = self._qkv(primary_latent, positions)
        primary_attended = self._attention(
            primary_query, primary_key, primary_value, causal=True
        ).transpose(1, 2).contiguous().view_as(primary_latent)
        secondary_latent = self.secondary_attention_input_projection(normalized)
        secondary_query, secondary_key, secondary_value = self._secondary_qkv(
            secondary_latent, positions
        )
        secondary_attended = self._attention(
            secondary_query, secondary_key, secondary_value, causal=True
        ).transpose(1, 2).contiguous().view_as(secondary_latent)
        hidden = hidden + self.attention_output_projection(self.o_proj(primary_attended))
        hidden = hidden + self.secondary_attention_output_projection(
            self.secondary_o_proj(secondary_attended)
        )
        hidden = hidden + self._mlp_delta(hidden, route_index)
        return (
            hidden,
            torch.cat((primary_key, secondary_key), dim=1),
            torch.cat((primary_value, secondary_value), dim=1),
        )

    def incremental(
        self,
        hidden: torch.Tensor,
        position: torch.Tensor,
        previous_key: torch.Tensor,
        previous_value: torch.Tensor,
        route_index: int,
    ):
        if previous_key.shape[1] != 2 * self.heads or previous_value.shape[1] != 2 * self.heads:
            raise ValueError("routed sparse cache topology changed")
        normalized = self.input_norm(hidden)
        primary_latent = self.attention_input_projection(normalized)
        primary_query, primary_key, primary_value = self._qkv(primary_latent, position)
        secondary_latent = self.secondary_attention_input_projection(normalized)
        secondary_query, secondary_key, secondary_value = self._secondary_qkv(
            secondary_latent, position
        )
        previous_primary_key, previous_secondary_key = previous_key.split(self.heads, dim=1)
        previous_primary_value, previous_secondary_value = previous_value.split(self.heads, dim=1)
        all_primary_keys = torch.cat((previous_primary_key, primary_key), dim=2)
        all_primary_values = torch.cat((previous_primary_value, primary_value), dim=2)
        all_secondary_keys = torch.cat((previous_secondary_key, secondary_key), dim=2)
        all_secondary_values = torch.cat((previous_secondary_value, secondary_value), dim=2)
        primary_attended = self._attention(
            primary_query, all_primary_keys, all_primary_values, causal=False
        ).transpose(1, 2).contiguous().view_as(primary_latent)
        secondary_attended = self._attention(
            secondary_query, all_secondary_keys, all_secondary_values, causal=False
        ).transpose(1, 2).contiguous().view_as(secondary_latent)
        hidden = hidden + self.attention_output_projection(self.o_proj(primary_attended))
        hidden = hidden + self.secondary_attention_output_projection(
            self.secondary_o_proj(secondary_attended)
        )
        hidden = hidden + self._mlp_delta(hidden, route_index)
        return (
            hidden,
            torch.cat((all_primary_keys, all_secondary_keys), dim=1),
            torch.cat((all_primary_values, all_secondary_values), dim=1),
        )


class RoutedSparseRank768ProgressiveCore(DualPathProgressiveCore):
    def __init__(
        self,
        *,
        fixed_vocab_size: int,
        full_width: int = 3072,
        bottleneck_width: int = 192,
        attention_heads: int = 2,
        replacement_layers: int = 32,
        intermediate_size: int = 768,
        residual_rank: int = 768,
        sparse_width: int = 384,
        route_names: tuple[str, ...] | list[str] = ("generic", "abstention", "conversation"),
        rms_epsilon: float = 1e-5,
        rope_theta: float = 10000.0,
        maximum_source_actions: int = 192,
        maximum_target_actions: int = 320,
        maximum_sequence_actions: int = 512,
    ) -> None:
        count = int(replacement_layers)
        names = tuple(str(value) for value in route_names)
        if names != ("generic", "abstention", "conversation"):
            raise ValueError("routed sparse route identity changed")
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
        self.replacement_layers = count
        self.residual_rank = int(residual_rank)
        self.sparse_width = int(sparse_width)
        self.route_names = names
        self.router = nn.Linear(self.full_width, len(names), bias=True)
        self.layers = nn.ModuleList(
            RoutedSparseRank768ProgressiveLayer(
                self.full_width,
                self.bottleneck_width,
                self.attention_heads,
                self.intermediate_size,
                residual_rank=self.residual_rank,
                sparse_width=self.sparse_width,
                routes=len(names),
                rms_epsilon=self.rms_epsilon,
                rope_theta=self.rope_theta,
            )
            for _ in range(count)
        )

    def _select_route(self, source_ids: torch.Tensor) -> int:
        if source_ids.ndim != 2 or source_ids.shape[0] != 1 or not source_ids.shape[1]:
            raise ValueError("routed sparse host requires one nonempty request")
        feature = self.token_embedding(source_ids).float().mean(dim=1)
        feature = feature / torch.linalg.vector_norm(feature, dim=-1, keepdim=True).clamp_min(1e-8)
        return int(self.router(feature).argmax(dim=-1).item())

    def forward_routed(self, input_ids: torch.Tensor, route_index: int) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1] or input_ids.shape[1] > self.maximum_sequence_actions:
            raise ValueError("routed sparse input has invalid shape")
        hidden = self.token_embedding(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        for layer in self.layers:
            hidden, _, _ = layer.forward_with_cache(hidden, positions, route_index)
        return self.lm_head(self.final_norm(hidden))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_routed(input_ids, self._select_route(input_ids))

    def prefill_ids(self, source_ids: list[int], source_lexemes: list[bytes]) -> ProgressiveReplacementCoreState:
        if not source_ids or len(source_ids) > self.maximum_source_actions:
            raise ValueError("routed sparse source is empty or exceeds bound")
        device = self.token_embedding.weight.device
        values = torch.tensor([source_ids], dtype=torch.long, device=device)
        route_index = self._select_route(values)
        positions = torch.arange(values.shape[1], device=device)
        hidden = self.token_embedding(values)
        keys = []
        cached_values = []
        for layer in self.layers:
            hidden, key, value = layer.forward_with_cache(hidden, positions, route_index)
            keys.append(key); cached_values.append(value)
        logits = self.lm_head(self.final_norm(hidden[:, -1]))
        state = ProgressiveReplacementCoreState(
            values, source_lexemes, [], tuple(keys), tuple(cached_values), logits, values.shape[1]
        )
        state.route_index = route_index
        return state

    @torch.inference_mode()
    def decode_step(self, state: ProgressiveReplacementCoreState):
        if not hasattr(state, "route_index"):
            raise ValueError("routed sparse state has no persistent route")
        from .portable_token_plan import EOS_ID

        if state.complete:
            raise ValueError("routed sparse request is already complete")
        action = int(state.next_logits.argmax(dim=-1).item())
        state.generated_actions.append(action)
        state.complete = action == EOS_ID or len(state.generated_actions) >= self.maximum_target_actions
        if state.complete or state.sequence_length >= self.maximum_sequence_actions:
            state.complete = True
            return action, state
        device = self.token_embedding.weight.device
        token = torch.tensor([[action]], dtype=torch.long, device=device)
        position = torch.tensor([state.sequence_length], dtype=torch.long, device=device)
        hidden = self.token_embedding(token)
        keys = []
        values = []
        for layer, previous_key, previous_value in zip(self.layers, state.layer_keys, state.layer_values):
            hidden, key, value = layer.incremental(
                hidden, position, previous_key, previous_value, int(state.route_index)
            )
            keys.append(key); values.append(value)
        state.layer_keys = tuple(keys); state.layer_values = tuple(values)
        state.next_logits = self.lm_head(self.final_norm(hidden[:, -1]))
        state.sequence_length += 1
        return action, state

    def canonical_config(self) -> dict:
        value = super().canonical_config()
        value.update(
            {
                "residual_rank": self.residual_rank,
                "sparse_width": self.sparse_width,
                "route_names": list(self.route_names),
            }
        )
        return value

    @staticmethod
    def parameter_count_for_config(
        *,
        fixed_vocab_size: int,
        full_width: int,
        bottleneck_width: int,
        replacement_layers: int,
        intermediate_size: int,
        residual_rank: int,
        sparse_width: int,
        routes: int = 3,
    ) -> int:
        del intermediate_size
        copied = 2 * fixed_vocab_size * full_width + (2 * replacement_layers + 1) * full_width
        attention = 2 * full_width * bottleneck_width + 4 * bottleneck_width * bottleneck_width + bottleneck_width
        residual = (
            2 * full_width * sparse_width
            + full_width * residual_rank
            + routes * sparse_width * residual_rank
            + full_width * residual_rank
            + full_width
        )
        router = full_width * routes + routes
        return copied + replacement_layers * (2 * attention + residual) + router
