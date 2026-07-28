"""Self-causal neural action plans attached through the semantic cake ABI."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .portable_domain import state_dict_hash


SEMANTIC_ACTION_PLAN_FORMAT = "layercake-semantic-self-action-plan/1"
BOS_ACTION = -1


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class SemanticActionPlanState:
    prompt_states: torch.Tensor
    encoded_prompt: torch.Tensor
    prompt_padding: torch.Tensor
    previous_action: torch.Tensor
    layer_self_attention_inputs: tuple[torch.Tensor, ...]
    planned_actions: list[int]
    response_steps: int
    complete: bool = False


class SemanticActionPlanResidual(nn.Module):
    """Plan from semantic prompt states without host-response exposure drift."""

    def __init__(
        self,
        *,
        fixed_token_ids: Sequence[int],
        eos_token_id: int,
        d_abi: int = 768,
        model_width: int = 192,
        attention_heads: int = 6,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        feedforward_width: int = 768,
        pointer_width: int = 128,
        dropout: float = 0.1,
        maximum_prompt_units: int = 128,
        maximum_response_units: int = 192,
        max_residual: float = 8.0,
    ) -> None:
        super().__init__()
        values = tuple(int(value) for value in fixed_token_ids)
        if not values or len(values) != len(set(values)):
            raise ValueError("fixed token ids must be non-empty and unique")
        if tuple(sorted(values)) != values:
            raise ValueError("fixed token ids must use ascending order")
        if eos_token_id not in values:
            raise ValueError("EOS must be included in fixed token ids")
        if min(
            d_abi,
            model_width,
            attention_heads,
            encoder_layers,
            decoder_layers,
            feedforward_width,
            pointer_width,
            maximum_prompt_units,
            maximum_response_units,
        ) <= 0:
            raise ValueError("semantic action-plan dimensions must be positive")
        if model_width % attention_heads:
            raise ValueError("model width must divide attention heads")
        if max_residual <= 0:
            raise ValueError("maximum residual must be positive")
        self.fixed_token_ids = values
        self.eos_token_id = int(eos_token_id)
        self.eos_action = values.index(self.eos_token_id)
        self.fixed_action_count = len(values)
        self.d_abi = int(d_abi)
        self.model_width = int(model_width)
        self.attention_heads = int(attention_heads)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.feedforward_width = int(feedforward_width)
        self.pointer_width = int(pointer_width)
        self.dropout = float(dropout)
        self.maximum_prompt_units = int(maximum_prompt_units)
        self.maximum_response_units = int(maximum_response_units)
        self.max_residual = float(max_residual)

        self.input_norm = nn.LayerNorm(self.d_abi)
        self.source_input = nn.Linear(self.d_abi, self.model_width, bias=False)
        self.source_position = nn.Embedding(
            self.maximum_prompt_units, self.model_width
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_width,
            nhead=self.attention_heads,
            dim_feedforward=self.feedforward_width,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_width,
            nhead=self.attention_heads,
            dim_feedforward=self.feedforward_width,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.encoder_layers,
            norm=nn.LayerNorm(self.model_width),
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.decoder_layers,
            norm=nn.LayerNorm(self.model_width),
        )
        self.fixed_action_embedding = nn.Embedding(
            self.fixed_action_count, self.model_width
        )
        self.bos_embedding = nn.Parameter(torch.empty(self.model_width))
        self.pointer_input = nn.Linear(
            self.model_width, self.model_width, bias=False
        )
        self.target_position = nn.Embedding(
            self.maximum_response_units, self.model_width
        )
        self.output_norm = nn.LayerNorm(self.model_width)
        self.fixed_action_output = nn.Linear(
            self.model_width, self.fixed_action_count
        )
        self.pointer_key = nn.Linear(
            self.model_width, self.pointer_width, bias=False
        )
        self.pointer_query = nn.Linear(
            self.model_width, self.pointer_width, bias=False
        )
        self.pointer_gate = nn.Linear(self.model_width, 1)
        self.fixed_semantic_value = nn.Embedding(
            self.fixed_action_count, self.d_abi
        )
        self.copy_semantic_value = nn.Linear(
            self.d_abi, self.d_abi, bias=False
        )
        self.plan_correction = nn.Linear(
            self.model_width, self.d_abi, bias=False
        )
        nn.init.normal_(self.bos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.pointer_input.weight)
        nn.init.zeros_(self.pointer_gate.weight)
        nn.init.constant_(self.pointer_gate.bias, -3.0)
        nn.init.zeros_(self.plan_correction.weight)

    def canonical_config(self) -> dict[str, Any]:
        return {
            "fixed_token_ids": list(self.fixed_token_ids),
            "eos_token_id": self.eos_token_id,
            "d_abi": self.d_abi,
            "model_width": self.model_width,
            "attention_heads": self.attention_heads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "feedforward_width": self.feedforward_width,
            "pointer_width": self.pointer_width,
            "dropout": self.dropout,
            "maximum_prompt_units": self.maximum_prompt_units,
            "maximum_response_units": self.maximum_response_units,
            "max_residual": self.max_residual,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def token_to_fixed_action(self) -> dict[int, int]:
        return {
            token_id: action
            for action, token_id in enumerate(self.fixed_token_ids)
        }

    def _validate_states(self, value: torch.Tensor, name: str) -> None:
        if value.ndim != 3 or value.shape[-1] != self.d_abi:
            raise ValueError(
                f"{name} must have shape [batch, sequence, {self.d_abi}]"
            )

    def encode_prompt(
        self,
        prompt_states: torch.Tensor,
        prompt_padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_states(prompt_states, "prompt states")
        if prompt_states.shape[1] > self.maximum_prompt_units:
            raise ValueError("prompt exceeds semantic action-plan limit")
        if prompt_padding is None:
            prompt_padding = torch.zeros(
                prompt_states.shape[:2],
                dtype=torch.bool,
                device=prompt_states.device,
            )
        positions = torch.arange(
            prompt_states.shape[1], device=prompt_states.device
        )
        hidden = (
            self.source_input(self.input_norm(prompt_states))
            + self.source_position(positions)[None]
        )
        return (
            self.encoder(hidden, src_key_padding_mask=prompt_padding),
            prompt_padding,
        )

    def _action_embeddings(
        self,
        previous_actions: torch.Tensor,
        encoded_prompt: torch.Tensor,
    ) -> torch.Tensor:
        if previous_actions.ndim != 2:
            raise ValueError("previous actions must be [batch, target]")
        fixed = (
            previous_actions.ge(0)
            & previous_actions.lt(self.fixed_action_count)
        )
        bos = previous_actions.eq(BOS_ACTION)
        fixed_ids = previous_actions.clamp(
            min=0, max=self.fixed_action_count - 1
        )
        embedded = self.fixed_action_embedding(fixed_ids)
        if bos.any():
            embedded = torch.where(
                bos[:, :, None],
                self.bos_embedding[None, None],
                embedded,
            )
        pointer = ~(fixed | bos)
        if pointer.any():
            positions = (
                previous_actions - self.fixed_action_count
            ).clamp(min=0, max=encoded_prompt.shape[1] - 1)
            selected = torch.gather(
                encoded_prompt,
                1,
                positions[:, :, None].expand(
                    -1, -1, encoded_prompt.shape[-1]
                ),
            )
            embedded = torch.where(
                pointer[:, :, None],
                self.pointer_input(selected),
                embedded,
            )
        target_positions = torch.arange(
            previous_actions.shape[1], device=previous_actions.device
        )
        return embedded + self.target_position(target_positions)[None]

    def _action_embedding_step(
        self,
        previous_action: torch.Tensor,
        encoded_prompt: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        return self._action_embeddings(
            previous_action[:, None], encoded_prompt
        ) + (
            self.target_position(
                torch.tensor([position], device=previous_action.device)
            )[None]
            - self.target_position(
                torch.tensor([0], device=previous_action.device)
            )[None]
        )

    def _action_log_probs(
        self,
        decoded: torch.Tensor,
        encoded_prompt: torch.Tensor,
        prompt_padding: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.output_norm(decoded)
        fixed_log = F.log_softmax(
            self.fixed_action_output(normalized), dim=-1
        )
        pointer_scores = torch.matmul(
            self.pointer_query(normalized),
            self.pointer_key(encoded_prompt).transpose(1, 2),
        ) / (self.pointer_width ** 0.5)
        pointer_scores = pointer_scores.masked_fill(
            prompt_padding[:, None],
            torch.finfo(pointer_scores.dtype).min,
        )
        pointer_log = F.log_softmax(pointer_scores, dim=-1)
        gate = self.pointer_gate(normalized)
        return torch.cat(
            (
                F.logsigmoid(-gate) + fixed_log,
                F.logsigmoid(gate) + pointer_log,
            ),
            dim=-1,
        )

    def _realize(
        self,
        actions: torch.Tensor,
        decoded: torch.Tensor,
        current_states: torch.Tensor,
        prompt_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fixed = actions.lt(self.fixed_action_count)
        fixed_ids = actions.clamp(
            min=0, max=self.fixed_action_count - 1
        )
        semantic_value = self.fixed_semantic_value(fixed_ids)
        if (~fixed).any():
            positions = (
                actions - self.fixed_action_count
            ).clamp(min=0, max=prompt_states.shape[1] - 1)
            selected = torch.gather(
                prompt_states,
                1,
                positions[:, :, None].expand(
                    -1, -1, prompt_states.shape[-1]
                ),
            )
            copy_value = self.copy_semantic_value(
                self.input_norm(selected)
            )
            semantic_value = torch.where(
                fixed[:, :, None], semantic_value, copy_value
            )
        correction = self.plan_correction(self.output_norm(decoded))
        residual = self.max_residual * torch.tanh(
            semantic_value + correction
        )
        return {
            "residual": residual,
            "adapted": current_states + residual,
            "semantic_value": semantic_value,
        }

    def training_forward(
        self,
        prompt_states: torch.Tensor,
        current_states: torch.Tensor,
        target_actions: torch.Tensor,
        *,
        prompt_padding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_states(current_states, "current states")
        encoded, prompt_padding = self.encode_prompt(
            prompt_states, prompt_padding
        )
        previous = torch.full_like(target_actions, BOS_ACTION)
        if target_actions.shape[1] > 1:
            previous[:, 1:] = target_actions[:, :-1]
        target = self._action_embeddings(previous, encoded)
        causal = torch.triu(
            torch.ones(
                target.shape[1],
                target.shape[1],
                dtype=torch.bool,
                device=target.device,
            ),
            diagonal=1,
        )
        decoded = self.decoder(
            target,
            encoded,
            tgt_mask=causal,
            memory_key_padding_mask=prompt_padding,
        )
        realized = self._realize(
            target_actions, decoded, current_states, prompt_states
        )
        return {
            **realized,
            "action_log_probs": self._action_log_probs(
                decoded, encoded, prompt_padding
            ),
            "decoded": decoded,
        }

    def _incremental_decoded(
        self,
        state: SemanticActionPlanState,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden = self._action_embedding_step(
            state.previous_action,
            state.encoded_prompt,
            state.response_steps,
        )
        next_caches = []
        for layer_index, layer in enumerate(self.decoder.layers):
            normalized = layer.norm1(hidden)
            previous = state.layer_self_attention_inputs[layer_index]
            memory = torch.cat((previous, normalized), dim=1)
            attended = layer.self_attn(
                normalized, memory, memory, need_weights=False
            )[0]
            hidden = hidden + layer.dropout1(attended)
            cross_input = layer.norm2(hidden)
            crossed = layer.multihead_attn(
                cross_input,
                state.encoded_prompt,
                state.encoded_prompt,
                key_padding_mask=state.prompt_padding,
                need_weights=False,
            )[0]
            hidden = hidden + layer.dropout2(crossed)
            feedforward_input = layer.norm3(hidden)
            feedforward = layer.linear2(
                layer.dropout(
                    layer.activation(layer.linear1(feedforward_input))
                )
            )
            hidden = hidden + layer.dropout3(feedforward)
            next_caches.append(memory)
        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        return hidden, tuple(next_caches)

    @torch.inference_mode()
    def prefill(
        self, prompt_states: torch.Tensor
    ) -> tuple[torch.Tensor, SemanticActionPlanState, int]:
        encoded, padding = self.encode_prompt(prompt_states)
        state = SemanticActionPlanState(
            prompt_states=prompt_states,
            encoded_prompt=encoded,
            prompt_padding=padding,
            previous_action=torch.full(
                (prompt_states.shape[0],),
                BOS_ACTION,
                dtype=torch.long,
                device=prompt_states.device,
            ),
            layer_self_attention_inputs=tuple(
                torch.empty(
                    prompt_states.shape[0],
                    0,
                    self.model_width,
                    dtype=encoded.dtype,
                    device=encoded.device,
                )
                for _ in self.decoder.layers
            ),
            planned_actions=[],
            response_steps=0,
        )
        return self.step(prompt_states[:, -1], state)

    @torch.inference_mode()
    def step(
        self,
        current_state: torch.Tensor,
        state: SemanticActionPlanState,
    ) -> tuple[torch.Tensor, SemanticActionPlanState, int]:
        if state.complete:
            raise ValueError("semantic action plan is already complete")
        if state.response_steps >= self.maximum_response_units:
            raise ValueError("response exceeds semantic action-plan limit")
        decoded, caches = self._incremental_decoded(state)
        log_probs = self._action_log_probs(
            decoded, state.encoded_prompt, state.prompt_padding
        )[:, 0]
        action_tensor = log_probs.argmax(dim=-1)
        action = int(action_tensor.item())
        realized = self._realize(
            action_tensor[:, None],
            decoded,
            current_state[:, None],
            state.prompt_states,
        )
        state.layer_self_attention_inputs = caches
        state.previous_action = action_tensor
        state.planned_actions.append(action)
        state.response_steps += 1
        state.complete = action == self.eos_action
        return realized["residual"][:, 0], state, action


def build_semantic_action_plan_artifact(
    model: SemanticActionPlanResidual,
    *,
    abi_version: str,
    abi_sha256: str,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    metadata = {
        "format": SEMANTIC_ACTION_PLAN_FORMAT,
        "architecture": model.canonical_config(),
        "abi_version": abi_version,
        "abi_sha256": abi_sha256,
        "training": dict(training),
    }
    return {
        **metadata,
        "spec_sha256": _canonical_hash(metadata),
        "payload_hash": state_dict_hash(state),
        "state_dict": state,
    }


def load_semantic_action_plan_artifact(
    source: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SemanticActionPlanResidual, dict[str, Any]]:
    if isinstance(source, Mapping):
        artifact = dict(source)
    else:
        artifact = torch.load(
            Path(source), map_location=map_location, weights_only=False
        )
    if artifact.get("format") != SEMANTIC_ACTION_PLAN_FORMAT:
        raise ValueError("unsupported semantic action-plan artifact")
    required = {
        "format",
        "architecture",
        "abi_version",
        "abi_sha256",
        "training",
        "spec_sha256",
        "payload_hash",
        "state_dict",
    }
    if set(artifact) != required:
        raise ValueError("semantic action-plan artifact is incomplete")
    metadata = {
        key: artifact[key]
        for key in (
            "format",
            "architecture",
            "abi_version",
            "abi_sha256",
            "training",
        )
    }
    if _canonical_hash(metadata) != artifact["spec_sha256"]:
        raise ValueError("semantic action-plan specification hash mismatch")
    if state_dict_hash(artifact["state_dict"]) != artifact["payload_hash"]:
        raise ValueError("semantic action-plan payload hash mismatch")
    model = SemanticActionPlanResidual(**artifact["architecture"])
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return model, artifact
