"""Canonical-ABI semantic token-plan residual cakes.

This module deliberately accepts only public 768-wide semantic states.  It
factorizes immutable prompt encoding from causal response planning and returns
a bounded same-shape residual for the frozen host's tied language-model head.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .portable_domain import state_dict_hash


SEMANTIC_TOKEN_PLAN_FORMAT = "layercake-semantic-token-plan-residual/1"


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class SemanticTokenPlanState:
    """Persistent prompt and causal decoder state for one selected cake."""

    prompt_states: torch.Tensor
    encoded_prompt: torch.Tensor
    prompt_padding: torch.Tensor
    layer_self_attention_inputs: tuple[torch.Tensor, ...]
    response_steps: int


class SemanticTokenPlanResidual(nn.Module):
    """Whole-prompt neural planner attached through the canonical semantic ABI."""

    def __init__(
        self,
        *,
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
            raise ValueError("semantic token-plan dimensions must be positive")
        if model_width % attention_heads:
            raise ValueError("model width must divide attention heads")
        if max_residual <= 0:
            raise ValueError("maximum residual must be positive")
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
        self.target_input = nn.Linear(self.d_abi, self.model_width, bias=False)
        self.source_position = nn.Embedding(
            self.maximum_prompt_units, self.model_width
        )
        self.target_position = nn.Embedding(
            self.maximum_response_units, self.model_width
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
        self.output_norm = nn.LayerNorm(self.model_width)
        self.residual_output = nn.Linear(
            self.model_width, self.d_abi, bias=False
        )
        self.pointer_key = nn.Linear(
            self.model_width, self.pointer_width, bias=False
        )
        self.pointer_query = nn.Linear(
            self.model_width, self.pointer_width, bias=False
        )
        self.copy_value = nn.Linear(self.d_abi, self.d_abi, bias=False)
        self.copy_gate = nn.Linear(self.model_width, 1)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.xavier_uniform_(self.copy_value.weight)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias, -3.0)

    def canonical_config(self) -> dict[str, Any]:
        return {
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

    def _validate_states(
        self, states: torch.Tensor, *, name: str
    ) -> None:
        if states.ndim != 3 or states.shape[-1] != self.d_abi:
            raise ValueError(
                f"{name} must have shape [batch, sequence, {self.d_abi}]"
            )

    def encode_prompt(
        self,
        prompt_states: torch.Tensor,
        prompt_padding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_states(prompt_states, name="prompt states")
        if prompt_states.shape[1] > self.maximum_prompt_units:
            raise ValueError("prompt exceeds semantic token-plan limit")
        if prompt_padding is None:
            prompt_padding = torch.zeros(
                prompt_states.shape[:2],
                dtype=torch.bool,
                device=prompt_states.device,
            )
        if prompt_padding.shape != prompt_states.shape[:2]:
            raise ValueError("prompt padding shape mismatch")
        positions = torch.arange(
            prompt_states.shape[1], device=prompt_states.device
        )
        hidden = (
            self.source_input(self.input_norm(prompt_states))
            + self.source_position(positions)[None]
        )
        encoded = self.encoder(
            hidden, src_key_padding_mask=prompt_padding
        )
        return encoded, prompt_padding

    def _target_embeddings(
        self, response_states: torch.Tensor
    ) -> torch.Tensor:
        self._validate_states(response_states, name="response states")
        if response_states.shape[1] > self.maximum_response_units:
            raise ValueError("response exceeds semantic token-plan limit")
        positions = torch.arange(
            response_states.shape[1], device=response_states.device
        )
        return (
            self.target_input(self.input_norm(response_states))
            + self.target_position(positions)[None]
        )

    def _project_outputs(
        self,
        decoded: torch.Tensor,
        response_states: torch.Tensor,
        prompt_states: torch.Tensor,
        encoded_prompt: torch.Tensor,
        prompt_padding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        normalized = self.output_norm(decoded)
        pointer_scores = torch.matmul(
            self.pointer_query(normalized),
            self.pointer_key(encoded_prompt).transpose(1, 2),
        ) / (self.pointer_width ** 0.5)
        pointer_scores = pointer_scores.masked_fill(
            prompt_padding[:, None],
            torch.finfo(pointer_scores.dtype).min,
        )
        pointer_weights = F.softmax(pointer_scores, dim=-1)
        copy_source = torch.matmul(pointer_weights, prompt_states)
        main_value = self.residual_output(normalized)
        copy_value = self.copy_value(self.input_norm(copy_source))
        gate_logits = self.copy_gate(normalized)
        combined_value = main_value + torch.sigmoid(gate_logits) * copy_value
        residual = self.max_residual * torch.tanh(combined_value)
        return {
            "residual": residual,
            "adapted": response_states + residual,
            "pointer_scores": pointer_scores,
            "pointer_weights": pointer_weights,
            "copy_source": copy_source,
            "copy_value": copy_value,
            "gate_logits": gate_logits.squeeze(-1),
            "decoded": decoded,
        }

    def training_forward(
        self,
        prompt_states: torch.Tensor,
        response_states: torch.Tensor,
        *,
        prompt_padding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded, prompt_padding = self.encode_prompt(
            prompt_states, prompt_padding
        )
        target = self._target_embeddings(response_states)
        causal_mask = torch.triu(
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
            tgt_mask=causal_mask,
            memory_key_padding_mask=prompt_padding,
        )
        return self._project_outputs(
            decoded,
            response_states,
            prompt_states,
            encoded,
            prompt_padding,
        )

    def _incremental_decoded(
        self,
        current_state: torch.Tensor,
        state: SemanticTokenPlanState,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if current_state.ndim != 2 or current_state.shape[-1] != self.d_abi:
            raise ValueError(
                f"current ABI state must have shape [batch, {self.d_abi}]"
            )
        if state.response_steps >= self.maximum_response_units:
            raise ValueError("response exceeds semantic token-plan limit")
        position = torch.tensor(
            [state.response_steps],
            dtype=torch.long,
            device=current_state.device,
        )
        hidden = (
            self.target_input(self.input_norm(current_state))[:, None]
            + self.target_position(position)[None]
        )
        next_caches: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.decoder.layers):
            normalized = layer.norm1(hidden)
            previous = state.layer_self_attention_inputs[layer_index]
            memory = torch.cat((previous, normalized), dim=1)
            attended = layer.self_attn(
                normalized,
                memory,
                memory,
                need_weights=False,
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
    ) -> tuple[torch.Tensor, SemanticTokenPlanState]:
        """Encode an immutable prompt once and return its first residual."""

        self._validate_states(prompt_states, name="prompt states")
        encoded, padding = self.encode_prompt(prompt_states)
        state = SemanticTokenPlanState(
            prompt_states=prompt_states,
            encoded_prompt=encoded,
            prompt_padding=padding,
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
            response_steps=0,
        )
        residual, state = self.step(prompt_states[:, -1], state)
        return residual, state

    @torch.inference_mode()
    def step(
        self,
        current_state: torch.Tensor,
        state: SemanticTokenPlanState,
    ) -> tuple[torch.Tensor, SemanticTokenPlanState]:
        """Return one residual while extending only persistent cake caches."""

        decoded, caches = self._incremental_decoded(current_state, state)
        projected = self._project_outputs(
            decoded,
            current_state[:, None],
            state.prompt_states,
            state.encoded_prompt,
            state.prompt_padding,
        )
        state.layer_self_attention_inputs = caches
        state.response_steps += 1
        return projected["residual"][:, 0], state


def build_semantic_token_plan_artifact(
    model: SemanticTokenPlanResidual,
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
        "format": SEMANTIC_TOKEN_PLAN_FORMAT,
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


def load_semantic_token_plan_artifact(
    source: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SemanticTokenPlanResidual, dict[str, Any]]:
    if isinstance(source, Mapping):
        artifact = dict(source)
    else:
        artifact = torch.load(
            Path(source), map_location=map_location, weights_only=False
        )
    if artifact.get("format") != SEMANTIC_TOKEN_PLAN_FORMAT:
        raise ValueError("unsupported semantic token-plan artifact format")
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
        raise ValueError("semantic token-plan artifact is incomplete")
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
        raise ValueError("semantic token-plan specification hash mismatch")
    state = artifact["state_dict"]
    if state_dict_hash(state) != artifact["payload_hash"]:
        raise ValueError("semantic token-plan payload hash mismatch")
    model = SemanticTokenPlanResidual(**artifact["architecture"])
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, artifact
