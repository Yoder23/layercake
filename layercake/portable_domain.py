"""Versioned, core-independent domain payloads for exact transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .canonical_anchors import canonical_byte_table, causal_byte_anchors


PORTABLE_DOMAIN_FORMAT = "layercake-portable-domain/1"
CANONICAL_ANCHOR_VERSION = "lc-causal-byte-anchor/1"
POINTER_TRANSITION_OFFSETS = (-2, -1, 0, 1, 2, 3, 4)


def canonical_json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PortableDomainSpec:
    domain_id: str
    feature_width: int = 64
    hidden_width: int = 256
    architecture: str = "anchor_mlp"
    embedding_width: int = 64
    pointer_width: int = 64
    format_version: str = PORTABLE_DOMAIN_FORMAT
    anchor_version: str = CANONICAL_ANCHOR_VERSION
    input_mode: str = "byte"
    byte_vocab_size: int = 256
    quantization: str = "fp32"

    def __post_init__(self) -> None:
        if not self.domain_id:
            raise ValueError("domain_id must be non-empty")
        if self.feature_width <= 0 or self.hidden_width <= 0:
            raise ValueError("decoder widths must be positive")
        if self.architecture not in {
            "anchor_mlp",
            "byte_gru",
            "byte_gru_pointer",
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
            "byte_attentive_seq2seq",
        }:
            raise ValueError(f"unsupported decoder architecture: {self.architecture}")
        if self.embedding_width <= 0 or self.pointer_width <= 0:
            raise ValueError("embedding and pointer widths must be positive")
        if self.format_version != PORTABLE_DOMAIN_FORMAT:
            raise ValueError(f"unsupported format: {self.format_version}")
        if self.anchor_version != CANONICAL_ANCHOR_VERSION:
            raise ValueError(f"unsupported anchor contract: {self.anchor_version}")
        if self.input_mode != "byte" or self.byte_vocab_size != 256:
            raise ValueError("portable domain v1 requires raw 256-value bytes")
        if self.quantization not in {"fp32", "int8_symmetric_per_tensor"}:
            raise ValueError(f"unsupported quantization: {self.quantization}")

    def canonical_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return canonical_json_hash(self.canonical_dict())


class PortableDomainDecoder(nn.Module):
    """Predict next bytes from deterministic causal anchors only.

    Host-core weights are intentionally absent from this path. Two runtimes loading
    the same verified artifact therefore produce identical logits and greedy output.
    """

    def __init__(
        self,
        feature_width: int = 64,
        hidden_width: int = 256,
        architecture: str = "anchor_mlp",
        embedding_width: int = 64,
        pointer_width: int = 64,
        *,
        d_abi: int | None = None,
        hidden: int | None = None,
    ):
        super().__init__()
        # Legacy aliases keep early research artifacts loadable.
        self.feature_width = d_abi if d_abi is not None else feature_width
        self.hidden_width = hidden if hidden is not None else hidden_width
        self.architecture = architecture
        self.embedding_width = embedding_width
        self.pointer_width = pointer_width
        if architecture == "anchor_mlp":
            self.decoder = nn.Sequential(
                nn.LayerNorm(self.feature_width),
                nn.Linear(self.feature_width, self.hidden_width),
                nn.GELU(),
                nn.Linear(self.hidden_width, 256),
            )
        elif architecture == "byte_attentive_seq2seq":
            self.byte_embedding = nn.Embedding(256, embedding_width)
            self.seq_encoder = nn.GRU(
                embedding_width,
                self.feature_width,
                batch_first=True,
                bidirectional=True,
            )
            self.seq_initial_hidden = nn.Linear(
                2 * self.feature_width, self.hidden_width
            )
            self.seq_start = nn.Parameter(torch.zeros(embedding_width))
            self.seq_decoder = nn.GRU(
                embedding_width,
                self.hidden_width,
                batch_first=True,
            )
            self.seq_key = nn.Linear(
                2 * self.feature_width, pointer_width, bias=False
            )
            self.seq_query = nn.Linear(
                self.hidden_width, pointer_width, bias=False
            )
            self.seq_output = nn.Linear(
                self.hidden_width + 2 * self.feature_width, 256
            )
            self.seq_copy_gate = nn.Linear(
                self.hidden_width + 2 * self.feature_width, 1
            )
            self.seq_input_dropout = nn.Dropout(0.1)
            nn.init.normal_(self.seq_start, mean=0.0, std=0.02)
            nn.init.xavier_uniform_(self.seq_key.weight)
            nn.init.xavier_uniform_(self.seq_query.weight)
            nn.init.zeros_(self.seq_copy_gate.weight)
            nn.init.constant_(self.seq_copy_gate.bias, -4.0)
        elif architecture in {
            "byte_gru",
            "byte_gru_pointer",
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
        }:
            self.byte_embedding = nn.Embedding(256, embedding_width)
            self.recurrent = nn.GRU(
                embedding_width + self.feature_width,
                self.hidden_width,
                batch_first=True,
            )
            self.decoder = nn.Sequential(
                nn.LayerNorm(self.hidden_width),
                nn.Linear(self.hidden_width, 256),
            )
            if architecture in {
                "byte_gru_pointer",
                "byte_gru_pointer_transition",
                "byte_gru_pointer_self_transition",
                "byte_gru_pointer_markov",
                "byte_gru_pointer_markov_max",
            }:
                self.copy_query = nn.Linear(
                    self.hidden_width, pointer_width, bias=False
                )
                self.copy_key = nn.Linear(
                    self.hidden_width, pointer_width, bias=False
                )
                self.copy_gate = nn.Linear(self.hidden_width, 1)
                nn.init.xavier_uniform_(self.copy_query.weight)
                nn.init.xavier_uniform_(self.copy_key.weight)
                nn.init.zeros_(self.copy_gate.weight)
                nn.init.constant_(self.copy_gate.bias, -4.0)
                if architecture in {
                    "byte_gru_pointer_transition",
                    "byte_gru_pointer_self_transition",
                    "byte_gru_pointer_markov",
                    "byte_gru_pointer_markov_max",
                }:
                    self.copy_transition_logits = nn.Parameter(
                        torch.zeros(len(POINTER_TRANSITION_OFFSETS))
                    )
                if architecture in {
                    "byte_gru_pointer_transition",
                    "byte_gru_pointer_self_transition",
                }:
                    self.copy_transition_gate = nn.Linear(
                        self.hidden_width, 1
                    )
                    nn.init.zeros_(self.copy_transition_gate.weight)
                    nn.init.constant_(self.copy_transition_gate.bias, -4.0)
        else:
            raise ValueError(f"unsupported decoder architecture: {architecture}")

    @property
    def d_abi(self) -> int:
        return self.feature_width

    @property
    def hidden(self) -> int:
        return self.hidden_width

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        if self.architecture == "byte_attentive_seq2seq":
            lengths = torch.full(
                (byte_ids.shape[0],),
                byte_ids.shape[1],
                dtype=torch.long,
                device=byte_ids.device,
            )
            state = self._seq2seq_prefill(byte_ids, lengths)
            return state["next_logits"][:, None]
        anchors = causal_byte_anchors(byte_ids, self.feature_width)
        if self.architecture in {
            "byte_gru",
            "byte_gru_pointer",
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
        }:
            embedded = self.byte_embedding(byte_ids)
            hidden, _ = self.recurrent(torch.cat([embedded, anchors], dim=-1))
            if self.architecture in {
                "byte_gru_pointer",
                "byte_gru_pointer_transition",
                "byte_gru_pointer_self_transition",
                "byte_gru_pointer_markov",
                "byte_gru_pointer_markov_max",
            }:
                return self.pointer_forward(byte_ids, hidden)["logits"]
            return self.decoder(hidden)
        return self.decoder(anchors)

    def _seq2seq_encode(
        self,
        prompt_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.architecture != "byte_attentive_seq2seq":
            raise ValueError("attentive encoder-decoder path is disabled")
        if prompt_ids.ndim != 2:
            raise ValueError("prompt ids must have shape [batch, sequence]")
        if prompt_lengths.ndim != 1 or prompt_lengths.shape[0] != prompt_ids.shape[0]:
            raise ValueError("prompt lengths must have shape [batch]")
        embedded = self.byte_embedding(prompt_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            prompt_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded_packed, encoder_hidden = self.seq_encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded_packed,
            batch_first=True,
            total_length=prompt_ids.shape[1],
        )
        final = torch.cat(
            (encoder_hidden[-2], encoder_hidden[-1]), dim=-1
        )
        initial_hidden = torch.tanh(
            self.seq_initial_hidden(final)
        ).unsqueeze(0)
        positions = torch.arange(
            prompt_ids.shape[1], device=prompt_ids.device
        )[None]
        source_mask = positions < prompt_lengths[:, None]
        return {
            "encoder_outputs": encoded,
            "attention_keys": self.seq_key(encoded),
            "source_ids": prompt_ids,
            "source_mask": source_mask,
            "recurrent_hidden": initial_hidden,
        }

    def _seq2seq_project(
        self,
        decoder_states: torch.Tensor,
        encoded: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        queries = self.seq_query(decoder_states)
        scores = torch.matmul(
            queries, encoded["attention_keys"].transpose(1, 2)
        ) / (self.pointer_width ** 0.5)
        scores = scores.masked_fill(
            ~encoded["source_mask"][:, None],
            torch.finfo(scores.dtype).min,
        )
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, encoded["encoder_outputs"])
        joined = torch.cat((decoder_states, context), dim=-1)
        language = torch.log_softmax(self.seq_output(joined), dim=-1)
        pointer = attention.new_zeros(
            attention.shape[0], attention.shape[1], 256
        )
        source_values = encoded["source_ids"][:, None].expand_as(scores)
        pointer.scatter_add_(2, source_values, attention)
        gate_logits = self.seq_copy_gate(joined)
        logits = torch.logaddexp(
            F.logsigmoid(-gate_logits) + language,
            F.logsigmoid(gate_logits)
            + torch.log(pointer.clamp_min(1e-12)),
        )
        return {
            "logits": logits,
            "pointer_scores": scores,
            "pointer_attention": attention,
            "pointer_distribution": pointer,
            "gate_logits": gate_logits,
            "context": context,
        }

    def seq2seq_forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced response logits for the attentive portable decoder."""
        encoded = self._seq2seq_encode(prompt_ids, prompt_lengths)
        starts = self.seq_start[None, None].expand(
            response_ids.shape[0], 1, -1
        )
        if response_ids.shape[1] > 1:
            previous = self.byte_embedding(response_ids[:, :-1])
            decoder_inputs = torch.cat((starts, previous), dim=1)
        else:
            decoder_inputs = starts
        decoder_inputs = self.seq_input_dropout(decoder_inputs)
        decoder_states, hidden = self.seq_decoder(
            decoder_inputs, encoded["recurrent_hidden"]
        )
        result = self._seq2seq_project(decoder_states, encoded)
        result.update(encoded)
        result["recurrent_hidden"] = hidden
        return result

    def _seq2seq_prefill(
        self,
        prompt_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self._seq2seq_encode(prompt_ids, prompt_lengths)
        decoder_input = self.seq_start[None, None].expand(
            prompt_ids.shape[0], 1, -1
        )
        decoder_state, hidden = self.seq_decoder(
            decoder_input, encoded["recurrent_hidden"]
        )
        projected = self._seq2seq_project(decoder_state, encoded)
        return {
            "seq_encoder_outputs": encoded["encoder_outputs"],
            "seq_attention_keys": encoded["attention_keys"],
            "seq_source_ids": encoded["source_ids"],
            "seq_source_mask": encoded["source_mask"],
            "recurrent_hidden": hidden,
            "pointer_attention": projected["pointer_attention"][:, 0],
            "pointer_gate_logits": projected["gate_logits"][:, 0],
            "next_logits": projected["logits"][:, 0],
        }

    def _scatter_pointer_probabilities(
        self,
        probabilities: torch.Tensor,
        byte_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = probabilities.new_zeros(
            *probabilities.shape[:-1], 256
        )
        values = byte_ids[:, None, :].expand_as(probabilities)
        return output.scatter_add(2, values, probabilities)

    def _pointer_distribution(
        self,
        scores: torch.Tensor,
        byte_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self._scatter_pointer_probabilities(
            torch.softmax(scores, dim=-1), byte_ids
        )

    def _max_pointer_distribution(
        self,
        probabilities: torch.Tensor,
        byte_ids: torch.Tensor,
    ) -> torch.Tensor:
        log_probabilities = torch.log(probabilities.clamp_min(1e-12))
        output = log_probabilities.new_full(
            (*log_probabilities.shape[:-1], 256),
            torch.finfo(log_probabilities.dtype).min,
        )
        values = byte_ids[:, None, :].expand_as(log_probabilities)
        output.scatter_reduce_(
            2,
            values,
            log_probabilities,
            reduce="amax",
            include_self=True,
        )
        return torch.softmax(output, dim=-1)

    def _shift_pointer_probabilities(
        self,
        previous: torch.Tensor,
        output_length: int,
    ) -> torch.Tensor:
        weights = torch.softmax(self.copy_transition_logits, dim=0)
        result = previous.new_zeros(previous.shape[0], output_length)
        source_length = previous.shape[1]
        for weight, offset in zip(weights, POINTER_TRANSITION_OFFSETS):
            source_start = max(0, -offset)
            source_stop = min(source_length, output_length - offset)
            if source_stop <= source_start:
                continue
            destination_start = source_start + offset
            destination_stop = source_stop + offset
            result[:, destination_start:destination_stop] += (
                weight * previous[:, source_start:source_stop]
            )
        return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _transition_pointer_probabilities(
        self,
        content_probabilities: torch.Tensor,
        recurrent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_steps = []
        transition_gate_steps = []
        previous = None
        previous_copy_gate = None
        for index in range(content_probabilities.shape[1]):
            content = content_probabilities[:, index, : index + 1]
            current_copy_gate = torch.sigmoid(
                self.copy_gate(recurrent[:, index])
            )
            if previous is None:
                attention = content
                gate_logits = torch.full_like(current_copy_gate, -30.0)
            else:
                transitioned = self._shift_pointer_probabilities(
                    previous, index + 1
                )
                if (
                    self.architecture
                    in {
                        "byte_gru_pointer_self_transition",
                        "byte_gru_pointer_markov",
                        "byte_gru_pointer_markov_max",
                    }
                ):
                    gate = (
                        current_copy_gate * previous_copy_gate
                    ).clamp(1e-12, 1.0 - 1e-7)
                    gate_logits = torch.logit(gate)
                else:
                    gate_logits = self.copy_transition_gate(
                        recurrent[:, index]
                    )
                    gate = torch.sigmoid(gate_logits)
                attention = (1.0 - gate) * content + gate * transitioned
                attention = attention / attention.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
            padded = F.pad(
                attention,
                (0, content_probabilities.shape[2] - index - 1),
            )
            attention_steps.append(padded)
            transition_gate_steps.append(gate_logits)
            previous = attention
            previous_copy_gate = current_copy_gate
        return (
            torch.stack(attention_steps, dim=1),
            torch.stack(transition_gate_steps, dim=1),
        )

    def _mix_pointer_logits(
        self,
        recurrent: torch.Tensor,
        pointer_distribution: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        language = torch.log_softmax(self.decoder(recurrent), dim=-1)
        gate_logits = self.copy_gate(recurrent)
        pointer = torch.log(pointer_distribution.clamp_min(1e-12))
        mixed = torch.logaddexp(
            F.logsigmoid(-gate_logits) + language,
            F.logsigmoid(gate_logits) + pointer,
        )
        return mixed, gate_logits

    def pointer_forward(
        self,
        byte_ids: torch.Tensor,
        recurrent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.architecture not in {
            "byte_gru_pointer",
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
        }:
            raise ValueError("neural pointer path is disabled")
        if recurrent is None:
            anchors = causal_byte_anchors(byte_ids, self.feature_width)
            recurrent, _ = self.recurrent(
                torch.cat([self.byte_embedding(byte_ids), anchors], dim=-1)
            )
        query = self.copy_query(recurrent)
        key = self.copy_key(recurrent)
        scores = torch.matmul(query, key.transpose(1, 2)) / (
            self.pointer_width ** 0.5
        )
        length = byte_ids.shape[1]
        future = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=byte_ids.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(
            future, torch.finfo(scores.dtype).min
        )
        content_probabilities = torch.softmax(scores, dim=-1)
        transition_gate_logits = None
        pointer_probabilities = content_probabilities
        if self.architecture in {
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
        }:
            (
                pointer_probabilities,
                transition_gate_logits,
            ) = self._transition_pointer_probabilities(
                content_probabilities, recurrent
            )
        if self.architecture == "byte_gru_pointer_markov_max":
            pointer = self._max_pointer_distribution(
                pointer_probabilities, byte_ids
            )
        else:
            pointer = self._scatter_pointer_probabilities(
                pointer_probabilities, byte_ids
            )
        logits, gate_logits = self._mix_pointer_logits(
            recurrent, pointer
        )
        return {
            "logits": logits,
            "pointer_scores": torch.log(
                pointer_probabilities.clamp_min(1e-12)
            ),
            "content_pointer_scores": scores,
            "pointer_attention": pointer_probabilities,
            "pointer_distribution": pointer,
            "gate_logits": gate_logits,
            "transition_gate_logits": transition_gate_logits,
            "recurrent": recurrent,
        }

    def _pointer_step_logits(
        self,
        recurrent: torch.Tensor,
        pointer_keys: torch.Tensor,
        pointer_byte_ids: torch.Tensor,
        previous_pointer_attention: torch.Tensor | None = None,
        previous_copy_gate_probability: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        query = self.copy_query(recurrent)
        scores = torch.matmul(
            query[:, None], pointer_keys.transpose(1, 2)
        ) / (self.pointer_width ** 0.5)
        pointer_attention = torch.softmax(scores[:, 0], dim=-1)
        transition_gate_logits = None
        current_copy_gate_probability = torch.sigmoid(
            self.copy_gate(recurrent)
        )
        if self.architecture in {
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
        }:
            if previous_pointer_attention is None:
                transition_gate_logits = torch.full(
                    (recurrent.shape[0], 1),
                    -30.0,
                    dtype=recurrent.dtype,
                    device=recurrent.device,
                )
            else:
                transitioned = self._shift_pointer_probabilities(
                    previous_pointer_attention,
                    pointer_attention.shape[1],
                )
                if (
                    self.architecture
                    in {
                        "byte_gru_pointer_self_transition",
                        "byte_gru_pointer_markov",
                        "byte_gru_pointer_markov_max",
                    }
                ):
                    if previous_copy_gate_probability is None:
                        raise ValueError(
                            "self transition requires previous copy gate"
                        )
                    transition_gate = (
                        current_copy_gate_probability
                        * previous_copy_gate_probability
                    ).clamp(1e-12, 1.0 - 1e-7)
                    transition_gate_logits = torch.logit(transition_gate)
                else:
                    transition_gate_logits = self.copy_transition_gate(
                        recurrent
                    )
                    transition_gate = torch.sigmoid(
                        transition_gate_logits
                    )
                pointer_attention = (
                    (1.0 - transition_gate) * pointer_attention
                    + transition_gate * transitioned
                )
                pointer_attention = pointer_attention / pointer_attention.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
        if self.architecture == "byte_gru_pointer_markov_max":
            pointer = self._max_pointer_distribution(
                pointer_attention[:, None], pointer_byte_ids
            )[:, 0]
        else:
            pointer = self._scatter_pointer_probabilities(
                pointer_attention[:, None], pointer_byte_ids
            )[:, 0]
        logits, gate_logits = self._mix_pointer_logits(
            recurrent, pointer
        )
        return (
            logits,
            gate_logits,
            pointer_attention,
            transition_gate_logits,
            current_copy_gate_probability,
        )

    def prefill_incremental(
        self, byte_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Create persistent byte-GRU state without changing model mathematics."""
        if self.architecture == "byte_attentive_seq2seq":
            if byte_ids.ndim != 2 or byte_ids.shape[1] == 0:
                raise ValueError(
                    "prefill requires non-empty [batch, sequence] bytes"
                )
            lengths = torch.full(
                (byte_ids.shape[0],),
                byte_ids.shape[1],
                dtype=torch.long,
                device=byte_ids.device,
            )
            return self._seq2seq_prefill(byte_ids, lengths)
        if self.architecture not in {
            "byte_gru",
            "byte_gru_pointer",
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
        }:
            raise ValueError("persistent incremental state requires byte_gru")
        if byte_ids.ndim != 2 or byte_ids.shape[1] == 0:
            raise ValueError("prefill requires non-empty [batch, sequence] bytes")
        table = canonical_byte_table(
            self.feature_width,
            byte_ids.device,
            self.byte_embedding.weight.dtype,
        )
        embedded_anchors = table[byte_ids]
        anchor_state = embedded_anchors.new_zeros(
            byte_ids.shape[0], self.feature_width
        )
        anchors = []
        for index in range(byte_ids.shape[1]):
            anchor_state = 0.875 * anchor_state + embedded_anchors[:, index]
            anchors.append(
                torch.nn.functional.layer_norm(
                    anchor_state, (self.feature_width,)
                )
            )
        anchor_sequence = torch.stack(anchors, dim=1)
        recurrent, hidden = self.recurrent(
            torch.cat(
                [self.byte_embedding(byte_ids), anchor_sequence], dim=-1
            )
        )
        result = {
            "anchor_state": anchor_state,
            "recurrent_hidden": hidden,
            "next_logits": self.decoder(recurrent[:, -1]),
        }
        if self.architecture in {
            "byte_gru_pointer",
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
        }:
            pointer_keys = self.copy_key(recurrent)
            if self.architecture in {
                "byte_gru_pointer_transition",
                "byte_gru_pointer_self_transition",
                "byte_gru_pointer_markov",
                "byte_gru_pointer_markov_max",
            }:
                pointer_result = self.pointer_forward(byte_ids, recurrent)
                next_logits = pointer_result["logits"][:, -1]
                gate_logits = pointer_result["gate_logits"][:, -1]
                pointer_attention = pointer_result[
                    "pointer_attention"
                ][:, -1]
                transition_gate_logits = pointer_result[
                    "transition_gate_logits"
                ][:, -1]
                current_copy_gate_probability = torch.sigmoid(
                    pointer_result["gate_logits"][:, -1]
                )
            else:
                (
                    next_logits,
                    gate_logits,
                    pointer_attention,
                    transition_gate_logits,
                    current_copy_gate_probability,
                ) = self._pointer_step_logits(
                    recurrent[:, -1],
                    pointer_keys,
                    byte_ids,
                )
            result.update(
                {
                    "pointer_keys": pointer_keys,
                    "pointer_byte_ids": byte_ids,
                    "pointer_attention": pointer_attention,
                    "pointer_gate_logits": gate_logits,
                    "pointer_transition_gate_logits": (
                        transition_gate_logits
                    ),
                    "previous_copy_gate_probability": (
                        current_copy_gate_probability
                    ),
                    "next_logits": next_logits,
                }
            )
        return result

    def decode_incremental(
        self,
        byte_ids: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Advance persistent state by one observed generated byte."""
        if self.architecture == "byte_attentive_seq2seq":
            if byte_ids.ndim == 1:
                byte_ids = byte_ids[:, None]
            if byte_ids.ndim != 2 or byte_ids.shape[1] != 1:
                raise ValueError(
                    "incremental decode requires [batch, 1] bytes"
                )
            decoder_state, hidden = self.seq_decoder(
                self.byte_embedding(byte_ids),
                state["recurrent_hidden"],
            )
            encoded = {
                "encoder_outputs": state["seq_encoder_outputs"],
                "attention_keys": state["seq_attention_keys"],
                "source_ids": state["seq_source_ids"],
                "source_mask": state["seq_source_mask"],
            }
            projected = self._seq2seq_project(decoder_state, encoded)
            logits = projected["logits"][:, 0]
            state["recurrent_hidden"] = hidden
            state["pointer_attention"] = projected[
                "pointer_attention"
            ][:, 0]
            state["pointer_gate_logits"] = projected["gate_logits"][:, 0]
            state["next_logits"] = logits
            return logits
        if self.architecture not in {
            "byte_gru",
            "byte_gru_pointer",
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
        }:
            raise ValueError("persistent incremental state requires byte_gru")
        if byte_ids.ndim == 1:
            byte_ids = byte_ids[:, None]
        if byte_ids.ndim != 2 or byte_ids.shape[1] != 1:
            raise ValueError("incremental decode requires [batch, 1] bytes")
        table = canonical_byte_table(
            self.feature_width,
            byte_ids.device,
            self.byte_embedding.weight.dtype,
        )
        anchor_state = (
            0.875 * state["anchor_state"] + table[byte_ids[:, 0]]
        )
        anchor = torch.nn.functional.layer_norm(
            anchor_state, (self.feature_width,)
        )
        recurrent, hidden = self.recurrent(
            torch.cat(
                [self.byte_embedding(byte_ids), anchor[:, None]], dim=-1
            ),
            state["recurrent_hidden"],
        )
        recurrent_last = recurrent[:, -1]
        logits = self.decoder(recurrent_last)
        if self.architecture in {
            "byte_gru_pointer",
            "byte_gru_pointer_transition",
            "byte_gru_pointer_self_transition",
            "byte_gru_pointer_markov",
            "byte_gru_pointer_markov_max",
        }:
            pointer_keys = torch.cat(
                (
                    state["pointer_keys"],
                    self.copy_key(recurrent),
                ),
                dim=1,
            )
            pointer_byte_ids = torch.cat(
                (state["pointer_byte_ids"], byte_ids), dim=1
            )
            (
                logits,
                gate_logits,
                pointer_attention,
                transition_gate_logits,
                current_copy_gate_probability,
            ) = self._pointer_step_logits(
                recurrent_last,
                pointer_keys,
                pointer_byte_ids,
                state.get("pointer_attention"),
                state.get("previous_copy_gate_probability"),
            )
            state["pointer_keys"] = pointer_keys
            state["pointer_byte_ids"] = pointer_byte_ids
            state["pointer_attention"] = pointer_attention
            state["pointer_gate_logits"] = gate_logits
            state["pointer_transition_gate_logits"] = (
                transition_gate_logits
            )
            state["previous_copy_gate_probability"] = (
                current_copy_gate_probability
            )
        state["anchor_state"] = anchor_state
        state["recurrent_hidden"] = hidden
        state["next_logits"] = logits
        return logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def quantized_state_hash(quantized_state: Mapping[str, dict]) -> str:
    digest = hashlib.sha256()
    for name in sorted(quantized_state):
        item = quantized_state[name]
        values = item["values"].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(values.numpy().tobytes())
        digest.update(float(item["scale"]).hex().encode("ascii"))
    return digest.hexdigest()


def build_portable_artifact(
    model: PortableDomainDecoder,
    spec: PortableDomainSpec,
    *,
    training: dict | None = None,
    evaluation: dict | None = None,
) -> dict:
    if model.feature_width != spec.feature_width:
        raise ValueError("model feature width does not match artifact spec")
    if model.hidden_width != spec.hidden_width:
        raise ValueError("model hidden width does not match artifact spec")
    if model.architecture != spec.architecture:
        raise ValueError("model architecture does not match artifact spec")
    if model.embedding_width != spec.embedding_width:
        raise ValueError("model embedding width does not match artifact spec")
    if model.pointer_width != spec.pointer_width:
        raise ValueError("model pointer width does not match artifact spec")
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    artifact = {
        "format": PORTABLE_DOMAIN_FORMAT,
        "spec": spec.canonical_dict(),
        "spec_hash": spec.hash(),
        "training": training or {},
        "evaluation": evaluation or {},
    }
    if spec.quantization == "fp32":
        artifact["state_dict"] = state
        artifact["payload_hash"] = state_dict_hash(state)
        return artifact
    quantized_state = {}
    for name, tensor in state.items():
        scale = max(tensor.abs().max().item() / 127.0, 1e-12)
        quantized_state[name] = {
            "values": (tensor / scale).round().clamp(-127, 127).to(torch.int8),
            "scale": scale,
        }
    artifact["quantized_state"] = quantized_state
    artifact["payload_hash"] = quantized_state_hash(quantized_state)
    return artifact


def load_portable_artifact(
    artifact: dict, device: torch.device | str = "cpu"
) -> tuple[PortableDomainSpec, PortableDomainDecoder]:
    if artifact.get("format") != PORTABLE_DOMAIN_FORMAT:
        raise ValueError("unsupported portable domain artifact")
    raw_spec = artifact["spec"]
    spec = PortableDomainSpec(**raw_spec)
    if artifact.get("spec_hash") not in {
        spec.hash(),
        canonical_json_hash(raw_spec),
    }:
        raise ValueError("portable domain spec hash mismatch")
    if spec.quantization == "fp32":
        if artifact.get("payload_hash") != state_dict_hash(artifact["state_dict"]):
            raise ValueError("portable domain payload hash mismatch")
        state_dict = artifact["state_dict"]
    else:
        quantized_state = artifact["quantized_state"]
        if artifact.get("payload_hash") != quantized_state_hash(quantized_state):
            raise ValueError("portable domain payload hash mismatch")
        state_dict = {
            name: item["values"].float() * float(item["scale"])
            for name, item in quantized_state.items()
        }
    model = PortableDomainDecoder(
        feature_width=spec.feature_width,
        hidden_width=spec.hidden_width,
        architecture=spec.architecture,
        embedding_width=spec.embedding_width,
        pointer_width=spec.pointer_width,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return spec, model


def quantize_portable_artifact(artifact: dict) -> dict:
    spec, model = load_portable_artifact(artifact, "cpu")
    if spec.quantization != "fp32":
        raise ValueError("only fp32 portable artifacts can be quantized")
    quantized_spec = PortableDomainSpec(
        **{**spec.canonical_dict(), "quantization": "int8_symmetric_per_tensor"}
    )
    return build_portable_artifact(
        model,
        quantized_spec,
        training=artifact.get("training"),
        evaluation=artifact.get("evaluation"),
    )


def artifact_payload_bytes(artifact: dict) -> int:
    if "state_dict" in artifact:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in artifact["state_dict"].values()
        )
    return sum(
        item["values"].numel() * item["values"].element_size() + 8
        for item in artifact["quantized_state"].values()
    )


class LayerCakeRuntime:
    """Runtime selecting either host-core or exact portable-domain predictions."""

    def __init__(self, core: nn.Module | None = None):
        self.core = core
        self.domains: dict[str, tuple[PortableDomainSpec, PortableDomainDecoder]] = {}

    def install_portable_domain(
        self, artifact: dict, device: torch.device | str = "cpu"
    ) -> str:
        spec, decoder = load_portable_artifact(artifact, device)
        self.domains[spec.domain_id] = (spec, decoder)
        return spec.domain_id

    def logits(self, byte_ids: torch.Tensor, *, domain_id: str | None = None):
        if domain_id is not None:
            try:
                return self.domains[domain_id][1](byte_ids)
            except KeyError as exc:
                raise KeyError(f"portable domain is not installed: {domain_id}") from exc
        if self.core is None:
            raise ValueError("base inference requires a host core")
        output = self.core(byte_ids)
        return output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def generate(
        self,
        prompt: bytes | str | torch.Tensor,
        *,
        max_new_bytes: int,
        domain_id: str,
        context_bytes: int = 256,
    ) -> torch.Tensor:
        if isinstance(prompt, str):
            prompt = prompt.encode("utf-8")
        if isinstance(prompt, bytes):
            prompt = torch.tensor(list(prompt), dtype=torch.long).unsqueeze(0)
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("generation currently requires one [1, sequence] prompt")
        decoder = self.domains[domain_id][1]
        device = next(decoder.parameters()).device
        generated = prompt.to(device)
        for _ in range(max_new_bytes):
            context = generated[:, -context_bytes:]
            next_byte = decoder(context)[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_byte], dim=1)
        return generated

    @torch.no_grad()
    def generate_incremental(
        self,
        prompt: bytes | str | torch.Tensor,
        *,
        max_new_bytes: int,
        domain_id: str,
    ) -> torch.Tensor:
        """Generate with persistent recurrent and canonical-anchor state."""
        if isinstance(prompt, str):
            prompt = prompt.encode("utf-8")
        if isinstance(prompt, bytes):
            prompt = torch.tensor(
                list(prompt), dtype=torch.long
            ).unsqueeze(0)
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("generation currently requires one [1, sequence] prompt")
        decoder = self.domains[domain_id][1]
        if decoder.architecture != "byte_gru":
            raise ValueError("persistent generation requires a byte_gru domain")
        device = next(decoder.parameters()).device
        generated = prompt.to(device)
        state = decoder.prefill_incremental(generated)
        for _ in range(max_new_bytes):
            next_byte = state["next_logits"].argmax(
                dim=-1, keepdim=True
            )
            generated = torch.cat([generated, next_byte], dim=1)
            decoder.decode_incremental(next_byte, state)
        return generated
