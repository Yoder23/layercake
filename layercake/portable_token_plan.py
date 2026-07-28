"""Lossless portable token-plan domain artifacts.

The private representation is a sequence of exact UTF-8 lexemes.  Ordinary
output lexemes are generated from a fixed vocabulary; prompt-specific values
are generated as neural source-position pointer actions.  Both action types
decode losslessly to bytes behind the unchanged external byte boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .portable_domain import canonical_json_hash, state_dict_hash


TOKEN_PLAN_FORMAT = "layercake-portable-token-plan/1"
TOKENIZER_FORMAT = "layercake-lossless-python-lexeme-pointer/1"
LEXEME_PATTERN = (
    rb"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|//|\*\*|->|\s+|."
)
LEXEME_REGEX = re.compile(LEXEME_PATTERN, re.DOTALL)
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_COUNT = 4


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LosslessLexemePointerTokenizer:
    """Training-locked fixed lexemes plus exact dynamic source lexemes."""

    def __init__(self, fixed_lexemes: Iterable[bytes]) -> None:
        values = tuple(fixed_lexemes)
        if any(not value for value in values):
            raise ValueError("fixed lexemes must be non-empty")
        if len(values) != len(set(values)):
            raise ValueError("fixed lexemes must be unique")
        if tuple(sorted(values)) != values:
            raise ValueError("fixed lexemes must use canonical byte ordering")
        self.fixed_lexemes = values
        self.lexeme_to_id = {
            value: index
            for index, value in enumerate(values, start=SPECIAL_COUNT)
        }
        self.id_to_lexeme = {
            index: value
            for index, value in enumerate(values, start=SPECIAL_COUNT)
        }

    @property
    def vocab_size(self) -> int:
        return SPECIAL_COUNT + len(self.fixed_lexemes)

    @staticmethod
    def split(value: bytes | str) -> list[bytes]:
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not value:
            return []
        pieces = LEXEME_REGEX.findall(value)
        if b"".join(pieces) != value:
            raise RuntimeError("lexeme representation is not lossless")
        return pieces

    @classmethod
    def build(cls, rows: Iterable[Mapping[str, Any]]):
        rows = list(rows)
        excluded = {
            str(row["function_name"]).encode("utf-8") for row in rows
        }
        values: set[bytes] = set()
        for row in rows:
            for field in ("prompt", "response"):
                values.update(
                    piece
                    for piece in cls.split(str(row[field]))
                    if piece not in excluded
                )
        return cls(sorted(values))

    def encode_source(
        self, value: bytes | str
    ) -> tuple[list[int], list[bytes]]:
        pieces = self.split(value)
        return [
            self.lexeme_to_id.get(piece, UNK_ID) for piece in pieces
        ], pieces

    def encode_target(
        self,
        response: bytes | str,
        *,
        function_name: str,
        source_lexemes: list[bytes],
    ) -> list[int]:
        identifier = function_name.encode("utf-8")
        source_positions = [
            index
            for index, value in enumerate(source_lexemes)
            if value == identifier
        ]
        if len(source_positions) != 1:
            raise ValueError(
                "function identifier must occur exactly once as a source lexeme"
            )
        pointer_action = self.vocab_size + source_positions[0]
        actions = []
        for piece in self.split(response):
            if piece == identifier:
                actions.append(pointer_action)
                continue
            try:
                actions.append(self.lexeme_to_id[piece])
            except KeyError as error:
                raise ValueError(
                    f"target lexeme is outside fixed vocabulary: {piece!r}"
                ) from error
        actions.append(EOS_ID)
        return actions

    def decode_actions(
        self,
        actions: Iterable[int],
        source_lexemes: list[bytes],
    ) -> bytes:
        output = []
        for raw_action in actions:
            action = int(raw_action)
            if action == EOS_ID:
                break
            if action >= self.vocab_size:
                position = action - self.vocab_size
                if not 0 <= position < len(source_lexemes):
                    raise ValueError("pointer action is outside source sequence")
                output.append(source_lexemes[position])
                continue
            try:
                output.append(self.id_to_lexeme[action])
            except KeyError as error:
                raise ValueError(
                    "special action cannot be realized as output bytes"
                ) from error
        return b"".join(output)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "format": TOKENIZER_FORMAT,
            "lexeme_pattern_ascii": LEXEME_PATTERN.decode("ascii"),
            "special_ids": {
                "pad": PAD_ID,
                "bos": BOS_ID,
                "eos": EOS_ID,
                "unknown_source": UNK_ID,
            },
            "fixed_lexemes_hex": [
                value.hex() for value in self.fixed_lexemes
            ],
            "fixed_vocabulary_order": "ascending_utf8_bytes",
            "external_input_output": "UTF-8 bytes",
        }

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any]
    ) -> "LosslessLexemePointerTokenizer":
        allowed = {
            "format",
            "lexeme_pattern_ascii",
            "special_ids",
            "fixed_lexemes_hex",
            "fixed_vocabulary_order",
            "external_input_output",
        }
        if set(document) != allowed:
            raise ValueError(
                "token-plan tokenizer document is incomplete or ambiguous"
            )
        if document.get("format") != TOKENIZER_FORMAT:
            raise ValueError("unsupported token-plan tokenizer format")
        if document.get("lexeme_pattern_ascii") != LEXEME_PATTERN.decode(
            "ascii"
        ):
            raise ValueError("token-plan lexeme pattern mismatch")
        expected_specials = {
            "pad": PAD_ID,
            "bos": BOS_ID,
            "eos": EOS_ID,
            "unknown_source": UNK_ID,
        }
        if document.get("special_ids") != expected_specials:
            raise ValueError("token-plan special ids mismatch")
        if (
            document.get("fixed_vocabulary_order")
            != "ascending_utf8_bytes"
        ):
            raise ValueError("token-plan fixed vocabulary order mismatch")
        if document.get("external_input_output") != "UTF-8 bytes":
            raise ValueError("token-plan external boundary mismatch")
        values = document.get("fixed_lexemes_hex")
        if not isinstance(values, list) or any(
            not isinstance(value, str) for value in values
        ):
            raise ValueError("token-plan fixed lexemes must be hex strings")
        try:
            fixed_lexemes = [bytes.fromhex(value) for value in values]
        except ValueError as error:
            raise ValueError(
                "token-plan fixed lexemes contain invalid hex"
            ) from error
        return cls(
            fixed_lexemes
        )

    def hash(self) -> str:
        return _canonical_hash(self.canonical_dict())


@dataclass
class PortableTokenPlanState:
    """Persistent source encoding and causal action history for one request."""

    source_ids: torch.Tensor
    source_lexemes: list[bytes]
    encoded: torch.Tensor
    source_padding: torch.Tensor
    previous_actions: torch.Tensor
    generated_actions: list[int]
    layer_self_attention_inputs: tuple[torch.Tensor, ...]
    complete: bool = False


class PortableTokenPlan(nn.Module):
    """Transformer action planner with a neural fixed/pointer mixture."""

    def __init__(
        self,
        *,
        fixed_vocab_size: int,
        model_width: int = 192,
        attention_heads: int = 6,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        feedforward_width: int = 768,
        pointer_width: int = 128,
        dropout: float = 0.1,
        maximum_source_lexemes: int = 64,
        maximum_target_actions: int = 96,
    ) -> None:
        super().__init__()
        if fixed_vocab_size <= SPECIAL_COUNT:
            raise ValueError("fixed vocabulary is too small")
        if model_width % attention_heads:
            raise ValueError("model width must divide attention heads")
        self.fixed_vocab_size = int(fixed_vocab_size)
        self.model_width = int(model_width)
        self.attention_heads = int(attention_heads)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.feedforward_width = int(feedforward_width)
        self.pointer_width = int(pointer_width)
        self.dropout = float(dropout)
        self.maximum_source_lexemes = int(maximum_source_lexemes)
        self.maximum_target_actions = int(maximum_target_actions)
        self.lexeme_embedding = nn.Embedding(
            self.fixed_vocab_size,
            self.model_width,
            padding_idx=PAD_ID,
        )
        self.source_position = nn.Embedding(
            self.maximum_source_lexemes, self.model_width
        )
        self.target_position = nn.Embedding(
            self.maximum_target_actions, self.model_width
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
        self.pointer_input = nn.Linear(
            self.model_width, self.model_width, bias=False
        )
        self.pointer_key = nn.Linear(
            self.model_width, self.pointer_width, bias=False
        )
        self.pointer_query = nn.Linear(
            self.model_width, self.pointer_width, bias=False
        )
        self.fixed_output = nn.Linear(
            self.model_width, self.fixed_vocab_size
        )
        self.pointer_gate = nn.Linear(self.model_width, 1)
        nn.init.xavier_uniform_(self.pointer_input.weight)
        nn.init.xavier_uniform_(self.pointer_key.weight)
        nn.init.xavier_uniform_(self.pointer_query.weight)
        nn.init.zeros_(self.pointer_gate.weight)
        nn.init.constant_(self.pointer_gate.bias, -3.0)
        self.tokenizer: LosslessLexemePointerTokenizer | None = None

    def canonical_config(self) -> dict[str, Any]:
        return {
            "fixed_vocab_size": self.fixed_vocab_size,
            "model_width": self.model_width,
            "attention_heads": self.attention_heads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "feedforward_width": self.feedforward_width,
            "pointer_width": self.pointer_width,
            "dropout": self.dropout,
            "maximum_source_lexemes": self.maximum_source_lexemes,
            "maximum_target_actions": self.maximum_target_actions,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def bind_tokenizer(
        self, tokenizer: LosslessLexemePointerTokenizer
    ) -> "PortableTokenPlan":
        if tokenizer.vocab_size != self.fixed_vocab_size:
            raise ValueError("model and tokenizer vocabulary sizes differ")
        self.tokenizer = tokenizer
        return self

    def encode(
        self, source_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_ids.ndim != 2:
            raise ValueError("source ids must have shape [batch, source]")
        if source_ids.shape[1] > self.maximum_source_lexemes:
            raise ValueError("source exceeds maximum lexeme count")
        source_padding = source_ids.eq(PAD_ID)
        positions = torch.arange(
            source_ids.shape[1], device=source_ids.device
        )
        embedded = (
            self.lexeme_embedding(source_ids)
            + self.source_position(positions)[None]
        )
        encoded = self.encoder(
            embedded, src_key_padding_mask=source_padding
        )
        return encoded, source_padding

    def _action_embeddings(
        self,
        previous_actions: torch.Tensor,
        encoded: torch.Tensor,
    ) -> torch.Tensor:
        fixed = previous_actions < self.fixed_vocab_size
        fixed_ids = previous_actions.clamp(
            min=0, max=self.fixed_vocab_size - 1
        )
        embedded = self.lexeme_embedding(fixed_ids)
        if (~fixed).any():
            positions = (
                previous_actions - self.fixed_vocab_size
            ).clamp(min=0, max=encoded.shape[1] - 1)
            selected = torch.gather(
                encoded,
                1,
                positions[:, :, None].expand(
                    -1, -1, encoded.shape[-1]
                ),
            )
            embedded = torch.where(
                fixed[:, :, None],
                embedded,
                self.pointer_input(selected),
            )
        target_positions = torch.arange(
            previous_actions.shape[1],
            device=previous_actions.device,
        )
        return embedded + self.target_position(target_positions)[None]

    def _action_embedding_step(
        self,
        action: torch.Tensor,
        encoded: torch.Tensor,
        *,
        position: int,
    ) -> torch.Tensor:
        fixed = action < self.fixed_vocab_size
        fixed_ids = action.clamp(
            min=0, max=self.fixed_vocab_size - 1
        )
        embedded = self.lexeme_embedding(fixed_ids)
        if (~fixed).any():
            source_positions = (
                action - self.fixed_vocab_size
            ).clamp(min=0, max=encoded.shape[1] - 1)
            batch = torch.arange(
                action.shape[0], device=action.device
            )
            selected = encoded[batch, source_positions]
            embedded = torch.where(
                fixed[:, None],
                embedded,
                self.pointer_input(selected),
            )
        target_position = torch.tensor(
            [position], dtype=torch.long, device=action.device
        )
        return (
            embedded[:, None]
            + self.target_position(target_position)[None]
        )

    def _incremental_action_log_probs(
        self,
        state: PortableTokenPlanState,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Evaluate one action position without recomputing prior positions."""

        action = state.previous_actions[:, -1]
        hidden = self._action_embedding_step(
            action,
            state.encoded,
            position=len(state.generated_actions),
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
                state.encoded,
                state.encoded,
                key_padding_mask=state.source_padding,
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
        fixed_log = F.log_softmax(self.fixed_output(hidden), dim=-1)
        pointer_scores = torch.matmul(
            self.pointer_query(hidden),
            self.pointer_key(state.encoded).transpose(1, 2),
        ) / (self.pointer_width ** 0.5)
        pointer_scores = pointer_scores.masked_fill(
            state.source_padding[:, None],
            torch.finfo(pointer_scores.dtype).min,
        )
        pointer_log = F.log_softmax(pointer_scores, dim=-1)
        gate_logits = self.pointer_gate(hidden)
        extended = torch.cat(
            (
                F.logsigmoid(-gate_logits) + fixed_log,
                F.logsigmoid(gate_logits) + pointer_log,
            ),
            dim=-1,
        )
        return extended[:, 0], tuple(next_caches)

    def action_log_probs(
        self,
        source_ids: torch.Tensor,
        previous_actions: torch.Tensor,
        *,
        encoded: torch.Tensor | None = None,
        source_padding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if previous_actions.ndim != 2:
            raise ValueError("previous actions must have shape [batch, target]")
        if (
            previous_actions.shape[1] > self.maximum_target_actions
        ):
            raise ValueError("target exceeds maximum action count")
        if encoded is None or source_padding is None:
            encoded, source_padding = self.encode(source_ids)
        target = self._action_embeddings(previous_actions, encoded)
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
            memory_key_padding_mask=source_padding,
        )
        fixed_log = F.log_softmax(self.fixed_output(decoded), dim=-1)
        pointer_scores = torch.matmul(
            self.pointer_query(decoded),
            self.pointer_key(encoded).transpose(1, 2),
        ) / (self.pointer_width ** 0.5)
        pointer_scores = pointer_scores.masked_fill(
            source_padding[:, None],
            torch.finfo(pointer_scores.dtype).min,
        )
        pointer_log = F.log_softmax(pointer_scores, dim=-1)
        gate_logits = self.pointer_gate(decoded)
        extended = torch.cat(
            (
                F.logsigmoid(-gate_logits) + fixed_log,
                F.logsigmoid(gate_logits) + pointer_log,
            ),
            dim=-1,
        )
        return {
            "log_probs": extended,
            "fixed_log_probs": fixed_log,
            "pointer_log_probs": pointer_log,
            "pointer_scores": pointer_scores,
            "gate_logits": gate_logits,
            "encoded": encoded,
            "source_padding": source_padding,
        }

    def forward(
        self,
        source_ids: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if target_actions.ndim != 2:
            raise ValueError("target actions must have shape [batch, target]")
        previous = torch.full_like(target_actions, PAD_ID)
        previous[:, 0] = BOS_ID
        if target_actions.shape[1] > 1:
            shifted = target_actions[:, :-1]
            previous[:, 1:] = torch.where(
                shifted.ge(0), shifted, torch.full_like(shifted, PAD_ID)
            )
        return self.action_log_probs(source_ids, previous)

    @torch.inference_mode()
    def generate_actions(
        self,
        source_ids: torch.Tensor,
        *,
        maximum_actions: int | None = None,
    ) -> list[list[int]]:
        if source_ids.ndim != 2:
            raise ValueError("source ids must have shape [batch, source]")
        limit = (
            self.maximum_target_actions
            if maximum_actions is None
            else min(maximum_actions, self.maximum_target_actions)
        )
        encoded, source_padding = self.encode(source_ids)
        batch = source_ids.shape[0]
        previous = torch.full(
            (batch, 1),
            BOS_ID,
            dtype=torch.long,
            device=source_ids.device,
        )
        generated: list[list[int]] = [[] for _ in range(batch)]
        active = torch.ones(batch, dtype=torch.bool, device=source_ids.device)
        for _ in range(limit):
            result = self.action_log_probs(
                source_ids,
                previous,
                encoded=encoded,
                source_padding=source_padding,
            )
            action = result["log_probs"][:, -1].argmax(dim=-1)
            for index, value in enumerate(action.tolist()):
                if bool(active[index]):
                    generated[index].append(int(value))
            active &= action.ne(EOS_ID)
            if not bool(active.any()):
                break
            previous = torch.cat((previous, action[:, None]), dim=1)
        return generated

    @torch.inference_mode()
    def prefill_bytes(
        self, prompt: bytes | str
    ) -> PortableTokenPlanState:
        """Encode a byte-facing prompt once and retain its immutable state."""

        if self.tokenizer is None:
            raise ValueError("token-plan tokenizer is not bound")
        source_ids, source_lexemes = self.tokenizer.encode_source(prompt)
        if not source_ids:
            raise ValueError("token-plan prompt must contain at least one lexeme")
        device = next(self.parameters()).device
        source = torch.tensor(
            [source_ids], dtype=torch.long, device=device
        )
        encoded, source_padding = self.encode(source)
        previous = torch.full(
            (1, 1), BOS_ID, dtype=torch.long, device=device
        )
        return PortableTokenPlanState(
            source_ids=source,
            source_lexemes=source_lexemes,
            encoded=encoded,
            source_padding=source_padding,
            previous_actions=previous,
            generated_actions=[],
            layer_self_attention_inputs=tuple(
                torch.empty(
                    1,
                    0,
                    self.model_width,
                    dtype=encoded.dtype,
                    device=device,
                )
                for _ in self.decoder.layers
            ),
        )

    @torch.inference_mode()
    def decode_step(
        self, state: PortableTokenPlanState
    ) -> tuple[int, PortableTokenPlanState]:
        """Select one neural action while retaining source/action state."""

        if state.complete:
            raise ValueError("token-plan request is already complete")
        log_probs, caches = self._incremental_action_log_probs(state)
        action_tensor = log_probs.argmax(dim=-1)
        action = int(action_tensor.item())
        state.layer_self_attention_inputs = caches
        state.generated_actions.append(action)
        state.complete = (
            action == EOS_ID
            or len(state.generated_actions) >= self.maximum_target_actions
        )
        if not state.complete:
            state.previous_actions = torch.cat(
                (state.previous_actions, action_tensor[:, None]), dim=1
            )
        return action, state

    @torch.inference_mode()
    def generate_bytes(
        self,
        prompt: bytes | str,
        *,
        maximum_actions: int | None = None,
    ) -> bytes:
        """Generate autonomous actions and losslessly realize exact bytes."""

        if self.tokenizer is None:
            raise ValueError("token-plan tokenizer is not bound")
        state = self.prefill_bytes(prompt)
        limit = (
            self.maximum_target_actions
            if maximum_actions is None
            else min(int(maximum_actions), self.maximum_target_actions)
        )
        while not state.complete and len(state.generated_actions) < limit:
            self.decode_step(state)
        return self.tokenizer.decode_actions(
            state.generated_actions, state.source_lexemes
        )


def build_token_plan_artifact(
    model: PortableTokenPlan,
    tokenizer: LosslessLexemePointerTokenizer,
    *,
    domain_id: str = "python",
    training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if model.fixed_vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabulary sizes differ")
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    tokenizer_document = tokenizer.canonical_dict()
    config = model.canonical_config()
    spec = {
        "domain_id": domain_id,
        "architecture": "portable_token_plan_pointer_transformer",
        "model": config,
        "tokenizer": tokenizer_document,
        "tokenizer_sha256": tokenizer.hash(),
        "external_input_output": "UTF-8 bytes",
        "quantization": "fp32",
    }
    return {
        "format": TOKEN_PLAN_FORMAT,
        "spec": spec,
        "spec_hash": canonical_json_hash(spec),
        "state_dict": state,
        "payload_hash": state_dict_hash(state),
        "training": dict(training or {}),
    }


def load_token_plan_artifact(
    artifact: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[
    Mapping[str, Any],
    LosslessLexemePointerTokenizer,
    PortableTokenPlan,
]:
    if artifact.get("format") != TOKEN_PLAN_FORMAT:
        raise ValueError("unsupported portable token-plan artifact")
    spec = artifact["spec"]
    if artifact.get("spec_hash") != canonical_json_hash(spec):
        raise ValueError("portable token-plan spec hash mismatch")
    tokenizer = LosslessLexemePointerTokenizer.from_document(
        spec["tokenizer"]
    )
    if tokenizer.hash() != spec["tokenizer_sha256"]:
        raise ValueError("portable token-plan tokenizer hash mismatch")
    state = artifact["state_dict"]
    if artifact.get("payload_hash") != state_dict_hash(state):
        raise ValueError("portable token-plan payload hash mismatch")
    model = PortableTokenPlan(**spec["model"]).to(device)
    model.load_state_dict(state)
    model.bind_tokenizer(tokenizer)
    model.eval()
    return spec, tokenizer, model
