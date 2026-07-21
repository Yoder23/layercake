"""Tokenless latent-span language model over raw bytes."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class LatentSpanCakeLM(nn.Module):
    """Model normalized raw-byte spans with marginalized latent cake states.

    Input bytes are never tokenized.  Complete fixed-size spans are encoded by
    a recurrent slow state.  The next raw-byte span is a mixture of normalized
    product distributions; the latent state is summed out exactly in the loss.
    """

    def __init__(
        self,
        *,
        span_bytes: int = 8,
        d_byte: int = 24,
        d_model: int = 192,
        layers: int = 2,
        latent_states: int = 128,
        d_abi: int = 64,
        emission_mode: str = "product",
        local_width: int = 128,
    ) -> None:
        super().__init__()
        if span_bytes <= 0 or d_byte <= 0 or d_model <= 0:
            raise ValueError("span and model widths must be positive")
        if layers <= 0 or latent_states <= 1 or d_abi <= 0:
            raise ValueError("layers, latent states, and ABI width must be positive")
        self.span_bytes = int(span_bytes)
        self.d_byte = int(d_byte)
        self.d_model = int(d_model)
        self.layers = int(layers)
        self.latent_states = int(latent_states)
        self.d_abi = int(d_abi)
        if emission_mode not in {"product", "autoregressive"}:
            raise ValueError("emission_mode must be product or autoregressive")
        if local_width <= 0:
            raise ValueError("local_width must be positive")
        self.emission_mode = str(emission_mode)
        self.local_width = int(local_width)
        self.byte_embedding = nn.Embedding(256, self.d_byte)
        self.span_projection = nn.Linear(
            self.span_bytes * self.d_byte, self.d_model
        )
        self.span_norm = nn.LayerNorm(self.d_model)
        self.slow_core = nn.GRU(
            self.d_model,
            self.d_model,
            num_layers=self.layers,
            batch_first=True,
        )
        self.to_abi = nn.Linear(self.d_model, self.d_abi, bias=False)
        self.from_abi = nn.Linear(self.d_abi, self.d_model, bias=False)
        self.context_norm = nn.LayerNorm(self.d_model)
        self.latent_gate = nn.Linear(self.d_model, self.latent_states)
        if self.emission_mode == "product":
            self.emission_high_logits = nn.Parameter(
                torch.empty(self.latent_states, self.span_bytes, 16)
            )
            self.emission_low_logits = nn.Parameter(
                torch.empty(self.latent_states, self.span_bytes, 16, 16)
            )
        else:
            self.context_to_local = nn.Linear(self.d_model, self.local_width)
            self.latent_initial = nn.Embedding(self.latent_states, self.local_width)
            self.local_core = nn.GRU(
                self.d_byte, self.local_width, batch_first=True
            )
            self.local_norm = nn.LayerNorm(self.local_width)
            self.high_head = nn.Linear(self.local_width, 16)
            self.high_embedding = nn.Embedding(16, self.local_width)
            self.high_scale = nn.Embedding(16, self.local_width)
            self.low_norm = nn.LayerNorm(self.local_width)
            self.low_head = nn.Linear(self.local_width, 16)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.emission_mode == "product":
            nn.init.normal_(self.emission_high_logits, std=0.02)
            nn.init.normal_(self.emission_low_logits, std=0.02)
        else:
            nn.init.zeros_(self.high_scale.weight)
        nn.init.zeros_(self.from_abi.weight)

    @property
    def logical_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _spans(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.ndim != 2:
            raise ValueError("rows must have shape [batch, bytes]")
        complete = rows.shape[1] // self.span_bytes
        if complete < 2:
            raise ValueError("rows must contain at least two complete spans")
        return rows[:, : complete * self.span_bytes].to(torch.long).reshape(
            rows.shape[0], complete, self.span_bytes
        )

    def _contexts(self, source_spans: torch.Tensor) -> torch.Tensor:
        embedded = self.byte_embedding(source_spans).flatten(-2)
        encoded = torch.tanh(self.span_norm(self.span_projection(embedded)))
        hidden, _ = self.slow_core(encoded)
        hidden = hidden + self.from_abi(self.to_abi(hidden))
        return self.context_norm(hidden)

    def _emission_log_probability(
        self,
        target_spans: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        teacher_spans: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return log p(target span | latent) with shape [B, P, K]."""
        if self.emission_mode == "autoregressive":
            if context is None or teacher_spans is None:
                raise ValueError("autoregressive emissions require context and teachers")
            batch, patches, span = target_spans.shape
            embedded = self.byte_embedding(teacher_spans)
            expanded = embedded.unsqueeze(2).expand(
                batch, patches, self.latent_states, span, self.d_byte
            )
            expanded = expanded.reshape(
                batch * patches * self.latent_states, span, self.d_byte
            )
            base_initial = self.context_to_local(context).unsqueeze(2)
            latent_initial = self.latent_initial.weight.reshape(
                1, 1, self.latent_states, self.local_width
            )
            initial = (base_initial + latent_initial).reshape(
                1, batch * patches * self.latent_states, self.local_width
            )
            hidden, _ = self.local_core(expanded, initial.contiguous())
            hidden = self.local_norm(hidden)
            target = target_spans.unsqueeze(2).expand(
                batch, patches, self.latent_states, span
            ).reshape(batch * patches * self.latent_states, span)
            high_target = target.bitwise_right_shift(4)
            low_target = target.bitwise_and(15)
            high_log = F.log_softmax(self.high_head(hidden).float(), dim=-1)
            observed_high = high_log.gather(
                -1, high_target.unsqueeze(-1)
            ).squeeze(-1)
            low_hidden = self.low_norm(
                hidden * (1.0 + self.high_scale(high_target))
                + self.high_embedding(high_target)
            )
            low_log = F.log_softmax(self.low_head(low_hidden).float(), dim=-1)
            observed_low = low_log.gather(
                -1, low_target.unsqueeze(-1)
            ).squeeze(-1)
            return (observed_high + observed_low).sum(-1).reshape(
                batch, patches, self.latent_states
            )
        high_log = F.log_softmax(self.emission_high_logits.float(), dim=-1)
        low_log = F.log_softmax(self.emission_low_logits.float(), dim=-1)
        byte_log = (
            high_log.unsqueeze(-1) + low_log
        ).reshape(self.latent_states, self.span_bytes, 256)
        batch, patches, _ = target_spans.shape
        score = torch.zeros(
            batch,
            patches,
            self.latent_states,
            device=target_spans.device,
            dtype=byte_log.dtype,
        )
        for position in range(self.span_bytes):
            target = target_spans[:, :, position].reshape(-1)
            selected = byte_log[:, position, :].index_select(1, target)
            score = score + selected.transpose(0, 1).reshape(
                batch, patches, self.latent_states
            )
        return score

    def span_log_probs(self, rows: torch.Tensor) -> torch.Tensor:
        """Return normalized log probabilities of every observed target span."""
        spans = self._spans(rows)
        context = self._contexts(spans[:, :-1])
        target = spans[:, 1:]
        teacher = torch.cat(
            [spans[:, :-1, -1:].clone(), target[:, :, :-1]], dim=-1
        )
        gate = F.log_softmax(self.latent_gate(context).float(), dim=-1)
        emission = self._emission_log_probability(
            target, context=context, teacher_spans=teacher
        )
        return torch.logsumexp(gate + emission, dim=-1)

    def loss(self, rows: torch.Tensor) -> torch.Tensor:
        """Return negative log likelihood per raw byte."""
        return -self.span_log_probs(rows).mean() / self.span_bytes

    @torch.no_grad()
    def generate_spans(
        self,
        prompt: torch.Tensor,
        *,
        spans: int,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> torch.Tensor:
        """Generate whole raw-byte spans without a tokenizer."""
        if prompt.ndim != 2 or prompt.shape[1] < self.span_bytes:
            raise ValueError("prompt must contain at least one complete span")
        generated = prompt.to(torch.long)
        for _ in range(spans):
            source = self._spans(generated)
            context = self._contexts(source)[:, -1]
            gate_logits = self.latent_gate(context) / max(temperature, 1e-6)
            if sample:
                latent = torch.multinomial(F.softmax(gate_logits, dim=-1), 1).squeeze(1)
            else:
                latent = gate_logits.argmax(dim=-1)
            if self.emission_mode == "autoregressive":
                local = (
                    self.context_to_local(context)
                    + self.latent_initial(latent)
                ).unsqueeze(0)
                teacher = generated[:, -1]
                emitted = []
                for _position in range(self.span_bytes):
                    local_input = self.byte_embedding(teacher).unsqueeze(1)
                    hidden, local = self.local_core(local_input, local)
                    hidden = self.local_norm(hidden[:, 0])
                    high_logits = self.high_head(hidden)
                    if sample:
                        high = torch.multinomial(
                            F.softmax(
                                high_logits / max(temperature, 1e-6), dim=-1
                            ),
                            1,
                        ).squeeze(1)
                    else:
                        high = high_logits.argmax(dim=-1)
                    low_hidden = self.low_norm(
                        hidden * (1.0 + self.high_scale(high))
                        + self.high_embedding(high)
                    )
                    low_logits = self.low_head(low_hidden)
                    if sample:
                        low = torch.multinomial(
                            F.softmax(
                                low_logits / max(temperature, 1e-6), dim=-1
                            ),
                            1,
                        ).squeeze(1)
                    else:
                        low = low_logits.argmax(dim=-1)
                    teacher = high.bitwise_left_shift(4) + low
                    emitted.append(teacher)
                next_span = torch.stack(emitted, dim=1)
                generated = torch.cat([generated, next_span], dim=1)
                continue
            high_logits = self.emission_high_logits[latent]
            if sample:
                high = torch.multinomial(
                    F.softmax(high_logits / max(temperature, 1e-6), dim=-1).reshape(-1, 16),
                    1,
                ).reshape(generated.shape[0], self.span_bytes)
            else:
                high = high_logits.argmax(dim=-1)
            batch_index = torch.arange(generated.shape[0], device=generated.device)[:, None]
            position = torch.arange(self.span_bytes, device=generated.device)[None, :]
            low_logits = self.emission_low_logits[latent][batch_index, position, high]
            if sample:
                low = torch.multinomial(
                    F.softmax(low_logits / max(temperature, 1e-6), dim=-1).reshape(-1, 16),
                    1,
                ).reshape(generated.shape[0], self.span_bytes)
            else:
                low = low_logits.argmax(dim=-1)
            next_span = high.bitwise_left_shift(4) + low
            generated = torch.cat([generated, next_span], dim=1)
        return generated


def parameter_count(**kwargs) -> int:
    """Build on CPU and return the exact trainable parameter count."""
    return LatentSpanCakeLM(**kwargs).logical_parameters
