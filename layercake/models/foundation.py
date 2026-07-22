"""Compact causal byte-patch foundation with physically sparse routed capacity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .routed_cakes import Top1RoutedFoundationCakes
from .incremental_state import (
    IncrementalState,
    fingerprint_state_dict,
    restore_incremental_state,
    serialize_state_with_metadata,
)


@dataclass(frozen=True)
class FoundationConfig:
    patch_size: int = 4
    d_byte: int = 48
    d_model: int = 192
    recurrent_layers: int = 2
    local_kernel: int = 5
    routed_experts: int = 16
    expert_expansion: int = 4
    dropout: float = 0.0
    abi_width: int = 64
    architecture_version: str = "layercake-sparse-patch-foundation/1"

    def __post_init__(self) -> None:
        if self.patch_size <= 0 or self.d_byte <= 0 or self.d_model <= 0:
            raise ValueError("foundation widths and patch size must be positive")
        if self.recurrent_layers <= 0 or self.local_kernel <= 1:
            raise ValueError("foundation requires recurrent layers and a local kernel > 1")
        if self.routed_experts < 2 or self.abi_width <= 0:
            raise ValueError("foundation requires routed experts and a positive ABI width")

    def canonical_dict(self) -> dict:
        return asdict(self)


class CausalLocalByteBlock(nn.Module):
    def __init__(self, width: int, kernel: int):
        super().__init__()
        self.left_padding = kernel - 1
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel, groups=width)
        self.mix = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = self.norm(hidden).transpose(1, 2)
        value = self.depthwise(F.pad(value, (self.left_padding, 0))).transpose(1, 2)
        return hidden + self.mix(F.silu(value))


class LayerCakeFoundation(nn.Module):
    """Tokenizer-free LM with patch-rate recurrence and byte-rate local decoding.

    Patch summaries affect only the following patch. The byte-local convolution is
    left padded. Therefore logits at position ``t`` depend only on bytes ``<= t``.
    """

    def __init__(self, config: FoundationConfig | None = None):
        super().__init__()
        self.config = config or FoundationConfig()
        cfg = self.config
        self.byte_embedding = nn.Embedding(256, cfg.d_byte)
        self.local_in = nn.Linear(cfg.d_byte, cfg.d_model)
        self.local = CausalLocalByteBlock(cfg.d_model, cfg.local_kernel)
        self.patch_projection = nn.Linear(cfg.patch_size * cfg.d_byte, cfg.d_model)
        self.global_recurrent = nn.GRU(
            cfg.d_model, cfg.d_model, num_layers=cfg.recurrent_layers,
            batch_first=True, dropout=cfg.dropout if cfg.recurrent_layers > 1 else 0.0,
        )
        self.routed_cakes = Top1RoutedFoundationCakes(
            cfg.d_model, cfg.routed_experts, expansion=cfg.expert_expansion
        )
        self.global_norm = nn.LayerNorm(cfg.d_model)
        self.to_abi = nn.Linear(cfg.d_model, cfg.abi_width, bias=False)
        self.from_abi = nn.Linear(cfg.abi_width, cfg.d_model, bias=False)
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, 256, bias=False)
        self._model_fingerprint: str | None = None
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def set_route(self, route: int | None) -> None:
        self.routed_cakes.set_route(route)

    def _patch_global(self, embedded: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor]:
        batch, length, width = embedded.shape
        pad = (-length) % self.config.patch_size
        if pad:
            embedded = F.pad(embedded, (0, 0, 0, pad))
        patches = embedded.reshape(batch, -1, self.config.patch_size * width)
        projected = self.patch_projection(patches)
        global_hidden, _ = self.global_recurrent(projected)
        global_hidden, balance_loss = self.routed_cakes(global_hidden, return_aux_loss=True)
        return global_hidden, pad, balance_loss

    def forward(
        self,
        byte_ids: torch.Tensor,
        *,
        host_residual: nn.Module | None = None,
        return_aux: bool = False,
    ):
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, sequence]")
        if byte_ids.dtype != torch.long:
            byte_ids = byte_ids.long()
        if byte_ids.numel() and (int(byte_ids.min()) < 0 or int(byte_ids.max()) > 255):
            raise ValueError("byte_ids must be in [0, 255]")
        embedded = self.byte_embedding(byte_ids)
        global_hidden, pad, balance_loss = self._patch_global(embedded)
        # A completed patch becomes available only to the subsequent patch.
        initial = global_hidden.new_zeros(global_hidden.shape[0], 1, global_hidden.shape[-1])
        causal_patch_context = torch.cat([initial, global_hidden[:, :-1]], dim=1)
        abi = self.to_abi(self.global_norm(causal_patch_context))
        if host_residual is not None:
            abi = host_residual(abi)
        causal_patch_context = causal_patch_context + self.from_abi(abi)
        expanded = causal_patch_context.repeat_interleave(self.config.patch_size, dim=1)
        expanded = expanded[:, : byte_ids.shape[1]]
        local = self.local(self.local_in(embedded))
        logits = self.output(self.output_norm(local + expanded))
        if return_aux:
            return logits, {"routing_balance_loss": balance_loss, "abi": abi, "padding": pad}
        return logits

    @torch.inference_mode()
    def generate(self, prompt: bytes | str | torch.Tensor, max_new_bytes: int, context: int = 512):
        del context  # Retained for source compatibility; stateful decoding has no context rerun.
        state = self.prefill(prompt, capture_generated=True)
        _, state = self.decode_many(state, max_new_bytes)
        if isinstance(prompt, str):
            prompt = prompt.encode("utf-8")
        if isinstance(prompt, bytes):
            prompt = torch.tensor(list(prompt), dtype=torch.long)[None]
        return torch.cat([prompt.to(state.generated_bytes.device), state.generated_bytes], dim=1)

    def _fingerprint(self) -> str:
        if self._model_fingerprint is None:
            self._model_fingerprint = fingerprint_state_dict(self.state_dict())
        return self._model_fingerprint

    def reset_state(
        self,
        *,
        batch_size: int = 1,
        route: int = 0,
        sampler: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 0,
        sampler_seed: int = 0,
        capture_generated: bool = False,
    ) -> IncrementalState:
        if sampler not in {"greedy", "sample"}:
            raise ValueError("sampler must be 'greedy' or 'sample'")
        if temperature <= 0 or top_k < 0:
            raise ValueError("invalid sampler settings")
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        cfg = self.config
        return IncrementalState(
            local_history=torch.zeros(batch_size, cfg.local_kernel - 1, cfg.d_model, device=device, dtype=dtype),
            incomplete_patch=torch.zeros(batch_size, 0, cfg.d_byte, device=device, dtype=dtype),
            recurrent_hidden=torch.zeros(cfg.recurrent_layers, batch_size, cfg.d_model, device=device, dtype=dtype),
            completed_patch_summary=torch.zeros(batch_size, cfg.d_model, device=device, dtype=dtype),
            canonical_state=torch.zeros(batch_size, cfg.abi_width, device=device, dtype=dtype),
            next_logits=None,
            generated_bytes=torch.zeros(batch_size, 0, device=device, dtype=torch.long),
            route=int(route),
            model_fingerprint=self._fingerprint(),
            sampler=sampler,
            temperature=float(temperature),
            top_k=int(top_k),
            sampler_seed=int(sampler_seed),
            capture_generated=bool(capture_generated),
        )

    def _consume_byte(
        self,
        state: IncrementalState,
        byte_ids: torch.Tensor,
        *,
        host_residual: nn.Module | None = None,
    ) -> IncrementalState:
        if byte_ids.ndim == 1:
            byte_ids = byte_ids[:, None]
        if byte_ids.shape != (state.batch_size, 1):
            raise ValueError("incremental byte_ids must have shape [batch, 1]")
        embedded = self.byte_embedding(byte_ids.long())
        local_hidden = self.local_in(embedded)
        normalized = self.local.norm(local_hidden)
        window = torch.cat([state.local_history, normalized], dim=1)
        convolved = self.local.depthwise(window.transpose(1, 2)).transpose(1, 2)
        local = local_hidden + self.local.mix(F.silu(convolved))

        patch_context = state.completed_patch_summary[:, None]
        abi = self.to_abi(self.global_norm(patch_context))
        if host_residual is not None:
            abi = host_residual(abi)
        fused_context = patch_context + self.from_abi(abi)
        logits = self.output(self.output_norm(local + fused_context))[:, 0]

        state.local_history = window[:, 1:].detach()
        state.canonical_state = abi[:, 0].detach()
        state.next_logits = logits
        state.incomplete_patch = torch.cat([state.incomplete_patch, embedded], dim=1)
        if state.incomplete_patch.shape[1] == self.config.patch_size:
            projected = self.patch_projection(state.incomplete_patch.flatten(1))[:, None]
            recurrent, hidden = self.global_recurrent(projected, state.recurrent_hidden)
            self.routed_cakes.set_route(state.route)
            routed = self.routed_cakes(recurrent)
            state.completed_patch_summary = routed[:, 0].detach()
            state.recurrent_hidden = hidden.detach()
            state.incomplete_patch = state.incomplete_patch[:, :0].detach()
        return state

    @torch.inference_mode()
    def prefill(
        self,
        prompt: bytes | str | torch.Tensor,
        *,
        route: int = 0,
        host_residual: nn.Module | None = None,
        sampler: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 0,
        sampler_seed: int = 0,
        capture_generated: bool = False,
    ) -> IncrementalState:
        if isinstance(prompt, str):
            prompt = prompt.encode("utf-8")
        if isinstance(prompt, bytes):
            prompt = torch.tensor(list(prompt), dtype=torch.long)[None]
        if prompt.ndim != 2 or prompt.shape[1] == 0:
            raise ValueError("prefill requires a non-empty [batch, sequence] prompt")
        state = self.reset_state(
            batch_size=prompt.shape[0], route=route, sampler=sampler,
            temperature=temperature, top_k=top_k, sampler_seed=sampler_seed,
            capture_generated=capture_generated,
        )
        prompt = prompt.to(state.local_history.device)
        self.routed_cakes.set_route(route)
        for index in range(prompt.shape[1]):
            state = self._consume_byte(state, prompt[:, index:index + 1], host_residual=host_residual)
        state.prompt_bytes = int(prompt.numel())
        return state

    @staticmethod
    def _sample(state: IncrementalState, logits: torch.Tensor) -> torch.Tensor:
        if state.sampler == "greedy":
            return logits.argmax(-1)
        values = logits / state.temperature
        if state.top_k:
            count = min(state.top_k, values.shape[-1])
            threshold = values.topk(count, dim=-1).values[:, -1:]
            values = values.masked_fill(values < threshold, float("-inf"))
        # A per-step CPU generator makes sampling resumable across devices.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(state.sampler_seed + state.sampler_counter)
        probabilities = torch.softmax(values.float().cpu(), dim=-1)
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(1).to(logits.device)

    @torch.inference_mode()
    def decode_step(
        self,
        state: IncrementalState,
        *,
        next_byte: torch.Tensor | None = None,
        host_residual: nn.Module | None = None,
    ) -> tuple[torch.Tensor, IncrementalState]:
        if state.model_fingerprint != self._fingerprint():
            raise ValueError("incremental state belongs to different model weights")
        if state.next_logits is None:
            raise ValueError("state has not been prefilled")
        logits = state.next_logits
        selected = self._sample(state, logits) if next_byte is None else next_byte.to(logits.device).long().flatten()
        if selected.shape != (state.batch_size,):
            raise ValueError("next_byte must provide one byte per batch item")
        if state.capture_generated:
            state.generated_bytes = torch.cat([state.generated_bytes, selected[:, None]], dim=1)
        state.decoded_bytes += state.batch_size
        state.sampler_counter += 1
        state = self._consume_byte(state, selected[:, None], host_residual=host_residual)
        return logits, state

    @torch.inference_mode()
    def decode_many(
        self,
        state: IncrementalState,
        count: int,
        *,
        host_residual: nn.Module | None = None,
    ) -> tuple[torch.Tensor, IncrementalState]:
        if count < 0:
            raise ValueError("count must be non-negative")
        rows = []
        for _ in range(count):
            logits, state = self.decode_step(state, host_residual=host_residual)
            rows.append(logits[:, None])
        empty = state.local_history.new_zeros(state.batch_size, 0, 256)
        return (torch.cat(rows, dim=1) if rows else empty), state

    def serialize_state(self, state: IncrementalState) -> bytes:
        if state.model_fingerprint != self._fingerprint():
            raise ValueError("incremental state belongs to different model weights")
        return serialize_state_with_metadata(state)

    def restore_state(self, payload: bytes) -> IncrementalState:
        return restore_incremental_state(
            payload,
            expected_model_fingerprint=self._fingerprint(),
            device=next(self.parameters()).device,
        )

    def parameter_report(self, route: int = 0) -> dict[str, float | int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        expert_total = sum(
            parameter.numel() for expert in self.routed_cakes.experts for parameter in expert.parameters()
        )
        one_expert = sum(parameter.numel() for parameter in self.routed_cakes.experts[route].parameters())
        always_active = total - expert_total
        active = always_active + one_expert
        return {
            "total_parameters": total,
            "active_parameters_per_homogeneous_batch": active,
            "active_fraction": active / total,
            "routed_experts": self.config.routed_experts,
        }


class SparseOptimizerFactory:
    """Create an optimizer whose state contains only shared and one pinned expert."""

    @staticmethod
    def parameters(model: LayerCakeFoundation, route: int) -> Iterable[nn.Parameter]:
        expert_ids = {id(parameter) for parameter in model.routed_cakes.experts.parameters()}
        yield from (parameter for parameter in model.parameters() if id(parameter) not in expert_ids)
        yield from model.routed_cakes.expert_parameters(route)

    @classmethod
    def adamw(
        cls, model: LayerCakeFoundation, route: int, *, lr: float = 3e-4, weight_decay: float = 0.01
    ) -> torch.optim.AdamW:
        model.set_route(route)
        return torch.optim.AdamW(cls.parameters(model, route), lr=lr, weight_decay=weight_decay)
