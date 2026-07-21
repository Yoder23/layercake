"""Compact causal byte-patch foundation with physically sparse routed capacity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .routed_cakes import Top1RoutedFoundationCakes


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
        if isinstance(prompt, str):
            prompt = prompt.encode("utf-8")
        if isinstance(prompt, bytes):
            prompt = torch.tensor(list(prompt), dtype=torch.long)[None]
        device = next(self.parameters()).device
        generated = prompt.to(device)
        for _ in range(max_new_bytes):
            logits = self(generated[:, -context:])
            generated = torch.cat([generated, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return generated

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
