"""Portable specialist residual that relies on a compatible English host."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PortableFusionConfig:
    abi_width: int = 64
    byte_width: int = 16
    hidden_width: int = 64
    rank: int = 32
    max_logit_residual: float = 4.0
    architecture_version: str = "portable-fusion/1"
    combination_rule: str = "bounded-additive-logit-residual"

    def __post_init__(self) -> None:
        if min(self.abi_width, self.byte_width, self.hidden_width, self.rank) <= 0:
            raise ValueError("portable fusion widths must be positive")
        if self.max_logit_residual <= 0:
            raise ValueError("max_logit_residual must be positive")

    def canonical_dict(self) -> dict:
        return asdict(self)


class PortableFusionCake(nn.Module):
    """A compact recurrent cake that corrects, but never replaces, host logits."""

    def __init__(self, config: PortableFusionConfig | None = None):
        super().__init__()
        self.config = config or PortableFusionConfig()
        cfg = self.config
        self.byte_embedding = nn.Embedding(256, cfg.byte_width)
        self.input_norm = nn.LayerNorm(cfg.abi_width + cfg.byte_width)
        self.recurrent = nn.GRU(
            cfg.abi_width + cfg.byte_width, cfg.hidden_width, batch_first=True
        )
        self.down = nn.Linear(cfg.hidden_width + cfg.abi_width, cfg.rank, bias=False)
        self.up = nn.Linear(cfg.rank, 256, bias=False)
        self.confidence = nn.Linear(cfg.hidden_width + cfg.abi_width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        for name, parameter in self.recurrent.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.confidence.weight)
        nn.init.constant_(self.confidence.bias, -1.0)

    def residual(
        self,
        canonical_state: torch.Tensor,
        byte_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if canonical_state.shape[:-1] != byte_ids.shape:
            raise ValueError("canonical state and byte ids must share batch/sequence dimensions")
        if canonical_state.shape[-1] != self.config.abi_width:
            raise ValueError("canonical ABI width mismatch")
        inputs = torch.cat([canonical_state, self.byte_embedding(byte_ids.long())], dim=-1)
        recurrent, hidden = self.recurrent(self.input_norm(inputs), hidden)
        features = torch.cat([recurrent, canonical_state], dim=-1)
        raw = self.up(F.silu(self.down(features)))
        confidence = torch.sigmoid(self.confidence(features))
        bounded = self.config.max_logit_residual * confidence * torch.tanh(raw)
        return bounded, hidden

    def forward(
        self,
        core_logits: torch.Tensor,
        canonical_state: torch.Tensor,
        byte_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual, hidden = self.residual(canonical_state, byte_ids, hidden)
        return core_logits + residual, hidden

    def parameter_report(self, core_total_parameters: int, core_active_parameters: int) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "cake_parameters": total,
            "fraction_of_core_total": total / core_total_parameters,
            "fraction_of_core_active": total / core_active_parameters,
            "combination_rule": self.config.combination_rule,
        }


def portable_fusion_manifest_architecture(config: PortableFusionConfig) -> dict:
    return {"name": "portable_fusion", **config.canonical_dict()}

