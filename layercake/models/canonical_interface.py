"""A versioned, semantically fixed interface shared by independent hosts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import nn


@dataclass(frozen=True)
class CanonicalInterfaceConfig:
    width: int = 64
    logit_width: int = 32
    anchor_decay: float = 0.875
    version: str = "lc-canonical-byte-semantics/2"
    precision_contract: str = "fp32-tolerance-1e-5"

    def __post_init__(self) -> None:
        if self.width < 16 or self.logit_width <= 0 or self.logit_width >= self.width:
            raise ValueError("canonical interface widths are invalid")
        if not 0.0 <= self.anchor_decay < 1.0:
            raise ValueError("anchor_decay must be in [0, 1)")

    def contract(self) -> dict:
        return {
            **asdict(self),
            "coordinates": {
                "predicted_byte_distribution": [0, self.logit_width],
                "causal_byte_anchor": [self.logit_width, self.width],
            },
            "normalization": "unit-rms-per-component-block",
            "combination": "bounded-additive-logit-residual",
            "patch_independence": True,
        }

    def abi_hash(self) -> str:
        raw = json.dumps(self.contract(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b"LAYERCAKE-CANONICAL-ABI-V2\0" + raw).hexdigest()


def _semantic_codebook(rows: int, width: int) -> torch.Tensor:
    byte = torch.arange(rows, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(width, dtype=torch.float32)[None] * (-math.log(10000.0) / max(width - 1, 1))
    )
    phase = byte * frequency
    codebook = torch.where(
        (torch.arange(width)[None] % 2) == 0,
        torch.sin(phase),
        torch.cos(phase),
    )
    return codebook / codebook.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)


class CanonicalByteInterface(nn.Module):
    """Map universal byte semantics into stable coordinates.

    The first block is an expectation over a fixed byte codebook under the
    host's predicted distribution.  The second block is a causal exponentially
    weighted summary of observed bytes in another fixed codebook.  Neither
    block depends on private hidden coordinates, depth, width, or random seed.
    """

    def __init__(self, config: CanonicalInterfaceConfig | None = None):
        super().__init__()
        self.config = config or CanonicalInterfaceConfig()
        anchor_width = self.config.width - self.config.logit_width
        self.register_buffer(
            "prediction_codebook", _semantic_codebook(256, self.config.logit_width), persistent=True
        )
        self.register_buffer(
            "anchor_codebook", _semantic_codebook(256, anchor_width).roll(17, dims=0), persistent=True
        )

    @staticmethod
    def _normalize(value: torch.Tensor) -> torch.Tensor:
        return value / value.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)

    def forward(self, core_logits: torch.Tensor, byte_ids: torch.Tensor) -> torch.Tensor:
        if core_logits.shape[:-1] != byte_ids.shape or core_logits.shape[-1] != 256:
            raise ValueError("canonical inputs require logits [..., 256] and matching byte ids")
        probabilities = torch.softmax(core_logits.float(), dim=-1)
        predicted = probabilities @ self.prediction_codebook.float()
        anchors = self.anchor_codebook[byte_ids.long()]
        running = torch.zeros_like(anchors[:, 0])
        rows = []
        decay = self.config.anchor_decay
        for index in range(anchors.shape[1]):
            running = decay * running + (1.0 - decay) * anchors[:, index]
            rows.append(running)
        causal_anchor = torch.stack(rows, dim=1)
        return torch.cat([self._normalize(predicted), self._normalize(causal_anchor)], dim=-1).to(core_logits.dtype)

    def step(
        self,
        core_logits: torch.Tensor,
        byte_ids: torch.Tensor,
        anchor_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if core_logits.shape[-1] != 256:
            raise ValueError("core logits must have width 256")
        if byte_ids.ndim != 1:
            byte_ids = byte_ids.flatten()
        probabilities = torch.softmax(core_logits.float(), dim=-1)
        predicted = probabilities @ self.prediction_codebook.float()
        anchor = self.anchor_codebook[byte_ids.long()]
        updated = self.config.anchor_decay * anchor_state + (1.0 - self.config.anchor_decay) * anchor
        canonical = torch.cat([self._normalize(predicted), self._normalize(updated)], dim=-1)
        return canonical.to(core_logits.dtype), updated

