"""Small reference byte-patch LayerCake model for smoke experiments."""

from __future__ import annotations

import torch
from torch import nn

from .abi import ABISpec
from .byte_patch import BytePatchDecoder, BytePatchEncoder
from .domain_bricks import DomainOperator


class BytePatchLayerCake(nn.Module):
    def __init__(
        self,
        abi_spec: ABISpec,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        patcher=None,
        domain_brick: DomainOperator | None = None,
    ):
        super().__init__()
        self.abi_spec = abi_spec
        self.frontend = BytePatchEncoder(d_model=d_model, patcher=patcher)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.core = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.core_to_abi = nn.Linear(d_model, abi_spec.d_abi)
        self.abi_to_core = nn.Linear(abi_spec.d_abi, d_model)
        self.abi_norm = nn.LayerNorm(abi_spec.d_abi)
        self.domain_brick = domain_brick
        self.decoder = BytePatchDecoder(d_model)

    def encode_abi(self, byte_ids: torch.Tensor):
        patches, metadata = self.frontend(byte_ids)
        positions = patches.shape[1]
        mask = torch.triu(
            torch.full((positions, positions), float("-inf"), device=patches.device),
            diagonal=1,
        )
        core = self.core(patches, mask=mask)
        abi = self.abi_norm(self.core_to_abi(core))
        return core, abi, metadata

    def forward(self, byte_ids: torch.Tensor):
        _, abi, metadata = self.encode_abi(byte_ids)
        if self.domain_brick is not None:
            abi = self.domain_brick(abi, self.abi_spec)
        patch_states = self.abi_to_core(abi)
        byte_logits = self.decoder(patch_states, metadata)
        return byte_logits, abi, metadata
