#!/usr/bin/env python3
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTransformerBlock(nn.Module):
    """IDENTICAL block to LayerCake's core — ensures apples-to-apples comparison."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.ln1(x)
        x = x + self.attn(normed, normed, normed, attn_mask=attn_mask)[0]
        x = x + self.ff(self.ln2(x))
        return x


class BaselineTransformerLM(nn.Module):
    """
    Plain GPT-ish transformer LM using the SAME SimpleTransformerBlock as LayerCake.
    
    This ensures any performance differences come from ARCHITECTURE, not implementation.
    
    Supports optional extra_d_ff to match total parameter count with LayerCake's
    ABI overhead (projections + domain modules).

    Shapes:
      input_ids: [B, T]
      logits:    [B, T, V]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        extra_d_ff: int = 0,  # Extra FF width to match LayerCake param count
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Use SAME block class as LayerCake (apples-to-apples)
        actual_d_ff = d_ff + extra_d_ff
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, actual_d_ff)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self._causal_mask_cache = None
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight,      mean=0.0, std=0.02)
        
        for blk in self.blocks:
            for module in blk.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cached = self._causal_mask_cache
        if cached is not None:
            mask, cached_len, cached_dev = cached
            if cached_len >= seq_len and cached_dev == device:
                return mask[:seq_len, :seq_len]
        
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        self._causal_mask_cache = (mask, seq_len, device)
        return mask

    def forward(self, input_ids: torch.LongTensor):
        """
        input_ids: [B, T]
        returns: logits [B, T, V]
        """
        B, T = input_ids.shape
        device = input_ids.device

        if T > self.max_seq_len:
            raise ValueError(f"seq_len={T} > max_seq_len={self.max_seq_len}")

        pos = torch.arange(0, T, device=device, dtype=torch.long).unsqueeze(0)  # [1, T]
        x = self.token_emb(input_ids) + self.pos_emb(pos)  # [B, T, C]

        causal_mask = self._get_causal_mask(T, device=device)
        
        for blk in self.blocks:
            x = blk(x, attn_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]
        return logits
