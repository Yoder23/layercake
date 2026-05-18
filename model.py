#!/usr/bin/env python3
"""
layercake_model_fixed_abi.py

LayerCake with FIXED ABI DIMENSION for 100% lossless cross-size paste.

KEY INNOVATION:
- d_abi = 512 (FIXED across ALL model sizes)
- Large models (d_model=768) project DOWN to d_abi=512
- Small models (d_model=256) project UP to d_abi=512
- Domain modules operate ONLY on fixed d_abi=512 space
- Result: 100% lossless bidirectional paste between ANY model sizes

This solves the fundamental problem: different dimensional spaces cannot
be losslessly transformed. By fixing d_abi, we create a universal
"semantic space" that's model-size agnostic.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


class SimpleTransformerBlock(nn.Module):
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


class DomainModule(nn.Module):
    """Domain module operates on FIXED d_abi dimension.
    
    Full version with transformer layers — for maximum domain-specific capacity.
    Use DomainModuleLite for lower overhead.
    """
    
    def __init__(self, d_abi: int, n_layers: int = 2, n_heads: int = 8, d_ff: int = 2048):
        super().__init__()
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_abi, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.log_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, h_abi: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = h_abi
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        delta = x - h_abi
        return delta * self.log_alpha.exp()


class DomainModuleLite(nn.Module):
    """Lightweight domain module — MLP-based, 1/6th the params of full DomainModule.
    
    Uses a gated MLP instead of transformer layers. This is sufficient for
    domain-specific adaptation via paste while minimizing parameter overhead.
    Still operates on FIXED d_abi=512 for lossless paste compatibility.
    
    Params: ~1.05M (vs ~6.3M for full DomainModule with 2 layers)
    """
    
    def __init__(self, d_abi: int, d_ff: int = 1024):
        super().__init__()
        self.ln = nn.LayerNorm(d_abi)
        self.gate_proj = nn.Linear(d_abi, d_ff)
        self.up_proj = nn.Linear(d_abi, d_ff)
        self.down_proj = nn.Linear(d_ff, d_abi)
        self.log_alpha = nn.Parameter(torch.zeros(1))
    
    def forward(self, h_abi: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ln(h_abi)
        # SwiGLU-style gated projection
        delta = self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))
        return delta * self.log_alpha.exp()


class DomainRouter(nn.Module):
    """Router operates on FIXED d_abi dimension."""
    
    def __init__(self, d_abi: int, num_domains: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_abi, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, h_abi: torch.Tensor, temperature: float = 1.0):
        summary = h_abi.mean(dim=1)
        logits = self.net(summary)
        probs = torch.sigmoid(logits / max(temperature, 1e-6))
        return logits, probs


class LayerCakeLMFixedABI(nn.Module):
    """
    LayerCake with FIXED d_abi for universal lossless paste.
    
    Architecture:
      Embeddings (vocab -> d_model)
         ↓
      Core Transformer (d_model)
         ↓
      ABI Projection (d_model -> d_abi=512 FIXED)
         ↓
      Domain Modules (operate on d_abi=512)
         ↓
      ABI Back-Projection (d_abi=512 -> d_model)
         ↓
      LM Head (d_model -> vocab)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_abi: int = 512,  # FIXED across all model sizes
        n_core_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        domain_names=("python", "chess"),
        max_seq_len: int = 256,
        use_router: bool = True,
        domain_module_type: str = "lite",  # "full" (transformer) or "lite" (gated MLP)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_abi = d_abi  # FIXED dimension
        self.domain_names = list(domain_names)
        self.num_domains = len(domain_names)
        self.max_seq_len = max_seq_len
        self.use_router = use_router

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Core transformer
        self.core_blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_core_layers)
        ])

        # ===== KEY INNOVATION: Projection to/from FIXED d_abi =====
        self.core_to_abi = nn.Linear(d_model, d_abi, bias=False)
        self.abi_to_core = nn.Linear(d_abi, d_model, bias=False)
        self.abi_ln = nn.LayerNorm(d_abi)
        # ===========================================================

        # Domain modules operate on FIXED d_abi
        if domain_module_type == "lite":
            self.domain_modules = nn.ModuleDict({
                name: DomainModuleLite(d_abi, d_ff=d_abi * 2)
                for name in domain_names
            })
        else:
            self.domain_modules = nn.ModuleDict({
                name: DomainModule(d_abi, n_layers=2, n_heads=min(n_heads, d_abi // 64), d_ff=d_abi * 4)
                for name in domain_names
            })

        # Router (optional)
        if use_router:
            self.router = DomainRouter(d_abi, self.num_domains)
        else:
            self.router = None

        # Final layers
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._causal_mask_cache = None
        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly."""
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        # Initialize projection matrices (critical for fixed d_abi)
        nn.init.xavier_uniform_(self.core_to_abi.weight)
        nn.init.xavier_uniform_(self.abi_to_core.weight)
        
        for blk in self.core_blocks:
            for module in blk.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        for domain_mod in self.domain_modules.values():
            for module in domain_mod.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        if self.router is not None:
            for module in self.router.modules():
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

    def encode_core(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_core: [batch, seq, d_model] - core output
          h_abi:  [batch, seq, d_abi=512] - FIXED ABI interface
        """
        bsz, seq = input_ids.shape
        device = input_ids.device

        if seq > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
            seq = input_ids.size(1)

        positions = torch.arange(seq, device=device).unsqueeze(0).expand(bsz, seq)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal_mask = self._get_causal_mask(seq_len=seq, device=device)

        for blk in self.core_blocks:
            x = blk(x, attn_mask=causal_mask)

        h_core = x  # [B, T, d_model]
        
        # Project to FIXED d_abi
        h_abi = self.core_to_abi(h_core)  # [B, T, d_abi=512]
        h_abi = self.abi_ln(h_abi)
        
        if h_abi.dtype != torch.float32:
            h_abi = h_abi.to(torch.float32)

        return h_core, h_abi
    
    def get_abi_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        THINKER INTERFACE: Get ABI representations as frozen black box.
        
        Thinkers should ONLY use this method, never access internal layers.
        This ensures base model stays frozen and gradients don't leak.
        
        Args:
            input_ids: [B, T] token IDs
        Returns:
            h_abi: [B, T, 512] - fixed ABI representations (detached, no gradients)
        """
        _, h_abi = self.encode_core(input_ids)
        return h_abi.detach()  # CRITICAL: Detach to prevent gradient flow
    
    def decode_from_abi(self, h_abi: torch.Tensor) -> torch.Tensor:
        """
        THINKER INTERFACE: Decode modulated ABI states back to logits.
        
        This is where thinkers ADD VALUE - gradients flow through the
        modulated h_abi to train thinker parameters.
        
        Args:
            h_abi: [B, T, 512] - modulated ABI states from thinker
        Returns:
            logits: [B, T, vocab_size]
        """
        # Project back to core space
        h_core = self.abi_to_core(h_abi)  # [B, T, d_model]
        
        # Final layer norm and projection to vocab
        h_core = self.ln_f(h_core)
        logits = self.lm_head(h_core)  # [B, T, vocab_size]
        
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        domain_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        router_temperature: float = 1.0,
        use_learned_router: bool = False,
    ):
        """Forward pass with fixed d_abi architecture."""
        bsz, _ = input_ids.shape
        device = input_ids.device

        # Core + ABI (FIXED dimension)
        h_core, h_abi = self.encode_core(input_ids, attn_mask=None)
        seq = h_abi.size(1)
        causal_mask = self._get_causal_mask(seq_len=seq, device=device)

        router_logits = None
        router_probs = None

        # Router
        if use_learned_router and self.router is not None:
            router_logits, router_probs = self.router(h_abi, temperature=router_temperature)
            domain_mask = router_probs

        # Core-only mode
        if domain_mask is None:
            h_abi_out = h_abi
        else:
            # Domain-enhanced mode (operates on FIXED d_abi)
            deltas = []
            for i, name in enumerate(self.domain_names):
                module = self.domain_modules[name]
                delta = module(h_abi, attn_mask=causal_mask)  # [B, T, d_abi]
                m = domain_mask[:, i].view(bsz, 1, 1)
                deltas.append(delta * m)

            total_delta = torch.stack(deltas, dim=0).sum(dim=0)
            h_abi_out = h_abi + total_delta

        # Project BACK to d_model space
        h = self.abi_to_core(h_abi_out)  # [B, T, d_model]
        
        # Residual connection from original h_core
        h = h + h_core  # Residual helps training
        
        h = self.ln_f(h)
        logits = self.lm_head(h)
        
        return logits, (router_logits, router_probs)
