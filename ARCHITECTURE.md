# LayerCake Architecture

## Overview

LayerCake is a transformer-based language model with a fixed-dimension bottleneck interface
called the **ABI** (Architecture Bottleneck Interface). The ABI sits between the core
transformer and any domain-specific extensions, enabling domain modules to be shared across
models of different sizes.

---

## Core Architectural Components

### 1. Core Transformer

The core follows a standard pre-norm causal transformer architecture:

```
Input IDs [B, T]
    → token_emb [vocab × d_model]  +  pos_emb [max_seq_len × d_model]
    → N × SimpleTransformerBlock
```

**`SimpleTransformerBlock`** (identical in LayerCake and the baseline):
```
x = x + Attn(LayerNorm(x))   # Pre-norm multi-head self-attention
x = x + FFN(LayerNorm(x))    # Pre-norm feed-forward (GELU)
```

All blocks use pre-norm (LayerNorm before attention/FFN), which provides better training
stability than post-norm.

### 2. ABI Projection (the key innovation)

After the core transformer, a linear projection maps to a **fixed** d_abi=512 dimension:

```
h_core  [B, T, d_model]
    → core_to_abi  Linear(d_model, d_abi=512, bias=False)
    → LayerNorm(d_abi)
    → h_abi  [B, T, 512]
```

For the 48M config, `d_model = d_abi = 512` (the projection is the identity up to learned
rotation). For the 150M config, `d_model = 768` projects down to 512. For 350M, `d_model = 1024`
projects down to 512.

**Why 512?** It is large enough to carry rich semantic information while being small enough
that domain modules are parameter-efficient (~1.05M for `DomainModuleLite`).

### 3. Domain Modules

Domain modules operate entirely in `d_abi=512` space. They compute a residual delta:

$$h_\text{out} = h_\text{abi} + \exp(\alpha) \cdot \bigl(F(h_\text{abi}) - h_\text{abi}\bigr)$$

- $\alpha$ is a learned scalar, initialized to 0 (so `exp(0) = 1.0`, identity delta at init)
- $F$ is the domain module network
- The delta is weighted by $\exp(\alpha)$ which is trained to scale the domain contribution

Two implementations are provided:

**`DomainModuleLite`** (~1.05M params) — recommended for most use cases:
```
h → LayerNorm → SwiGLU(Linear(d_abi → d_ff), Linear(d_abi → d_ff)) → Linear(d_ff → d_abi)
delta = output - h_abi (weighted by exp(α))
```

**`DomainModule`** (~6.3M params) — for maximum domain capacity:
```
h → 2 × SimpleTransformerBlock(d_abi)
delta = output - h_abi (weighted by exp(α))
```

Because `d_abi=512` is fixed, all domain modules have **exactly the same weight shapes**
regardless of which model size they are embedded in. This is what makes the state dict
directly copyable.

### 4. ABI Back-Projection

After domain modules apply their deltas, the representation is projected back to `d_model`:

```
h_abi_out  [B, T, 512]
    → abi_to_core  Linear(512, d_model, bias=False)
    → h  [B, T, d_model]
    → h = h + h_core   (residual from core output — helps training)
    → LayerNorm → LM Head → logits  [B, T, vocab]
```

### 5. Domain Router (optional)

An optional learned router selects which domain modules to activate:

```
h_abi.mean(dim=1)  [B, d_abi]
    → Linear(d_abi, hidden=256) → GELU → Linear(256, num_domains)
    → sigmoid(logits / temperature)
    → domain_mask  [B, num_domains]   (soft mask in [0, 1])
```

When `use_router=False` (default), domain masks are passed explicitly or set to None
(core-only mode).

---

## Parameter Counts

For the 48M configuration (`d_model=512, d_abi=512, n_layers=6, n_heads=8, d_ff=2048, vocab=16K`):

| Component | Parameters |
|-----------|-----------|
| Token embedding | 512 × 16000 = 8.19M |
| Positional embedding | 512 × 256 = 0.13M |
| Core blocks (6×) | 6 × ~6.3M = ~37.8M |
| core_to_abi | 512 × 512 = 0.26M |
| abi_to_core | 512 × 512 = 0.26M |
| ABI LayerNorm | 512 × 2 = ~0 |
| LM head (tied) | shared with token_emb |
| Domain modules (per domain, Lite) | ~1.05M |
| Router (optional) | ~0.13M |
| **Total (core-only, no domains)** | **~35.96M** |

For comparison, the baseline transformer with matched parameters has 35.96M parameters using
the same block class. The ABI adds projections but they are matched by slight d_ff expansion
in the baseline for a true apples-to-apples comparison.

---

## Forward Pass (Core-Only Mode)

```python
# No domain modules active — equivalent to standard transformer
logits, _ = model(input_ids, domain_mask=None)
```

In core-only mode, the ABI projections are identity operations with no semantic effect
(at init, before training). The model behaves as a standard causal LM.

## Forward Pass (Domain-Active Mode)

```python
# Activate chess domain with weight 1.0
domain_mask = torch.zeros(batch_size, num_domains)
domain_mask[:, domain_names.index("chess")] = 1.0
logits, (router_logits, router_probs) = model(input_ids, domain_mask=domain_mask)
```

## Forward Pass (Learned Router)

```python
logits, (router_logits, router_probs) = model(
    input_ids,
    use_learned_router=True,
    router_temperature=1.0,
)
```

---

## ABI Interface for Thinkers

The ABI can also be used as a frozen black-box interface for "thinker" modules that
modulate representations without touching the core:

```python
# Get frozen ABI states (gradients detached — core stays frozen)
h_abi = model.get_abi_hidden_states(input_ids)  # [B, T, 512]

# Thinker applies modulations to h_abi (not shown here)
h_abi_modulated = thinker(h_abi)

# Decode modulated states back to logits (gradients flow through thinker only)
logits = model.decode_from_abi(h_abi_modulated)
```

This allows thinker modules to be trained while the core weights are completely frozen.

---

## Why the ABI Enables Lossless Portability

Consider two models:

- Model A: d_model=512, d_abi=512. The `core_to_abi` projection is 512→512.
- Model B: d_model=768, d_abi=512. The `core_to_abi` projection is 768→512.

The domain module in both is `DomainModuleLite(d_abi=512)`. Its weight matrices have shapes:
- `ln.weight`: [512]
- `gate_proj.weight`: [1024, 512]
- `up_proj.weight`: [1024, 512]
- `down_proj.weight`: [512, 1024]
- `log_alpha`: [1]

These shapes are **identical** in both models. There is no `d_model` in any of them.

A `state_dict` copy therefore transfers all weights bit-for-bit with no mismatches.

The `results/paste_proof.json` file records MD5 checksums of each tensor for a 37.5M model
and a 112.2M model, confirming they are identical.

---

## Comparison to Standard Transformer (Baseline)

The `baseline_lm.py` uses the same `SimpleTransformerBlock` as LayerCake's core, ensuring
that any performance difference is attributable to architecture (ABI + domains), not to
implementation details like weight initialization or block variants.

At 20K training steps with matched parameters (35.96M each):
- LayerCake adds `core_to_abi` and `abi_to_core` projections and an ABI LayerNorm
- The baseline has slightly wider FFN layers to compensate for parameter matching
- Both share the same causal masking, optimizer, scheduler, and data

See `results/fair_comparison.json` for the full benchmark.
