# Domain Paste Guide

## What Domain Paste Does

A LayerCake "domain brick" is a small neural module (~1M params) that was trained to
specialize the model for a specific domain (chess notation, Python code, legal text, etc.).
Because all domain bricks live in the same fixed 512-dimensional ABI space, their weight
tensors have the same shapes regardless of the surrounding model's size.

Pasting a domain brick from Model A to Model B means:
1. Extract the domain module's state dict from Model A's checkpoint.
2. Load it directly into Model B's checkpoint at the same key path.
3. Save Model B with the pasted domain brick.

No calibration. No alignment. No retraining. The weights are bit-identical.

---

## Quick Start

```bash
# Paste the "chess" domain from a 37.5M model to a 112.2M model
python paste_domain.py \
    --source checkpoints/small_37M.pt \
    --target checkpoints/large_112M.pt \
    --domain chess \
    --out    checkpoints/large_with_chess.pt
```

Then use the pasted model:
```python
from model import LayerCakeLMFixedABI
import torch

ckpt = torch.load("checkpoints/large_with_chess.pt", map_location="cpu")
model = LayerCakeLMFixedABI(vocab_size=16000, d_model=768, d_abi=512, ...)
model.load_state_dict(ckpt["model"])

# Activate chess domain
domain_mask = torch.zeros(1, num_domains)
domain_mask[0, domain_names.index("chess")] = 1.0
logits, _ = model(input_ids, domain_mask=domain_mask)
```

---

## Python API

```python
from paste_domain import paste_domain_brick, paste_domains

# Single domain paste
paste_domain_brick(
    source_path="checkpoints/small.pt",   # Model with trained chess domain
    target_path="checkpoints/large.pt",   # Target model (any size)
    domain_name="chess",
    output_path="checkpoints/large_chess.pt",
)

# Multiple domain paste
paste_domains(
    source_path="checkpoints/small.pt",
    target_path="checkpoints/large.pt",
    domains=["chess", "python"],
    output_path="checkpoints/large_chess_python.pt",
)
```

---

## Prerequisites

Both models must:
1. Have `d_abi=512` in their config (all standard LayerCake configs do)
2. Have the domain name in `domain_names` (e.g., `"chess"` must appear in the model's
   domain_names list — see configs or pass it when building the model)
3. Use the same `max_seq_len` (positional embeddings in the core must match)

The source model must have a trained domain module for the domain being pasted. If the
domain module was never trained, it will copy the randomly-initialized (near-identity)
weights, which is valid but will provide no domain benefit.

---

## How It Works

The paste is a direct state dict operation:

```python
source_state = checkpoint_A["model"]
target_state = checkpoint_B["model"]

# Copy all tensors belonging to the domain module
for key in source_state:
    if key.startswith(f"domain_modules.{domain_name}."):
        target_state[key] = source_state[key].clone()
```

No transformation is applied because the domain module weights have no `d_model`
dependence — they operate entirely in `d_abi=512` space.

**Formal proof of losslessness:**  
`results/paste_proof.json` records MD5 checksums of all domain module tensor values for
both a 37.5M model and a 112.2M model. The checksums are identical, confirming the weights
are bit-exact across model sizes.

---

## Paste Fidelity Verification

Run the self-consistency test at any time:

```bash
python tests/test_paste_lossless.py
```

Output:
```
MSE:             1.638141e-28
RMSE:            1.279899e-14
Max abs diff:    1.429412e-13
Mean cosine sim: 1.000000
Min cosine sim:  1.000000
RESULT: PASS  (MSE < 1e-20)
```

This test creates two domain modules from the same checkpoint, performs a paste, and
verifies that the pasted weights produce numerically identical outputs (within floating
point precision).

---

## Training a Domain Module

To get a meaningful domain module, train on domain-specific data with the core frozen:

```bash
# Tokenize your domain corpus first (see DATA.md)
python train_domain.py \
    --core_ckpt  runs/48M_core/best.pt \
    --domain_name chess \
    --train_data data/tokens/chess_train.npy \
    --eval_data  data/tokens/chess_val.npy \
    --max_steps  5000 \
    --lr         5e-4 \
    --out_dir    runs/chess_domain
```

The core weights are frozen. Only the domain module and router parameters are updated.
After training, `runs/chess_domain/domain_chess.pt` contains the trained domain module
ready for paste.

---

## Combining Multiple Domain Modules

Domain modules are additive deltas. You can activate multiple simultaneously:

```python
# Activate both chess and python domains
domain_mask = torch.ones(1, 2)  # All domains at full strength
logits, _ = model(input_ids, domain_mask=domain_mask)

# Partial activation (chess at 80%, python at 40%)
domain_mask = torch.tensor([[0.8, 0.4]])
logits, _ = model(input_ids, domain_mask=domain_mask)
```

The router (if trained) learns to select domains automatically:

```python
logits, (logits_r, probs_r) = model(input_ids, use_learned_router=True)
```

---

## Known Limitations

**Structural vs. functional portability:**  
The paste guarantee is structural: the weight tensors are bit-identical across model sizes.
Whether the domain module's learned specialization *performs equally well* after paste into
a differently-scaled model (with different core representations) is an open research
question. Initial experiments suggest performance is preserved when the ABI projections
are well-conditioned, but systematic evaluation at scale has not been completed.

**Architecture compatibility:**  
- Both models must have `d_abi=512`
- Both models must declare the domain name in their `domain_names` list
- Positional embedding shape (`max_seq_len`) must match (or use config-matching)

**Domain bleeding:**  
Domain modules are isolated in a `ModuleDict`. They do not share parameters and do not
affect each other when inactive (`domain_mask = 0`). In core-only mode
(`domain_mask = None`), no domain module is ever called.
