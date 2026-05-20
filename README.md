# LayerCake

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Status: Research Preview](https://img.shields.io/badge/status-research%20preview-yellow.svg)]()
[![Paste Tests](https://github.com/Yoder23/layercake/actions/workflows/tests.yml/badge.svg)](https://github.com/Yoder23/layercake/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yoder23/layercake/blob/master/notebooks/layercake_demo.ipynb)

> **This repository is LayerCake** — the native modular architecture. For post-hoc alignment across independently trained or cross-architecture models, see the companion [ABI repository](https://github.com/Yoder23/abi).

**A modular language-model architecture with a fixed-dimensional ABI bottleneck, enabling bit-exact domain-module paste across model sizes.**

LayerCake separates the general LM core from domain-specific modules. All domain modules operate in a fixed `d_abi=512` space, so their weights can be copied directly between LayerCake models of any `d_model` size.

The paste is **bit-exact and function-preserving**: weights match exactly, forward outputs match with `max_diff = 0.0`, and same-core autoregressive generation is token-identical.

Functional transfer across independently trained cores (different seeds) is a separate problem — different seeds produce different ABI coordinate systems. That requires ABI alignment, handled in the [companion ABI project](#relationship-to-abi).

---

## Start Here

```bash
# No GPU. No data. No checkpoints. Runs in ~3 seconds.
git clone https://github.com/Yoder23/layercake
cd layercake
pip install -e .
python verify_paste.py
```

Expected output:
```
  [weight identity — 9 tensors, cross-size 48M → 150M]
    ✓  9 tensors — max_diff = 0.0 (bit-exact)
  [forward-pass identity — same core + pasted domain]
    ✓  logit max_diff = 0.000000e+00 (bit-exact)
  [generation identity — 50-token autoregressive sequence]
    ✓  50 tokens generated — zero divergence
  [cross-size function identity — domain module 48M → 150M]
    ✓  domain output max_diff = 0.000000e+00 (bit-exact, d_model 512→768)

  ALL CHECKS PASSED  (~3s)
```

If you want to paste between real trained checkpoints:

```python
from paste_domain import paste_domain_brick

# Copy a trained chess domain module from a 48M model to a 150M model.
paste_domain_brick(
    source_path="checkpoints/48M_core.pt",
    target_path="checkpoints/150M_core.pt",
    domain_name="chess",
    output_path="checkpoints/150M_with_chess.pt",
)
```

This works because LayerCake routes all domain computation through a **fixed 512-dimensional
bottleneck** (the ABI). Domain modules live entirely in that fixed space — the same
tensor whether the surrounding model is 48M or 150M parameters.

**Note on functional transfer:** Paste fidelity is proven (max diff = 0.0). Functional transfer
across independently-trained cores also requires ABI alignment, since different training seeds
produce different `core_to_abi` projection geometries. See [Claim 1](#-claim-1--domain-paste-is-bit-exact-and-function-preserving) for full details.

---

## What LayerCake Claims

### Claim Ladder

| Claim | Status | Scope |
|-------|--------|-------|
| Bit-exact paste (weights) | Validated | Same `d_abi`, any `d_model` |
| Same-core forward-pass identity (`max_diff = 0.0`) | Validated | Same core representations |
| Same-core generation identity (token-for-token) | Validated | Same core + pasted domain, 50 tokens |
| Cross-size structural portability (weights + function) | Validated | Domain module weights/function transfer exactly |
| Cross-seed functional transfer | **Not validated** | Requires ABI alignment |
| Raw LM parity with transformer | Validated at 48M/20K steps | Not tested at large scale |
| Domain adaptation efficiency | Validated on chess | 6.7× fewer trainable params |

Full evidence: [CLAIMS.md](CLAIMS.md) — Common objections: [SKEPTICS.md](SKEPTICS.md)

---

### ✅ Claim 1 — Domain paste is bit-exact and function-preserving

Domain module paste is **bit-exact and function-preserving** at three levels:

**Level 1 — Weight identity:** All 25 domain module tensors are bit-exact after paste, across any target model size with `d_abi=512`.

**Level 2 — Forward-pass identity:** When a pasted domain module is evaluated on the same core representations, it produces **bit-identical outputs** (`max_diff = 0.0`). This holds both same-size and cross-size.

**Level 3 — Generation identity (the full proof):** Autoregressive generation — token-by-token, feeding each output back as input — produces **identical token sequences** at every single step. No divergence across 50 steps. This is the strongest possible proof: lossless at the output level, not just the weight level.

| Test | Result |
|------|--------|
| Weight identity: 9 tensors (cross-size 48M→150M) | bit-exact ✓ |
| Forward pass: same-size, same core + pasted domain (max diff = 0.0) | bit-exact ✓ |
| Forward pass: cross-size domain module on d_abi input (max diff = 0.0) | bit-exact ✓ |
| **Generation: 50-token autoregressive sequence, token-by-token** | **identical ✓** |

Run to reproduce:
```bash
python tests/test_paste_lossless.py
# All 6 tests pass. Generation sequences match at every token.
```

**Why paste is bit-exact:** Domain modules compute exclusively in `d_abi=512` space. Paste is a direct `tensor.clone()` — no transformation, no quantization, no interpolation. The domain module is the same function before and after paste. We call this *lossless paste*: weights, domain-function outputs, and same-core generation are all bit-identical. This does **not** imply functional compatibility under different ABI input distributions.

**Why cross-seed PPL degrades (and why it doesn't contradict losslessness):**  
Paste fidelity and functional compatibility are orthogonal questions. The domain module is
copied *perfectly* — but it was trained to interpret `h_abi` vectors from a specific core
(seed9000). A different-seed core (seed6000) produces geometrically different `h_abi` vectors.
The domain module runs correctly; it simply receives inputs it has never seen.
This is analogous to copying a chess engine's weights to a different input encoding — the copy
is lossless but the inputs are mismatched. ABI alignment resolves this; it is left as future work.

Both models share `d_abi = 512`. The domain module state dict is the same in both.
Verified: [`results/paste_proof.json`](results/paste_proof.json), [`results/domain_paste_functional.json`](results/domain_paste_functional.json), [`tests/test_paste_lossless.py`](tests/test_paste_lossless.py)

### ✅ Claim 2 — Raw LM quality at par with a standard transformer

At 20,000 training steps with exactly matched parameters, the same optimizer, the same seed,
and the same data, LayerCake is within measurement noise of a plain transformer:

| Metric | LayerCake | Baseline | Δ |
|--------|-----------|----------|---|
| C4 val PPL (20K steps) | 45.01 | 44.89 | +0.27% overhead |
| WikiText2 PPL | 174.69 | 171.82 | +1.67% overhead |
| HellaSwag accuracy | **27.0%** | 26.0% | **+3.85%** |
| Parameters | 35.96M | 35.96M | matched |

Fairness controls applied: same block class, same seed (42), same optimizer (AdamW), same LR
schedule, same data sampling, same training steps.
Full results: [`results/fair_comparison.json`](results/fair_comparison.json)

### ✅ Claim 3 — Domain adaptation is 6.7× more parameter-efficient

Training only a domain module (6.3M params, 15% of model) achieves nearly identical
domain-specific quality to full model fine-tuning (42.3M params, 100% of model):

| Method | Trainable params | Chess PPL (5K steps) | Training time |
|--------|-----------------|---------------------|---------------|
| **LayerCake domain module** | **6.30M (15%)** | **2.50** | **583s** |
| Full model fine-tune | 42.26M (100%) | 2.42 | 648s |

6.7× fewer parameters trained. ~10% faster. Equivalent final quality.
The core transformer is frozen — only the 6.3M domain module is updated.

Training starting point: untrained domain (PPL 45.7 before, **2.50 after**).
Full results: [`results/domain_paste_functional.json`](results/domain_paste_functional.json)

### How domain modules compare to LoRA

| | LoRA | LayerCake domain module |
|---|---|---|
| Portable across model sizes? | ❌ No — rank matrices are tied to `d_model` | ✅ Yes — operates only in fixed `d_abi=512` |
| Paste is bit-exact? | ❌ No | ✅ Yes — `max_diff = 0.0` proven |
| Generation-identical after paste? | ❌ No | ✅ Yes — token-for-token identical |
| Core stays frozen during domain training? | ✅ Yes | ✅ Yes |
| Works with independently trained cores? | ✔ Any model | ⚠️ Same-ABI cores only (different seeds require alignment) |
| Typical trainable params | 0.1–1% of model | ~15% of model (domain module only) |

The key difference: LoRA parameters are shaped `(d_model, rank)` — they cannot move between
models with different `d_model`. LayerCake domain modules are shaped `(d_abi, d_abi)` with
`d_abi=512` fixed across all model sizes, so they paste directly with no transformation.

---

## What LayerCake Does NOT Claim

- **Does not outperform** standard transformers on raw language modeling at current scale.
  The overhead is 0.27–1.67% PPL — within measurement noise — not a win.
- **Domain modules do not help general LM quality.** They are designed for domain-specific
  perplexity (chess, Python, medical text) when trained on domain data, not for general text.
- **Cross-seed functional transfer does not work without alignment.** Structural paste is
  bit-exact, but domain task performance is not preserved when pasting to a core trained with
  a different random seed. Different seeds → different `core_to_abi` projection spaces →
  incompatible representations. Measured: chess PPL jumps from 31 → 2619 on a different-seed
  core. This is an **open research direction** — ABI alignment is left to future work.
- **Not production-ready.** This is research code, tested single-GPU up to 350M parameters.
- **No large-scale results yet.** Competitive benchmarks (HellaSwag, MMLU) at 1B+ parameters
  have not been run.

---

## Why This Matters

Plain transformers cannot share learned specializations. A chess-tuned model and a Python-tuned
model of different sizes have incompatible internal representations — there is no well-defined
way to copy domain knowledge between them.

LayerCake fixes this by making domain computation happen in a **fixed-dimension interface**:

```
Small model (d_model=512)              Large model (d_model=768)
─────────────────────────────────────────────────────────────────
token_emb          [512]               token_emb          [768]
core_blocks      [512×512]             core_blocks      [768×768]
core_to_abi   (512 → d_abi=512)        core_to_abi   (768 → d_abi=512)
                    ↓                                       ↓
           domain_module["chess"] [512] ←── SAME WEIGHTS ──→ domain_module["chess"] [512]
                    ↓                                       ↓
abi_to_core   (512 → 512)              abi_to_core   (512 → 768)
lm_head        [512×vocab]             lm_head        [768×vocab]
```

Both models project into the same `d_abi=512` space, so `domain_module["chess"]` is literally
the same tensor — **domain module weights are bit-exactly portable across LayerCake model
sizes that share `d_abi=512`.** No affine transformation needed — direct state dict copy.

---

## Architecture

```
Input IDs  [B, T]
    ↓
Token + Positional Embeddings  [B, T, d_model]
    ↓
Core Transformer Blocks × N  [B, T, d_model]   ← trained on general corpus
    ↓
core_to_abi  [B, T, d_abi=512]                 ← FIXED dimension (all sizes)
    ↓
LayerNorm
    ↓
Domain Modules (optional residual deltas)       ← portable across sizes
    ↓
abi_to_core  [B, T, d_model]
    ↓  + residual from core output
Final LayerNorm → LM Head  [B, T, vocab]
```

**Domain modules** are residual delta networks:

$$h_\text{out} = h_\text{abi} + \exp(\alpha) \cdot \bigl(F(h_\text{abi}) - h_\text{abi}\bigr)$$

where $\alpha$ is a learned scalar initialized to 0 (identity at init, trained to add a
domain-specific delta).

Two variants are available:

| Type | Params | Implementation |
|------|--------|----------------|
| `DomainModuleLite` | ~1.05M | SwiGLU gated MLP — recommended |
| `DomainModule` | ~6.3M | 2-layer transformer — high capacity |

---

## Supported Configurations

| Config | d_model | d_abi | Est. params | Min GPU |
|--------|---------|-------|-------------|---------|
| `configs/48M.json` | 512 | **512** | 48M | RTX 3060 8GB |
| `configs/150M.json` | 768 | **512** | 150M | RTX 3080 10GB |
| `configs/350M.json` | 1024 | **512** | 350M | A100 40GB |

All configs share `d_abi=512`. Domain modules trained on any one of these are compatible
with any other.

---

## Installation

```bash
git clone https://github.com/Yoder23/layercake.git
cd layercake
pip install -e .
```

**Requirements:** Python 3.10+, PyTorch 2.0+, NumPy, SentencePiece (for tokenizer)

---

## Reproduce the Results

### Table 1 — Fair LM Comparison (20K steps)

> Requires pre-tokenized C4 and WikiText-2 token arrays.
> See [DATA.md](DATA.md) for data preparation instructions.

```bash
python compare_vs_baseline.py \
    --train_data data/tokens/c4_train.npy \
    --eval_data  data/tokens/c4_val.npy \
    --wikitext_data data/tokens/wikitext2.npy \
    --steps 20000 \
    --seed  42 \
    --out   results/my_fair_comparison.json
```

Expected: LayerCake C4 PPL ≈ 45.01, Baseline ≈ 44.89 (see `results/fair_comparison.json`)

### Table 3 — Domain Adaptation Efficiency + Structural Portability

> Requires pre-tokenized chess and Python domain token arrays (see DATA.md).

```bash
python experiment_domain_paste.py
# Expected: chess domain PPL 45.7 → 2.50 (6.3M params, 583s)
#           python domain PPL 37.5 → 12.96 (6.3M params, 557s)
#           full fine-tune chess PPL → 2.42 (42.3M params, 648s)
#           cross-size paste checksums: bit-identical (25 tensors)
```

See `results/domain_paste_functional.json`.

### Exhibit A — Paste Proof (no data required)

```bash
python tests/test_paste_lossless.py
# Expected: MSE < 1e-20, cosine_sim = 1.000000
```

### Train a Core (48M, 20K steps)

```bash
python train_core.py \
    --config   configs/48M.json \
    --train_data data/tokens/c4_train.npy \
    --steps    20000 \
    --out_dir  runs/48M_core
```

### Train a Domain Module

```bash
python train_domain.py \
    --core_ckpt  runs/48M_core/best.pt \
    --domain_name chess \
    --train_data data/tokens/chess_train.npy \
    --eval_data  data/tokens/chess_val.npy \
    --out_dir    runs/chess_domain
```

### Paste Domain to Another Model

```bash
python paste_domain.py \
    --source runs/48M_core/best.pt \
    --target runs/150M_core/best.pt \
    --domain chess \
    --out    checkpoints/150M_with_chess.pt
```

---

## Relationship to ABI

LayerCake and the companion [ABI project](https://github.com/Yoder23/abi) solve adjacent problems:

| | LayerCake | ABI |
|---|---|---|
| **What it is** | A model architecture with a fixed ABI bottleneck | A cross-model alignment framework |
| **What it proves** | Exact structural portability when the interface is shared | Functional transfer when interface geometry differs |
| **When to use** | Designing a model family that should share domain modules by construction | Aligning independently trained or cross-architecture models post-hoc |

**LayerCake is the clean design principle.** When you build with LayerCake from the start, domain modules paste bit-exactly — by construction, not by luck.

**ABI is the alignment layer** for when the clean assumption breaks: different seeds, different architectures, post-hoc alignment of existing models.

LayerCake proves exact module portability is achievable when the model is built around a fixed ABI. ABI then generalizes the idea to mismatched geometries.

---

## Project Structure

```
layercake/
├── model.py              # LayerCakeLMFixedABI — the canonical model
├── baseline_lm.py        # BaselineTransformerLM — for fair comparison
├── data.py               # LM1DDataset — pre-tokenized .npy streaming
├── train_core.py         # Train the core transformer from scratch
├── train_domain.py       # Train domain modules on a frozen core
├── paste_domain.py       # Copy domain modules between models
├── compare_vs_baseline.py # Fair head-to-head benchmark script
├── experiment_domain_paste.py  # Domain efficiency + paste experiment
│
├── configs/              # Size configs (48M, 150M, 350M)
│   ├── 48M.json
│   ├── 150M.json
│   └── 350M.json
│
├── results/              # Locked benchmark results (do not modify)
│   ├── fair_comparison.json        # Table 1 — LM quality comparison
│   ├── paste_proof.json            # Exhibit A — bit-exact portability
│   ├── domain_paste_functional.json # Table 3 — domain efficiency experiment
│   ├── thinker_v3.json             # Thinker V3 ablation
│   └── abi_diagnosis.json          # ABI overhead diagnosis
│
└── tests/
    └── test_paste_lossless.py  # Verifies paste fidelity (MSE < 1e-20)
```

---

## Honest Known Limitations

| Limitation | Detail |
|------------|--------|
| Parameter scale | Only tested up to 350M. Competitive benchmarks need 1B+. |
| Domain modules at small scale | Active domain modules add ~3% PPL overhead on general text at 48M (see `results/abi_diagnosis.json`). |
| Functional cross-seed transfer | Structural paste is bit-exact. Functional transfer requires ABI alignment when cores are trained with different seeds. Direct paste to a different-seed core degrades performance (chess: 31→2619 PPL). Open research direction. |
| Cross-size functional transfer | 150M target was only trained 10K steps (PPL ~374) — insufficient baseline for meaningful evaluation. Structural paste confirmed. Functional evaluation requires a fully-trained large core. |
| Thinker V3 | Adds only +0.034% C4 improvement — negligible at this scale. |
| Training data | Domain results use 2M tokens, 5K steps per domain. Not a large training run. |

---

## Citation

```bibtex
@software{layercake2025,
  author  = {Yoder, Sam},
  title   = {{LayerCake}: Modular Language Models via a Fixed-Dimension ABI Bottleneck},
  year    = {2025},
  url     = {https://github.com/Yoder23/layercake},
  license = {Apache-2.0},
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
