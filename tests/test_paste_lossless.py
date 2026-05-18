#!/usr/bin/env python3
"""
test_paste_lossless.py — Verify that domain module paste is lossless.

This test does NOT require checkpoints or data. It creates two LayerCake models
from the same random initialization, performs a paste, and verifies that the
pasted weights produce numerically identical outputs (within floating point precision).

Expected output:
    MSE:             < 1e-20
    Mean cosine sim: 1.000000
    Min cosine sim:  1.000000
    RESULT: PASS

Run with:
    python tests/test_paste_lossless.py
    # or:
    pytest tests/test_paste_lossless.py
"""

import sys
from pathlib import Path

# Allow running from repo root or tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import pytest

from model import LayerCakeLMFixedABI


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

VOCAB_SIZE = 1000
D_ABI = 512
SEQ_LEN = 32
BATCH_SIZE = 4
N_SAMPLES = 100
SEED = 42

# Two different model sizes — both share d_abi=512
SMALL_CFG = dict(d_model=512, n_layers=4, n_heads=8)
LARGE_CFG = dict(d_model=768, n_layers=6, n_heads=12)

DOMAIN_NAMES = ["chess", "python"]


def build_model(d_model: int, n_layers: int, n_heads: int) -> LayerCakeLMFixedABI:
    return LayerCakeLMFixedABI(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        d_abi=D_ABI,        # Fixed across all sizes
        n_core_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4,
        domain_names=DOMAIN_NAMES,
        max_seq_len=SEQ_LEN,
        use_router=False,
        domain_module_type="lite",
    )


def paste_domain(src_model: LayerCakeLMFixedABI,
                 tgt_model: LayerCakeLMFixedABI,
                 domain_name: str) -> LayerCakeLMFixedABI:
    """Direct state dict copy of domain module weights."""
    src_state = src_model.state_dict()
    tgt_state = tgt_model.state_dict()

    prefix = f"domain_modules.{domain_name}."
    copied = 0
    for key, tensor in src_state.items():
        if key.startswith(prefix):
            tgt_state[key] = tensor.clone()
            copied += 1

    if copied == 0:
        raise ValueError(f"No keys found for domain '{domain_name}' in source model.")

    tgt_model.load_state_dict(tgt_state)
    return tgt_model


# ---------------------------------------------------------------------------
# Self-consistency test (same architecture)
# ---------------------------------------------------------------------------

def test_self_consistency_paste():
    """Paste within identical architecture. Expect MSE ≈ 0."""
    torch.manual_seed(SEED)

    model_a = build_model(**SMALL_CFG)
    model_b = build_model(**SMALL_CFG)
    torch.manual_seed(SEED)  # Same init as model_a
    model_b.load_state_dict(model_a.state_dict())  # Make identical

    # Perturb model_b's core (to make it non-trivially different)
    with torch.no_grad():
        for name, p in model_b.named_parameters():
            if not name.startswith("domain_modules."):
                p.add_(torch.randn_like(p) * 0.01)

    # Now paste the domain module from A into B
    paste_domain(model_a, model_b, "chess")

    # Extract domain module state dicts
    a_state = {k: v for k, v in model_a.state_dict().items()
               if k.startswith("domain_modules.chess.")}
    b_state = {k: v for k, v in model_b.state_dict().items()
               if k.startswith("domain_modules.chess.")}

    # Compute per-tensor MSE and cosine similarity
    mse_values = []
    cos_values = []

    for key in a_state:
        a_t = a_state[key].float()
        b_t = b_state[key].float()
        mse = ((a_t - b_t) ** 2).mean().item()
        mse_values.append(mse)
        a_flat = a_t.flatten()
        b_flat = b_t.flatten()
        if a_flat.norm() > 1e-12:
            cos = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
            cos_values.append(cos)

    avg_mse = sum(mse_values) / len(mse_values)
    min_cos = min(cos_values) if cos_values else 1.0
    mean_cos = sum(cos_values) / len(cos_values) if cos_values else 1.0

    print(f"\n[self-consistency paste]")
    print(f"  MSE (avg over tensors): {avg_mse:.6e}")
    print(f"  Mean cosine sim:        {mean_cos:.6f}")
    print(f"  Min cosine sim:         {min_cos:.6f}")

    assert avg_mse < 1e-20, f"MSE too high: {avg_mse}"
    assert min_cos > 0.9999999, f"Min cosine sim too low: {min_cos}"


# ---------------------------------------------------------------------------
# Cross-size paste test (different d_model)
# ---------------------------------------------------------------------------

def test_cross_size_paste_weight_identity():
    """
    Paste domain module from small (d_model=512) to large (d_model=768).
    Domain module weights must be IDENTICAL (bit-exact) because they only
    depend on d_abi=512, not on d_model.
    """
    torch.manual_seed(SEED)
    small_model = build_model(**SMALL_CFG)

    torch.manual_seed(SEED + 1)
    large_model = build_model(**LARGE_CFG)

    # Paste chess domain from small to large
    paste_domain(small_model, large_model, "chess")

    # Verify weight identity
    small_state = {k: v for k, v in small_model.state_dict().items()
                   if k.startswith("domain_modules.chess.")}
    large_state = {k: v for k, v in large_model.state_dict().items()
                   if k.startswith("domain_modules.chess.")}

    assert set(small_state.keys()) == set(large_state.keys()), \
        "Domain module key sets differ!"

    for key in small_state:
        a = small_state[key]
        b = large_state[key]
        assert a.shape == b.shape, f"Shape mismatch at {key}: {a.shape} vs {b.shape}"
        max_diff = (a - b).abs().max().item()
        assert max_diff == 0.0, f"Non-zero diff at {key}: max_diff={max_diff}"

    n_keys = len(small_state)
    print(f"\n[cross-size paste]  {n_keys} tensors — all bit-identical ✓")
    print(f"  Small d_model={SMALL_CFG['d_model']}, Large d_model={LARGE_CFG['d_model']}")
    print(f"  d_abi={D_ABI} (fixed in both)")


# ---------------------------------------------------------------------------
# Router paste test
# ---------------------------------------------------------------------------

def test_router_paste():
    """Router also lives in d_abi space — should paste cleanly."""
    torch.manual_seed(SEED)
    src = build_model(**SMALL_CFG)

    torch.manual_seed(SEED + 10)
    tgt = build_model(**LARGE_CFG)

    src_state = src.state_dict()
    tgt_state = tgt.state_dict()

    # Paste router
    copied = 0
    for key in src_state:
        if key.startswith("router."):
            tgt_state[key] = src_state[key].clone()
            copied += 1

    if copied > 0:
        tgt.load_state_dict(tgt_state)
        src_router = {k: v for k, v in src.state_dict().items() if k.startswith("router.")}
        tgt_router = {k: v for k, v in tgt.state_dict().items() if k.startswith("router.")}
        for key in src_router:
            max_diff = (src_router[key] - tgt_router[key]).abs().max().item()
            assert max_diff == 0.0, f"Router diff at {key}: {max_diff}"
        print(f"\n[router paste]  {copied} tensors — bit-identical ✓")
    else:
        print(f"\n[router paste]  No router found (use_router=False) — skipped")


# ---------------------------------------------------------------------------
# Forward-pass losslessness (THE definitive mathematical proof)
# ---------------------------------------------------------------------------

def test_forward_pass_lossless():
    """
    THE DEFINITIVE LOSSLESSNESS PROOF.

    After pasting a domain module from model A into a fresh copy of the same
    core (model A'), the full forward pass — including domain transformation —
    produces IDENTICAL outputs.

    This proves that paste loses zero information: not just that weights are
    copied correctly, but that the pasted module computes exactly the same
    function on the same representations.

    Setup:
      1. Build model A (core + domain "chess").
      2. Clone model A's CORE into model A' (same core weights, fresh random domain).
      3. Paste domain from A into A'.
      4. Run identical inputs through both models with domain active.
      5. Assert logits are bit-identical (max absolute difference == 0.0).

    If this passes, paste is provably lossless: the receiving model with the pasted
    domain module is mathematically equivalent to the original.
    """
    torch.manual_seed(SEED)
    model_a = build_model(**SMALL_CFG)

    # Simulate "trained" by perturbing domain weights away from init
    with torch.no_grad():
        for name, p in model_a.named_parameters():
            if name.startswith("domain_modules.chess."):
                p.add_(torch.randn_like(p) * 0.5)

    # Build model A': same core weights, independent (random) domain init
    torch.manual_seed(SEED + 77)
    model_a_prime = build_model(**SMALL_CFG)

    # Copy core weights only (not domain) from A into A'
    a_state  = model_a.state_dict()
    ap_state = model_a_prime.state_dict()
    for key in a_state:
        if not key.startswith("domain_modules."):
            ap_state[key] = a_state[key].clone()
    model_a_prime.load_state_dict(ap_state)

    # Verify that domain modules differ BEFORE paste (proves A' started different)
    for key in a_state:
        if key.startswith("domain_modules.chess."):
            before_diff = (a_state[key] - model_a_prime.state_dict()[key]).abs().max().item()
            if before_diff > 0:
                break  # Confirmed: domain weights differ before paste
    else:
        raise AssertionError("Domain weights were already identical before paste — test setup error")

    # Paste domain from A into A'
    paste_domain(model_a, model_a_prime, "chess")

    # Run forward pass on identical inputs with domain active
    torch.manual_seed(999)
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    n_domains = len(DOMAIN_NAMES)
    domain_mask = torch.zeros(BATCH_SIZE, n_domains)
    chess_idx = DOMAIN_NAMES.index("chess")
    domain_mask[:, chess_idx] = 1.0

    model_a.eval()
    model_a_prime.eval()
    with torch.no_grad():
        logits_a,  _ = model_a(x, domain_mask=domain_mask)
        logits_ap, _ = model_a_prime(x, domain_mask=domain_mask)

    max_diff = (logits_a - logits_ap).abs().max().item()
    mean_diff = (logits_a - logits_ap).abs().mean().item()

    print(f"\n[forward-pass lossless paste]")
    print(f"  Core weights:   identical (A == A')")
    print(f"  Domain weights: pasted from A → A'")
    print(f"  Input shape:    {list(x.shape)}")
    print(f"  Max logit diff: {max_diff:.6e}   (must be 0.0)")
    print(f"  Mean logit diff:{mean_diff:.6e}   (must be 0.0)")
    print(f"  Result: {'PASS — outputs are bit-identical ✓' if max_diff == 0.0 else 'FAIL'}")

    assert max_diff == 0.0, (
        f"Forward-pass outputs differ after paste! max_diff={max_diff:.6e}\n"
        f"This means the paste changed the domain module's computation — it is NOT lossless."
    )


def test_cross_size_forward_pass_lossless():
    """
    Cross-size forward-pass losslessness.

    Paste chess domain from a 48M model (d_model=512) to a 150M model (d_model=768).
    With the same core-to-abi projection in the large model fixed to the small model's
    projection weights, the domain module must produce identical ABI-space outputs.

    This proves that the DOMAIN MODULE ITSELF is a lossless function regardless of
    surrounding model size — it only ever sees d_abi=512 vectors.
    """
    torch.manual_seed(SEED)
    small = build_model(**SMALL_CFG)   # d_model=512

    torch.manual_seed(SEED + 99)
    large = build_model(**LARGE_CFG)   # d_model=768

    # Simulate trained domain in small model
    with torch.no_grad():
        for name, p in small.named_parameters():
            if name.startswith("domain_modules.chess."):
                p.add_(torch.randn_like(p) * 0.5)

    # Also copy the ABI layer weights (core_to_abi, abi_to_core, abi_ln) from small → large
    # so both models produce the SAME h_abi for the same ABI-space input
    # (This is the key: same ABI projection → same h_abi → same domain output)
    small_state = small.state_dict()
    large_state = large.state_dict()

    abi_keys = [k for k in small_state if k in ('core_to_abi.weight', 'abi_to_core.weight',
                                                  'abi_ln.weight', 'abi_ln.bias')]
    for k in abi_keys:
        if k in large_state and small_state[k].shape == large_state[k].shape:
            large_state[k] = small_state[k].clone()

    large.load_state_dict(large_state)
    paste_domain(small, large, "chess")

    # Inject identical ABI-space inputs directly and check domain module outputs match
    torch.manual_seed(42)
    h_abi = torch.randn(BATCH_SIZE, SEQ_LEN, D_ABI)  # Both models see same d_abi=512 vectors

    small.eval()
    large.eval()
    with torch.no_grad():
        small_domain_out = small.domain_modules["chess"](h_abi)
        large_domain_out = large.domain_modules["chess"](h_abi)

    max_diff  = (small_domain_out - large_domain_out).abs().max().item()
    mean_diff = (small_domain_out - large_domain_out).abs().mean().item()

    print(f"\n[cross-size domain output lossless]")
    print(f"  Small d_model={SMALL_CFG['d_model']}, Large d_model={LARGE_CFG['d_model']}")
    print(f"  Domain input:   d_abi={D_ABI} (same for both)")
    print(f"  Max output diff: {max_diff:.6e}   (must be 0.0)")
    print(f"  Mean output diff:{mean_diff:.6e}   (must be 0.0)")
    print(f"  Result: {'PASS — domain outputs are bit-identical ✓' if max_diff == 0.0 else 'FAIL'}")

    assert max_diff == 0.0, (
        f"Domain module outputs differ after cross-size paste! max_diff={max_diff:.6e}\n"
        f"The domain module is NOT computing the same function — paste is lossy."
    )


# ---------------------------------------------------------------------------
# Generation identity test (the final proof: token sequences are identical)
# ---------------------------------------------------------------------------

def test_generation_identity():
    """
    GENERATION IDENTITY PROOF.

    Proves that autoregressive text generation is token-for-token identical after paste.
    This is the strongest possible form of losslessness: not just weights, not just
    a single forward pass, but an entire generation sequence (including feeding each
    model's own outputs back in as inputs) produces identical tokens at every step.

    Why this must be true: identical logits → identical argmax at every step →
    identical next token → identical subsequent context → identical next logits → ...

    Setup:
      1. Build model A (core + "trained" chess domain).
      2. Build model A': same CORE weights, different DOMAIN init.
      3. Paste domain from A into A'.
      4. Run autoregressive greedy decoding from the same prompt on both.
      5. Assert: every generated token is identical. No divergence at any step.

    Scope: applies when the receiving model has the same core weights.
    Cross-size and cross-seed are addressed separately (test 5 and CLAIMS.md).
    """
    GEN_STEPS = 50   # tokens to generate
    PROMPT_LEN = 8   # tokens in the prompt

    torch.manual_seed(SEED)
    model_a = build_model(**SMALL_CFG)

    # Simulate "trained" domain
    with torch.no_grad():
        for name, p in model_a.named_parameters():
            if name.startswith("domain_modules.chess."):
                p.add_(torch.randn_like(p) * 0.5)

    # Build A': same core, different domain init
    torch.manual_seed(SEED + 111)
    model_a_prime = build_model(**SMALL_CFG)

    a_state = model_a.state_dict()
    ap_state = model_a_prime.state_dict()
    for key in a_state:
        if not key.startswith("domain_modules."):
            ap_state[key] = a_state[key].clone()
    model_a_prime.load_state_dict(ap_state)

    # Paste domain
    paste_domain(model_a, model_a_prime, "chess")

    # Greedy autoregressive generation
    torch.manual_seed(999)
    prompt = torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN))  # [1, PROMPT_LEN]

    n_domains = len(DOMAIN_NAMES)
    chess_idx = DOMAIN_NAMES.index("chess")

    model_a.eval()
    model_a_prime.eval()

    tokens_a  = prompt.clone()
    tokens_ap = prompt.clone()

    with torch.no_grad():
        for step in range(GEN_STEPS):
            # Both models receive same context at every step (enforced — divergence
            # is impossible because next token is deterministic and identical, but
            # we feed from each model's own sequence to prove they don't diverge)
            mask_a  = torch.zeros(1, n_domains)
            mask_a[:, chess_idx] = 1.0
            mask_ap = mask_a.clone()

            logits_a,  _ = model_a(tokens_a,  domain_mask=mask_a)
            logits_ap, _ = model_a_prime(tokens_ap, domain_mask=mask_ap)

            # Greedy: take the last token's argmax
            next_a  = logits_a[:, -1, :].argmax(dim=-1, keepdim=True)   # [1, 1]
            next_ap = logits_ap[:, -1, :].argmax(dim=-1, keepdim=True)

            assert next_a.item() == next_ap.item(), (
                f"Generation diverged at step {step+1}! "
                f"Model A predicted token {next_a.item()}, "
                f"Model A' predicted token {next_ap.item()}"
            )

            tokens_a  = torch.cat([tokens_a,  next_a],  dim=1)
            tokens_ap = torch.cat([tokens_ap, next_ap], dim=1)

    generated_a  = tokens_a[0, PROMPT_LEN:].tolist()
    generated_ap = tokens_ap[0, PROMPT_LEN:].tolist()

    assert generated_a == generated_ap, (
        f"Generated sequences differ!\nA:  {generated_a}\nA': {generated_ap}"
    )

    print(f"\n[generation identity]")
    print(f"  Prompt length:    {PROMPT_LEN} tokens")
    print(f"  Steps generated:  {GEN_STEPS} tokens")
    print(f"  Generated (A):  {generated_a[:10]}...")
    print(f"  Generated (A'): {generated_ap[:10]}...")
    print(f"  Sequences match:  {generated_a == generated_ap}")
    print(f"  Result: PASS — generation is token-for-token identical ✓")


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LayerCake — Paste Losslessness Tests")
    print("=" * 60)

    test_self_consistency_paste()
    test_cross_size_paste_weight_identity()
    test_router_paste()
    test_forward_pass_lossless()
    test_cross_size_forward_pass_lossless()
    test_generation_identity()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("RESULT: ALL TESTS PASSED")
    print("=" * 60)
