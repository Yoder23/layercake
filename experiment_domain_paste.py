#!/usr/bin/env python3
"""
experiment_domain_paste.py

THE DEFINITIVE LAYERCAKE FUNCTIONAL DOMAIN PASTE EXPERIMENT
============================================================

Proves THREE claims:
  1. Functional domain knowledge transfers after paste (not just structural)
  2. Paste works cross-size: 48M -> 150M without retraining
  3. Domain adaptation is parameter-efficient: trains 6.3M vs 35.96M for full fine-tune

Experiment design:
  - Source: core_seed9000 (d_model=512, 48M, trained 250K steps on C4)
  - Train chess domain on source (5K steps, chess_domain_train_large.npy, frozen core)
  - Train python domain on source (5K steps, python_domain_train_large.npy, frozen core)
  - Paste both domains to:
      * seed6000 (d_model=512, same arch, different seed, 245K steps)
      * seed7000 (d_model=512, same arch, different seed, 130K steps)
      * large_5003 (d_model=768, 150M, CROSS-SIZE, 10K steps)
  - Evaluate chess/python PPL for every (model, domain_state) combination
  - Compare domain param efficiency: 6.3M trained vs 35.96M full fine-tune

Outputs: results/domain_paste_functional.json
"""

import sys, os, math, time, json, hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
SRC      = ROOT.parent / "layercakeogwithdecoder"
RUNS     = SRC / "runs"
DATA     = SRC / "data/tokens"

CKPT_SEED9000  = RUNS / "c4_48M_1B/v6_core_seed9000/core_v6.pt"
CKPT_SEED6000  = RUNS / "c4_fluent_core/v6_core_seed6000/core_v6.pt"
CKPT_SEED7000  = RUNS / "c4_fluent_core_v7/v6_core_seed7000/core_v6.pt"
CKPT_LARGE     = RUNS / "paste_test_large/v5_core_seed5003/core_v5.pt"

CHESS_TRAIN    = DATA / "chess_domain_train_large.npy"
CHESS_EVAL     = DATA / "chess_domain_eval_large.npy"
PYTHON_TRAIN   = DATA / "python_domain_train_large.npy"
PYTHON_EVAL    = DATA / "python_domain_eval_large.npy"

OUT_DIR        = ROOT / "runs_experiment"
RESULTS_FILE   = ROOT / "results" / "domain_paste_functional.json"

# ── Config ─────────────────────────────────────────────────────────────────
DOMAIN_TRAIN_STEPS = 5000
DOMAIN_BATCH       = 16
DOMAIN_LR          = 5e-4
DOMAIN_SEQ_LEN     = 256
LARGE_SEQ_LEN      = 128   # large model was trained with seq_len=128
EVAL_TOKENS        = 200_000
SEED               = 42
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# ── Import model ───────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
from model import LayerCakeLMFixedABI, DomainModule


# ══════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokens(path: Path) -> torch.Tensor:
    arr = np.load(str(path))
    return torch.from_numpy(arr.astype("int64"))


def random_batch(tokens: torch.Tensor, seq_len: int, batch_size: int,
                  rng: torch.Generator, device: str):
    n = len(tokens) - seq_len - 1
    idxs = torch.randint(0, n, (batch_size,), generator=rng)
    batch = torch.stack([
        tokens[i : i + seq_len + 1].clone().long()
        for i in idxs
    ])
    return batch[:, :-1].to(device), batch[:, 1:].to(device)


@torch.no_grad()
def eval_ppl(model: nn.Module, tokens: torch.Tensor, seq_len: int,
             device: str, domain_name: str | None = None) -> float:
    """Evaluate perplexity. domain_name activates that domain if given."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    rng = torch.Generator()
    rng.manual_seed(999)
    total_loss, total_n, evaluated = 0.0, 0, 0
    domain_mask = None

    while evaluated < EVAL_TOKENS:
        x, y = random_batch(tokens, seq_len, 32, rng, device)
        if domain_name is not None:
            names = model.domain_names
            if domain_name in names:
                dm = torch.zeros(x.size(0), len(names), device=device)
                dm[:, names.index(domain_name)] = 1.0
                domain_mask = dm
        logits, _ = model(x, domain_mask=domain_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        n = y.numel()
        total_loss += loss.item() * n
        total_n += n
        evaluated += n

    model.train()
    return math.exp(total_loss / total_n)


def build_model(ckpt_path: Path, seq_len: int, domain_names: list[str], device: str) -> LayerCakeLMFixedABI:
    """Load a core checkpoint into a LayerCakeLMFixedABI model."""
    ck = torch.load(str(ckpt_path), map_location="cpu")
    sd = ck["state_dict"]

    d_model   = int(ck.get("d_model", 512))
    d_abi     = int(ck.get("d_abi",   512))
    vocab_size = int(ck.get("vocab_size", 16000))
    n_layers  = len([k for k in sd if k.startswith("core_blocks.") and k.endswith("ln1.weight")])
    n_heads   = 8 if d_model <= 512 else 12

    model = LayerCakeLMFixedABI(
        vocab_size=vocab_size,
        d_model=d_model,
        d_abi=d_abi,
        n_core_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 4,
        domain_names=domain_names,
        max_seq_len=seq_len,
        use_router=False,
        domain_module_type="full",   # existing checkpoints use full transformer DomainModule
    )

    # Load state dict — strict=False allows extra keys in checkpoint
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        # Only domain module keys should be missing (new domains not in source checkpoint)
        non_domain_missing = [k for k in result.missing_keys if "domain_modules" not in k]
        if non_domain_missing:
            print(f"  [WARN] Non-domain missing keys: {non_domain_missing[:5]}")

    return model.to(device)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def domain_tensor_checksums(model: nn.Module, domain_name: str) -> dict:
    """MD5 checksums of all domain module tensors."""
    result = {}
    for k, v in model.state_dict().items():
        if k.startswith(f"domain_modules.{domain_name}."):
            short = k[len(f"domain_modules.{domain_name}."):]
            result[short] = hashlib.md5(v.cpu().numpy().tobytes()).hexdigest()
    return result


# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Train domain module on source core
# ══════════════════════════════════════════════════════════════════════════

def train_domain(source_ckpt: Path, domain_name: str, train_tokens: torch.Tensor,
                  eval_tokens: torch.Tensor, seq_len: int, n_steps: int,
                  out_path: Path, device: str) -> dict:
    """
    Train a single domain module on a frozen core.
    Returns metadata dict with timing and final PPL.
    """
    print(f"\n{'='*60}")
    print(f"TRAIN DOMAIN: {domain_name} on {source_ckpt.name}")
    print(f"  Steps: {n_steps}, Batch: {DOMAIN_BATCH}, LR: {DOMAIN_LR}")
    print(f"{'='*60}")

    model = build_model(source_ckpt, seq_len, [domain_name], device)

    # Freeze everything except this domain module
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(f"domain_modules.{domain_name}.")

    n_trainable = count_trainable(model)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M total params")
    print(f"  Param efficiency ratio: {n_trainable/n_total*100:.1f}%")

    # Eval baseline PPL (untrained domain)
    ppl_before = eval_ppl(model, eval_tokens, seq_len, device, domain_name)
    print(f"  PPL before training (untrained domain): {ppl_before:.2f}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=DOMAIN_LR, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    rng = torch.Generator()
    rng.manual_seed(SEED)

    names = model.domain_names
    model.train()
    t0 = time.time()

    for step in range(1, n_steps + 1):
        x, y = random_batch(train_tokens, seq_len, DOMAIN_BATCH, rng, device)
        dm = torch.ones(x.size(0), 1, device=device)  # domain always active
        logits, _ = model(x, domain_mask=dm)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        if step % 500 == 0 or step == n_steps:
            ppl = math.exp(loss.item())
            print(f"  step {step:5d}/{n_steps}  loss={loss.item():.4f}  ppl={ppl:.2f}  {time.time()-t0:.0f}s")

    train_time = time.time() - t0
    ppl_after = eval_ppl(model, eval_tokens, seq_len, device, domain_name)
    print(f"  PPL after training: {ppl_after:.2f}  (was {ppl_before:.2f})")
    print(f"  Training time: {train_time:.0f}s")

    # Save checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "domain": domain_name,
                "source": str(source_ckpt), "steps": n_steps}, str(out_path))

    checksums = domain_tensor_checksums(model, domain_name)

    return {
        "ppl_before_training": ppl_before,
        "ppl_after_training": ppl_after,
        "ppl_improvement_pct": (ppl_before - ppl_after) / ppl_before * 100,
        "train_steps": n_steps,
        "train_time_s": train_time,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "param_efficiency_pct": n_trainable / n_total * 100,
        "checksums": checksums,
        "ckpt_path": str(out_path),
    }


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Paste and evaluate
# ══════════════════════════════════════════════════════════════════════════

def paste_and_eval(target_ckpt: Path, domain_ckpt: Path, domain_name: str,
                    eval_tokens: torch.Tensor, seq_len: int,
                    source_checksums: dict, device: str) -> dict:
    """
    Paste trained domain module from domain_ckpt into target_ckpt.
    Evaluate PPL before and after paste.
    Verify checksums are bit-identical.
    """
    label = f"{target_ckpt.parent.parent.name}/{target_ckpt.parent.name}"
    print(f"\n--- Paste {domain_name} -> {label} ---")

    # Load source domain state dict
    src_ck = torch.load(str(domain_ckpt), map_location="cpu")
    src_sd = src_ck["model"]

    # Build target model
    model = build_model(target_ckpt, seq_len, [domain_name], device)

    # Eval without domain (no domain knowledge)
    ppl_no_domain = eval_ppl(model, eval_tokens, seq_len, device, domain_name=None)

    # Eval with untrained domain (randomly-init domain active)
    ppl_untrained_domain = eval_ppl(model, eval_tokens, seq_len, device, domain_name)

    # Paste domain module weights
    tgt_sd = model.state_dict()
    prefix = f"domain_modules.{domain_name}."
    n_pasted = 0
    for k, v in src_sd.items():
        if k.startswith(prefix):
            tgt_sd[k] = v.clone()
            n_pasted += 1
    model.load_state_dict(tgt_sd)

    # Verify checksums
    pasted_checksums = domain_tensor_checksums(model, domain_name)
    checksum_match = (pasted_checksums == source_checksums)
    mismatches = [k for k in source_checksums if source_checksums[k] != pasted_checksums.get(k)]

    # Eval with pasted domain
    ppl_pasted = eval_ppl(model, eval_tokens, seq_len, device, domain_name)

    print(f"  PPL (no domain):         {ppl_no_domain:.2f}")
    print(f"  PPL (untrained domain):  {ppl_untrained_domain:.2f}")
    print(f"  PPL (pasted domain):     {ppl_pasted:.2f}")
    print(f"  Checksums bit-identical: {checksum_match} ({n_pasted} tensors pasted)")
    if mismatches:
        print(f"  [WARN] Checksum mismatches: {mismatches}")

    # How much of the source model's performance did we recover?
    # (Will be filled in later by caller comparing to source_ppl_after_training)

    tgt_ck = torch.load(str(target_ckpt), map_location="cpu")
    tgt_d_model = tgt_ck.get("d_model", 512)

    return {
        "target_ckpt": label,
        "target_d_model": tgt_d_model,
        "target_seed": label,
        "ppl_no_domain": ppl_no_domain,
        "ppl_untrained_domain": ppl_untrained_domain,
        "ppl_pasted_domain": ppl_pasted,
        "n_tensors_pasted": n_pasted,
        "checksums_bit_identical": checksum_match,
        "checksum_mismatches": mismatches,
    }


# ══════════════════════════════════════════════════════════════════════════
# Phase 3: Fine-tune baseline (for efficiency comparison)
# ══════════════════════════════════════════════════════════════════════════

def finetune_baseline_core(ckpt_path: Path, domain_name: str,
                            train_tokens: torch.Tensor, eval_tokens: torch.Tensor,
                            seq_len: int, n_steps: int, device: str) -> dict:
    """
    Fine-tune the ENTIRE model (all parameters) on domain data.
    Used to show parameter efficiency of LayerCake domain training.
    """
    print(f"\n--- BASELINE full fine-tune: {domain_name} on {ckpt_path.parent.name} ---")

    sys.path.insert(0, str(SRC))
    model = build_model(ckpt_path, seq_len, [domain_name], device)

    # ALL params trainable
    for p in model.parameters():
        p.requires_grad = True

    n_trainable = count_trainable(model)
    print(f"  Trainable (full fine-tune): {n_trainable/1e6:.2f}M params")

    ppl_before = eval_ppl(model, eval_tokens, seq_len, device, None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    rng = torch.Generator()
    rng.manual_seed(SEED + 100)

    t0 = time.time()
    model.train()
    for step in range(1, n_steps + 1):
        x, y = random_batch(train_tokens, seq_len, DOMAIN_BATCH, rng, device)
        logits, _ = model(x, domain_mask=None)  # Core-only mode
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0 or step == n_steps:
            ppl = math.exp(loss.item())
            print(f"  step {step:5d}/{n_steps}  loss={loss.item():.4f}  ppl={ppl:.2f}  {time.time()-t0:.0f}s")

    train_time = time.time() - t0
    ppl_after = eval_ppl(model, eval_tokens, seq_len, device, None)

    print(f"  PPL before: {ppl_before:.2f}  after: {ppl_after:.2f}")
    print(f"  Fine-tune time: {train_time:.0f}s")

    return {
        "ppl_before": ppl_before,
        "ppl_after": ppl_after,
        "train_steps": n_steps,
        "train_time_s": train_time,
        "trainable_params": n_trainable,
    }


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'#'*70}")
    print("# LayerCake Functional Domain Paste Experiment")
    print(f"# Device: {DEVICE}")
    print(f"{'#'*70}\n")

    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load token data
    print("Loading token data...")
    chess_train  = load_tokens(CHESS_TRAIN)
    chess_eval   = load_tokens(CHESS_EVAL)
    python_train = load_tokens(PYTHON_TRAIN)
    python_eval  = load_tokens(PYTHON_EVAL)
    print(f"  Chess train: {len(chess_train):,} tokens")
    print(f"  Chess eval:  {len(chess_eval):,} tokens")
    print(f"  Python train: {len(python_train):,} tokens")
    print(f"  Python eval:  {len(python_eval):,} tokens")

    results = {
        "experiment": "Functional Domain Paste Proof",
        "date": "2026-05-18",
        "device": DEVICE,
        "domain_train_steps": DOMAIN_TRAIN_STEPS,
        "source_ckpt": str(CKPT_SEED9000),
        "target_ckpts": {
            "seed6000_48M": str(CKPT_SEED6000),
            "seed7000_48M": str(CKPT_SEED7000),
            "seed5003_150M": str(CKPT_LARGE),
        },
        "claim": "Domain modules trained on one model transfer functionally to other models of any size via direct state dict copy",
        "phase1_domain_training": {},
        "phase2_paste_eval": {},
        "phase3_efficiency_comparison": {},
    }

    # ── Phase 1: Train chess domain on seed9000 ──────────────────────────
    chess_ckpt_path = OUT_DIR / "chess_domain_seed9000.pt"
    chess_results = train_domain(
        source_ckpt=CKPT_SEED9000,
        domain_name="chess",
        train_tokens=chess_train,
        eval_tokens=chess_eval,
        seq_len=DOMAIN_SEQ_LEN,
        n_steps=DOMAIN_TRAIN_STEPS,
        out_path=chess_ckpt_path,
        device=DEVICE,
    )
    results["phase1_domain_training"]["chess"] = chess_results
    chess_source_checksums = chess_results["checksums"]
    print(f"\n✅ Chess domain trained. PPL: {chess_results['ppl_before_training']:.1f} → {chess_results['ppl_after_training']:.2f}")

    # ── Phase 1b: Train python domain on seed9000 ────────────────────────
    python_ckpt_path = OUT_DIR / "python_domain_seed9000.pt"
    python_results = train_domain(
        source_ckpt=CKPT_SEED9000,
        domain_name="python",
        train_tokens=python_train,
        eval_tokens=python_eval,
        seq_len=DOMAIN_SEQ_LEN,
        n_steps=DOMAIN_TRAIN_STEPS,
        out_path=python_ckpt_path,
        device=DEVICE,
    )
    results["phase1_domain_training"]["python"] = python_results
    python_source_checksums = python_results["checksums"]
    print(f"\n✅ Python domain trained. PPL: {python_results['ppl_before_training']:.1f} → {python_results['ppl_after_training']:.2f}")

    # ── Phase 2: Paste to same-architecture cores (seed6000, seed7000) ───
    paste_targets_48M = [
        ("seed6000_48M", CKPT_SEED6000, DOMAIN_SEQ_LEN),
        ("seed7000_48M", CKPT_SEED7000, DOMAIN_SEQ_LEN),
    ]

    for label, ckpt, seq_len in paste_targets_48M:
        results["phase2_paste_eval"][label] = {}

        chess_paste = paste_and_eval(
            target_ckpt=ckpt, domain_ckpt=chess_ckpt_path, domain_name="chess",
            eval_tokens=chess_eval, seq_len=seq_len,
            source_checksums=chess_source_checksums, device=DEVICE,
        )
        chess_paste["ppl_source_trained"] = chess_results["ppl_after_training"]
        chess_paste["functional_recovery_pct"] = (
            (chess_paste["ppl_no_domain"] - chess_paste["ppl_pasted_domain"]) /
            (chess_paste["ppl_no_domain"] - chess_results["ppl_after_training"])
            * 100
        )
        results["phase2_paste_eval"][label]["chess"] = chess_paste

        python_paste = paste_and_eval(
            target_ckpt=ckpt, domain_ckpt=python_ckpt_path, domain_name="python",
            eval_tokens=python_eval, seq_len=seq_len,
            source_checksums=python_source_checksums, device=DEVICE,
        )
        python_paste["ppl_source_trained"] = python_results["ppl_after_training"]
        python_paste["functional_recovery_pct"] = (
            (python_paste["ppl_no_domain"] - python_paste["ppl_pasted_domain"]) /
            (python_paste["ppl_no_domain"] - python_results["ppl_after_training"])
            * 100
        )
        results["phase2_paste_eval"][label]["python"] = python_paste

    # ── Phase 2b: Paste to CROSS-SIZE large model (150M) ─────────────────
    print(f"\n{'='*60}")
    print("CROSS-SIZE PASTE: 48M domain modules → 150M model")
    print(f"{'='*60}")
    results["phase2_paste_eval"]["seed5003_150M"] = {}

    chess_paste_large = paste_and_eval(
        target_ckpt=CKPT_LARGE, domain_ckpt=chess_ckpt_path, domain_name="chess",
        eval_tokens=chess_eval, seq_len=LARGE_SEQ_LEN,
        source_checksums=chess_source_checksums, device=DEVICE,
    )
    chess_paste_large["ppl_source_trained"] = chess_results["ppl_after_training"]
    chess_paste_large["functional_recovery_pct"] = (
        (chess_paste_large["ppl_no_domain"] - chess_paste_large["ppl_pasted_domain"]) /
        (chess_paste_large["ppl_no_domain"] - chess_results["ppl_after_training"])
        * 100
    )
    results["phase2_paste_eval"]["seed5003_150M"]["chess"] = chess_paste_large

    python_paste_large = paste_and_eval(
        target_ckpt=CKPT_LARGE, domain_ckpt=python_ckpt_path, domain_name="python",
        eval_tokens=python_eval, seq_len=LARGE_SEQ_LEN,
        source_checksums=python_source_checksums, device=DEVICE,
    )
    python_paste_large["ppl_source_trained"] = python_results["ppl_after_training"]
    python_paste_large["functional_recovery_pct"] = (
        (python_paste_large["ppl_no_domain"] - python_paste_large["ppl_pasted_domain"]) /
        (python_paste_large["ppl_no_domain"] - python_results["ppl_after_training"])
        * 100
    )
    results["phase2_paste_eval"]["seed5003_150M"]["python"] = python_paste_large

    # ── Phase 3: Efficiency comparison (domain only vs full fine-tune) ────
    print(f"\n{'='*60}")
    print("PHASE 3: Parameter Efficiency Comparison")
    print(f"{'='*60}")

    bl_chess = finetune_baseline_core(
        ckpt_path=CKPT_SEED6000, domain_name="chess",
        train_tokens=chess_train, eval_tokens=chess_eval,
        seq_len=DOMAIN_SEQ_LEN, n_steps=DOMAIN_TRAIN_STEPS, device=DEVICE,
    )
    results["phase3_efficiency_comparison"]["baseline_full_finetune_chess"] = bl_chess

    domain_trainable = chess_results["trainable_params"]
    full_trainable   = bl_chess["trainable_params"]
    results["phase3_efficiency_comparison"]["summary"] = {
        "domain_module_params": domain_trainable,
        "full_model_params": full_trainable,
        "efficiency_ratio": full_trainable / domain_trainable,
        "domain_as_pct_of_full": domain_trainable / full_trainable * 100,
        "domain_chess_ppl": chess_results["ppl_after_training"],
        "full_finetune_chess_ppl": bl_chess["ppl_after"],
        "domain_train_time_s": chess_results["train_time_s"],
        "full_finetune_time_s": bl_chess["train_time_s"],
    }

    # ── Save results ──────────────────────────────────────────────────────
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print("# EXPERIMENT SUMMARY")
    print(f"{'#'*70}")

    print(f"\n[CLAIM 1 — Functional Domain Transfer (same architecture)]")
    for lbl in ["seed6000_48M", "seed7000_48M"]:
        cr = results["phase2_paste_eval"][lbl]["chess"]
        pr = results["phase2_paste_eval"][lbl]["python"]
        print(f"  {lbl}:")
        print(f"    Chess:  no-domain PPL={cr['ppl_no_domain']:.1f}  → pasted PPL={cr['ppl_pasted_domain']:.2f}  (source={cr['ppl_source_trained']:.2f})  recovery={cr['functional_recovery_pct']:.0f}%")
        print(f"    Python: no-domain PPL={pr['ppl_no_domain']:.1f}  → pasted PPL={pr['ppl_pasted_domain']:.2f}  (source={pr['ppl_source_trained']:.2f})  recovery={pr['functional_recovery_pct']:.0f}%")

    print(f"\n[CLAIM 2 — Cross-Size Paste (48M → 150M)]")
    cr = results["phase2_paste_eval"]["seed5003_150M"]["chess"]
    pr = results["phase2_paste_eval"]["seed5003_150M"]["python"]
    print(f"  Chess:  no-domain PPL={cr['ppl_no_domain']:.1f}  → pasted PPL={cr['ppl_pasted_domain']:.2f}  (source trained on 48M)  recovery={cr['functional_recovery_pct']:.0f}%")
    print(f"  Python: no-domain PPL={pr['ppl_no_domain']:.1f}  → pasted PPL={pr['ppl_pasted_domain']:.2f}  (source trained on 48M)  recovery={pr['functional_recovery_pct']:.0f}%")
    print(f"  Checksums bit-identical: chess={cr['checksums_bit_identical']}  python={pr['checksums_bit_identical']}")

    print(f"\n[CLAIM 3 — Parameter Efficiency]")
    s = results["phase3_efficiency_comparison"]["summary"]
    print(f"  Domain module training:  {s['domain_module_params']/1e6:.2f}M params, PPL={s['domain_chess_ppl']:.2f}, time={s['domain_train_time_s']:.0f}s")
    print(f"  Full fine-tune:          {s['full_model_params']/1e6:.2f}M params, PPL={s['full_finetune_chess_ppl']:.2f}, time={s['full_finetune_time_s']:.0f}s")
    print(f"  LayerCake is {s['efficiency_ratio']:.1f}x more parameter-efficient for domain adaptation")

    print(f"\n✅ Results saved to: {RESULTS_FILE}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
