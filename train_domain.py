#!/usr/bin/env python3
"""
train_domain_fixed_abi.py

Train domain-specific modules on a FROZEN LayerCake Fixed ABI core.

Architecture:
  - Core (English): frozen, pre-trained on general text via train_layercake_core_cleanv6.py
  - core_to_abi / abi_to_core: frozen, trained with core (the universal bridge)
  - Domain modules: TRAINABLE, specialized for task (chess, python, medical, legal, etc.)
  - Router (optional): TRAINABLE, learns to select domain modules

After training, the domain module can be copy/pasted to ANY other LayerCake model
with the same d_abi=512. This is the killer feature.

Usage:
  # Train chess domain on frozen core
  python train_domain_fixed_abi.py ^
    --core_ckpt runs/c4_48M_1B/v6_core_seed9000/core_v6.pt ^
    --domain_name chess ^
    --train_tokens data/tokens/chess_domain_train.npy ^
    --eval_tokens data/tokens/chess_domain_eval.npy ^
    --out_dir runs/domains/chess ^
    --seq_len 256 --batch_size 16 --max_steps 5000 --lr 5e-4

  # Train python domain
  python train_domain_fixed_abi.py ^
    --core_ckpt runs/c4_48M_1B/v6_core_seed9000/core_v6.pt ^
    --domain_name python ^
    --train_tokens data/tokens/python_domain_train.npy ^
    --eval_tokens data/tokens/python_domain_eval.npy ^
    --out_dir runs/domains/python ^
    --seq_len 256 --batch_size 16 --max_steps 5000 --lr 5e-4

  # Paste chess domain from Model A to Model B
  python train_domain_fixed_abi.py --paste ^
    --source_ckpt runs/domains/chess/domain_chess.pt ^
    --target_ckpt runs/c4_124M_8B/v6_core_seed8000/core_v6.pt ^
    --domain_name chess ^
    --out_dir runs/domains/chess_pasted_to_124M
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import LayerCakeLMFixedABI


# ---- Dataset ----

class LMDataset(Dataset):
    """Next-token LM dataset over a 1D token stream."""

    def __init__(self, token_ids: torch.LongTensor, seq_len: int):
        super().__init__()
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.max_start = len(self.token_ids) - (self.seq_len + 1)
        if self.max_start <= 0:
            raise ValueError(
                f"Not enough tokens ({len(self.token_ids)}) for seq_len={self.seq_len}"
            )

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.seq_len]
        y = self.token_ids[idx + 1 : idx + 1 + self.seq_len]
        return x, y


# ---- Model Setup ----

def load_core_model(ckpt_path: str, device: torch.device) -> LayerCakeLMFixedABI:
    """Load a trained LayerCake Fixed ABI model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    vocab_size = ckpt.get("vocab_size", 16000)
    d_model = ckpt.get("d_model", 512)
    d_abi = ckpt.get("d_abi", 512)
    seq_len = ckpt.get("seq_len", 256)
    step = ckpt.get("step", 0)
    loss = ckpt.get("loss", 0)
    
    # Infer architecture from state_dict
    sd = ckpt["state_dict"]
    n_layers = sum(1 for k in sd if k.startswith("core_blocks.") and k.endswith(".ln1.weight"))
    n_heads_key = [k for k in sd if "attn.in_proj_weight" in k and "core_blocks.0" in k]
    if n_heads_key:
        in_proj_shape = sd[n_heads_key[0]].shape[0]
        n_heads = in_proj_shape // (3 * (d_model // (in_proj_shape // (3 * d_model)))) if d_model > 0 else 8
    else:
        n_heads = 8
    
    # Get d_ff from first core block
    ff_key = [k for k in sd if "core_blocks.0.ff.0.weight" in k]
    d_ff = sd[ff_key[0]].shape[0] if ff_key else d_model * 4
    
    # Get domain names from state dict
    domain_names = set()
    for k in sd:
        if k.startswith("domain_modules."):
            parts = k.split(".")
            if len(parts) >= 2:
                domain_names.add(parts[1])
    domain_names = sorted(domain_names) if domain_names else ["python", "chess"]
    
    print(f"[LOAD] Core checkpoint: {ckpt_path}")
    print(f"  d_model={d_model}, d_abi={d_abi}, n_layers={n_layers}, d_ff={d_ff}")
    print(f"  Step={step:,}, Loss={loss:.4f}")
    print(f"  Domain slots: {domain_names}")
    
    model = LayerCakeLMFixedABI(
        vocab_size=vocab_size,
        d_model=d_model,
        d_abi=d_abi,
        n_core_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max(seq_len, 256),
        domain_names=domain_names,
    )
    
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        print(f"  Warning: {len(result.missing_keys)} missing keys (new domain modules?)")
    if result.unexpected_keys:
        print(f"  Warning: {len(result.unexpected_keys)} unexpected keys")
    
    model.to(device)
    return model


def freeze_core_only(model: LayerCakeLMFixedABI, domain_name: str):
    """Freeze everything EXCEPT the target domain module and router."""
    # Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze target domain module
    if domain_name in model.domain_modules:
        for param in model.domain_modules[domain_name].parameters():
            param.requires_grad = True
        print(f"  [UNFREEZE] domain_modules.{domain_name}")
    else:
        raise ValueError(f"Domain '{domain_name}' not found. Available: {list(model.domain_modules.keys())}")
    
    # Unfreeze router (so it learns to route to this domain)
    if model.router is not None:
        for param in model.router.parameters():
            param.requires_grad = True
        print(f"  [UNFREEZE] router")
    
    # Also unfreeze core_to_abi and abi_to_core for fine-tuning the bridge
    # (optional — comment these out for strict frozen-core training)
    # for param in model.core_to_abi.parameters():
    #     param.requires_grad = True
    # for param in model.abi_to_core.parameters():
    #     param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


# ---- Training ----

def train_domain(
    model: LayerCakeLMFixedABI,
    domain_name: str,
    train_ds: Dataset,
    eval_ds: Dataset,
    batch_size: int,
    max_steps: int,
    lr: float,
    weight_decay: float,
    out_dir: Path,
    device: torch.device,
    eval_every: int = 500,
    checkpoint_every: int = 1000,
):
    """Train a single domain module with frozen core."""
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_eval = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.1)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_eval_loss = float("inf")
    global_step = 0
    
    # Build domain mask: activate only the target domain
    domain_idx = list(model.domain_modules.keys()).index(domain_name)
    num_domains = len(model.domain_modules)
    
    print(f"\n{'='*70}")
    print(f"  DOMAIN TRAINING: {domain_name}")
    print(f"  Domain index: {domain_idx}/{num_domains}")
    print(f"  LR={lr}, WD={weight_decay}, BS={batch_size}, Steps={max_steps}")
    print(f"{'='*70}\n")

    t_start = time.time()
    running_loss = 0.0
    running_tokens = 0

    while global_step < max_steps:
        for (x, y) in dl_train:
            if global_step >= max_steps:
                break
            
            x = x.to(device)
            y = y.to(device)
            bsz, T = x.shape

            # Build domain mask: [B, num_domains] with 1.0 for target domain
            domain_mask = torch.zeros(bsz, num_domains, device=device)
            domain_mask[:, domain_idx] = 1.0

            # Forward with domain enabled
            out = model(x, domain_mask=domain_mask)
            logits = out[0] if isinstance(out, tuple) else out

            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * bsz * T
            running_tokens += bsz * T
            global_step += 1

            # Log
            if global_step % 100 == 0:
                avg_loss = running_loss / running_tokens
                ppl = math.exp(min(avg_loss, 20))  # cap to prevent overflow
                elapsed = time.time() - t_start
                toks_per_sec = running_tokens / elapsed
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[{domain_name}] step={global_step}/{max_steps} "
                    f"loss={avg_loss:.4f} ppl={ppl:.1f} "
                    f"lr={current_lr:.2e} toks/s={toks_per_sec:.0f}"
                )
                running_loss = 0.0
                running_tokens = 0

            # Eval
            if global_step % eval_every == 0:
                eval_loss = evaluate(model, dl_eval, criterion, domain_mask, device)
                eval_ppl = math.exp(min(eval_loss, 20))
                print(f"  [EVAL] step={global_step} loss={eval_loss:.4f} ppl={eval_ppl:.1f}")
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    save_domain_checkpoint(
                        model, domain_name, global_step, eval_loss,
                        out_dir / f"domain_{domain_name}_best.pt"
                    )
                    print(f"  [BEST] New best eval loss!")

            # Checkpoint
            if global_step % checkpoint_every == 0:
                save_domain_checkpoint(
                    model, domain_name, global_step, running_loss / max(running_tokens, 1),
                    out_dir / f"domain_{domain_name}.pt"
                )

    # Final save
    save_domain_checkpoint(
        model, domain_name, global_step, best_eval_loss,
        out_dir / f"domain_{domain_name}.pt"
    )
    
    elapsed = time.time() - t_start
    print(f"\n[DONE] Domain '{domain_name}' training complete!")
    print(f"  Steps: {global_step:,}")
    print(f"  Best eval loss: {best_eval_loss:.4f} (PPL {math.exp(min(best_eval_loss, 20)):.1f})")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Saved to: {out_dir}")


def evaluate(model, dl_eval, criterion, domain_mask_template, device):
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dl_eval:
            x = x.to(device)
            y = y.to(device)
            bsz, T = x.shape
            
            # Resize domain mask for this batch
            domain_mask = domain_mask_template[:1].expand(bsz, -1).contiguous()
            
            out = model(x, domain_mask=domain_mask)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            
            total_loss += loss.item() * bsz * T
            total_tokens += bsz * T
    
    model.train()
    return total_loss / total_tokens


def save_domain_checkpoint(model, domain_name, step, loss, path):
    """Save domain module checkpoint (just the domain weights, not the full model)."""
    # Extract only the domain module state dict
    domain_state = {}
    for k, v in model.state_dict().items():
        if f"domain_modules.{domain_name}" in k:
            domain_state[k] = v
    
    # Also save router if it exists
    router_state = {}
    for k, v in model.state_dict().items():
        if k.startswith("router."):
            router_state[k] = v
    
    payload = {
        "domain_name": domain_name,
        "domain_state": domain_state,
        "router_state": router_state,
        "d_abi": model.d_abi,
        "step": step,
        "loss": loss,
        "source_d_model": model.d_model,
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"  [SAVE] Domain checkpoint: {path} ({len(domain_state)} keys)")


# ---- Paste ----

def paste_domain(source_ckpt: str, target_ckpt: str, domain_name: str, out_dir: str, device: torch.device):
    """Paste a trained domain module from one model to another."""
    print(f"\n{'='*70}")
    print(f"  DOMAIN PASTE: {domain_name}")
    print(f"  Source: {source_ckpt}")
    print(f"  Target: {target_ckpt}")
    print(f"{'='*70}\n")
    
    # Load source domain checkpoint
    src = torch.load(source_ckpt, map_location="cpu")
    if "domain_state" in src:
        # It's a domain-only checkpoint
        domain_state = src["domain_state"]
        src_d_abi = src.get("d_abi", 512)
        print(f"[SOURCE] Domain-only checkpoint, d_abi={src_d_abi}, {len(domain_state)} keys")
    else:
        # It's a full model checkpoint - extract domain
        domain_state = {k: v for k, v in src["state_dict"].items() if f"domain_modules.{domain_name}" in k}
        src_d_abi = src.get("d_abi", 512)
        print(f"[SOURCE] Full model checkpoint, extracted {len(domain_state)} domain keys")
    
    if not domain_state:
        raise ValueError(f"No domain module '{domain_name}' found in source checkpoint!")
    
    # Load target model
    target_model = load_core_model(target_ckpt, device)
    tgt_d_abi = target_model.d_abi
    
    print(f"[TARGET] d_abi={tgt_d_abi}")
    
    if src_d_abi != tgt_d_abi:
        raise ValueError(f"d_abi mismatch! Source={src_d_abi}, Target={tgt_d_abi}. Cannot paste.")
    
    # Paste: load domain state into target model
    target_sd = target_model.state_dict()
    pasted = 0
    for k, v in domain_state.items():
        if k in target_sd:
            if target_sd[k].shape == v.shape:
                target_sd[k] = v
                pasted += 1
            else:
                print(f"  [SKIP] Shape mismatch: {k} ({v.shape} vs {target_sd[k].shape})")
        else:
            print(f"  [SKIP] Key not in target: {k}")
    
    target_model.load_state_dict(target_sd)
    print(f"\n[PASTE] Pasted {pasted}/{len(domain_state)} keys successfully!")
    
    # Save pasted model
    out_path = Path(out_dir) / f"pasted_{domain_name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": target_model.state_dict(),
        "vocab_size": target_model.vocab_size,
        "d_model": target_model.d_model,
        "d_abi": target_model.d_abi,
        "paste_source": source_ckpt,
        "paste_target": target_ckpt,
        "domain_name": domain_name,
    }, out_path)
    print(f"[SAVE] Pasted model saved: {out_path}")
    
    return target_model


# ---- Main ----

def main():
    ap = argparse.ArgumentParser(description="Train/paste domain modules on frozen LayerCake Fixed ABI core")
    
    # Mode
    ap.add_argument("--paste", action="store_true", help="Paste mode (copy domain from source to target)")
    
    # Training args
    ap.add_argument("--core_ckpt", default="", help="Path to trained core checkpoint")
    ap.add_argument("--domain_name", required=True, help="Domain name (chess, python, etc.)")
    ap.add_argument("--train_tokens", default="", help="Domain training tokens (.npy)")
    ap.add_argument("--eval_tokens", default="", help="Domain eval tokens (.npy)")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--checkpoint_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="runs/domains/out")
    
    # Paste args
    ap.add_argument("--source_ckpt", default="", help="Source domain checkpoint for paste")
    ap.add_argument("--target_ckpt", default="", help="Target model for paste")
    
    args = ap.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.paste:
        # Paste mode
        if not args.source_ckpt or not args.target_ckpt:
            raise SystemExit("--paste requires --source_ckpt and --target_ckpt")
        paste_domain(args.source_ckpt, args.target_ckpt, args.domain_name, args.out_dir, device)
    else:
        # Training mode
        if not args.core_ckpt:
            raise SystemExit("Training mode requires --core_ckpt")
        if not args.train_tokens or not args.eval_tokens:
            raise SystemExit("Training mode requires --train_tokens and --eval_tokens")
        
        # Load model
        model = load_core_model(args.core_ckpt, device)
        
        # Freeze core, unfreeze domain
        print(f"\n[SETUP] Freezing core, enabling domain '{args.domain_name}'")
        freeze_core_only(model, args.domain_name)
        
        # Load data
        train_np = np.load(args.train_tokens)
        eval_np = np.load(args.eval_tokens)
        train_ids = torch.from_numpy(train_np.astype("int64"))
        eval_ids = torch.from_numpy(eval_np.astype("int64"))
        
        print(f"[DATA] Train: {len(train_ids):,} tokens, Eval: {len(eval_ids):,} tokens")
        
        train_ds = LMDataset(train_ids, seq_len=args.seq_len)
        eval_ds = LMDataset(eval_ids, seq_len=args.seq_len)
        
        # Train
        train_domain(
            model=model,
            domain_name=args.domain_name,
            train_ds=train_ds,
            eval_ds=eval_ds,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_dir=Path(args.out_dir),
            device=device,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
        )


if __name__ == "__main__":
    main()
