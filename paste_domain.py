#!/usr/bin/env python3
"""
paste_domain_brick.py

100% LOSSLESS DOMAIN BRICK COPY/PASTE

Copy individual domain bricks (chess, python, english, etc.) between ANY
LayerCake models, regardless of core size (Tiny, Small, Medium, Large, XLarge).

WORKS BECAUSE:
- All domain bricks are d_abi=512 (fixed, universal)
- Domain bricks are isolated in ModuleDict (no bleeding)
- Direct state_dict copy = 100% lossless

USAGE:
    # Copy chess domain from Small to Large
    paste_domain_brick(
        source_path="checkpoints/small_50k.pt",
        target_path="checkpoints/large_10k.pt", 
        domain_name="chess",
        output_path="checkpoints/large_with_chess.pt"
    )
    
    # Copy multiple domains
    paste_domains(
        source_path="...",
        target_path="...",
        domains=["chess", "python"],
        output_path="..."
    )
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load model checkpoint with metadata."""
    print(f"📂 Loading: {path}")
    ckpt = torch.load(path, map_location="cpu")
    
    # Extract metadata
    metadata = {
        "step": ckpt.get("step", "unknown"),
        "loss": ckpt.get("loss", "unknown"),
    }
    
    # Get model state
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    
    # Detect architecture
    # First try to get from checkpoint metadata
    if "d_model" in ckpt and "d_abi" in ckpt:
        d_model = ckpt["d_model"]
        d_abi = ckpt["d_abi"]
    elif "core_to_abi.weight" in state_dict:
        d_model = state_dict["core_to_abi.weight"].shape[1]
        d_abi = state_dict["core_to_abi.weight"].shape[0]
    elif "token_emb.weight" in state_dict:
        # Fallback: check embedding dimension
        d_model = state_dict["token_emb.weight"].shape[1]
        d_abi = d_model  # Assume no ABI projection
    else:
        raise ValueError("Cannot detect d_model/d_abi from checkpoint")
    
    # Detect available domains
    domains = []
    for key in state_dict.keys():
        if key.startswith("domain_modules."):
            domain = key.split(".")[1]
            if domain not in domains:
                domains.append(domain)
    
    print(f"  ✓ d_model={d_model}, d_abi={d_abi}")
    print(f"  ✓ Domains: {', '.join(domains)}")
    print(f"  ✓ Step: {metadata['step']}, Loss: {metadata['loss']}")
    
    return {
        "checkpoint": ckpt,
        "state_dict": state_dict,
        "metadata": metadata,
        "d_model": d_model,
        "d_abi": d_abi,
        "domains": domains,
    }


def extract_domain_brick(state_dict: Dict[str, torch.Tensor], domain_name: str) -> Dict[str, torch.Tensor]:
    """
    Extract a single domain brick from state_dict.
    
    Returns all parameters for domain_modules.{domain_name}.*
    """
    prefix = f"domain_modules.{domain_name}."
    brick = {}
    
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # Remove prefix to get relative key
            relative_key = key[len(prefix):]
            brick[relative_key] = value.clone()
    
    if not brick:
        raise ValueError(f"Domain '{domain_name}' not found in state_dict")
    
    return brick


def paste_domain_brick(
    source_path: str,
    target_path: str,
    domain_name: str,
    output_path: str,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Copy a single domain brick from source to target model.
    
    Args:
        source_path: Path to source checkpoint (e.g., small_50k.pt)
        target_path: Path to target checkpoint (e.g., large_10k.pt)
        domain_name: Domain to copy (e.g., "chess", "python")
        output_path: Where to save target with pasted domain
        verify: Whether to verify paste quality
    
    Returns:
        Dictionary with paste results and verification
    """
    print("\n" + "="*70)
    print(f"DOMAIN BRICK PASTE: {domain_name}")
    print("="*70)
    
    # Load models
    print("\n[1] LOADING MODELS")
    source = load_checkpoint(source_path)
    target = load_checkpoint(target_path)
    
    # Verify domain exists in source
    if domain_name not in source["domains"]:
        raise ValueError(f"Domain '{domain_name}' not in source. Available: {source['domains']}")
    
    # Verify d_abi compatibility
    print(f"\n[2] CHECKING COMPATIBILITY")
    if source["d_abi"] != target["d_abi"]:
        raise ValueError(
            f"ABI mismatch! Source d_abi={source['d_abi']}, "
            f"target d_abi={target['d_abi']}. Both must be 512 for lossless paste."
        )
    print(f"  ✓ Both models use d_abi={source['d_abi']} (compatible!)")
    
    # Extract domain brick from source
    print(f"\n[3] EXTRACTING DOMAIN BRICK: {domain_name}")
    brick = extract_domain_brick(source["state_dict"], domain_name)
    
    brick_params = sum(p.numel() for p in brick.values())
    brick_size_mb = brick_params * 4 / (1024**2)  # float32
    print(f"  ✓ Extracted {len(brick)} parameter tensors")
    print(f"  ✓ Total parameters: {brick_params:,}")
    print(f"  ✓ Size: {brick_size_mb:.2f} MB")
    
    # Paste brick into target
    print(f"\n[4] PASTING DOMAIN BRICK")
    prefix = f"domain_modules.{domain_name}."
    pasted_keys = []
    
    for relative_key, value in brick.items():
        full_key = prefix + relative_key
        target["state_dict"][full_key] = value
        pasted_keys.append(full_key)
    
    print(f"  ✓ Pasted {len(pasted_keys)} parameters")
    
    # Verify paste quality
    verification = None
    if verify:
        print(f"\n[5] VERIFYING PASTE QUALITY")
        verification = verify_domain_paste(
            source["state_dict"],
            target["state_dict"],
            domain_name
        )
        
        if verification["perfect_match"]:
            print(f"  ✅ PERFECT MATCH (100% lossless)")
        else:
            print(f"  ⚠️  Differences detected:")
            for key, diff in verification["differences"].items():
                print(f"      {key}: max_diff={diff:.2e}")
    
    # Save target with pasted domain
    print(f"\n[6] SAVING OUTPUT")
    output_ckpt = target["checkpoint"].copy()
    if "model" in output_ckpt:
        output_ckpt["model"] = target["state_dict"]
    elif "model_state_dict" in output_ckpt:
        output_ckpt["model_state_dict"] = target["state_dict"]
    else:
        output_ckpt = target["state_dict"]
    
    # Add paste metadata
    if isinstance(output_ckpt, dict):
        output_ckpt["paste_metadata"] = {
            "source_path": source_path,
            "domain_pasted": domain_name,
            "source_step": source["metadata"]["step"],
            "target_step": target["metadata"]["step"],
            "d_abi": source["d_abi"],
        }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_ckpt, output_path)
    print(f"  ✓ Saved: {output_path}")
    
    # Results
    results = {
        "success": True,
        "source_path": source_path,
        "target_path": target_path,
        "output_path": output_path,
        "domain_name": domain_name,
        "brick_params": brick_params,
        "brick_size_mb": brick_size_mb,
        "source_d_model": source["d_model"],
        "target_d_model": target["d_model"],
        "d_abi": source["d_abi"],
        "verification": verification,
    }
    
    print("\n" + "="*70)
    print("✅ DOMAIN PASTE COMPLETE")
    print("="*70)
    
    return results


def paste_domains(
    source_path: str,
    target_path: str,
    domains: List[str],
    output_path: str,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Copy multiple domain bricks from source to target.
    
    Args:
        source_path: Path to source checkpoint
        target_path: Path to target checkpoint  
        domains: List of domains to copy (e.g., ["chess", "python"])
        output_path: Where to save result
        verify: Whether to verify each paste
    
    Returns:
        Dictionary with results for each domain
    """
    print("\n" + "="*70)
    print(f"MULTI-DOMAIN PASTE: {', '.join(domains)}")
    print("="*70)
    
    # Load once
    source = load_checkpoint(source_path)
    target = load_checkpoint(target_path)
    
    # Verify compatibility
    if source["d_abi"] != target["d_abi"]:
        raise ValueError(
            f"ABI mismatch! Source d_abi={source['d_abi']}, "
            f"target d_abi={target['d_abi']}"
        )
    
    # Paste each domain
    results = {}
    for domain_name in domains:
        print(f"\n{'='*70}")
        print(f"Pasting: {domain_name}")
        print(f"{'='*70}")
        
        if domain_name not in source["domains"]:
            print(f"  ⚠️  Domain '{domain_name}' not in source, skipping")
            results[domain_name] = {"success": False, "error": "not_in_source"}
            continue
        
        # Extract and paste
        brick = extract_domain_brick(source["state_dict"], domain_name)
        prefix = f"domain_modules.{domain_name}."
        
        for relative_key, value in brick.items():
            full_key = prefix + relative_key
            target["state_dict"][full_key] = value
        
        # Verify
        verification = None
        if verify:
            verification = verify_domain_paste(
                source["state_dict"],
                target["state_dict"],
                domain_name
            )
        
        results[domain_name] = {
            "success": True,
            "params": sum(p.numel() for p in brick.values()),
            "verification": verification,
        }
        
        if verification and verification["perfect_match"]:
            print(f"  ✅ {domain_name}: PERFECT")
        else:
            print(f"  ✓ {domain_name}: pasted")
    
    # Save
    print(f"\n[SAVING]")
    output_ckpt = target["checkpoint"].copy()
    if "model" in output_ckpt:
        output_ckpt["model"] = target["state_dict"]
    elif "model_state_dict" in output_ckpt:
        output_ckpt["model_state_dict"] = target["state_dict"]
    else:
        output_ckpt = target["state_dict"]
    
    if isinstance(output_ckpt, dict):
        output_ckpt["paste_metadata"] = {
            "source_path": source_path,
            "domains_pasted": domains,
            "source_step": source["metadata"]["step"],
            "target_step": target["metadata"]["step"],
        }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_ckpt, output_path)
    print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "="*70)
    print(f"✅ PASTED {len(domains)} DOMAINS")
    print("="*70)
    
    return results


def verify_domain_paste(
    source_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    domain_name: str,
) -> Dict[str, Any]:
    """
    Verify that domain brick was copied perfectly (100% lossless).
    """
    prefix = f"domain_modules.{domain_name}."
    differences = {}
    
    for key in source_state.keys():
        if key.startswith(prefix):
            if key not in target_state:
                differences[key] = float('inf')
                continue
            
            source_param = source_state[key]
            target_param = target_state[key]
            
            # Check exact match
            max_diff = (source_param - target_param).abs().max().item()
            
            if max_diff > 1e-9:  # Tolerance for float precision
                differences[key] = max_diff
    
    perfect_match = len(differences) == 0
    
    return {
        "perfect_match": perfect_match,
        "differences": differences,
        "num_params_checked": sum(1 for k in source_state.keys() if k.startswith(prefix)),
    }


def list_domains(checkpoint_path: str) -> List[str]:
    """List all domains available in a checkpoint."""
    info = load_checkpoint(checkpoint_path)
    return info["domains"]


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paste domain bricks between LayerCake models")
    parser.add_argument("--source", required=True, help="Source checkpoint path")
    parser.add_argument("--target", required=True, help="Target checkpoint path")
    parser.add_argument("--domain", required=True, help="Domain to copy (e.g., 'chess')")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    
    args = parser.parse_args()
    
    result = paste_domain_brick(
        source_path=args.source,
        target_path=args.target,
        domain_name=args.domain,
        output_path=args.output,
        verify=not args.no_verify,
    )
    
    print("\n" + "="*70)
    print("PASTE SUMMARY")
    print("="*70)
    print(f"Domain: {result['domain_name']}")
    print(f"Brick size: {result['brick_size_mb']:.2f} MB ({result['brick_params']:,} params)")
    print(f"Source d_model: {result['source_d_model']}")
    print(f"Target d_model: {result['target_d_model']}")
    print(f"ABI dimension: {result['d_abi']}")
    
    if result["verification"]:
        if result["verification"]["perfect_match"]:
            print(f"Verification: ✅ PERFECT MATCH (100% lossless)")
        else:
            print(f"Verification: ⚠️  {len(result['verification']['differences'])} differences")


if __name__ == "__main__":
    main()
