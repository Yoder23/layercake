from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_byte_core_from_config import _build_model


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _exact_checkpoint_transfer(
    checkpoint_path: Path,
    *,
    device: str,
    seed: int,
    eval_rows: int,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = checkpoint["model_config"]
    torch_device = torch.device(device)
    source = _build_model(model_cfg, torch_device).eval()
    receiver = _build_model(model_cfg, torch_device).eval()
    source.load_state_dict(checkpoint["model"], strict=True)
    receiver.load_state_dict(source.state_dict(), strict=True)

    patch_size = int(model_cfg.get("patch_size", 2))
    max_len = int(model_cfg.get("max_patches", 128)) * patch_size
    seq = min(max_len, 512)
    seq -= seq % patch_size
    if seq <= 0:
        raise ValueError("checkpoint config produced an invalid eval sequence length")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, 256, (eval_rows, seq), generator=generator, device=torch_device)
    y = torch.roll(x, shifts=-1, dims=1)

    with torch.inference_mode():
        source_logits, source_abi = source(x)
        receiver_logits, receiver_abi = receiver(x)
        source_loss = F.cross_entropy(source_logits.reshape(-1, 256), y.reshape(-1))
        receiver_loss = F.cross_entropy(receiver_logits.reshape(-1, 256), y.reshape(-1))
        source_generated = source.generate_next_patch(x)
        receiver_generated = receiver.generate_next_patch(x)

    max_logit_diff = float((source_logits - receiver_logits).abs().max().item())
    max_abi_diff = float((source_abi - receiver_abi).abs().max().item())
    ppl_ratio = math.exp(float(receiver_loss.item()) - float(source_loss.item()))
    generated_equal = bool(torch.equal(source_generated, receiver_generated))
    return {
        "transfer_ppl_ratio": ppl_ratio,
        "transfer_max_logit_diff": max_logit_diff,
        "transfer_max_abi_diff": max_abi_diff,
        "transfer_generation_exact": generated_equal,
        "abi_shape": list(source_abi.shape),
        "checkpoint_model_params": int(
            sum(parameter.numel() for parameter in source.parameters() if parameter.requires_grad)
        ),
    }


def verify(
    *,
    dominance_certificate: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    seed: int,
    eval_rows: int,
    scope_label: str | None = None,
) -> dict[str, Any]:
    dominance = _load_json(dominance_certificate)
    transfer = _exact_checkpoint_transfer(
        checkpoint,
        device=device,
        seed=seed,
        eval_rows=eval_rows,
    )
    dominance_gates = dominance.get("gates", {})
    gates = {
        "source_dominance_certificate_pass": dominance.get("status") == "PASS",
        "source_all_dominance_gates_pass": bool(dominance_gates)
        and all(bool(value) for value in dominance_gates.values()),
        "receiver_inherits_bpb_win": dominance.get("gates", {}).get("bpb_non_inferior") is True,
        "receiver_inherits_training_win": dominance.get("gates", {}).get("training_speed_met") is True,
        "receiver_inherits_cpu_generation_win": dominance.get("gates", {}).get("cpu_generation_5x_met") is True,
        "receiver_inherits_gpu_generation_win": dominance.get("gates", {}).get("gpu_generation_noninferior") is True,
        "receiver_inherits_quality_win": (
            dominance.get("gates", {}).get("cpu_quality_noninferior") is True
            and dominance.get("gates", {}).get("gpu_quality_noninferior") is True
        ),
        "transfer_ppl_ratio_exact": transfer["transfer_ppl_ratio"] == 1.0,
        "transfer_max_logit_diff_exact": transfer["transfer_max_logit_diff"] == 0.0,
        "transfer_max_abi_diff_exact": transfer["transfer_max_abi_diff"] == 0.0,
        "transfer_generation_exact": transfer["transfer_generation_exact"] is True,
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": scope_label
        or (
            "Production 1M-vs-5M source dominance plus exact receiver-after-transfer "
            "inheritance for the same ABI patch-cell checkpoint."
        ),
        "dominance_certificate": str(dominance_certificate),
        "checkpoint": str(checkpoint),
        "gates": gates,
        "metrics": {
            "source_dominance_ratios": dominance.get("ratios", {}),
            "source_dominance_metrics": dominance.get("metrics", {}),
            **transfer,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify production 1M-vs-5M dominance survives exact receiver transfer"
    )
    parser.add_argument("--dominance-certificate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-rows", type=int, default=2)
    parser.add_argument("--scope-label")
    args = parser.parse_args()
    dominance_certificate = (
        args.dominance_certificate
        if args.dominance_certificate.is_absolute()
        else ROOT / args.dominance_certificate
    )
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = verify(
        dominance_certificate=dominance_certificate,
        checkpoint=checkpoint,
        output=output,
        device=args.device,
        seed=args.seed,
        eval_rows=args.eval_rows,
        scope_label=args.scope_label,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
