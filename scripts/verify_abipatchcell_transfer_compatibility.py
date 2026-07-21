from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_byte_core_from_config import _build_model


def verify(config: dict, *, seed: int = 1234, device: str = "cpu") -> dict:
    torch.manual_seed(seed)
    model_cfg = config["model"]
    torch_device = torch.device(device)
    source = _build_model(model_cfg, torch_device).eval()
    receiver = _build_model(model_cfg, torch_device).eval()
    receiver.load_state_dict(source.state_dict(), strict=True)
    seq = int(config.get("training", {}).get("seq_len", 256))
    seq = min(seq, int(model_cfg.get("max_patches", 128)) * int(model_cfg.get("patch_size", 2)))
    if seq % int(model_cfg.get("patch_size", 2)):
        seq -= seq % int(model_cfg.get("patch_size", 2))
    x = torch.randint(0, 256, (2, seq), device=torch_device)
    y = torch.roll(x, shifts=-1, dims=1)
    with torch.inference_mode():
        source_logits, source_abi = source(x)
        receiver_logits, receiver_abi = receiver(x)
        source_generated = source.generate_next_patch(x)
        receiver_generated = receiver.generate_next_patch(x)
        source_loss = F.cross_entropy(source_logits.reshape(-1, 256), y.reshape(-1))
        receiver_loss = F.cross_entropy(receiver_logits.reshape(-1, 256), y.reshape(-1))
    max_logit_diff = float((source_logits - receiver_logits).abs().max().item())
    max_abi_diff = float((source_abi - receiver_abi).abs().max().item())
    ppl_ratio = math.exp(float(receiver_loss.item()) - float(source_loss.item()))
    generated_equal = torch.equal(source_generated, receiver_generated)
    gates = {
        "local_decoder_is_abi_patch_cell": model_cfg.get("local_decoder") == "abi_patch_cell",
        "abi_shape_equal": tuple(source_abi.shape) == tuple(receiver_abi.shape),
        "transfer_ppl_ratio_exact": ppl_ratio == 1.0,
        "transfer_max_logit_diff_exact": max_logit_diff == 0.0,
        "transfer_max_abi_diff_exact": max_abi_diff == 0.0,
        "transfer_generation_exact": generated_equal,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "ABI patch-cell source/receiver exact transfer compatibility",
        "gates": gates,
        "metrics": {
            "transfer_ppl_ratio": ppl_ratio,
            "transfer_max_logit_diff": max_logit_diff,
            "transfer_max_abi_diff": max_abi_diff,
            "generated_bytes_equal": generated_equal,
            "abi_shape": list(source_abi.shape),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ABI patch-cell source/receiver transfer compatibility")
    parser.add_argument("--config", type=Path, default=Path("configs/production_cpu_game_1m_abipatchcell_layercake.json"))
    parser.add_argument("--output", type=Path, default=Path("results/production_cpu_game/1m_vs_5m_abipatchcell/abipatchcell_transfer_certificate.json"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result = verify(json.loads(config_path.read_text(encoding="utf-8")), seed=args.seed, device=args.device)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
