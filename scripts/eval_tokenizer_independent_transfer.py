from __future__ import annotations

import argparse
import torch

from _common import emit
from layercake.abi_alignment import abi_distribution_drift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(42)
    # Smoke diagnostic: paired pooled states. Real mode should load checkpoints.
    tokenized = torch.randn(8, 32)
    byte_patch = tokenized + 0.05 * torch.randn_like(tokenized)
    drift = abi_distribution_drift(tokenized, byte_patch)
    emit(
        {
            "source_input_mode": "tokenized",
            "target_input_mode": "byte_patch",
            "abi_shape": list(tokenized.shape),
            "abi_drift": drift["total"],
            "status": "MEASURED_SMOKE_NOT_SCIENTIFIC_CLAIM",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
