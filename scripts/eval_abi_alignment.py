from __future__ import annotations

import argparse
import torch

from _common import emit
from layercake.abi_alignment import (
    abi_distribution_drift,
    abi_pairwise_alignment_loss,
    abi_whitening_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-a", type=int, default=42)
    parser.add_argument("--seed-b", type=int, default=314)
    parser.add_argument("--output")
    args = parser.parse_args()
    generator_a = torch.Generator().manual_seed(args.seed_a)
    generator_b = torch.Generator().manual_seed(args.seed_b)
    a = torch.randn(4, 16, 32, generator=generator_a)
    b = torch.randn(4, 16, 32, generator=generator_b)
    emit(
        {
            "pairwise_mse": abi_pairwise_alignment_loss(a, b).item(),
            "whitening_a": abi_whitening_loss(a).item(),
            "whitening_b": abi_whitening_loss(b).item(),
            "drift": abi_distribution_drift(a, b),
            "diagnostic_only": True,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
