from __future__ import annotations

import argparse
import torch

from _common import emit
from layercake.abi_alignment import (
    abi_anchor_loss,
    abi_pairwise_alignment_loss,
    abi_whitening_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy canonical-ABI alignment smoke trainer")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(42)
    source = torch.randn(64, 32)
    target = torch.nn.Linear(32, 32, bias=False)
    optimizer = torch.optim.AdamW(target.parameters(), lr=1e-2)
    history = []
    for step in range(args.steps):
        optimizer.zero_grad()
        aligned = target(source)
        loss = (
            abi_anchor_loss(aligned, source)
            + abi_pairwise_alignment_loss(aligned, source)
            + 0.01 * abi_whitening_loss(aligned)
        )
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    emit(
        {
            "steps": args.steps,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "status": "SMOKE_ALIGNMENT_ONLY",
        },
        args.output,
    )


if __name__ == "__main__":
    main()
