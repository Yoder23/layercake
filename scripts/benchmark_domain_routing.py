from __future__ import annotations

import argparse
import time
import torch

from _common import emit
from layercake.abi import ABISpec
from layercake.domain_bricks import SparseLowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec


def measure(brick, states, iterations):
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            brick(states)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output")
    args = parser.parse_args()
    spec = ABISpec(
        version="lc-abi/2",
        d_abi=64,
        input_interface=InputInterfaceSpec(mode="tokenized", vocab_size=1000),
    )
    states = torch.randn(2, 32, 64)
    rows = []
    for installed in (4, 32, 128):
        brick = SparseLowRankDomainOperator(
            spec, rank=4, num_experts=installed, top_k=2, alpha_init=1.0
        ).eval()
        rows.append(
            {
                "installed_bricks": installed,
                "active_bricks": 2,
                "wall_seconds": measure(brick, states, args.iterations),
                "estimated_flops_per_position": brick.estimated_flops_per_position(),
            }
        )
    emit(
        {
            "benchmark": "installed_vs_active_bricks",
            "invariant": "active compute is top_k bounded; router cost still scales with installed count",
            "rows": rows,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
