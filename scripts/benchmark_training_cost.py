from __future__ import annotations

import argparse

from _common import emit
from layercake.abi import ABISpec
from layercake.domain_bricks import LowRankDomainOperator, SparseLowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    spec = ABISpec(
        version="lc-abi/2",
        d_abi=512,
        input_interface=InputInterfaceSpec(mode="tokenized", vocab_size=32000),
    )
    low = LowRankDomainOperator(spec, rank=16)
    sparse = SparseLowRankDomainOperator(spec, rank=8, num_experts=8, top_k=2)
    emit(
        {
            "benchmark": "training_cost_static",
            "dense_abi_baseline_parameters": 512 * 512 + 512,
            "low_rank_parameters": low.parameter_count(),
            "sparse_low_rank_parameters": sparse.parameter_count(),
            "low_rank_flops_per_position": low.estimated_flops_per_position(),
            "sparse_flops_per_position": sparse.estimated_flops_per_position(),
            "wall_clock_required_for_real_claim": True,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
