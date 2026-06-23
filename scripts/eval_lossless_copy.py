from __future__ import annotations

import argparse
import torch

from _common import emit
from layercake.abi import ABISpec
from layercake.domain_bricks import SparseLowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec
from layercake.transfer import classify_transfer, state_dict_max_diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.manual_seed(42)
    spec = ABISpec(
        version="lc-abi/2",
        d_abi=32,
        input_interface=InputInterfaceSpec(mode="tokenized", vocab_size=1000),
    )
    source = SparseLowRankDomainOperator(
        spec, rank=4, num_experts=4, top_k=1, alpha_init=1.0
    )
    target = source.copy_lossless()
    states = torch.randn(2, 8, 32)
    weight_diff = state_dict_max_diff(source, target)
    function_diff = (source(states) - target(states)).abs().max().item()
    emit(
        {
            "weight_max_diff": weight_diff,
            "function_max_diff": function_diff,
            "status": classify_transfer(
                weight_max_diff=weight_diff,
                function_max_diff=function_diff,
                degradation_ratio=None,
            ),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
