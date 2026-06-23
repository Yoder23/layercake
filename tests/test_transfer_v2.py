import torch

from layercake.abi import ABISpec
from layercake.domain_bricks import LowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec
from layercake.transfer import (
    classify_transfer,
    perplexity_from_logits,
    ppl_degradation_ratio,
    state_dict_max_diff,
)


def spec():
    return ABISpec(
        version="lc-abi/2",
        d_abi=16,
        input_interface=InputInterfaceSpec(mode="tokenized", vocab_size=100),
    )


def test_copy_and_function_lossless():
    source = LowRankDomainOperator(spec(), rank=4, alpha_init=1.0)
    target = source.copy_lossless()
    h = torch.randn(2, 3, 16)
    assert state_dict_max_diff(source, target) == 0.0
    assert (source(h) - target(h)).abs().max().item() == 0.0


def test_ppl_contract_flags_regression():
    logits = torch.zeros(1, 2, 4)
    targets = torch.tensor([[0, 1]])
    ppl = perplexity_from_logits(logits, targets)
    assert abs(ppl - 4.0) < 1e-5
    assert ppl_degradation_ratio(10.0, 12.0) == 1.2
    assert classify_transfer(
        weight_max_diff=0.0, function_max_diff=0.0, degradation_ratio=1.2
    ) == "PPL_REGRESSION"
