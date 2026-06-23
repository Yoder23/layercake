import torch

from layercake.abi import ABISpec
from layercake.domain_bricks import LowRankDomainOperator, SparseLowRankDomainOperator
from layercake.input_interfaces import InputInterfaceSpec


def abi_spec(d_abi=32):
    return ABISpec(
        version="lc-abi/2",
        d_abi=d_abi,
        input_interface=InputInterfaceSpec(mode="tokenized", vocab_size=1000),
    )


def test_low_rank_shape_noop_and_parameter_advantage():
    spec = abi_spec()
    brick = LowRankDomainOperator(spec, rank=4, alpha_init=0.0)
    h = torch.randn(2, 5, 32)
    out = brick(h)
    assert out.shape == h.shape
    assert torch.equal(out, h)
    dense_baseline = 32 * 32 + 32
    assert brick.parameter_count() < dense_baseline


def test_disabled_brick_is_exact_identity():
    brick = LowRankDomainOperator(abi_spec(), rank=4, alpha_init=1.0, enabled=False)
    h = torch.randn(2, 5, 32)
    assert torch.equal(brick(h), h)


def test_sparse_top_k_behavior():
    brick = SparseLowRankDomainOperator(
        abi_spec(), rank=4, num_experts=6, top_k=2, alpha_init=1.0
    )
    h = torch.randn(2, 5, 32)
    out, weights = brick(h, return_routing=True)
    assert out.shape == h.shape
    assert (weights > 0).sum(dim=-1).eq(2).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5))


def test_sparse_kernel_only_updates_selected_experts():
    brick = SparseLowRankDomainOperator(
        abi_spec(), rank=4, num_experts=8, top_k=1, alpha_init=1.0
    )
    h = torch.randn(1, 1, 32)
    _, weights = brick(h, return_routing=True)
    selected = weights.argmax(dim=-1).item()
    brick(h).sum().backward()
    expert_grad = brick.up.grad.abs().sum(dim=(1, 2))
    nonzero = torch.nonzero(expert_grad, as_tuple=False).flatten().tolist()
    assert nonzero == [selected]


def test_lossless_brick_copy_function():
    brick = SparseLowRankDomainOperator(
        abi_spec(), rank=4, num_experts=4, top_k=1, alpha_init=1.0
    )
    copied = brick.copy_lossless()
    h = torch.randn(2, 3, 32)
    assert torch.equal(brick(h), copied(h))
