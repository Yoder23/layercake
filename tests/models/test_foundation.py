import torch
import torch.nn.functional as F

from layercake.models.foundation import FoundationConfig, LayerCakeFoundation, SparseOptimizerFactory


def tiny():
    return LayerCakeFoundation(FoundationConfig(
        patch_size=4, d_byte=16, d_model=32, recurrent_layers=1, local_kernel=3,
        routed_experts=8, expert_expansion=2, abi_width=16,
    ))


def test_foundation_is_causal_before_modified_byte():
    torch.manual_seed(1)
    model = tiny().eval()
    model.set_route(0)
    first = torch.randint(0, 256, (1, 20))
    second = first.clone()
    second[:, 12:] = torch.randint(0, 256, second[:, 12:].shape)
    with torch.inference_mode():
        left = model(first)
        right = model(second)
    # Fused CPU GRU kernels may change last-bit rounding when later timesteps differ.
    # The prefix must nevertheless be numerically invariant.
    assert torch.allclose(left[:, :12], right[:, :12], atol=1e-6, rtol=1e-6)


def test_sparse_optimizer_and_backward_touch_only_selected_expert():
    torch.manual_seed(2)
    model = tiny()
    optimizer = SparseOptimizerFactory.adamw(model, 3)
    ids = torch.randint(0, 256, (2, 20))
    logits = model(ids[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 256), ids[:, 1:].reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.routed_cakes.experts[3].parameters())
    for index, expert in enumerate(model.routed_cakes.experts):
        if index != 3:
            assert all(parameter.grad is None for parameter in expert.parameters())


def test_default_active_fraction_meets_sparse_target():
    report = LayerCakeFoundation().parameter_report()
    assert 0.10 <= report["active_fraction"] <= 0.20
