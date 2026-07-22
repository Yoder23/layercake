import torch

from layercake.training.patch_campaign import _build_model, _forward


def test_variable_patch_campaign_model_contract():
    model = _build_model({
        "kind": "variable",
        "model": {
            "max_patch_size": 8,
            "d_byte": 8,
            "d_model": 32,
            "d_abi": 16,
            "layers": 1,
            "heads": 4,
            "max_patches": 32,
        },
    })
    inputs = torch.tensor([list(b"a causal variable patch input")], dtype=torch.long)
    logits, metadata = _forward(model, inputs)
    assert logits.shape == (*inputs.shape, 256)
    assert int(metadata["valid_patches"].sum()) < inputs.numel()


def test_adaptive_patch_campaign_model_contract():
    model = _build_model({
        "kind": "adaptive_two_four",
        "model": {
            "d_byte": 8,
            "d_model": 32,
            "d_abi": 16,
            "layers": 1,
            "local_layers": 1,
            "heads": 4,
            "max_patches": 16,
            "local_window": 4,
        },
    })
    inputs = torch.tensor([list(b"abcdefghijklmno")], dtype=torch.long)
    logits, metadata = _forward(model, inputs)
    assert logits.shape == (*inputs.shape, 256)
    assert int(metadata["valid_patches"].sum()) <= inputs.numel() // 2


def test_adaptive_routed_patch_dispatch_is_sparse_and_trainable():
    model = _build_model({
        "kind": "adaptive_two_four",
        "model": {
            "d_byte": 8,
            "d_model": 32,
            "d_abi": 16,
            "layers": 1,
            "local_layers": 1,
            "heads": 4,
            "max_patches": 16,
            "local_window": 4,
            "routed_experts": 4,
            "expert_expansion": 1,
            "routing_mode": "learned_top2",
        },
    })
    inputs = torch.tensor([list(b"abcdefghijklmnop")], dtype=torch.long)
    logits, metadata = _forward(model, inputs)
    assert logits.shape == (*inputs.shape, 256)
    assert metadata["routing"] is not None
    assert int((metadata["routing"]["assignment_counts"] > 0).sum()) <= 4
    logits.float().sum().backward()
    experts_with_gradients = sum(
        any(parameter.grad is not None for parameter in expert.parameters())
        for expert in model.routed.experts
    )
    assert 1 <= experts_with_gradients <= 4
