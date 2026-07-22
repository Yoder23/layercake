import torch

from layercake.models.foundation_v2 import FoundationV2Config, LayerCakeFoundationV2
from layercake.models.routed_experts import CausalRoutedFoundationExperts


def test_top1_route_is_causal_and_physically_sparse():
    torch.manual_seed(4)
    routed = CausalRoutedFoundationExperts(8, experts=4, expansion=2, mode="learned_top1")
    with torch.no_grad():
        routed.router[1].weight.zero_()
    calls = [0, 0, 0, 0]
    handles = [
        expert.register_forward_hook(lambda _m, _i, _o, index=index: calls.__setitem__(index, calls[index] + 1))
        for index, expert in enumerate(routed.experts)
    ]
    values = torch.randn(2, 5, 8)
    original = routed(values)
    changed = values.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 50
    modified = routed(changed)
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(original[:, :3], modified[:, :3])
    assert calls[0] == 2
    assert calls[1:] == [0, 0, 0]


def test_top2_executes_only_selected_experts():
    routed = CausalRoutedFoundationExperts(8, experts=4, expansion=2, mode="learned_top2")
    with torch.no_grad():
        routed.router[1].weight.zero_()
    _output, aux = routed(torch.randn(1, 3, 8), return_aux=True)
    assert int((aux["assignment_counts"] > 0).sum()) == 2
    assert int(aux["assignment_counts"].sum()) == 6


def test_learned_top1_incremental_matches_full_context():
    torch.manual_seed(9)
    model = LayerCakeFoundationV2(FoundationV2Config(
        d_byte=8, d_local=12, d_global=16, local_kernel=3,
        fast_patch_size=2, slow_patch_size=4, routed_experts=4,
        expert_expansion=2, abi_width=16, routing_mode="learned_top1",
    )).eval()
    sequence = torch.tensor([[2, 7, 1, 9, 4, 3, 8, 5]], dtype=torch.long)
    with torch.inference_mode():
        full = model(sequence, route=-1)
        state = model.prefill(sequence[:, :3], route=-1)
        for position in range(2, sequence.shape[1] - 1):
            torch.testing.assert_close(state.next_logits, full[:, position], rtol=1e-5, atol=1e-6)
            _logits, state = model.decode_step(state, next_byte=sequence[:, position + 1])


def test_fixed_mode_remains_backward_compatible_without_explicit_route():
    model = LayerCakeFoundationV2(FoundationV2Config(
        d_byte=8, d_local=12, d_global=16, local_kernel=3,
        fast_patch_size=2, slow_patch_size=4, routed_experts=4,
        expert_expansion=2, abi_width=16,
    ))
    assert model(torch.tensor([[1, 2, 3]])).shape == (1, 3, 256)


def test_auxiliary_future_prediction_is_training_only_capacity():
    model = LayerCakeFoundationV2(FoundationV2Config(
        d_byte=8, d_local=12, d_global=16, local_kernel=3,
        fast_patch_size=2, slow_patch_size=4, routed_experts=4,
        expert_expansion=2, abi_width=16, routing_mode="learned_top1",
        output_bias=True, auxiliary_horizons=(2, 4),
    ))
    logits, aux = model(torch.tensor([[1, 2, 3, 4, 5]]), route=-1, return_aux=True)
    assert logits.shape == (1, 5, 256)
    assert set(aux["future_logits"]) == {"2", "4"}
    assert all(value.shape == logits.shape for value in aux["future_logits"].values())
    report = model.parameter_report(-1)
    assert report["active_parameters_per_training_item"] > report["active_parameters_per_inference_token"]


def test_patch_attention_incremental_matches_full_context():
    torch.manual_seed(12)
    model = LayerCakeFoundationV2(FoundationV2Config(
        d_byte=8, d_local=12, d_global=16, local_kernel=3,
        fast_patch_size=2, slow_patch_size=4, global_layers=2,
        routed_experts=4, expert_expansion=2, abi_width=16,
        routing_mode="learned_top1", global_backend="attention",
        attention_heads=4, attention_expansion=2,
    )).eval()
    sequence = torch.tensor([[2, 7, 1, 9, 4, 3, 8, 5, 6, 11]], dtype=torch.long)
    with torch.inference_mode():
        full = model(sequence, route=-1)
        state = model.prefill(sequence[:, :3], route=-1)
        for position in range(2, sequence.shape[1] - 1):
            torch.testing.assert_close(state.next_logits, full[:, position], rtol=2e-5, atol=2e-6)
            _logits, state = model.decode_step(state, next_byte=sequence[:, position + 1])
        payload = model.serialize_state(state)
        restored = model.restore_state(payload)
        assert len(restored.fast_attention_keys) == 2
