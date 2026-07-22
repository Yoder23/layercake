from __future__ import annotations

import torch

from layercake.models.foundation import FoundationConfig, LayerCakeFoundation
from layercake.models.foundation_v2 import FoundationV2Config, LayerCakeFoundationV2
from layercake.models.portable_fusion import PortableFusionCake, PortableFusionConfig
from layercake.models.baseline_transformer import ModernBPETransformer, TransformerConfig
from layercake.runtime.cpu_optimized import CPUOptimizedRuntime
from layercake.runtime.cpu_reference import CPUReferenceRuntime


def _v1() -> LayerCakeFoundation:
    torch.manual_seed(19)
    return LayerCakeFoundation(FoundationConfig(
        patch_size=4, d_byte=8, d_model=16, recurrent_layers=1,
        local_kernel=3, routed_experts=4, expert_expansion=2, abi_width=8,
    )).eval()


def _v2() -> LayerCakeFoundationV2:
    torch.manual_seed(23)
    return LayerCakeFoundationV2(FoundationV2Config(
        d_byte=8, d_local=12, d_global=16, local_layers=1, local_kernel=3,
        fast_patch_size=4, slow_patch_size=8, global_layers=1,
        routed_experts=4, expert_expansion=2, abi_width=16,
    )).eval()


def test_v1_prefill_and_teacher_forced_steps_match_full_context() -> None:
    model = _v1()
    prompt = torch.tensor([[11, 7, 4, 99, 1, 2, 3]], dtype=torch.long)
    route = 2
    model.set_route(route)
    state = model.prefill(prompt, route=route)
    torch.testing.assert_close(state.next_logits, model(prompt)[:, -1], rtol=1e-5, atol=1e-6)
    prefix = prompt
    for next_byte in (8, 9, 10, 11, 12, 13):
        _, state = model.decode_step(state, next_byte=torch.tensor([next_byte]))
        prefix = torch.cat([prefix, torch.tensor([[next_byte]])], dim=1)
        torch.testing.assert_close(state.next_logits, model(prefix)[:, -1], rtol=1e-5, atol=1e-6)


def test_v1_state_round_trip_is_safe_and_exact() -> None:
    model = _v1()
    state = model.prefill(
        b"serialization crosses a patch", route=1, capture_generated=True
    )
    _, state = model.decode_many(state, 5)
    restored = model.restore_state(model.serialize_state(state))
    expected, state = model.decode_many(state, 7)
    actual, restored = model.decode_many(restored, 7)
    assert torch.equal(expected, actual)
    assert torch.equal(state.generated_bytes, restored.generated_bytes)


def test_v2_incremental_matches_full_context_across_both_patch_scales() -> None:
    model = _v2()
    prompt = torch.tensor([[5, 10, 15, 20, 25, 30, 35]], dtype=torch.long)
    route = 3
    state = model.prefill(prompt, route=route)
    torch.testing.assert_close(
        state.next_logits, model(prompt, route=route)[:, -1], rtol=1e-5, atol=1e-6
    )
    prefix = prompt
    for next_byte in range(40, 55):
        _, state = model.decode_step(state, next_byte=torch.tensor([next_byte]))
        prefix = torch.cat([prefix, torch.tensor([[next_byte]])], dim=1)
        torch.testing.assert_close(
            state.next_logits, model(prefix, route=route)[:, -1], rtol=2e-5, atol=2e-6
        )


def test_portable_fusion_uses_host_logits_and_runs_incrementally() -> None:
    model = _v2()
    cake = PortableFusionCake(PortableFusionConfig(
        abi_width=16, byte_width=4, hidden_width=8, rank=4
    )).eval()
    with torch.no_grad():
        cake.up.weight.normal_(std=0.02)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    state = model.prefill(prompt, route=1, fusion_cake=cake)
    full, aux = model(prompt, route=1, fusion_cake=cake, return_aux=True)
    torch.testing.assert_close(state.next_logits, full[:, -1], rtol=2e-5, atol=2e-6)
    assert not torch.equal(aux["core_logits"][:, -1], full[:, -1])
    restored = model.restore_state(model.serialize_state(state))
    assert restored.active_cake == state.active_cake
    try:
        model.decode_step(restored)
    except ValueError as error:
        assert "active cake" in str(error)
    else:
        raise AssertionError("cake-bound state decoded without its cake")
    expected, _ = model.decode_many(state, 4, fusion_cake=cake)
    actual, _ = model.decode_many(restored, 4, fusion_cake=cake)
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)


def test_v2_active_fraction_is_below_twenty_percent() -> None:
    model = LayerCakeFoundationV2()
    report = model.parameter_report()
    assert report["active_fraction"] <= 0.20


def test_v2_default_state_memory_does_not_grow_with_generated_output() -> None:
    model = _v2()
    state = model.prefill(bytes(range(8)), route=1)
    initial_bytes = state.state_bytes
    _, state = model.decode_many(state, 32)
    assert state.generated_bytes.numel() == 0
    assert state.decoded_bytes == 32
    assert state.state_bytes == initial_bytes


def test_reference_and_optimized_cpu_runtime_generate_same_bytes() -> None:
    model = _v2()
    prompt = b"incremental equivalence"
    reference = CPUReferenceRuntime(model, route=2).generate(prompt, 12)
    generated, metrics = CPUOptimizedRuntime(model, route=2).generate(prompt, 12)
    assert torch.equal(reference[:, len(prompt):], generated)
    assert metrics["incremental"] is True
    assert metrics["state_bytes"] > 0


def test_transformer_kv_cache_matches_full_context() -> None:
    torch.manual_seed(29)
    model = ModernBPETransformer(TransformerConfig(
        vocab_size=272, width=24, layers=2, heads=4, max_tokens=64, expansion=2
    )).eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    state = model.prefill(prompt)
    torch.testing.assert_close(state.next_logits, model(prompt)[:, -1], rtol=1e-5, atol=1e-6)
    prefix = prompt
    for token in (7, 8, 9, 10):
        _, state = model.decode_step(state, torch.tensor([token]))
        prefix = torch.cat([prefix, torch.tensor([[token]])], dim=1)
        torch.testing.assert_close(state.next_logits, model(prefix)[:, -1], rtol=1e-5, atol=1e-6)
