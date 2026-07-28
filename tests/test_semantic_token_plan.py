import inspect

import torch

from layercake.semantic_token_plan import (
    SemanticTokenPlanResidual,
    build_semantic_token_plan_artifact,
    load_semantic_token_plan_artifact,
)


def _model() -> SemanticTokenPlanResidual:
    model = SemanticTokenPlanResidual(
        d_abi=24,
        model_width=24,
        attention_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        feedforward_width=48,
        pointer_width=12,
        dropout=0.0,
        maximum_prompt_units=16,
        maximum_response_units=16,
        max_residual=4.0,
    )
    model.eval()
    return model


def test_semantic_token_plan_returns_bounded_same_shape_residual():
    torch.manual_seed(3)
    model = _model()
    prompt = torch.randn(2, 7, 24)
    response = torch.randn(2, 5, 24)
    result = model.training_forward(prompt, response)
    assert result["residual"].shape == response.shape
    assert result["adapted"].shape == response.shape
    assert result["pointer_scores"].shape == (2, 5, 7)
    assert torch.all(result["residual"].abs() <= 4.0)


def test_incremental_path_matches_full_causal_path():
    torch.manual_seed(5)
    model = _model()
    prompt = torch.randn(1, 6, 24)
    response = torch.randn(1, 4, 24)
    full = model.training_forward(prompt, response)["residual"]
    first, state = model.prefill(prompt)
    incremental = [first]
    for index in range(1, response.shape[1]):
        residual, state = model.step(response[:, index], state)
        incremental.append(residual)
    actual = torch.stack(incremental, dim=1)
    # The first response query at runtime is the final prompt ABI state.
    aligned = response.clone()
    aligned[:, 0] = prompt[:, -1]
    expected = model.training_forward(prompt, aligned)["residual"]
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert state.response_steps == response.shape[1]
    assert all(
        cache.shape[1] == response.shape[1]
        for cache in state.layer_self_attention_inputs
    )
    assert full.shape == actual.shape


def test_artifact_is_bound_to_exact_semantic_abi():
    model = _model()
    artifact = build_semantic_token_plan_artifact(
        model,
        abi_version="lc-test/1",
        abi_sha256="a" * 64,
        training={"seed": 7},
    )
    restored, payload = load_semantic_token_plan_artifact(artifact)
    assert restored.canonical_config() == model.canonical_config()
    assert payload["abi_version"] == "lc-test/1"
    assert payload["abi_sha256"] == "a" * 64


def test_semantic_token_plan_has_no_raw_byte_or_token_id_runtime_argument():
    parameters = inspect.signature(
        SemanticTokenPlanResidual.prefill
    ).parameters
    assert tuple(parameters) == ("self", "prompt_states")
