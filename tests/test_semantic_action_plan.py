import torch

from layercake.semantic_action_plan import (
    SemanticActionPlanResidual,
    build_semantic_action_plan_artifact,
    load_semantic_action_plan_artifact,
)


def _model() -> SemanticActionPlanResidual:
    model = SemanticActionPlanResidual(
        fixed_token_ids=[5, 7, 11],
        eos_token_id=11,
        d_abi=24,
        model_width=24,
        attention_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        feedforward_width=48,
        pointer_width=12,
        dropout=0.0,
        maximum_prompt_units=12,
        maximum_response_units=12,
        max_residual=4.0,
    )
    model.eval()
    return model


def test_training_path_returns_action_distribution_and_bounded_residual():
    torch.manual_seed(3)
    model = _model()
    prompt = torch.randn(2, 6, 24)
    current = torch.randn(2, 4, 24)
    actions = torch.tensor(
        [[0, 3 + 2, 1, 2], [1, 3 + 3, 0, 2]], dtype=torch.long
    )
    result = model.training_forward(prompt, current, actions)
    assert result["action_log_probs"].shape == (2, 4, 3 + 6)
    assert result["residual"].shape == current.shape
    assert torch.all(result["residual"].abs() <= 4.0)


def test_runtime_accepts_only_semantic_prompt_and_current_state():
    torch.manual_seed(7)
    model = _model()
    prompt = torch.randn(1, 5, 24)
    residual, state, action = model.prefill(prompt)
    assert residual.shape == (1, 24)
    assert isinstance(action, int)
    assert state.response_steps == 1
    assert len(state.planned_actions) == 1


def test_artifact_round_trip_preserves_abi_binding():
    model = _model()
    artifact = build_semantic_action_plan_artifact(
        model,
        abi_version="lc-test/1",
        abi_sha256="b" * 64,
        training={"seed": 5},
    )
    restored, payload = load_semantic_action_plan_artifact(artifact)
    assert restored.canonical_config() == model.canonical_config()
    assert payload["abi_version"] == "lc-test/1"
    assert payload["abi_sha256"] == "b" * 64


def test_transition_codec_is_zero_initialized_and_uses_adjacent_states():
    torch.manual_seed(11)
    parent = _model()
    config = parent.canonical_config()
    config["copy_transition_width"] = 8
    codec = SemanticActionPlanResidual(**config)
    missing, unexpected = codec.load_state_dict(
        parent.state_dict(), strict=False
    )
    assert not unexpected
    assert set(missing) == {
        "copy_transition_norm.weight",
        "copy_transition_norm.bias",
        "copy_transition_input.weight",
        "copy_transition_input.bias",
        "copy_transition_output.weight",
    }
    prompt = torch.randn(1, 4, 24)
    current = torch.randn(1, 1, 24)
    decoded = torch.randn(1, 1, 24)
    pointer_action = torch.tensor([[parent.fixed_action_count + 2]])
    parent_value = parent._realize(
        pointer_action, decoded, current, prompt
    )["residual"]
    codec_value = codec._realize(
        pointer_action, decoded, current, prompt
    )["residual"]
    assert torch.equal(parent_value, codec_value)

    with torch.no_grad():
        codec.copy_transition_output.weight.fill_(0.1)
    changed_previous = prompt.clone()
    changed_previous[:, 1] += 3.0
    original = codec._realize(
        pointer_action, decoded, current, prompt
    )["residual"]
    changed = codec._realize(
        pointer_action, decoded, current, changed_previous
    )["residual"]
    assert not torch.equal(original, changed)


def test_transition_only_channel_does_not_use_parent_linear_copy_value():
    config = _model().canonical_config()
    config["copy_transition_width"] = 8
    config["copy_transition_replaces_linear"] = True
    model = SemanticActionPlanResidual(**config).eval()
    prompt = torch.randn(1, 4, 24)
    current = torch.randn(1, 1, 24)
    decoded = torch.randn(1, 1, 24)
    pointer_action = torch.tensor([[model.fixed_action_count + 2]])
    before = model._realize(
        pointer_action, decoded, current, prompt
    )["residual"]
    with torch.no_grad():
        model.copy_semantic_value.weight.normal_(mean=50.0, std=10.0)
    after = model._realize(
        pointer_action, decoded, current, prompt
    )["residual"]
    assert torch.equal(before, after)
