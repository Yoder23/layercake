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
