import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.train_bpe_transformer_from_config import BPETokenTransformerLM
from scripts.train_byte_core_from_config import (
    _build_model,
    _load_config_with_extends,
    _weighted_schedule,
)


def _load_config(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _count_params(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_northstar_v6_layercake_and_bpe_configs_are_fair_paired():
    layercake_cfg = _load_config("northstar_v6_patch4_grounded_mix_layercake.json")
    bpe_cfg = _load_config("northstar_v6_patch4_grounded_mix_bpe.json")

    layercake_params = _count_params(
        _build_model(layercake_cfg["model"], torch.device("cpu"))
    )
    bpe_model_cfg = bpe_cfg["model"]
    bpe_params = _count_params(
        BPETokenTransformerLM(
            vocab_size=bpe_cfg["tokenizer"]["vocab_size"],
            d_model=bpe_model_cfg["d_model"],
            layers=bpe_model_cfg["layers"],
            heads=bpe_model_cfg["heads"],
            max_len=bpe_cfg["training"]["seq_len"],
            ff_mult=bpe_model_cfg["ff_mult"],
            dropout=bpe_model_cfg["dropout"],
        )
    )

    assert max(layercake_params, bpe_params) / min(layercake_params, bpe_params) <= 1.05

    layercake_mix = [
        (component["name"], component["weight"])
        for component in layercake_cfg["training"]["data_mix"]
    ]
    bpe_mix = [
        (component["name"], component["weight"])
        for component in bpe_cfg["training"]["data_mix"]
    ]
    assert layercake_mix == bpe_mix
    schedule = _weighted_schedule([weight for _, weight in layercake_mix])
    assert len(schedule) == 100
    assert schedule.count(0) == 74
    assert schedule.count(1) == 13
    assert schedule.count(2) == 13

    layercake_model_cfg = layercake_cfg["model"]
    assert layercake_model_cfg.get("domain_cache_order", 0) == 0
    assert layercake_model_cfg.get("domain_cache_logit_scale", 0.0) == 0.0
    assert layercake_model_cfg.get("copy_transducer", False) is False
    assert layercake_cfg["training"]["resume_from"].endswith("_polish5000.pt")


def test_northstar_v6_patchgen_only_config_freezes_base_lm():
    config = _load_config("northstar_v6_patch4_grounded_patchgen_only_layercake.json")
    training = config["training"]

    assert training["trainable_parameter_patterns"] == ["patch_generator.*"]
    assert training["patch_prediction_answer_only_loss"] is True
    assert training["patch_prediction_answer_loss_weight"] > 1.0
    assert training["answer_loss_weight"] == 1.0
    assert config["model"].get("domain_cache_order", 0) == 0
    assert config["model"].get("domain_cache_logit_scale", 0.0) == 0.0


def test_northstar_v6_copy_patchgen_config_adds_only_trainable_copy_path():
    config = _load_config("northstar_v6_patch4_grounded_copy_patchgen_layercake.json")
    model = config["model"]
    training = config["training"]

    assert model["patch_generation_copy_window"] >= 64
    assert model["patch_generation_copy_dim"] > 0
    assert model["patch_generation_copy_scale"] > 0.0
    assert training["resume_strict"] is False
    assert training["trainable_parameter_patterns"] == ["patch_generator.*"]
    assert training["patch_prediction_answer_only_loss"] is True
    assert model.get("domain_cache_order", 0) == 0


def test_northstar_v7_decoder_continuation_preserves_core_and_replays_lm():
    config = _load_config("northstar_v7_patch4_grounded_copy_patchgen_layercake.json")
    model = config["model"]
    training = config["training"]

    assert training["trainable_parameter_patterns"] == ["patch_generator.*"]
    assert training["patch_prediction_answer_only_loss"] is False
    assert training["patch_prediction_answer_loss_weight"] > 1.0
    assert training["lr_step_offset"] > 0
    assert training["resume_strict"] is True
    assert [component["name"] for component in training["data_mix"]] == [
        "lm_replay",
        "question_relevance",
        "schema_action_grounding",
    ]
    assert model["patch_generation_copy_window"] >= 64
    assert model.get("domain_cache_order", 0) == 0


def test_northstar_v8_uses_long_local_span_without_changing_core_size():
    config = _load_config("northstar_v8_patch4_span80_grounded_layercake.json")
    model = config["model"]
    training = config["training"]

    assert model["patch_size"] == 4
    assert model["patch_generation_bytes"] == 80
    assert model["patch_prediction_mode"] == "autoregressive"
    assert training["patch_prediction_answer_only_loss"] is True
    assert training["trainable_parameter_patterns"] == ["patch_generator.*"]
    assert training["resume_from"].endswith(
        "northstar_v7_patch4_grounded_copy_patchgen_layercake/latest.pt"
    )


def test_northstar_v9_continues_only_the_long_span_decoder():
    config = _load_config("northstar_v9_patch4_span80_grounded_continuation.json")
    assert config["model"]["patch_generation_bytes"] == 80
    assert config["training"]["trainable_parameter_patterns"] == [
        "patch_generator.*"
    ]
    assert config["training"]["resume_from"].endswith(
        "northstar_v8_patch4_span80_grounded_layercake/latest.pt"
    )
    assert config["training"]["lr_step_offset"] == 13000


def test_northstar_v10_targets_the_deployment_prompt_boundary():
    config = _load_config("northstar_v10_span80_prompt_boundary.json")
    assert config["extends"] == "northstar_v9_patch4_span80_grounded_continuation.json"
    assert config["training"]["patch_prediction_answer_start_only"] is True
    assert config["training"]["resume_from"].endswith(
        "northstar_v9_patch4_span80_grounded_continuation/latest.pt"
    )


def test_northstar_v11_inherits_boundary_training_and_enables_rollout():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v11_span80_rollout.json"
    )
    assert config["model"]["patch_generation_bytes"] == 80
    assert config["model"]["patch_prediction_rollout_training"] is True
    assert config["training"]["patch_prediction_answer_start_only"] is True
    assert config["training"]["trainable_parameter_patterns"] == [
        "patch_generator.*"
    ]


def test_northstar_v12_keeps_a_strong_teacher_anchor():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v12_span80_scheduled_rollout.json"
    )
    assert config["model"]["patch_prediction_rollout_training"] is True
    assert 0.0 < config["model"]["patch_prediction_rollout_mix"] <= 0.1
    assert config["training"]["patch_prediction_answer_start_only"] is True
    assert config["training"]["resume_from"].endswith(
        "northstar_v10_span80_prompt_boundary/latest.pt"
    )


def test_northstar_v13_expands_only_the_local_copy_decoder():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v13_span80_wide_position_copy.json"
    )
    assert config["model"]["patch_generation_width"] == 256
    assert config["model"]["patch_generation_position_copy"] is True
    assert config["training"]["resume_ignore_shape_mismatch"] is True
    assert config["training"]["trainable_parameter_patterns"] == [
        "patch_generator.*"
    ]


def test_northstar_v14_preserves_wide_head_on_unified_contract():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v14_unified_grounding.json"
    )
    assert config["model"]["patch_generation_width"] == 256
    assert config["model"]["patch_generation_position_copy"] is True
    assert config["training"]["resume_strict"] is True
    assert config["training"]["patch_prediction_answer_start_only"] is True


def test_northstar_v15_keeps_guarded_paraphrase_head_shape_exact():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v15_paraphrase_grounding.json"
    )
    assert config["model"]["patch_generation_width"] == 256
    assert config["training"]["resume_strict"] is True
    assert config["training"]["resume_from"].endswith(
        "northstar_v14_unified_grounding/latest.pt"
    )


def test_northstar_v16_preserves_role_copy_head_contract():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v16_role_copy_grounding.json"
    )
    assert config["model"]["patch_generation_position_copy"] is True
    assert config["training"]["resume_strict"] is True
    assert config["training"]["resume_from"].endswith(
        "northstar_v15_paraphrase_grounding/latest.pt"
    )


def test_northstar_v17_keeps_exact_holdout_combinations_excluded():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v17_compositional_holdout.json"
    )
    assert config["training"]["resume_strict"] is True
    assert config["training"]["resume_from"].endswith(
        "northstar_v16_role_copy_grounding/latest.pt"
    )


def test_northstar_v18_supervises_contextual_copy_only_locally():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v18_supervised_context_copy.json"
    )
    assert config["model"]["patch_generation_contextual_copy"] is True
    assert config["training"]["patch_prediction_copy_loss_weight"] > 0.0
    assert config["training"]["trainable_parameter_patterns"] == [
        "patch_generator.*"
    ]


def test_northstar_v19_casefolds_canonical_copy_targets():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v19_casefold_copy.json"
    )
    assert config["model"]["patch_generation_lowercase_copy"] is True
    assert config["training"]["patch_prediction_copy_loss_weight"] >= 1.0
    assert config["training"]["resume_strict"] is True


def test_northstar_v20_freezes_syntax_and_trains_pointer_only():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v20_pointer_only_casefold.json"
    )
    assert config["model"]["patch_generation_lowercase_copy"] is True
    assert config["training"]["trainable_parameter_patterns"] == [
        "patch_generator.copy_*"
    ]
    assert config["training"]["resume_from"].endswith(
        "northstar_v18_supervised_context_copy/latest.pt"
    )


def test_northstar_v21_adds_semantic_context_to_pointer_only():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v21_semantic_pointer.json"
    )
    assert config["model"]["patch_generation_semantic_copy"] is True
    assert config["model"]["patch_generation_lowercase_copy"] is True
    assert config["training"]["trainable_parameter_patterns"] == [
        "patch_generator.copy_*"
    ]


def test_northstar_v22_gives_resumed_bpe_the_corrected_fair_mix():
    config = _load_config_with_extends(
        ROOT / "configs" / "northstar_v22_fair_corrected_bpe.json"
    )
    training = config["training"]

    assert training["resume_from"].endswith(
        "northstar_v6_patch4_grounded_mix_bpe/latest.pt"
    )
    assert training["resume_step"] == 2000
    assert training["steps"] > training["resume_step"]
    assert training["micro_batch_size"] == 16
    weights = [component["weight"] for component in training["data_mix"]]
    assert sum(weights) == 1.0
    assert sum(weights[1:]) == 0.75
    schedule = _weighted_schedule(weights)
    assert [schedule.count(index) for index in range(3)] == [8, 9, 15]
