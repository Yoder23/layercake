from __future__ import annotations

import torch

from scripts.artifact_utils import build_models
from layercake.causal_byte_models import CausalBytePatchLM


def test_build_models_accepts_config_checkpoint_format():
    config = {
        "patch_size": 2,
        "d_byte": 8,
        "d_model": 32,
        "d_abi": 16,
        "layers": 1,
        "heads": 4,
        "max_patches": 8,
        "direct_global_context": True,
        "modern_blocks": True,
        "fused_attention": True,
        "local_decoder": "window_transformer",
        "local_layers": 1,
        "local_width": 32,
        "local_window": 8,
    }
    model = CausalBytePatchLM(**config)
    _, loaded = build_models(
        {"model_config": config, "model": model.state_dict()},
        torch.device("cpu"),
    )

    x = torch.randint(0, 256, (1, 8))
    assert loaded(x)[0].shape == (1, 8, 256)
