from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.train_bpe_transformer_from_config import BPETokenTransformerLM
from scripts.train_byte_core_from_config import _build_model


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_production_cpu_game_same_size_configs_are_parameter_matched():
    config_pairs = [
        ("1m", "production_cpu_game_same_size_1m_layercake.json", "production_cpu_game_same_size_1m_bpe.json"),
        ("2m", "production_cpu_game_same_size_2m_layercake.json", "production_cpu_game_same_size_2m_bpe.json"),
        ("5m", "production_cpu_game_same_size_5m_layercake.json", "production_cpu_game_same_size_5m_bpe.json"),
        ("10m", "production_cpu_game_same_size_10m_layercake.json", "production_cpu_game_same_size_10m_bpe.json"),
        ("1m_patch4", "production_cpu_game_same_size_1m_patch4_layercake.json", "production_cpu_game_same_size_1m_patch4_bpe.json"),
        ("1m_parallelpatch", "production_cpu_game_same_size_1m_parallelpatch_layercake.json", "production_cpu_game_same_size_1m_parallelpatch_bpe.json"),
    ]
    for tier, lc_name, bpe_name in config_pairs:
        lc_cfg = _load(lc_name)
        bpe_cfg = _load(bpe_name)
        lc_params = _count_params(_build_model(lc_cfg["model"], torch.device("cpu")))
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
        ratio = max(lc_params, bpe_params) / min(lc_params, bpe_params)

        assert ratio <= 1.10, (tier, lc_params, bpe_params, ratio)
        assert bpe_model_cfg["heads"] >= 4


def test_production_1m_vs_5m_abipatchcell_configs_have_required_param_ratio():
    lc_cfg = _load("production_cpu_game_1m_abipatchcell_layercake.json")
    bpe_cfg = _load("production_cpu_game_5m_bpe.json")
    lc_params = _count_params(_build_model(lc_cfg["model"], torch.device("cpu")))
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

    assert 900_000 <= lc_params <= 1_100_000
    assert bpe_params / lc_params >= 5.0
    assert lc_cfg["model"]["local_decoder"] == "abi_patch_cell"


def test_production_1m_vs_5m_selective_abipatchcell_configs_have_required_param_ratio():
    lc_cfg = _load("production_cpu_game_1m_selective_abipatchcell_layercake.json")
    bpe_cfg = _load("production_cpu_game_5m_bpe.json")
    lc_params = _count_params(_build_model(lc_cfg["model"], torch.device("cpu")))
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

    assert 900_000 <= lc_params <= 1_100_000
    assert bpe_params / lc_params >= 5.0
    assert lc_cfg["model"]["global_block"] == "selective_state_patch"
    assert lc_cfg["model"]["local_decoder"] == "abi_patch_cell"
