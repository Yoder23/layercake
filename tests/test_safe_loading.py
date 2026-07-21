import pytest
import torch

from layercake.safe_loading import CheckpointError, safe_torch_load


class Dangerous:
    pass


def test_safe_torch_load_accepts_tensor_mapping(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"weight": torch.ones(2, 3), "architecture": {"width": 3}}, path)
    loaded = safe_torch_load(path)
    assert torch.equal(loaded["weight"], torch.ones(2, 3))


def test_safe_torch_load_rejects_pickle_global(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save({"value": Dangerous()}, path)
    with pytest.raises(CheckpointError, match="restricted"):
        safe_torch_load(path)
