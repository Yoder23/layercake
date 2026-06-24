import pytest
import torch

from layercake.rolling.cherrypick import CherryPickError, cherry_pick_module
from layercake.rolling.commit import ModelCommit


def _commit(tmp_path, name, module):
    path = tmp_path / f"{name}.pt"
    torch.save(module.state_dict(), path)
    return ModelCommit.create(
        parent_commit_id=None,
        branch="main",
        status="passed",
        model_family_id="toy",
        abi_hash="abi",
        input_interface_hash="input",
        byte_patch_hash="patch",
        module_hashes={"m": name},
        artifact_paths={"m": str(path)},
        rubric_hash="r",
        message=name,
    )


def test_cherrypick_validates_tensor_shapes(tmp_path):
    source = _commit(tmp_path, "source", torch.nn.Linear(2, 2))
    target = _commit(tmp_path, "target", torch.nn.Linear(2, 2))
    result = cherry_pick_module(source, target, "m", output_dir=tmp_path / "picked")
    assert result["shape_match"] is True
    assert result["tensor_validation"]["keys_match"] is True
    bad = _commit(tmp_path, "bad", torch.nn.Linear(3, 2))
    with pytest.raises(CherryPickError, match="tensor shape mismatch"):
        cherry_pick_module(source, bad, "m", output_dir=tmp_path / "badpick")
