from pathlib import Path

import pytest

from layercake.rolling.cherrypick import CherryPickError, cherry_pick_module
from layercake.rolling.commit import ModelCommit


def _commit(tmp_path, abi="abi", input_hash="input"):
    artifact = tmp_path / f"{abi}_{input_hash}.pt"
    artifact.write_bytes(b"weights")
    return ModelCommit.create(
        parent_commit_id=None,
        branch="main",
        status="passed",
        model_family_id="toy",
        abi_hash=abi,
        input_interface_hash=input_hash,
        byte_patch_hash="patch",
        module_hashes={"brick": "hash"},
        artifact_paths={"brick": str(artifact)},
        rubric_hash="rubric",
        message="commit",
    )


def test_cherrypick_requires_compatible_interface(tmp_path):
    source = _commit(tmp_path, abi="abi")
    target = _commit(tmp_path, abi="abi")
    result = cherry_pick_module(source, target, "brick", output_dir=tmp_path / "picked")
    assert Path(result["artifact_path"]).exists()
    with pytest.raises(CherryPickError):
        cherry_pick_module(source, _commit(tmp_path, abi="other"), "brick", output_dir=tmp_path / "picked2")
