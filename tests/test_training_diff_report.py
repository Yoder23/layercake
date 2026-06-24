import torch

from layercake.rolling.commit import ModelCommit
from layercake.rolling.registry import ModuleRegistry
from layercake.rolling.reports import write_training_diff_report


def test_training_diff_report_contains_required_sections(tmp_path):
    registry = ModuleRegistry()
    registry.register("m", torch.nn.Linear(1, 1))
    parent = ModelCommit.create(
        parent_commit_id=None,
        branch="main",
        status="passed",
        model_family_id="toy",
        abi_hash="abi",
        input_interface_hash="input",
        byte_patch_hash="patch",
        module_hashes=registry.module_hashes(),
        artifact_paths={"m": registry.save_module("m", tmp_path / "p.pt")},
        rubric_hash="r",
        message="p",
    )
    commit = ModelCommit.create(
        parent_commit_id=parent.commit_id,
        branch="main",
        status="passed",
        model_family_id="toy",
        abi_hash="abi",
        input_interface_hash="input",
        byte_patch_hash="patch",
        module_hashes=registry.module_hashes(),
        artifact_paths={"m": registry.save_module("m", tmp_path / "c.pt")},
        rubric_hash="r",
        message="c",
    )
    path = write_training_diff_report(commit, parent, metrics_before={"bpb": 2}, metrics_after={"bpb": 1}, output_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "preview_summary" in text and "metrics_after" in text and "artifact_sizes" in text
