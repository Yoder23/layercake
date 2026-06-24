import torch

from layercake.rolling.commit import ModelCommit
from layercake.rolling.registry import ModuleRegistry
from layercake.rolling.rollback import preserve_failed_commit, rollback_failed_stage, rollback_to_commit


def _commit(registry, root, status="passed"):
    module_hashes = registry.module_hashes()
    commit = ModelCommit.create(
        parent_commit_id=None,
        branch="main",
        status=status,
        model_family_id="toy",
        abi_hash="abi",
        input_interface_hash="input",
        byte_patch_hash="patch",
        module_hashes=module_hashes,
        artifact_paths={},
        rubric_hash="rubric",
        message=status,
    )
    artifacts = {name: registry.save_module(name, root / f"{commit.commit_id}_{name}.pt") for name in registry.list_modules()}
    commit = ModelCommit.create(**{**commit.to_dict(), "commit_id": "", "artifact_paths": artifacts})
    commit.save(root)
    return commit


def test_rollback_restores_parent_and_preserves_failed(tmp_path):
    registry = ModuleRegistry()
    model = torch.nn.Linear(2, 1)
    registry.register("toy", model)
    parent = _commit(registry, tmp_path, "passed")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(3.0)
    failed = _commit(registry, tmp_path, "failed")
    report = rollback_failed_stage(failed, parent, registry)
    assert report["restored_commit"] == parent.commit_id
    assert registry.module_hash("toy") == parent.module_hashes["toy"]
    assert rollback_to_commit(parent, registry)["restored_modules"] == ["toy"]
    archived = preserve_failed_commit(failed, tmp_path / "archive")
    assert archived.exists()
