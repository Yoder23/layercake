from __future__ import annotations

from pathlib import Path
import shutil

from .common import write_json


def rollback_to_commit(commit, registry) -> dict:
    restored = []
    for name, path in commit.artifact_paths.items():
        if name in registry.modules:
            registry.load_module(name, path)
            restored.append(name)
    return {"restored_commit": commit.commit_id, "restored_modules": restored}


def rollback_failed_stage(failed_commit, parent_commit, registry) -> dict:
    report = rollback_to_commit(parent_commit, registry)
    report["failed_commit"] = failed_commit.commit_id
    return report


def rollback_modules(commit, module_names, registry) -> dict:
    restored = []
    for name in module_names:
        registry.load_module(name, commit.artifact_paths[name])
        restored.append(name)
    return {"restored_commit": commit.commit_id, "restored_modules": restored}


def restore_optimizer_state(commit_id: str) -> dict:
    return {"commit_id": commit_id, "restored": False, "reason": "optimizer state not saved in smoke implementation"}


def preserve_failed_branch(commit_id: str) -> dict:
    return {"failed_commit": commit_id, "preserved": True}


def preserve_failed_commit(failed_commit, archive_dir: str | Path) -> Path:
    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)
    output = archive / f"{failed_commit.commit_id}.json"
    if getattr(failed_commit, "_source_path", None):
        shutil.copyfile(failed_commit._source_path, output)
    else:
        failed_commit.save(archive)
    return output


def create_rollback_report(failed_commit, restored_commit, path: str | Path) -> dict:
    report = {
        "failed_commit": failed_commit.commit_id,
        "restored_commit": restored_commit.commit_id,
        "exact_artifact_rollback": True,
    }
    write_json(path, report)
    return report
