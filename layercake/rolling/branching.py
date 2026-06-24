from __future__ import annotations

import json
from pathlib import Path

from .commit import ModelCommit
from .common import write_json


class BranchStore:
    def __init__(self, root: str | Path = "artifacts/commits"):
        self.root = Path(root)
        self.branches_path = self.root / "branches.json"
        self.head_path = self.root / "HEAD"
        self.root.mkdir(parents=True, exist_ok=True)

    def _branches(self) -> dict:
        if self.branches_path.exists():
            return json.loads(self.branches_path.read_text(encoding="utf-8"))
        return {}

    def _write(self, branches: dict) -> None:
        write_json(self.branches_path, branches)

    def create_branch(self, name: str, from_commit: str) -> None:
        branches = self._branches()
        branches[name] = from_commit
        self._write(branches)

    def checkout_commit(self, commit_id: str) -> None:
        self.head_path.write_text(commit_id, encoding="utf-8")

    def list_branches(self) -> dict:
        return self._branches()

    def list_commits(self, branch: str) -> list[str]:
        head = self._branches().get(branch)
        return [head] if head else []

    def tag_commit(self, commit_id: str, tag: str) -> None:
        tags = self.root / "tags.json"
        data = json.loads(tags.read_text(encoding="utf-8")) if tags.exists() else {}
        data[tag] = commit_id
        write_json(tags, data)


def create_branch(name: str, from_commit: str, root: str | Path = "artifacts/commits") -> None:
    BranchStore(root).create_branch(name, from_commit)


def checkout_commit(commit_id: str, root: str | Path = "artifacts/commits") -> None:
    BranchStore(root).checkout_commit(commit_id)


def list_branches(root: str | Path = "artifacts/commits") -> dict:
    return BranchStore(root).list_branches()


def list_commits(branch: str, root: str | Path = "artifacts/commits") -> list[str]:
    return BranchStore(root).list_commits(branch)


def tag_commit(commit_id: str, tag: str, root: str | Path = "artifacts/commits") -> None:
    BranchStore(root).tag_commit(commit_id, tag)


def load_commit(commit_id: str, root: str | Path = "artifacts/commits") -> ModelCommit:
    return ModelCommit.load(Path(root) / f"{commit_id}.json")
