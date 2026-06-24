from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path

from .common import stable_hash, write_json

COMMIT_DIR = Path("artifacts/commits")


@dataclass(frozen=True)
class ModelCommit:
    commit_id: str
    parent_commit_id: str | None
    branch: str
    status: str
    model_family_id: str
    abi_hash: str
    input_interface_hash: str
    byte_patch_hash: str
    module_hashes: dict[str, str] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    optimizer_hash: str | None = None
    rng_state_hash: str | None = None
    rubric_hash: str | None = None
    dataset_manifest_hashes: dict[str, str] = field(default_factory=dict)
    eval_result_hashes: dict[str, str] = field(default_factory=dict)
    benchmark_result_hashes: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, **kwargs) -> "ModelCommit":
        payload = dict(kwargs)
        payload.setdefault("commit_id", "")
        payload.setdefault("status", "candidate")
        payload.setdefault("branch", "main")
        payload.setdefault("model_family_id", "toy")
        payload.setdefault("abi_hash", "")
        payload.setdefault("input_interface_hash", "")
        payload.setdefault("byte_patch_hash", "")
        commit = cls(**payload)
        return replace(commit, commit_id=commit.compute_hash()[:16])

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ModelCommit":
        return cls(**json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> "ModelCommit":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, root: str | Path = COMMIT_DIR) -> Path:
        path = Path(root) / f"{self.commit_id}.json"
        write_json(path, self.to_dict())
        return path

    def compute_hash(self) -> str:
        data = self.to_dict()
        data["commit_id"] = ""
        return stable_hash(data)

    def verify(self) -> bool:
        return bool(self.commit_id and self.branch and self.status)

    def compare_to_parent(self, parent: "ModelCommit") -> dict:
        changed = sorted(
            name
            for name in set(self.module_hashes) | set(parent.module_hashes)
            if self.module_hashes.get(name) != parent.module_hashes.get(name)
        )
        return {
            "changed_modules": changed,
            "abi_changed": self.abi_hash != parent.abi_hash,
            "input_interface_changed": self.input_interface_hash != parent.input_interface_hash,
            "rubric_changed": self.rubric_hash != parent.rubric_hash,
        }

    def with_status(self, status: str) -> "ModelCommit":
        return replace(self, status=status)

    def mark_passed(self) -> "ModelCommit":
        return self.with_status("passed")

    def mark_failed(self) -> "ModelCommit":
        return self.with_status("failed")

    def mark_promoted(self) -> "ModelCommit":
        return self.with_status("promoted")

    def mark_rolled_back(self) -> "ModelCommit":
        return self.with_status("rolled_back")
