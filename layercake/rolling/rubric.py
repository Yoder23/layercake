from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .common import load_structured, stable_hash


@dataclass(frozen=True)
class TrainingRubric:
    rubric_id: str
    description: str = ""
    target_capabilities: list[str] = field(default_factory=list)
    parent_commit_id: str | None = None
    branch: str = "main"
    trainable_modules: list[str] = field(default_factory=list)
    frozen_modules: list[str] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    losses: list[dict[str, Any]] = field(default_factory=list)
    optimizer: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 0
    max_tokens_or_bytes: int = 0
    stopping_rules: list[dict[str, Any]] = field(default_factory=list)
    gates: list[dict[str, Any]] = field(default_factory=list)
    rollback_policy: dict[str, Any] = field(default_factory=lambda: {"on_failure": True})
    promotion_policy: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingRubric":
        data = load_structured(path)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, text: str) -> "TrainingRubric":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingRubric":
        data = dict(data)
        if "name" in data and "description" not in data:
            data["description"] = data.pop("name")
        for extra in ("metric", "target", "direction"):
            data.pop(extra, None)
        valid = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in valid})

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def compute_hash(self) -> str:
        return stable_hash(self.to_dict())
