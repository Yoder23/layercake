from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .common import stable_hash, write_json
from .preview import RubricPreview
from .rubric import TrainingRubric


@dataclass(frozen=True)
class Syllabus:
    syllabus_id: str
    rubric_id: str
    preview_id: str
    ordered_data_buckets: list[dict[str, Any]]
    sampling_weights: list[float]
    stage_schedule: list[dict[str, Any]]
    trainable_modules: list[str]
    frozen_modules: list[str]
    loss_weights: dict[str, float]
    optimizer_overrides: dict[str, Any]
    early_stop_rules: list[dict[str, Any]]
    rollback_rules: list[dict[str, Any]]
    gate_thresholds: dict[str, Any]
    expected_improvement: float
    expected_cost: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    syllabus_hash: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["syllabus_hash"] = self.compute_hash()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "Syllabus":
        return cls(**json.loads(text))

    def compute_hash(self) -> str:
        data = asdict(self)
        data["syllabus_hash"] = ""
        return stable_hash(data)

    def save(self, path: str | Path | None = None) -> Path:
        output = Path(path) if path else Path("results/syllabi") / f"{self.syllabus_id}.json"
        write_json(output, self.to_dict())
        return output

    @classmethod
    def load(cls, path: str | Path) -> "Syllabus":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _ordered(preview: RubricPreview, mode: str) -> list[dict[str, Any]]:
    buckets = list(preview.difficulty_buckets)
    if mode == "hard_to_easy":
        return sorted(buckets, key=lambda item: item["difficulty"], reverse=True)
    if mode == "entropy_balanced":
        ordered = sorted(buckets, key=lambda item: item["difficulty"])
        low, high, mixed = 0, len(ordered) - 1, []
        while low <= high:
            mixed.append(ordered[low])
            low += 1
            if low <= high:
                mixed.append(ordered[high])
                high -= 1
        return mixed
    if mode == "rehearsal_interleaved":
        ordered = sorted(buckets, key=lambda item: item["difficulty"])
        return [
            {**bucket, "rehearsal": index % 2 == 1}
            for index, bucket in enumerate(ordered)
        ]
    return sorted(buckets, key=lambda item: item["difficulty"])


def compile_syllabus(
    rubric: TrainingRubric,
    preview: RubricPreview,
    *,
    mode: str | None = None,
    output_dir: str | Path = "results/syllabi",
) -> Syllabus:
    mode = mode or preview.recommended_curriculum
    ordered = _ordered(preview, mode)
    weights = [1.0 / max(len(ordered), 1)] * len(ordered)
    gates = {
        gate.get("name", gate.get("metric", "gate")): gate.get("threshold", gate.get("max_delta"))
        for gate in preview.recommended_gates
        if isinstance(gate, dict)
    }
    syllabus = Syllabus(
        syllabus_id=stable_hash({"rubric": rubric.compute_hash(), "preview": preview.compute_hash(), "mode": mode})[:16],
        rubric_id=rubric.rubric_id,
        preview_id=preview.preview_id,
        ordered_data_buckets=ordered,
        sampling_weights=weights,
        stage_schedule=[{"mode": mode, "steps": max(rubric.max_steps, 1)}],
        trainable_modules=preview.recommended_trainable_modules,
        frozen_modules=preview.recommended_frozen_modules,
        loss_weights=preview.recommended_loss_weights,
        optimizer_overrides={"lr": 0.025 if preview.byte_entropy < 5 else 0.01},
        early_stop_rules=[{"type": "compute_waste", "patience": 2, "min_delta": 0.0}],
        rollback_rules=[{"on": "gate_failure", "restore_parent": True}],
        gate_thresholds=gates,
        expected_improvement=0.01 + min(preview.byte_entropy, 8.0) / 1000.0,
        expected_cost=preview.estimated_wallclock_proxy,
    )
    syllabus.save(Path(output_dir) / f"{syllabus.syllabus_id}.json")
    return syllabus
