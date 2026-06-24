from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .common import stable_hash, write_json
from .gates import GateResult


@dataclass(frozen=True)
class SemanticCertificate:
    commit_id: str
    parent_commit_id: str | None
    rubric_id: str
    gate_results: list[dict]
    passed: bool
    regression_summary: dict = field(default_factory=dict)
    exactness_summary: dict = field(default_factory=dict)
    benchmark_references: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    certificate_hash: str = ""

    @classmethod
    def create(
        cls,
        commit_id: str,
        parent_commit_id: str | None,
        rubric_id: str,
        gate_results: list[GateResult],
        **extra,
    ) -> "SemanticCertificate":
        data = cls(
            commit_id=commit_id,
            parent_commit_id=parent_commit_id,
            rubric_id=rubric_id,
            gate_results=[result.to_dict() for result in gate_results],
            passed=all(result.passed for result in gate_results),
            **extra,
        )
        return cls(**{**asdict(data), "certificate_hash": stable_hash(asdict(data))})

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        write_json(path, self.to_dict())
        return Path(path)


def run_semantic_ci(commit, parent_commit, rubric, context: dict, output_path: str | Path):
    from .gates import run_gates

    gate_results = run_gates(rubric.gates, context)
    cert = SemanticCertificate.create(
        commit.commit_id,
        parent_commit.commit_id if parent_commit else None,
        rubric.rubric_id,
        gate_results,
    )
    cert.save(output_path)
    return cert
