from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .common import write_json


def append_capability_ledger(path: str | Path = "results/capability_ledger.jsonl", **row) -> dict:
    row = {
        "commit_id": row.get("commit_id"),
        "parent_commit_id": row.get("parent_commit_id"),
        "rubric_id": row.get("rubric_id"),
        "preview_id": row.get("preview_id"),
        "syllabus_id": row.get("syllabus_id"),
        "capability": row.get("capability", "smoke"),
        "metric": row.get("metric", "score"),
        "value": row.get("value"),
        "threshold": row.get("threshold"),
        "passed": bool(row.get("passed", False)),
        "delta_from_parent": row.get("delta_from_parent"),
        "delta_vs_transformer_baseline": row.get("delta_vs_transformer_baseline"),
        "created_at": row.get("created_at", datetime.now(timezone.utc).isoformat()),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def write_training_diff_report(
    commit,
    parent_commit,
    *,
    preview=None,
    syllabus=None,
    metrics_before=None,
    metrics_after=None,
    gate_results=None,
    transformer_baseline=None,
    rollback_report=None,
    warnings=None,
    output_dir: str | Path = "results/reports",
) -> Path:
    changed = commit.compare_to_parent(parent_commit) if parent_commit else {"changed_modules": sorted(commit.module_hashes)}
    artifact_sizes = {
        name: Path(path).stat().st_size
        for name, path in commit.artifact_paths.items()
        if Path(path).exists()
    }
    report = {
        "commit_id": commit.commit_id,
        "parent_commit_id": parent_commit.commit_id if parent_commit else None,
        "modules_changed": changed.get("changed_modules", []),
        "parameter_count_changed": None,
        "trainable_fraction": None,
        "dataset_used": getattr(preview, "dataset_manifest_hash", None),
        "preview_summary": preview.to_dict() if preview else None,
        "syllabus_summary": syllabus.to_dict() if syllabus else None,
        "metrics_before": metrics_before or {},
        "metrics_after": metrics_after or {},
        "gate_deltas": gate_results or [],
        "transformer_baseline_comparison": transformer_baseline or {},
        "rollback_safety": rollback_report or {},
        "artifact_sizes": artifact_sizes,
        "warnings": warnings or [],
    }
    output = Path(output_dir) / f"{commit.commit_id}_training_diff.json"
    write_json(output, report)
    return output
