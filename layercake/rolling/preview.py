from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from layercake.byte_patch import FixedBytePatcher

from .common import stable_hash, write_json
from .manifest import DatasetManifest
from .rubric import TrainingRubric


CODE_SYMBOLS = set(b"{}[]()<>+-=*/\\_:.;,#\"'`")


@dataclass(frozen=True)
class RubricPreview:
    preview_id: str
    rubric_id: str
    parent_commit_id: str | None
    dataset_manifest_hash: str
    input_mode: str
    patching_mode: str
    sample_count: int
    byte_count: int
    token_count: int | None
    byte_entropy: float
    byte_histogram_summary: dict[str, Any]
    rare_byte_rate: float
    unicode_rate: float
    code_symbol_rate: float
    average_sequence_length: float
    patch_count: int
    patch_compression_ratio: float
    current_model_bpb: float | None = None
    current_model_loss: float | None = None
    transformer_baseline_bpb: float | None = None
    byte_transformer_baseline_bpb: float | None = None
    abi_mean_norm: float | None = None
    abi_covariance_summary: dict[str, Any] = field(default_factory=dict)
    abi_drift_from_parent: float | None = None
    estimated_train_steps: int = 0
    estimated_trainable_params: int = 0
    estimated_wallclock_proxy: float = 0.0
    difficulty_buckets: list[dict[str, Any]] = field(default_factory=list)
    recommended_curriculum: str = "easy_to_hard"
    recommended_trainable_modules: list[str] = field(default_factory=list)
    recommended_frozen_modules: list[str] = field(default_factory=list)
    recommended_loss_weights: dict[str, float] = field(default_factory=dict)
    recommended_gates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    preview_hash: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["preview_hash"] = self.compute_hash()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RubricPreview":
        data = json.loads(text)
        return cls(**data)

    def compute_hash(self) -> str:
        data = asdict(self)
        data["preview_hash"] = ""
        data["created_at"] = ""
        return stable_hash(data)

    def save(self, path: str | Path | None = None) -> Path:
        output = Path(path) if path else Path("results/previews") / f"{self.preview_id}.json"
        write_json(output, self.to_dict())
        return output

    @classmethod
    def load(cls, path: str | Path) -> "RubricPreview":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def load_byte_samples(path: str | Path, *, max_samples: int = 8, max_bytes: int = 4096) -> list[bytes]:
    path = Path(path)
    raw = path.read_bytes() if path.is_file() else b"\n".join(
        p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()
    )
    raw = raw[:max_bytes]
    if not raw:
        raw = b" "
    chunks = [chunk for chunk in raw.splitlines() if chunk][:max_samples]
    return chunks or [raw[:max_bytes]]


def byte_entropy(byte_values: bytes) -> float:
    if not byte_values:
        return 0.0
    counts = torch.bincount(torch.tensor(list(byte_values), dtype=torch.long), minlength=256).float()
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * torch.log2(probs)).sum().item())


def histogram_summary(byte_values: bytes) -> dict[str, Any]:
    counts = torch.bincount(torch.tensor(list(byte_values), dtype=torch.long), minlength=256)
    top = torch.topk(counts, k=min(8, int((counts > 0).sum().item()) or 1))
    return {
        "unique_bytes": int((counts > 0).sum().item()),
        "top_bytes": [
            {"byte": int(index), "count": int(count)}
            for count, index in zip(top.values.tolist(), top.indices.tolist())
            if count
        ],
    }


def sequence_difficulty(sample: bytes) -> float:
    entropy = byte_entropy(sample)
    symbol_rate = sum(byte in CODE_SYMBOLS for byte in sample) / max(len(sample), 1)
    unicode_rate = sum(byte >= 128 for byte in sample) / max(len(sample), 1)
    return entropy + 2.0 * symbol_rate + unicode_rate


def eval_model_bpb(model, samples: list[bytes]) -> tuple[float | None, float | None, float | None, dict[str, Any]]:
    if model is None:
        return None, None, None, {}
    rows = [torch.tensor(list(sample), dtype=torch.long) for sample in samples if len(sample) > 1]
    if not rows:
        return None, None, None, {}
    max_len = max(row.numel() for row in rows)
    batch = torch.zeros(len(rows), max_len, dtype=torch.long)
    mask = torch.zeros_like(batch, dtype=torch.bool)
    for index, row in enumerate(rows):
        batch[index, : row.numel()] = row
        mask[index, : row.numel()] = True
    model.eval()
    with torch.no_grad():
        try:
            output = model(batch)
            logits, abi = output[0], output[1] if isinstance(output, tuple) and len(output) > 1 else None
            logits = logits[:, :-1]
            targets = batch[:, 1 : 1 + logits.shape[1]]
            target_mask = mask[:, 1 : 1 + logits.shape[1]]
            loss = F.cross_entropy(logits[target_mask], targets[target_mask])
            bpb = float(loss.item() / math.log(2.0))
            abi_norm = float(abi.norm(dim=-1).mean().item()) if abi is not None else None
            cov_summary = {}
            if abi is not None:
                flat = abi.reshape(-1, abi.shape[-1]).float()
                cov_summary = {
                    "mean_abs": float(flat.mean(dim=0).abs().mean().item()),
                    "std_mean": float(flat.std(dim=0, unbiased=False).mean().item()),
                }
            return bpb, float(loss.item()), abi_norm, cov_summary
        except Exception as exc:
            return None, None, None, {"warning": str(exc)}


def run_preview(
    rubric: TrainingRubric,
    dataset_path: str | Path,
    *,
    model=None,
    parent_commit=None,
    baseline_bpb: float | None = None,
    patch_size: int = 2,
    output_dir: str | Path = "results/previews",
    max_samples: int = 8,
    max_bytes: int = 4096,
) -> RubricPreview:
    samples = load_byte_samples(dataset_path, max_samples=max_samples, max_bytes=max_bytes)
    raw = b"\n".join(samples)
    manifest = DatasetManifest.from_path(dataset_path, name=Path(dataset_path).stem)
    patcher = FixedBytePatcher(patch_size)
    metas = [patcher.boundaries(sample) for sample in samples]
    difficulties = [
        {"index": i, "byte_count": len(sample), "difficulty": sequence_difficulty(sample)}
        for i, sample in enumerate(samples)
    ]
    difficulties.sort(key=lambda item: item["difficulty"])
    model_bpb, model_loss, abi_norm, cov_summary = eval_model_bpb(model, samples)
    estimated_params = (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if model is not None and hasattr(model, "parameters")
        else 0
    )
    warnings: list[str] = []
    if cov_summary.get("warning"):
        warnings.append(cov_summary["warning"])
    if byte_entropy(raw) > 7.5:
        warnings.append("high_entropy_data")
    preview = RubricPreview(
        preview_id=stable_hash({"rubric": rubric.compute_hash(), "dataset": manifest.compute_hash(), "patch_size": patch_size})[:16],
        rubric_id=rubric.rubric_id,
        parent_commit_id=getattr(parent_commit, "commit_id", None),
        dataset_manifest_hash=manifest.compute_hash(),
        input_mode="bytes",
        patching_mode=patcher.mode,
        sample_count=len(samples),
        byte_count=len(raw),
        token_count=None,
        byte_entropy=byte_entropy(raw),
        byte_histogram_summary=histogram_summary(raw),
        rare_byte_rate=sum(raw.count(bytes([b])) for b in range(256) if raw.count(bytes([b])) == 1) / max(len(raw), 1),
        unicode_rate=sum(byte >= 128 for byte in raw) / max(len(raw), 1),
        code_symbol_rate=sum(byte in CODE_SYMBOLS for byte in raw) / max(len(raw), 1),
        average_sequence_length=sum(len(s) for s in samples) / max(len(samples), 1),
        patch_count=sum(meta.patch_count for meta in metas),
        patch_compression_ratio=sum(meta.original_length for meta in metas) / max(sum(meta.patch_count for meta in metas), 1),
        current_model_bpb=model_bpb,
        current_model_loss=model_loss,
        transformer_baseline_bpb=baseline_bpb,
        byte_transformer_baseline_bpb=baseline_bpb,
        abi_mean_norm=abi_norm,
        abi_covariance_summary=cov_summary,
        abi_drift_from_parent=0.0 if parent_commit else None,
        estimated_train_steps=max(rubric.max_steps, 1),
        estimated_trainable_params=estimated_params,
        estimated_wallclock_proxy=float(max(rubric.max_steps, 1) * max(estimated_params, 1) * len(raw)) / 1e9,
        difficulty_buckets=difficulties,
        recommended_curriculum="entropy_balanced" if byte_entropy(raw) > 5.0 else "easy_to_hard",
        recommended_trainable_modules=rubric.trainable_modules or ["layercake_core"],
        recommended_frozen_modules=rubric.frozen_modules,
        recommended_loss_weights={"lm": 1.0, "abi_stability": 0.1},
        recommended_gates=rubric.gates,
        warnings=warnings,
    )
    preview.save(Path(output_dir) / f"{preview.preview_id}.json")
    return preview


def preview_summary(preview: RubricPreview) -> str:
    return (
        f"preview={preview.preview_id} rubric={preview.rubric_id} "
        f"bytes={preview.byte_count} entropy={preview.byte_entropy:.3f} "
        f"patch_ratio={preview.patch_compression_ratio:.2f} "
        f"curriculum={preview.recommended_curriculum}"
    )
