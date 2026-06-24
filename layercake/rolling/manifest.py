from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

from .common import file_sha256, stable_hash


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    source_path: str
    split: str = "train"
    byte_count: int = 0
    token_count: int | None = None
    file_hashes: dict[str, str] = field(default_factory=dict)
    preprocessing_hash: str = ""
    sample_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_path(cls, dataset_id: str | Path, path: str | Path | None = None, split: str = "train", name: str | None = None) -> "DatasetManifest":
        if path is None:
            path = dataset_id
            dataset_id = name or Path(path).stem
        elif name is not None:
            dataset_id = name
        path = Path(path)
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        byte_count = sum(p.stat().st_size for p in files)
        hashes = {str(p): file_sha256(p) for p in files}
        return cls(
            dataset_id=dataset_id,
            source_path=str(path),
            split=split,
            byte_count=byte_count,
            file_hashes=hashes,
            sample_count=len(files),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DatasetManifest":
        return cls(**json.loads(text))

    def hash(self) -> str:
        return stable_hash(self.to_dict())

    def compute_hash(self) -> str:
        return self.hash()

    @property
    def name(self) -> str:
        return self.dataset_id

    @property
    def total_bytes(self) -> int:
        return self.byte_count
