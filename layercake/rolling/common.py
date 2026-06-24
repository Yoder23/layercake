from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> str:
    if is_dataclass(data):
        data = asdict(data)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_structured(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for YAML rubrics. Install pyyaml or use JSON rubrics."
        ) from exc


def _simple_yaml(text: str) -> dict:
    """Small YAML subset for smoke rubrics: scalars and one-line lists."""
    result: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].strip()
            result[key.strip()] = [
                item.strip().strip("'\"")
                for item in items.split(",")
                if item.strip()
            ]
        elif value in {"true", "false"}:
            result[key.strip()] = value == "true"
        elif value == "":
            result[key.strip()] = []
        else:
            try:
                result[key.strip()] = int(value)
            except ValueError:
                try:
                    result[key.strip()] = float(value)
                except ValueError:
                    result[key.strip()] = value.strip("'\"")
    return result


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
