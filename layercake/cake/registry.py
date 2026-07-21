"""Content-addressed local cake registry with atomic metadata updates."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterator

from .manifest import canonical_json


REGISTRY_VERSION = "1"


class RegistryError(RuntimeError):
    pass


class CakeRegistry:
    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = Path(os.environ.get("LAYERCAKE_HOME", Path.home() / ".layercake")) / "cakes"
        self.root = Path(root).resolve()
        self.blobs = self.root / "blobs" / "sha256"
        self.index_path = self.root / "registry.json"
        self.lock_path = self.root / ".registry.lock"

    def initialize(self) -> None:
        self.blobs.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({"version": REGISTRY_VERSION, "installed": {}})

    def _read_index(self) -> dict[str, Any]:
        self.initialize()
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError("local cake registry is corrupt") from exc
        if (
            not isinstance(data, dict)
            or data.get("version") != REGISTRY_VERSION
            or not isinstance(data.get("installed"), dict)
        ):
            raise RegistryError("unsupported or malformed cake registry")
        return data

    def _write_index(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_name(f".{self.index_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(canonical_json(data) + b"\n")
            os.replace(temporary, self.index_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @contextmanager
    def locked(self, timeout: float = 10.0) -> Iterator[None]:
        self.initialize()
        deadline = time.monotonic() + timeout
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise RegistryError("timed out waiting for cake registry lock")
                time.sleep(0.025)
        try:
            yield
        finally:
            os.close(descriptor)
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def blob_path(self, archive_hash: str) -> Path:
        if len(archive_hash) != 64 or any(c not in "0123456789abcdef" for c in archive_hash):
            raise RegistryError("invalid content address")
        path = (self.blobs / f"{archive_hash}.cake").resolve()
        if path.parent != self.blobs.resolve():
            raise RegistryError("content address escaped the registry")
        return path

    def store_blob(self, source: Path, archive_hash: str) -> Path:
        self.initialize()
        destination = self.blob_path(archive_hash)
        if destination.exists():
            return destination
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def list(self) -> list[dict[str, Any]]:
        installed = self._read_index()["installed"]
        return [dict(installed[cake_id]) for cake_id in sorted(installed)]

    def get(self, cake_id: str) -> dict[str, Any] | None:
        value = self._read_index()["installed"].get(cake_id)
        return dict(value) if value is not None else None

    def search(self, query: str) -> list[dict[str, Any]]:
        terms = {term for term in query.casefold().split() if term}
        rows = []
        for item in self.list():
            haystack = " ".join(
                [item.get("cake_id", ""), item.get("name", ""), item.get("description", "")]
                + item.get("domains", [])
                + item.get("keywords", [])
            ).casefold()
            score = sum(term in haystack for term in terms)
            if not terms or score:
                rows.append((score, item))
        return [item for _, item in sorted(rows, key=lambda pair: (-pair[0], pair[1]["cake_id"]))]

    def activate(self, record: dict[str, Any]) -> dict[str, Any] | None:
        with self.locked():
            index = self._read_index()
            previous = index["installed"].get(record["cake_id"])
            history = list(previous.get("history", [])) if previous else []
            if previous:
                history.append({key: value for key, value in previous.items() if key != "history"})
            record = {**record, "history": history[-20:]}
            index["installed"][record["cake_id"]] = record
            self._write_index(index)
            return dict(previous) if previous else None

    def remove(self, cake_id: str) -> dict[str, Any]:
        with self.locked():
            index = self._read_index()
            try:
                previous = index["installed"].pop(cake_id)
            except KeyError as exc:
                raise RegistryError(f"cake is not installed: {cake_id}") from exc
            self._write_index(index)
            return previous

    def rollback(self, cake_id: str) -> dict[str, Any]:
        with self.locked():
            index = self._read_index()
            try:
                current = index["installed"][cake_id]
            except KeyError as exc:
                raise RegistryError(f"cake is not installed: {cake_id}") from exc
            history = list(current.get("history", []))
            if not history:
                raise RegistryError(f"cake has no rollback version: {cake_id}")
            target = history.pop()
            restored = {**current, **target, "history": history, "installed_at": time.time()}
            index["installed"][cake_id] = restored
            self._write_index(index)
            return restored
