from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Callable


@dataclass
class CacheEntry:
    value: Any
    size_bytes: int
    loaded_at: float
    last_used_at: float
    uses: int = 0


class CakeLRUCache:
    """LRU cache that distinguishes installed storage from active memory."""

    def __init__(
        self,
        max_bytes: int,
        loader: Callable[[dict], tuple[Any, int]],
        unloader: Callable[[Any], None] | None = None,
    ):
        if max_bytes <= 0:
            raise ValueError("cache size must be positive")
        self.max_bytes = int(max_bytes)
        self.loader = loader
        self.unloader = unloader or (lambda _value: None)
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.swap_count = 0

    @property
    def active_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries.values())

    def loaded(self, cake_id: str) -> bool:
        return cake_id in self.entries

    def get(self, record: dict) -> tuple[Any, bool, float]:
        cake_id = record["cake_id"]
        started = time.perf_counter()
        if cake_id in self.entries:
            entry = self.entries.pop(cake_id)
            entry.last_used_at = time.time()
            entry.uses += 1
            self.entries[cake_id] = entry
            return entry.value, False, (time.perf_counter() - started) * 1000
        value, size_bytes = self.loader(record)
        if size_bytes > self.max_bytes:
            self.unloader(value)
            raise MemoryError(f"cake {cake_id!r} exceeds the active memory budget")
        while self.entries and self.active_bytes + size_bytes > self.max_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.unloader(evicted.value)
            self.swap_count += 1
        now = time.time()
        self.entries[cake_id] = CacheEntry(value, int(size_bytes), now, now, uses=1)
        return value, True, (time.perf_counter() - started) * 1000

    def prefetch(self, records: list[dict]) -> list[str]:
        loaded: list[str] = []
        for record in records:
            if not self.loaded(record["cake_id"]):
                self.get(record)
                loaded.append(record["cake_id"])
        return loaded

    def unload(self, cake_id: str) -> bool:
        entry = self.entries.pop(cake_id, None)
        if entry is None:
            return False
        self.unloader(entry.value)
        self.swap_count += 1
        return True

    def state(self) -> dict:
        return {
            "loaded": list(self.entries),
            "active_bytes": self.active_bytes,
            "max_bytes": self.max_bytes,
            "swap_count": self.swap_count,
        }
