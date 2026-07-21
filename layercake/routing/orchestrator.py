"""End-to-end installed-cake orchestration with routing traces and LRU loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Callable

import torch

from layercake.cake.package import load_package
from layercake.cake.registry import CakeRegistry
from layercake.models.portable_decoder import load_cake_module

from .cache import CakeLRUCache
from .policies import RoutingPolicy
from .router import CakeRouter, RouteResult


@dataclass(frozen=True)
class OrchestrationResult:
    output: Any
    route: RouteResult
    execution_path: str
    loaded_cakes: tuple[str, ...]
    cold_loaded: tuple[str, ...]
    load_milliseconds: float
    execution_milliseconds: float
    end_to_end_milliseconds: float
    verifier_escalated: bool
    cache_state: dict

    def metrics(self) -> dict:
        result = asdict(self)
        result.pop("output", None)
        return result


def _module_bytes(module: torch.nn.Module) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in list(module.parameters()) + list(module.buffers())
    )


class LocalLayerCakeOrchestrator:
    def __init__(
        self,
        registry: CakeRegistry,
        *,
        policy: RoutingPolicy | None = None,
        trust_store: dict | None = None,
        device: str | torch.device = "cpu",
        loader: Callable[[dict], tuple[Any, int]] | None = None,
    ):
        self.registry = registry
        self.policy = policy or RoutingPolicy()
        self.router = CakeRouter(self.policy)
        self.trust_store = trust_store or {}
        self.device = torch.device(device)
        self._loader = loader or self._load_record
        self.cache = CakeLRUCache(
            self.policy.budget.max_loaded_bytes,
            self._loader,
            self._unload,
        )

    def _load_record(self, record: dict) -> tuple[torch.nn.Module, int]:
        package = load_package(
            Path(record["blob"]),
            trust_store=self.trust_store,
            require_signature=not bool(record.get("trusted_local")),
            allow_local_development=bool(record.get("trusted_local")),
        )
        module = load_cake_module(package).to(self.device)
        return module, _module_bytes(module)

    def _unload(self, value: Any) -> None:
        if isinstance(value, torch.nn.Module):
            value.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def route(
        self,
        prompt: str,
        *,
        top_k: int | None = None,
        forced: tuple[str, ...] | None = None,
    ) -> RouteResult:
        return self.router.route(
            prompt,
            self.registry.list(),
            loaded=set(self.cache.entries),
            top_k=top_k,
            forced=forced,
        )

    def prefetch(self, prompt: str, *, top_k: int | None = None) -> list[str]:
        route = self.route(prompt, top_k=top_k)
        records = {item["cake_id"]: item for item in self.registry.list()}
        return self.cache.prefetch([records[cake_id] for cake_id in route.selected])

    def execute(
        self,
        prompt: str,
        *,
        core_handler: Callable[[str], Any],
        cake_handler: Callable[[str, list[Any], RouteResult], Any],
        verifier_handler: Callable[[str, Any, RouteResult], Any] | None = None,
        top_k: int | None = None,
        forced: tuple[str, ...] | None = None,
    ) -> OrchestrationResult:
        started = time.perf_counter()
        route = self.route(prompt, top_k=top_k, forced=forced)
        records = {item["cake_id"]: item for item in self.registry.list()}
        selected_records = [records[cake_id] for cake_id in route.selected]
        if len(selected_records) > 1:
            composition_modes = {record.get("composition", "none") for record in selected_records}
            if len(composition_modes) != 1 or "none" in composition_modes:
                raise ValueError("selected cakes do not declare a common safe composition contract")
        modules: list[Any] = []
        cold: list[str] = []
        load_ms = 0.0
        for cake_id in route.selected:
            module, was_cold, elapsed = self.cache.get(records[cake_id])
            modules.append(module)
            load_ms += elapsed
            if was_cold:
                cold.append(cake_id)
        execution_started = time.perf_counter()
        if route.core_fallback:
            output = core_handler(prompt)
            path = "core_only_abstention"
        else:
            output = cake_handler(prompt, modules, route)
            path = "composed_cakes" if len(modules) > 1 else "selected_cake"
        escalated = False
        if route.escalate and verifier_handler is not None:
            output = verifier_handler(prompt, output, route)
            escalated = True
            path += "+verifier"
        execution_ms = (time.perf_counter() - execution_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        return OrchestrationResult(
            output=output,
            route=route,
            execution_path=path,
            loaded_cakes=route.selected,
            cold_loaded=tuple(cold),
            load_milliseconds=load_ms,
            execution_milliseconds=execution_ms,
            end_to_end_milliseconds=total_ms,
            verifier_escalated=escalated,
            cache_state=self.cache.state(),
        )
