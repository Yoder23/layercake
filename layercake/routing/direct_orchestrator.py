"""Sparse orchestration for the sealed direct neural-decoder cake ABI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

import torch

from layercake.models.direct_cake_host import DirectCakeHost

from .catalog_router import (
    ArchiveBoundProfile,
    CatalogProfileRouter,
    CatalogRoutingError,
)
from .policies import CakePermissionPolicy, RoutingBudget, RoutingPolicy
from .router import RouteResult


@dataclass(frozen=True)
class DirectOrchestrationResult:
    mode: str
    execution_path: str
    selected: tuple[str, ...]
    output: bytes | tuple[dict[str, Any], ...]
    route: RouteResult
    route_milliseconds: float
    execution_milliseconds: float
    end_to_end_milliseconds: float
    telemetry_delta: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        output: Any
        if isinstance(self.output, bytes):
            output = self.output.decode("utf-8", errors="strict")
        else:
            output = [
                {
                    **value,
                    "output": (
                        value["output"].decode("utf-8", errors="strict")
                        if isinstance(value.get("output"), bytes)
                        else value.get("output")
                    ),
                }
                for value in self.output
            ]
        return {
            "mode": self.mode,
            "execution_path": self.execution_path,
            "selected": list(self.selected),
            "output": output,
            "route": asdict(self.route),
            "route_milliseconds": self.route_milliseconds,
            "execution_milliseconds": self.execution_milliseconds,
            "end_to_end_milliseconds": self.end_to_end_milliseconds,
            "telemetry_delta": self.telemetry_delta,
        }


def _telemetry_delta(
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        cake_id: {
            key: int(values.get(key, 0)) - int(before.get(cake_id, {}).get(key, 0))
            for key in sorted(values)
        }
        for cake_id, values in sorted(after.items())
    }


def _canonical_direct_prompt(prompt: str) -> str:
    """Apply the frozen direct-decoder request terminator exactly once."""

    return prompt if prompt.endswith("\n") else prompt + "\n"


class DirectCakeOrchestrator:
    """Authenticate, discover, route, and sparsely execute direct cakes."""

    MODES = frozenset(
        {
            "core_only",
            "automatic_top1",
            "automatic_topk",
            "multidomain",
            "manual",
        }
    )

    def __init__(
        self,
        registry_root: str | Path,
        *,
        abi_version: str,
        abi_hash: str,
        trust_store: Mapping[str, bytes | str | Path],
        profiles: Iterable[ArchiveBoundProfile],
        device: torch.device | str = "cpu",
        maximum_loaded_cakes: int = 2,
    ) -> None:
        bound_profiles = tuple(profiles)
        self._profiles = {
            profile.cake_id: profile for profile in bound_profiles
        }
        self.host = DirectCakeHost(
            registry_root,
            abi_version=abi_version,
            abi_hash=abi_hash,
            trust_store=trust_store,
            device=device,
        )
        policy = RoutingPolicy(
            version="layercake-catalog-router/1",
            activation_threshold=0.55,
            abstention_margin=0.08,
            escalation_confidence=0.55,
            allow_composition=True,
            budget=RoutingBudget(
                max_loaded_bytes=512 * 1024 * 1024,
                max_cakes=maximum_loaded_cakes,
                max_route_milliseconds=5.0,
                cold_load_penalty=0.0,
            ),
            permissions=CakePermissionPolicy(
                allowed_permissions=frozenset({"local-inference"}),
            ),
        )
        self.router = CatalogProfileRouter(bound_profiles, policy=policy)
        self.refresh()

    def install(self, package: str | Path) -> dict[str, Any]:
        record = self.host.install(package)
        self.refresh()
        return record

    def refresh(self) -> dict[str, Any]:
        return self.router.refresh(self.host.registry.list())

    def plan(
        self,
        prompt: str,
        *,
        mode: str = "automatic_top1",
        manual: Iterable[str] | None = None,
    ) -> RouteResult:
        if mode not in self.MODES:
            raise CatalogRoutingError(f"unsupported orchestration mode: {mode}")
        if mode == "core_only":
            return RouteResult(
                selected=(),
                candidates=(),
                confidence=1.0,
                abstained=True,
                core_fallback=True,
                escalate=False,
                multidomain=False,
                reason="explicit_core_only",
                policy_version="layercake-catalog-router/1",
                route_milliseconds=0.0,
                trace=({"event": "explicit_core_only"},),
            )
        if mode == "manual":
            if manual is None:
                raise CatalogRoutingError("manual mode requires explicit cake identities")
            return self.router.route(prompt, forced=manual)
        return self.router.route(
            prompt,
            top_k=2 if mode in {"automatic_topk", "multidomain"} else 1,
        )

    def plan_labeled(
        self,
        prompt: str,
        *,
        destination_scope: str,
        domain: str | None = None,
    ) -> RouteResult:
        """Honor an authoritative destination label without parsing prompt text.

        This is the fail-closed boundary for callers that already possess a
        provenance-bound capability label.  Prompt text can never override the
        supplied label.  Missing selected domains do not fall through to the
        English core.
        """

        scope = str(destination_scope).strip().casefold()
        selected_domain = (
            str(domain).strip().casefold() if domain is not None else None
        )
        if scope == "english_core":
            if selected_domain not in {None, "", "domain_independent"}:
                raise CatalogRoutingError(
                    "english-core destination cannot carry a specialist domain"
                )
            result = self.plan(prompt, mode="core_only")
            return RouteResult(
                selected=result.selected,
                candidates=result.candidates,
                confidence=result.confidence,
                abstained=result.abstained,
                core_fallback=result.core_fallback,
                escalate=result.escalate,
                multidomain=result.multidomain,
                reason="authoritative_english_core_label",
                policy_version=result.policy_version,
                route_milliseconds=result.route_milliseconds,
                trace=(
                    {
                        "event": "authoritative_destination",
                        "scope": "english_core",
                    },
                ),
            )
        if scope == "quarantine":
            return RouteResult(
                selected=(),
                candidates=(),
                confidence=1.0,
                abstained=True,
                core_fallback=False,
                escalate=True,
                multidomain=False,
                reason="authoritative_quarantine_label",
                policy_version=self.router.policy.version,
                route_milliseconds=0.0,
                trace=(
                    {
                        "event": "authoritative_destination",
                        "scope": "quarantine",
                    },
                ),
            )
        if scope != "domain_cake" or not selected_domain:
            raise CatalogRoutingError(
                "authoritative destination must be english_core, domain_cake, or quarantine"
            )
        matches = tuple(
            sorted(
                profile.cake_id
                for profile in self._profiles.values()
                if selected_domain
                in {value.casefold() for value in profile.domains}
            )
        )
        if len(matches) != 1:
            return RouteResult(
                selected=(),
                candidates=(),
                confidence=0.0,
                abstained=True,
                core_fallback=False,
                escalate=True,
                multidomain=False,
                reason="unknown_or_ambiguous_selected_domain",
                policy_version=self.router.policy.version,
                route_milliseconds=0.0,
                trace=(
                    {
                        "event": "authoritative_destination",
                        "scope": "domain_cake",
                        "domain": selected_domain,
                        "matches": list(matches),
                    },
                ),
            )
        cake_id = matches[0]
        if cake_id not in self.router.eligible_ids:
            return RouteResult(
                selected=(),
                candidates=(),
                confidence=1.0,
                abstained=True,
                core_fallback=False,
                escalate=True,
                multidomain=False,
                reason="selected_domain_not_installed",
                policy_version=self.router.policy.version,
                route_milliseconds=0.0,
                trace=(
                    {
                        "event": "authoritative_destination",
                        "scope": "domain_cake",
                        "domain": selected_domain,
                        "cake_id": cake_id,
                        "installed": False,
                    },
                ),
            )
        return self.router.route(prompt, forced=(cake_id,))

    def execute_labeled(
        self,
        prompt: str,
        *,
        destination_scope: str,
        domain: str | None = None,
        core_handler: Callable[[str], bytes | str] | None = None,
    ) -> DirectOrchestrationResult:
        """Execute one authoritative destination or return a fail-closed result."""

        started = time.perf_counter()
        before = self.host.telemetry()
        route = self.plan_labeled(
            prompt, destination_scope=destination_scope, domain=domain
        )
        execution_started = time.perf_counter()
        if route.core_fallback:
            handler = core_handler or (lambda value: f"CORE:{value}")
            value = handler(prompt)
            output = value.encode("utf-8") if isinstance(value, str) else value
            path = "authoritative_english_core"
        elif route.selected:
            generated = self.host.generate(
                route.selected[0], _canonical_direct_prompt(prompt)
            )
            output = generated.output
            path = "authoritative_selected_domain"
        elif route.reason == "selected_domain_not_installed":
            output = (
                f"The requested {str(domain).strip()} capability is not installed."
            ).encode("utf-8")
            path = "authoritative_domain_missing"
        elif route.reason == "authoritative_quarantine_label":
            output = b"The request is quarantined and cannot be executed."
            path = "authoritative_quarantine"
        else:
            output = b"The requested capability is unavailable or ambiguous."
            path = "authoritative_domain_unavailable"
        execution_ms = (time.perf_counter() - execution_started) * 1000.0
        end_ms = (time.perf_counter() - started) * 1000.0
        return DirectOrchestrationResult(
            mode="authoritative_label",
            execution_path=path,
            selected=route.selected,
            output=output,
            route=route,
            route_milliseconds=route.route_milliseconds,
            execution_milliseconds=execution_ms,
            end_to_end_milliseconds=end_ms,
            telemetry_delta=_telemetry_delta(before, self.host.telemetry()),
        )

    def execute(
        self,
        prompt: str,
        *,
        mode: str = "automatic_top1",
        manual: Iterable[str] | None = None,
        subrequests: Iterable[str] | None = None,
        core_handler: Callable[[str], bytes | str] | None = None,
    ) -> DirectOrchestrationResult:
        started = time.perf_counter()
        before = self.host.telemetry()
        route = self.plan(prompt, mode=mode, manual=manual)
        route_ms = route.route_milliseconds
        execution_started = time.perf_counter()
        core_handler = core_handler or (lambda value: f"CORE:{value}")
        if route.core_fallback:
            core_output = core_handler(prompt)
            output = core_output.encode("utf-8") if isinstance(core_output, str) else core_output
            path = "core_only"
        elif mode in {"automatic_topk", "multidomain"}:
            parts = tuple(subrequests or ())
            if len(parts) < 2:
                raise CatalogRoutingError(
                    "top-k execution requires at least two explicit independent subrequests"
                )
            routed_parts: list[tuple[str, str, RouteResult]] = []
            for part in parts:
                part_route = self.router.route(part, top_k=1)
                route_ms += part_route.route_milliseconds
                if len(part_route.selected) != 1:
                    raise CatalogRoutingError("each multidomain subrequest must route unambiguously")
                routed_parts.append((part, part_route.selected[0], part_route))
            selected_parts = tuple(dict.fromkeys(value[1] for value in routed_parts))
            if set(selected_parts) != set(route.selected):
                raise CatalogRoutingError(
                    "combined top-k plan and independent subrequest plans disagree"
                )
            composed: list[dict[str, Any]] = []
            for index, (part, cake_id, _) in enumerate(routed_parts):
                generated = self.host.generate(
                    cake_id, _canonical_direct_prompt(part)
                )
                composed.append(
                    {
                        "index": index,
                        "cake_id": cake_id,
                        "prompt": part,
                        "output": generated.output,
                        "actions": list(generated.actions),
                    }
                )
            output = tuple(composed)
            path = "structured_multidomain"
        else:
            if len(route.selected) != 1:
                raise CatalogRoutingError("single-cake execution requires exactly one selected cake")
            generated = self.host.generate(
                route.selected[0], _canonical_direct_prompt(prompt)
            )
            output = generated.output
            path = "manual_cake" if mode == "manual" else "automatic_top1_cake"
        execution_ms = (time.perf_counter() - execution_started) * 1000.0
        end_ms = (time.perf_counter() - started) * 1000.0
        after = self.host.telemetry()
        return DirectOrchestrationResult(
            mode=mode,
            execution_path=path,
            selected=route.selected,
            output=output,
            route=route,
            route_milliseconds=route_ms,
            execution_milliseconds=execution_ms,
            end_to_end_milliseconds=end_ms,
            telemetry_delta=_telemetry_delta(before, after),
        )
