"""Additive fail-closed control for provenance-bound destination labels.

The sealed direct orchestrator remains byte-identical.  This wrapper adds a
separate call surface for hosts whose caller already has an authoritative
English, domain, or quarantine label.  Prompt text never participates in that
control decision.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Iterable, Mapping

import torch

from .catalog_router import ArchiveBoundProfile, CatalogRoutingError
from .direct_orchestrator import (
    DirectCakeOrchestrator,
    DirectOrchestrationResult,
    _canonical_direct_prompt,
    _telemetry_delta,
)
from .router import RouteResult


class AuthoritativeDestinationOrchestrator(DirectCakeOrchestrator):
    """Direct-cake orchestration governed by an out-of-band destination label."""

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
        self._authoritative_profiles = {
            profile.cake_id: profile for profile in bound_profiles
        }
        super().__init__(
            registry_root,
            abi_version=abi_version,
            abi_hash=abi_hash,
            trust_store=trust_store,
            profiles=bound_profiles,
            device=device,
            maximum_loaded_cakes=maximum_loaded_cakes,
        )

    def plan_labeled(
        self,
        prompt: str,
        *,
        destination_scope: str,
        domain: str | None = None,
    ) -> RouteResult:
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
            return self._terminal_route(
                reason="authoritative_quarantine_label",
                confidence=1.0,
                trace={"scope": "quarantine"},
            )
        if scope != "domain_cake" or not selected_domain:
            raise CatalogRoutingError(
                "authoritative destination must be english_core, domain_cake, or quarantine"
            )
        matches = tuple(
            sorted(
                profile.cake_id
                for profile in self._authoritative_profiles.values()
                if selected_domain
                in {value.casefold() for value in profile.domains}
            )
        )
        if len(matches) != 1:
            return self._terminal_route(
                reason="unknown_or_ambiguous_selected_domain",
                confidence=0.0,
                trace={
                    "scope": "domain_cake",
                    "domain": selected_domain,
                    "matches": list(matches),
                },
            )
        cake_id = matches[0]
        if cake_id not in self.router.eligible_ids:
            return self._terminal_route(
                reason="selected_domain_not_installed",
                confidence=1.0,
                trace={
                    "scope": "domain_cake",
                    "domain": selected_domain,
                    "cake_id": cake_id,
                    "installed": False,
                },
            )
        return self.router.route(prompt, forced=(cake_id,))

    def _terminal_route(
        self,
        *,
        reason: str,
        confidence: float,
        trace: dict,
    ) -> RouteResult:
        return RouteResult(
            selected=(),
            candidates=(),
            confidence=confidence,
            abstained=True,
            core_fallback=False,
            escalate=True,
            multidomain=False,
            reason=reason,
            policy_version=self.router.policy.version,
            route_milliseconds=0.0,
            trace=({"event": "authoritative_destination", **trace},),
        )

    def execute_labeled(
        self,
        prompt: str,
        *,
        destination_scope: str,
        domain: str | None = None,
        core_handler: Callable[[str], bytes | str] | None = None,
    ) -> DirectOrchestrationResult:
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
