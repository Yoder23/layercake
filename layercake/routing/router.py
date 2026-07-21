"""Sub-millisecond lexical-semantic router with calibrated abstention."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Iterable

from .policies import RoutingPolicy


_TERM = re.compile(r"[a-z0-9][a-z0-9_+#.-]*")
_CONTROL_PHRASES = re.compile(
    r"(?:ignore|disregard|override)\s+(?:the\s+)?(?:router|routing|cake|domain)",
    re.IGNORECASE,
)


def terms(text: str) -> set[str]:
    return set(_TERM.findall(text.casefold()))


@dataclass(frozen=True)
class RouteCandidate:
    cake_id: str
    score: float
    lexical_coverage: float
    loaded: bool
    domains: tuple[str, ...]


@dataclass(frozen=True)
class RouteResult:
    selected: tuple[str, ...]
    candidates: tuple[RouteCandidate, ...]
    confidence: float
    abstained: bool
    core_fallback: bool
    escalate: bool
    multidomain: bool
    reason: str
    policy_version: str
    route_milliseconds: float
    trace: tuple[dict, ...]


class CakeRouter:
    def __init__(self, policy: RoutingPolicy | None = None):
        self.policy = policy or RoutingPolicy()

    @staticmethod
    def _record_terms(record: dict) -> tuple[set[str], set[str]]:
        domains = terms(" ".join(record.get("domains", [])))
        keywords = terms(" ".join(record.get("keywords", [])))
        identity = terms(
            " ".join(
                [
                    str(record.get("cake_id", "")),
                    str(record.get("name", "")),
                    str(record.get("description", "")),
                ]
            )
        )
        return domains, keywords | identity

    def _score(self, prompt_terms: set[str], record: dict, loaded: bool) -> tuple[float, float]:
        domains, features = self._record_terms(record)
        strong_hits = len(prompt_terms & domains)
        hits = len(prompt_terms & features)
        coverage = hits / max(min(len(prompt_terms), max(len(features), 1)), 1)
        precision = hits / max(len(features), 1)
        domain_bonus = 0.25 * min(strong_hits, 2)
        score = (0.72 * coverage) + (0.28 * math.sqrt(precision)) + domain_bonus
        if not loaded:
            score -= self.policy.budget.cold_load_penalty
        return max(0.0, min(score, 1.0)), coverage

    def route(
        self,
        prompt: str,
        installed: Iterable[dict],
        *,
        loaded: set[str] | None = None,
        top_k: int | None = None,
        forced: Iterable[str] | None = None,
    ) -> RouteResult:
        started = time.perf_counter()
        loaded = loaded or set()
        records = {record["cake_id"]: record for record in installed}
        trace: list[dict] = []
        forced_ids = tuple(forced or ())
        if forced_ids:
            missing = [cake_id for cake_id in forced_ids if cake_id not in records]
            if missing:
                raise KeyError(f"forced cakes are not installed: {missing}")
            for cake_id in forced_ids:
                permitted, reason = self.policy.permissions.permits(records[cake_id])
                if not permitted:
                    raise PermissionError(f"forced cake {cake_id!r} rejected: {reason}")
            elapsed = (time.perf_counter() - started) * 1000
            return RouteResult(
                selected=forced_ids[: self.policy.budget.max_cakes],
                candidates=(), confidence=1.0, abstained=False, core_fallback=False,
                escalate=False, multidomain=len(forced_ids) > 1, reason="user_forced",
                policy_version=self.policy.version, route_milliseconds=elapsed,
                trace=({"event": "forced", "cakes": list(forced_ids)},),
            )
        injection_detected = _CONTROL_PHRASES.search(prompt) is not None
        prompt_terms = terms(_CONTROL_PHRASES.sub("", prompt))
        candidates: list[RouteCandidate] = []
        for record in records.values():
            permitted, denial = self.policy.permissions.permits(record)
            if not permitted:
                trace.append({"event": "rejected", "cake_id": record["cake_id"], "reason": denial})
                continue
            is_loaded = record["cake_id"] in loaded
            score, coverage = self._score(prompt_terms, record, is_loaded)
            candidates.append(
                RouteCandidate(
                    cake_id=record["cake_id"], score=score, lexical_coverage=coverage,
                    loaded=is_loaded, domains=tuple(record.get("domains", [])),
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.cake_id))
        if injection_detected:
            elapsed = (time.perf_counter() - started) * 1000
            trace.append({"event": "adversarial_control_phrase", "decision": "abstain"})
            return RouteResult(
                selected=(), candidates=tuple(candidates), confidence=0.0, abstained=True,
                core_fallback=True, escalate=False, multidomain=False,
                reason="adversarial_control_phrase", policy_version=self.policy.version,
                route_milliseconds=elapsed, trace=tuple(trace),
            )
        limit = min(top_k or self.policy.budget.max_cakes, self.policy.budget.max_cakes)
        selected_candidates = [
            item for item in candidates[:limit] if item.score >= self.policy.activation_threshold
        ]
        best = candidates[0].score if candidates else 0.0
        second = candidates[1].score if len(candidates) > 1 else 0.0
        margin = best - second
        abstained = not selected_candidates
        if selected_candidates and len(selected_candidates) == 1 and margin < self.policy.abstention_margin:
            abstained = True
            selected_candidates = []
        multidomain = len(selected_candidates) > 1
        if multidomain and not self.policy.allow_composition:
            selected_candidates = selected_candidates[:1]
            multidomain = False
        confidence = best if not abstained else max(0.0, best - self.policy.activation_threshold)
        elapsed = (time.perf_counter() - started) * 1000
        trace.extend(
            {"event": "candidate", "cake_id": item.cake_id, "score": item.score, "loaded": item.loaded}
            for item in candidates
        )
        trace.append(
            {
                "event": "decision", "selected": [item.cake_id for item in selected_candidates],
                "abstained": abstained, "confidence": confidence, "margin": margin,
            }
        )
        return RouteResult(
            selected=tuple(item.cake_id for item in selected_candidates),
            candidates=tuple(candidates), confidence=confidence, abstained=abstained,
            core_fallback=abstained, escalate=(not abstained and confidence < self.policy.escalation_confidence),
            multidomain=multidomain,
            reason="no_suitable_cake" if abstained else "matched_installed_cake",
            policy_version=self.policy.version, route_milliseconds=elapsed, trace=tuple(trace),
        )
