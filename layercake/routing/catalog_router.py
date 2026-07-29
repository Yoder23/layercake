"""Catalog-derived routing with archive-bound capability profiles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

from .policies import RoutingPolicy
from .router import RouteCandidate, RouteResult


_WORD = re.compile(r"[a-z0-9][a-z0-9_+#.-]*")
_CONTROL = re.compile(
    r"(?:ignore|disregard|override|bypass|change)\s+"
    r"(?:the\s+)?(?:router|routing|cake|domain|specialist)",
    re.IGNORECASE,
)


class CatalogRoutingError(ValueError):
    """Raised when catalog routing metadata fails closed validation."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class RoutingFeature:
    kind: str
    value: str
    weight: float

    def __post_init__(self) -> None:
        if self.kind not in {"token", "phrase", "regex"}:
            raise CatalogRoutingError(f"unsupported routing feature: {self.kind}")
        if not self.value or not 0.0 < self.weight <= 1.0:
            raise CatalogRoutingError("routing features require a value and weight in (0, 1]")
        if self.kind == "regex":
            try:
                re.compile(self.value, re.IGNORECASE)
            except re.error as error:
                raise CatalogRoutingError(f"invalid routing regex: {error}") from error

    def matches(self, normalized: str, words: set[str]) -> bool:
        if self.kind == "token":
            return self.value.casefold() in words
        if self.kind == "phrase":
            return self.value.casefold() in normalized
        return re.search(self.value, normalized, re.IGNORECASE) is not None


@dataclass(frozen=True)
class ArchiveBoundProfile:
    cake_id: str
    archive_sha256: str
    domains: tuple[str, ...]
    features: tuple[RoutingFeature, ...]
    negative_features: tuple[RoutingFeature, ...] = ()

    def __post_init__(self) -> None:
        if not self.cake_id or not self.domains or not self.features:
            raise CatalogRoutingError("routing profile identity, domains, and features are required")
        if (
            len(self.archive_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.archive_sha256)
        ):
            raise CatalogRoutingError("routing profile archive SHA-256 is malformed")

    def score(self, prompt: str) -> tuple[float, float]:
        normalized = " ".join(prompt.casefold().split())
        words = set(_WORD.findall(normalized))
        positive = sum(feature.weight for feature in self.features if feature.matches(normalized, words))
        negative = sum(
            feature.weight for feature in self.negative_features
            if feature.matches(normalized, words)
        )
        score = max(0.0, min(1.0, positive - negative))
        matched = sum(feature.matches(normalized, words) for feature in self.features)
        coverage = matched / len(self.features)
        return score, coverage


def load_archive_bound_profiles(path: str | Path) -> tuple[ArchiveBoundProfile, ...]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogRoutingError(f"cannot read routing profiles: {error}") from error
    if (
        not isinstance(document, dict)
        or document.get("format") != "layercake-archive-bound-routing-profiles/1"
        or document.get("status") != "FROZEN"
        or not isinstance(document.get("profiles"), list)
    ):
        raise CatalogRoutingError("routing profile document identity is invalid")
    profiles: list[ArchiveBoundProfile] = []
    seen: set[str] = set()
    for raw in document["profiles"]:
        if not isinstance(raw, dict):
            raise CatalogRoutingError("routing profile rows must be objects")
        cake_id = str(raw.get("cake_id", ""))
        if cake_id in seen:
            raise CatalogRoutingError(f"duplicate routing profile: {cake_id}")
        seen.add(cake_id)

        def features(key: str) -> tuple[RoutingFeature, ...]:
            values = raw.get(key, [])
            if not isinstance(values, list):
                raise CatalogRoutingError(f"{cake_id}.{key} must be an array")
            return tuple(
                RoutingFeature(
                    kind=str(item.get("kind", "")),
                    value=str(item.get("value", "")),
                    weight=float(item.get("weight", 0.0)),
                )
                for item in values
                if isinstance(item, dict)
            )

        profiles.append(
            ArchiveBoundProfile(
                cake_id=cake_id,
                archive_sha256=str(raw.get("archive_sha256", "")),
                domains=tuple(str(value) for value in raw.get("domains", [])),
                features=features("features"),
                negative_features=features("negative_features"),
            )
        )
    return tuple(sorted(profiles, key=lambda profile: profile.cake_id))


@dataclass(frozen=True)
class CatalogDescriptor:
    cake_id: str
    domains: tuple[str, ...]
    available: bool
    installed: bool
    promoted_capability: bool
    archive_sha256: str | None = None


class CapabilityCatalog:
    """Searchable catalog that keeps management descriptors out of execution."""

    def __init__(self, descriptors: Iterable[CatalogDescriptor]):
        values = list(descriptors)
        if len({value.cake_id for value in values}) != len(values):
            raise CatalogRoutingError("catalog cake identities must be unique")
        self._rows = tuple(sorted(values, key=lambda value: value.cake_id))
        self._tokens = {
            row.cake_id: set(_WORD.findall(" ".join((row.cake_id, *row.domains)).casefold()))
            for row in self._rows
        }

    def __len__(self) -> int:
        return len(self._rows)

    def list(self) -> tuple[CatalogDescriptor, ...]:
        return self._rows

    def search(self, query: str, *, limit: int = 20) -> tuple[CatalogDescriptor, ...]:
        words = set(_WORD.findall(query.casefold()))
        ranked = sorted(
            self._rows,
            key=lambda row: (
                -len(words & self._tokens[row.cake_id]),
                row.cake_id,
            ),
        )
        if words:
            ranked = [row for row in ranked if words & self._tokens[row.cake_id]]
        return tuple(ranked[: max(0, int(limit))])


class CatalogProfileRouter:
    """Route over eligible installed packages without a fixed domain head."""

    def __init__(
        self,
        profiles: Iterable[ArchiveBoundProfile],
        *,
        policy: RoutingPolicy | None = None,
    ) -> None:
        self.policy = policy or RoutingPolicy(
            activation_threshold=0.55,
            abstention_margin=0.08,
        )
        self._profiles = {profile.cake_id: profile for profile in profiles}
        self._eligible: dict[str, dict[str, Any]] = {}
        self._rejections: tuple[dict[str, str], ...] = ()

    @property
    def eligible_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._eligible))

    @property
    def profile_hashes(self) -> dict[str, str]:
        return {
            cake_id: profile.archive_sha256
            for cake_id, profile in sorted(self._profiles.items())
        }

    def refresh(self, installed: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        eligible: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, str]] = []
        for raw in installed:
            record = dict(raw)
            cake_id = str(record.get("cake_id", ""))
            allowed, reason = self.policy.permissions.permits(record)
            profile = self._profiles.get(cake_id)
            if not allowed:
                rejected.append({"cake_id": cake_id, "reason": str(reason)})
            elif profile is None:
                rejected.append({"cake_id": cake_id, "reason": "missing_archive_bound_profile"})
            elif record.get("archive_hash") != profile.archive_sha256:
                rejected.append({"cake_id": cake_id, "reason": "profile_archive_hash_mismatch"})
            elif set(profile.domains) != set(str(value) for value in record.get("domains", [])):
                rejected.append({"cake_id": cake_id, "reason": "profile_domain_mismatch"})
            else:
                eligible[cake_id] = record
        self._eligible = eligible
        self._rejections = tuple(sorted(rejected, key=lambda value: value["cake_id"]))
        return {
            "eligible": list(self.eligible_ids),
            "rejected": [dict(value) for value in self._rejections],
        }

    def _forced(self, forced: Iterable[str], started: float) -> RouteResult:
        selected = tuple(dict.fromkeys(str(value) for value in forced))
        missing = [cake_id for cake_id in selected if cake_id not in self._eligible]
        if missing:
            raise CatalogRoutingError(
                f"manual cakes lack an eligible archive-bound profile: {missing}"
            )
        if len(selected) > self.policy.budget.max_cakes:
            raise CatalogRoutingError("manual selection exceeds the configured cake budget")
        return RouteResult(
            selected=selected,
            candidates=(),
            confidence=1.0,
            abstained=False,
            core_fallback=False,
            escalate=False,
            multidomain=len(selected) > 1,
            reason="user_forced",
            policy_version="layercake-catalog-router/1",
            route_milliseconds=(time.perf_counter() - started) * 1000.0,
            trace=({"event": "forced", "cakes": list(selected)},),
        )

    def route(
        self,
        prompt: str,
        *,
        top_k: int = 1,
        forced: Iterable[str] | None = None,
        loaded: set[str] | None = None,
    ) -> RouteResult:
        started = time.perf_counter()
        if forced is not None:
            return self._forced(forced, started)
        if top_k <= 0:
            raise CatalogRoutingError("top_k must be positive")
        loaded = loaded or set()
        candidates: list[RouteCandidate] = []
        for cake_id, record in self._eligible.items():
            profile = self._profiles[cake_id]
            score, coverage = profile.score(prompt)
            candidates.append(
                RouteCandidate(
                    cake_id=cake_id,
                    score=score,
                    lexical_coverage=coverage,
                    loaded=cake_id in loaded,
                    domains=profile.domains,
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.cake_id))
        trace: list[dict[str, Any]] = [
            {"event": "catalog_rejection", **value} for value in self._rejections
        ]
        if _CONTROL.search(prompt):
            trace.append({"event": "adversarial_control_phrase", "decision": "abstain"})
            return RouteResult(
                selected=(),
                candidates=tuple(candidates),
                confidence=0.0,
                abstained=True,
                core_fallback=True,
                escalate=False,
                multidomain=False,
                reason="adversarial_control_phrase",
                policy_version="layercake-catalog-router/1",
                route_milliseconds=(time.perf_counter() - started) * 1000.0,
                trace=tuple(trace),
            )
        limit = min(int(top_k), self.policy.budget.max_cakes)
        accepted = [
            candidate
            for candidate in candidates
            if candidate.score >= self.policy.activation_threshold
        ][:limit]
        if limit == 1 and accepted:
            runner_up = candidates[1].score if len(candidates) > 1 else 0.0
            if accepted[0].score - runner_up < self.policy.abstention_margin:
                accepted = []
        confidence = accepted[0].score if accepted else 0.0
        trace.extend(
            {
                "event": "candidate",
                "cake_id": candidate.cake_id,
                "score": candidate.score,
                "coverage": candidate.lexical_coverage,
            }
            for candidate in candidates
        )
        trace.append(
            {
                "event": "decision",
                "selected": [candidate.cake_id for candidate in accepted],
                "top_k": limit,
            }
        )
        selected = tuple(candidate.cake_id for candidate in accepted)
        return RouteResult(
            selected=selected,
            candidates=tuple(candidates),
            confidence=confidence,
            abstained=not selected,
            core_fallback=not selected,
            escalate=False,
            multidomain=len(selected) > 1,
            reason="matched_archive_bound_profile" if selected else "no_suitable_cake",
            policy_version="layercake-catalog-router/1",
            route_milliseconds=(time.perf_counter() - started) * 1000.0,
            trace=tuple(trace),
        )
