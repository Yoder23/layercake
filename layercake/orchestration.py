"""CorticalSwarm-compatible local handoff packet and routing stub."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any


@dataclass(frozen=True)
class Claim:
    text: str
    confidence: float


@dataclass(frozen=True)
class Evidence:
    source: str
    summary: str


@dataclass(frozen=True)
class Uncertainty:
    score: float
    reason: str


@dataclass(frozen=True)
class Contradiction:
    claim_a: str
    claim_b: str
    severity: float


@dataclass(frozen=True)
class RouteDecision:
    model_id: str
    input_mode: str
    active_bricks: tuple[str, ...]
    selected_domain: str | None = None
    route_confidence: float = 0.0


@dataclass(frozen=True)
class NeededDomain:
    name: str
    rationale: str


@dataclass(frozen=True)
class EscalationReason:
    code: str
    detail: str


@dataclass(frozen=True)
class ABIStateSummary:
    dimensions: int
    mean_norm: float
    checksum: str | None = None


@dataclass(frozen=True)
class BytePatchSummary:
    byte_count: int
    patch_count: int
    compression_ratio: float


@dataclass
class HandoffPacket:
    abi_version: str
    input_mode: str
    source_model_id: str
    uncertainty: float
    patching_mode: str | None = None
    active_domain_bricks: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    needed_domain: str | None = None
    escalation_reason: str | None = None
    abi_state: ABIStateSummary | None = None
    byte_patch: BytePatchSummary | None = None
    compressed_state: list[float] | None = None
    parent_hash: str | None = None

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        envelope = {"payload": self.canonical_dict(), "hash": self.compute_hash()}
        return json.dumps(envelope, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "HandoffPacket":
        envelope = json.loads(raw)
        data = envelope["payload"]
        data["claims"] = [Claim(**item) for item in data.get("claims", [])]
        data["evidence"] = [Evidence(**item) for item in data.get("evidence", [])]
        data["contradictions"] = [
            Contradiction(**item) for item in data.get("contradictions", [])
        ]
        if data.get("abi_state"):
            data["abi_state"] = ABIStateSummary(**data["abi_state"])
        if data.get("byte_patch"):
            data["byte_patch"] = BytePatchSummary(**data["byte_patch"])
        packet = cls(**data)
        if packet.compute_hash() != envelope["hash"]:
            raise ValueError("handoff packet hash mismatch")
        return packet

    def validate(self) -> None:
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        if not self.abi_version or not self.source_model_id:
            raise ValueError("ABI version and source model id are required")


class LayerCakeOrchestrator:
    def __init__(self, escalation_threshold: float = 0.6):
        self.escalation_threshold = escalation_threshold

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", text.lower()))

    def _domain_score(self, task_terms: set[str], domain: str, aliases: list[str] | None = None) -> float:
        aliases = aliases or []
        domain_terms = self._terms(domain)
        alias_terms = set()
        for alias in aliases:
            alias_terms.update(self._terms(alias))
        terms = domain_terms | alias_terms
        if not terms:
            return 0.0
        hits = len(task_terms & terms)
        return hits / len(terms)

    def _brick_score(self, task_terms: set[str], brick: dict) -> float:
        aliases = list(brick.get("keywords", [])) + list(brick.get("aliases", []))
        return self._domain_score(task_terms, str(brick.get("domain", "")), aliases)

    def _model_score(self, task_terms: set[str], model: dict) -> float:
        domains = model.get("domains", [])
        if isinstance(domains, str):
            domains = [domains]
        aliases = list(model.get("keywords", [])) + list(model.get("aliases", []))
        scores = [
            self._domain_score(task_terms, str(domain), aliases)
            for domain in domains
        ]
        if not scores:
            return 0.0
        return max(scores)

    def route(
        self,
        task: str,
        models: list[dict],
        bricks: list[dict],
        uncertainty: float,
    ) -> dict:
        task_terms = self._terms(task)
        candidates = [model for model in models if model.get("available", True)]
        if not candidates:
            raise ValueError("no available LayerCake models")
        model_rows = [
            {
                "model": model,
                "score": self._model_score(task_terms, model),
                "cost": float(model.get("cost", float("inf"))),
                "capacity": float(model.get("capacity", model.get("cost", 0.0))),
            }
            for model in candidates
        ]
        best_score = max(row["score"] for row in model_rows)
        domain_matched = best_score > 0.0
        if uncertainty >= self.escalation_threshold:
            selected_row = max(
                [row for row in model_rows if row["score"] == best_score] if domain_matched else model_rows,
                key=lambda row: (row["capacity"], -row["cost"]),
            )
        else:
            selected_row = min(
                [row for row in model_rows if row["score"] == best_score] if domain_matched else model_rows,
                key=lambda row: (row["cost"], -row["capacity"]),
            )
        selected = selected_row["model"]
        brick_rows = [
            {"brick": brick, "score": self._brick_score(task_terms, brick)}
            for brick in bricks
            if brick.get("abi_version") == selected.get("abi_version")
        ]
        brick_rows = [row for row in brick_rows if row["score"] > 0.0]
        brick_rows.sort(key=lambda row: (-row["score"], str(row["brick"].get("id", ""))))
        compatible = [row["brick"]["id"] for row in brick_rows]
        selected_domain = None
        if brick_rows:
            selected_domain = str(brick_rows[0]["brick"].get("domain", ""))
        elif selected.get("domains"):
            domains = selected["domains"]
            if isinstance(domains, str):
                selected_domain = domains
            else:
                selected_domain = str(domains[0])
        return {
            "model_id": selected["id"],
            "input_mode": selected.get("input_mode", "byte_patch"),
            "active_bricks": compatible[: selected.get("top_k", 1)],
            "active_model_count": 1,
            "selected_domain": selected_domain,
            "route_confidence": selected_row["score"],
            "escalate": uncertainty >= self.escalation_threshold,
        }

    def handoff_packet(
        self,
        task: str,
        models: list[dict],
        bricks: list[dict],
        uncertainty: float,
        abi_state: ABIStateSummary | None = None,
        byte_patch: BytePatchSummary | None = None,
    ) -> HandoffPacket:
        decision = self.route(task, models, bricks, uncertainty)
        packet = HandoffPacket(
            abi_version=str(
                next(
                    model.get("abi_version", "")
                    for model in models
                    if model.get("id") == decision["model_id"]
                )
            ),
            input_mode=str(decision["input_mode"]),
            source_model_id=str(decision["model_id"]),
            uncertainty=float(uncertainty),
            active_domain_bricks=list(decision["active_bricks"]),
            needed_domain=decision.get("selected_domain"),
            escalation_reason=(
                "uncertainty_threshold"
                if decision.get("escalate")
                else None
            ),
            abi_state=abi_state,
            byte_patch=byte_patch,
        )
        packet.validate()
        return packet
