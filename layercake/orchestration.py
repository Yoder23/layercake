"""CorticalSwarm-compatible local handoff packet and routing stub."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
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

    def route(
        self,
        task: str,
        models: list[dict],
        bricks: list[dict],
        uncertainty: float,
    ) -> dict:
        candidates = sorted(
            (model for model in models if model.get("available", True)),
            key=lambda model: model.get("cost", float("inf")),
        )
        if not candidates:
            raise ValueError("no available LayerCake models")
        selected = candidates[-1] if uncertainty >= self.escalation_threshold else candidates[0]
        compatible = [
            brick["id"]
            for brick in bricks
            if brick.get("abi_version") == selected.get("abi_version")
            and brick.get("domain") in task.lower()
        ]
        return {
            "model_id": selected["id"],
            "input_mode": selected.get("input_mode", "byte_patch"),
            "active_bricks": compatible[: selected.get("top_k", 1)],
            "escalate": uncertainty >= self.escalation_threshold,
        }
