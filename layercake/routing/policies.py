from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoutingBudget:
    max_loaded_bytes: int = 512 * 1024 * 1024
    max_cakes: int = 2
    max_route_milliseconds: float = 5.0
    cold_load_penalty: float = 0.08

    def __post_init__(self) -> None:
        if self.max_loaded_bytes <= 0 or self.max_cakes <= 0:
            raise ValueError("routing memory and cake budgets must be positive")
        if self.max_route_milliseconds <= 0 or self.cold_load_penalty < 0:
            raise ValueError("routing latency must be positive and cold penalty non-negative")


@dataclass(frozen=True)
class CakePermissionPolicy:
    allowed_permissions: frozenset[str] = frozenset()
    allow_unsigned_local: bool = False
    denied_cakes: frozenset[str] = frozenset()

    def permits(self, record: dict) -> tuple[bool, str | None]:
        cake_id = str(record.get("cake_id", ""))
        if cake_id in self.denied_cakes:
            return False, "cake_denied"
        if not record.get("signed", False) and not (
            self.allow_unsigned_local and record.get("trusted_local", False)
        ):
            return False, "untrusted_cake"
        missing = set(record.get("permissions", [])) - set(self.allowed_permissions)
        if missing:
            return False, f"permissions_denied:{','.join(sorted(missing))}"
        return True, None


@dataclass(frozen=True)
class RoutingPolicy:
    version: str = "layercake-router/1"
    activation_threshold: float = 0.34
    abstention_margin: float = 0.08
    escalation_confidence: float = 0.55
    allow_composition: bool = True
    budget: RoutingBudget = field(default_factory=RoutingBudget)
    permissions: CakePermissionPolicy = field(default_factory=CakePermissionPolicy)

    def __post_init__(self) -> None:
        for field_name in (
            "activation_threshold",
            "abstention_margin",
            "escalation_confidence",
        ):
            value = getattr(self, field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be in [0, 1]")
