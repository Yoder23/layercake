from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyStopDecision:
    should_stop: bool
    reason: str = ""
    step: int = 0
    metric: str = "loss"
    value: float | None = None


class EarlyStopper:
    def __init__(self, *, patience: int = 2, min_delta: float = 0.0, divergence_factor: float = 2.0, max_abi_drift: float | None = None):
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_factor = divergence_factor
        self.max_abi_drift = max_abi_drift
        self.history: list[float] = []

    def update(self, value: float, *, step: int, abi_drift: float | None = None) -> EarlyStopDecision:
        if self.max_abi_drift is not None and abi_drift is not None and abi_drift > self.max_abi_drift:
            return EarlyStopDecision(True, "abi_drift", step, "abi_drift", abi_drift)
        if self.history and value > self.history[0] * self.divergence_factor:
            return EarlyStopDecision(True, "loss_diverged", step, "loss", value)
        self.history.append(float(value))
        if len(self.history) > self.patience:
            recent = self.history[-self.patience - 1 :]
            if recent[0] - min(recent[1:]) <= self.min_delta:
                return EarlyStopDecision(True, "no_improvement", step, "loss", value)
        return EarlyStopDecision(False, step=step, value=value)
