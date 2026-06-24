from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from pathlib import Path

from .common import load_structured


@dataclass(frozen=True)
class GateResult:
    gate_name: str
    passed: bool
    metric_name: str
    value: Any
    threshold: Any = None
    comparison: str = ""
    details: dict[str, Any] | None = None
    artifact_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Gate:
    name: str

    def run(self, context: dict) -> GateResult:
        raise NotImplementedError


class MaxMetricGate(Gate):
    def __init__(self, name: str, metric: str, threshold: float):
        self.name = name
        self.metric = metric
        self.threshold = threshold

    def run(self, context: dict) -> GateResult:
        value = _lookup(context, self.metric)
        return GateResult(self.name, value <= self.threshold, self.metric, value, self.threshold, "<=")


class MinMetricGate(Gate):
    def __init__(self, name: str, metric: str, threshold: float):
        self.name = name
        self.metric = metric
        self.threshold = threshold

    def run(self, context: dict) -> GateResult:
        value = _lookup(context, self.metric)
        return GateResult(self.name, value >= self.threshold, self.metric, value, self.threshold, ">=")


class RegressionGate(Gate):
    def __init__(self, name: str, metric: str, max_delta: float):
        self.name = name
        self.metric = metric
        self.max_delta = max_delta

    def run(self, context: dict) -> GateResult:
        value = _lookup(context, self.metric)
        parent = _lookup(context, f"parent.{self.metric}")
        delta = value - parent
        return GateResult(self.name, delta <= self.max_delta, self.metric, delta, self.max_delta, "delta<=", {"value": value, "parent": parent})


class ABIHashCompatibilityGate(Gate):
    def __init__(self, name: str = "abi_hash_compatible"):
        self.name = name

    def run(self, context: dict) -> GateResult:
        value = _lookup(context, "commit.abi_hash")
        parent = _lookup(context, "parent_commit.abi_hash")
        return GateResult(self.name, value == parent, "abi_hash", value, parent, "==")


class InputInterfaceCompatibilityGate(Gate):
    def __init__(self, name: str = "input_interface_hash_compatible"):
        self.name = name

    def run(self, context: dict) -> GateResult:
        value = _lookup(context, "commit.input_interface_hash")
        parent = _lookup(context, "parent_commit.input_interface_hash")
        return GateResult(self.name, value == parent, "input_interface_hash", value, parent, "==")


class BytePatchCompatibilityGate(Gate):
    def __init__(self, name: str = "byte_patch_hash_compatible"):
        self.name = name

    def run(self, context: dict) -> GateResult:
        value = _lookup(context, "commit.byte_patch_hash")
        parent = _lookup(context, "parent_commit.byte_patch_hash")
        return GateResult(self.name, value == parent, "byte_patch_hash", value, parent, "==")


class ABIDriftGate(MaxMetricGate):
    pass


class TransferExactnessGate(MaxMetricGate):
    pass


class CrossHostExactnessGate(MaxMetricGate):
    pass


class QuantizationDegradationGate(MaxMetricGate):
    pass


class LatencyRegressionGate(RegressionGate):
    pass


class MemoryRegressionGate(RegressionGate):
    pass


class BytePatchCompressionGate(MinMetricGate):
    pass


class InstalledVsActiveComputeGate(MaxMetricGate):
    pass


class SmokeTaskGate(MinMetricGate):
    pass


class DominanceGate(Gate):
    def __init__(self, name: str = "dominance", required: list[str] | None = None):
        self.name = name
        self.required = required or []

    def run(self, context: dict) -> GateResult:
        gates = _lookup_default(context, "dominance.gates", {})
        required = self.required or sorted(gates)
        failed = [gate for gate in required if not gates.get(gate, False)]
        return GateResult(self.name, not failed, "failed_dominance_gates", len(failed), 0, "==", {"failed": failed})


class ProtectedCapabilityGate(Gate):
    def __init__(self, name: str = "protected_capabilities", config_path: str = "rubrics/protected_capabilities.yaml"):
        self.name = name
        self.config_path = config_path

    def run(self, context: dict) -> GateResult:
        path = Path(self.config_path)
        if not path.exists():
            return GateResult(self.name, False, "protected_config_exists", False, True, "==")
        config = load_structured(path)
        metrics = config.get("protected_metrics", {})
        failures = []
        for metric, threshold in metrics.items():
            value = _lookup_default(context, metric, None)
            if value is not None and value > threshold:
                failures.append({"metric": metric, "value": value, "threshold": threshold})
        return GateResult(self.name, not failures, "protected_failures", len(failures), 0, "==", {"failures": failures})


def gate_from_config(config: dict) -> Gate:
    from .efficiency import (
        ComputeWasteGate,
        PreviewBenefitGate,
        QualityPerStepGate,
        QualityPerTrainableParamGate,
        RollbackRecoveryGate,
        TimeToMetricGate,
        TrainingRegressionGate,
        TransformerBaselineGate,
    )

    kind = config.get("type", "min_metric")
    name = config.get("name", config.get("metric", kind))
    metric = config.get("metric", "score")
    if kind in {"max_metric", "generic_max_metric"}:
        return MaxMetricGate(name, metric, config["threshold"])
    if kind in {"min_metric", "generic_min_metric"}:
        return MinMetricGate(name, metric, config.get("threshold", 0.0))
    if kind == "regression":
        return RegressionGate(name, metric, config.get("max_delta", 0.0))
    if kind in {"abi_hash", "abi_hash_compatibility"}:
        return ABIHashCompatibilityGate(name)
    if kind in {"input_interface_hash", "input_interface_compatibility"}:
        return InputInterfaceCompatibilityGate(name)
    if kind in {"byte_patch_hash", "byte_patch_compatibility"}:
        return BytePatchCompatibilityGate(name)
    if kind == "abi_drift":
        return ABIDriftGate(name, metric, config.get("threshold", config.get("max_delta", 0.0)))
    if kind == "transfer_exactness":
        return TransferExactnessGate(name, metric, config.get("threshold", 0.0))
    if kind == "cross_host_exactness":
        return CrossHostExactnessGate(name, metric, config.get("threshold", 0.0))
    if kind == "quantization_degradation":
        return QuantizationDegradationGate(name, metric, config.get("threshold", 0.0))
    if kind == "latency_regression":
        return LatencyRegressionGate(name, metric, config.get("max_delta", config.get("threshold", 0.0)))
    if kind == "memory_regression":
        return MemoryRegressionGate(name, metric, config.get("max_delta", config.get("threshold", 0.0)))
    if kind == "byte_patch_compression":
        return BytePatchCompressionGate(name, metric, config.get("threshold", 1.0))
    if kind == "installed_vs_active_compute":
        return InstalledVsActiveComputeGate(name, metric, config.get("threshold", 1.0))
    if kind == "smoke_task":
        return SmokeTaskGate(name, metric, config.get("threshold", 0.0))
    if kind == "time_to_metric":
        return TimeToMetricGate(name, metric, config.get("target", config.get("threshold", 0.0)), config.get("max_seconds", 60.0), config.get("max_steps"))
    if kind == "quality_per_step":
        return QualityPerStepGate(name, config.get("min_gain_per_step", config.get("threshold", 0.0)))
    if kind == "quality_per_trainable_param":
        return QualityPerTrainableParamGate(name, config.get("min_gain_per_param", config.get("threshold", 0.0)))
    if kind == "training_regression":
        return TrainingRegressionGate(name, config.get("max_regression", config.get("threshold", 0.0)))
    if kind == "compute_waste":
        return ComputeWasteGate(name, config.get("patience", 2), config.get("min_improvement", 0.0))
    if kind == "rollback_recovery":
        return RollbackRecoveryGate(name)
    if kind == "preview_benefit":
        return PreviewBenefitGate(name, metric, config.get("min_delta", 0.0))
    if kind == "transformer_baseline":
        return TransformerBaselineGate(name, metric, config.get("max_delta", 0.0))
    if kind == "protected_capabilities":
        return ProtectedCapabilityGate(name, config.get("config_path", "rubrics/protected_capabilities.yaml"))
    if kind == "dominance":
        return DominanceGate(name, config.get("required"))
    raise ValueError(f"unknown gate type: {kind}")


def run_gates(configs: list[dict], context: dict) -> list[GateResult]:
    gates = [config if isinstance(config, Gate) else gate_from_config(config) for config in configs]
    return [gate.run(context) for gate in gates]


def _lookup(context: dict, dotted: str):
    current = context
    for part in dotted.split("."):
        current = current[part]
    return current


def _lookup_default(context: dict, dotted: str, default=None):
    current = context
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
