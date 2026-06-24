from __future__ import annotations

from .gates import Gate, GateResult


def lookup(context: dict, dotted: str, default=None):
    current = context
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


class TimeToMetricGate(Gate):
    def __init__(self, name="time_to_metric", metric="bpb", target=2.0, max_seconds=60.0, max_steps=None):
        self.name, self.metric, self.target, self.max_seconds, self.max_steps = name, metric, target, max_seconds, max_steps

    def run(self, context: dict) -> GateResult:
        value = lookup(context, self.metric)
        seconds = lookup(context, "training_seconds", 0.0)
        steps = lookup(context, "steps", 0)
        passed = value is not None and value <= self.target and seconds <= self.max_seconds
        if self.max_steps is not None:
            passed = passed and steps <= self.max_steps
        return GateResult(self.name, passed, self.metric, value, self.target, "<=", {"seconds": seconds, "steps": steps})


class QualityPerStepGate(Gate):
    def __init__(self, name="quality_per_step", min_gain_per_step=0.0):
        self.name, self.min_gain_per_step = name, min_gain_per_step

    def run(self, context: dict) -> GateResult:
        gain = lookup(context, "parent.bpb", 0.0) - lookup(context, "bpb", 0.0)
        steps = max(lookup(context, "steps", 1), 1)
        value = gain / steps
        return GateResult(self.name, value >= self.min_gain_per_step, "gain_per_step", value, self.min_gain_per_step, ">=")


class QualityPerTrainableParamGate(Gate):
    def __init__(self, name="quality_per_trainable_param", min_gain_per_param=0.0):
        self.name, self.min_gain_per_param = name, min_gain_per_param

    def run(self, context: dict) -> GateResult:
        gain = lookup(context, "parent.bpb", 0.0) - lookup(context, "bpb", 0.0)
        params = max(lookup(context, "trainable_params", 1), 1)
        value = gain / params
        return GateResult(self.name, value >= self.min_gain_per_param, "gain_per_param", value, self.min_gain_per_param, ">=")


class TrainingRegressionGate(Gate):
    def __init__(self, name="training_regression", max_regression=0.0):
        self.name, self.max_regression = name, max_regression

    def run(self, context: dict) -> GateResult:
        regression = lookup(context, "bpb", 0.0) - lookup(context, "parent.bpb", 0.0)
        return GateResult(self.name, regression <= self.max_regression, "bpb_regression", regression, self.max_regression, "<=")


class ComputeWasteGate(Gate):
    def __init__(self, name="compute_waste", patience=2, min_improvement=0.0):
        self.name, self.patience, self.min_improvement = name, patience, min_improvement

    def run(self, context: dict) -> GateResult:
        history = lookup(context, "loss_history", []) or []
        if len(history) <= self.patience:
            return GateResult(self.name, True, "stalled_steps", 0, self.patience, "<=")
        recent = history[-self.patience - 1 :]
        improvement = recent[0] - min(recent[1:])
        passed = improvement > self.min_improvement
        return GateResult(self.name, passed, "recent_improvement", improvement, self.min_improvement, ">")


class RollbackRecoveryGate(Gate):
    def __init__(self, name="rollback_recovery"):
        self.name = name

    def run(self, context: dict) -> GateResult:
        value = bool(lookup(context, "rollback.exact_artifact_rollback", lookup(context, "rollback.restored_commit", None)))
        return GateResult(self.name, value, "rollback_recovered", value, True, "==")


class PreviewBenefitGate(Gate):
    def __init__(self, name="preview_benefit", metric="bpb", min_delta=0.0):
        self.name, self.metric, self.min_delta = name, metric, min_delta

    def run(self, context: dict) -> GateResult:
        guided = lookup(context, f"preview_guided.{self.metric}")
        blind = lookup(context, f"blind.{self.metric}")
        delta = blind - guided if guided is not None and blind is not None else 0.0
        return GateResult(self.name, delta >= self.min_delta, f"delta_{self.metric}", delta, self.min_delta, ">=")


class TransformerBaselineGate(Gate):
    def __init__(self, name="transformer_baseline", metric="bpb", max_delta=0.0):
        self.name, self.metric, self.max_delta = name, metric, max_delta

    def run(self, context: dict) -> GateResult:
        lc = lookup(context, f"layercake.{self.metric}", lookup(context, self.metric))
        baseline = lookup(context, f"transformer.{self.metric}", lookup(context, "transformer_baseline_bpb"))
        delta = lc - baseline if lc is not None and baseline is not None else 0.0
        return GateResult(self.name, delta <= self.max_delta, f"delta_vs_transformer_{self.metric}", delta, self.max_delta, "<=")
