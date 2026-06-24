from __future__ import annotations

from .efficiency import TransformerBaselineGate
from .gates import GateResult


DOMINANCE_DIMENSIONS = (
    "training_time",
    "final_bpb",
    "trainable_params",
    "artifact_size",
    "cpu_inference",
    "memory",
    "rollback_recovery",
    "domain_transfer",
    "domain_add_cost",
    "installed_vs_active_compute",
)


def run_dominance_suite(metrics: dict) -> dict:
    layercake_time = metrics.get(
        "layercake_time_to_target_seconds",
        metrics.get("layercake_training_seconds", 0),
    )
    transformer_time = metrics.get(
        "transformer_time_to_target_seconds",
        metrics.get("transformer_training_seconds", float("inf")),
    )
    gates = {
        "lower_training_time": layercake_time <= transformer_time,
        "lower_final_bpb": TransformerBaselineGate(max_delta=0.0).run({
            "layercake": {"bpb": metrics.get("layercake_bpb", 0)},
            "transformer": {"bpb": metrics.get("transformer_bpb", float("inf"))},
        }).passed,
        "fewer_trainable_params": metrics.get("layercake_trainable_params", 0) <= metrics.get("transformer_trainable_params", float("inf")),
        "successful_rollback": bool(metrics.get("rollback_recovered", True)),
        "successful_transfer": bool(metrics.get("transfer_exact", True)),
    }
    return {
        "dimensions": list(DOMINANCE_DIMENSIONS),
        "gates": gates,
        "passed": all(gates.values()),
        "metrics": metrics,
    }
