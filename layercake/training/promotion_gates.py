"""Predeclared fail-closed promotion gates for bounded research campaigns."""

from __future__ import annotations

import math


def classify_foundation_failure(run: dict, *, fixed_bpb: float, policy: dict) -> list[str]:
    failures: list[str] = []
    if run.get("status") != "PASS" or not math.isfinite(float(run.get("selection_bpb", math.inf))):
        return ["optimization_failure"]
    routing = run.get("routing", {})
    if run.get("routed_candidate"):
        if routing.get("router_collapsed"):
            failures.append("router_collapse")
        if not routing.get("all_experts_meaningfully_trained"):
            failures.append("unused_expert_capacity")
        if float(routing.get("maximum_load_fraction", 1.0)) > float(policy["maximum_router_load_fraction"]):
            failures.append("load_imbalance")
    if float(run["active_fraction"]) > float(policy["maximum_active_fraction"]):
        failures.append("active_compute_exceeded")
    if float(run["selection_bpb"]) > fixed_bpb * float(policy["maximum_selection_bpb_ratio_to_fixed"]):
        failures.append("quality_regression")
    return failures


def eligible_foundation_run(run: dict, *, fixed_bpb: float, policy: dict) -> bool:
    return not classify_foundation_failure(run, fixed_bpb=fixed_bpb, policy=policy)

