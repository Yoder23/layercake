from __future__ import annotations

from typing import Iterable

from layercake.routing.router import CakeRouter


def evaluate_routes(
    router: CakeRouter,
    installed: list[dict],
    examples: Iterable[dict],
) -> dict:
    rows = []
    top1 = topk = false_activation = correct_abstention = 0
    for example in examples:
        expected = set(example.get("expected", []))
        result = router.route(example["prompt"], installed, top_k=example.get("top_k", 2))
        selected = set(result.selected)
        if (result.selected[0] if result.selected else None) in expected:
            top1 += 1
        if expected and selected & expected:
            topk += 1
        if not expected and selected:
            false_activation += 1
        if not expected and result.abstained:
            correct_abstention += 1
        rows.append({
            "prompt": example["prompt"], "expected": sorted(expected),
            "selected": list(result.selected), "confidence": result.confidence,
            "abstained": result.abstained, "route_milliseconds": result.route_milliseconds,
        })
    count = len(rows)
    no_domain = sum(not row["expected"] for row in rows)
    with_domain = count - no_domain
    return {
        "examples": count,
        "route_accuracy": top1 / max(with_domain, 1),
        "top_k_recall": topk / max(with_domain, 1),
        "false_activation_rate": false_activation / max(no_domain, 1),
        "abstention_accuracy": correct_abstention / max(no_domain, 1),
        "mean_route_milliseconds": sum(row["route_milliseconds"] for row in rows) / max(count, 1),
        "rows": rows,
    }
