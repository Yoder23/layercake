"""Catalog-size stress test separated from real-domain capability evidence."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from safetensors.torch import load_file

from layercake.routing.learned_router import CompactSemanticRouter, DOMAINS


CATALOG_SIZES = (0, 1, 5, 10, 50, 100)


def benchmark_catalog_scaling(
    router_path: str | Path,
    cake_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Measure registry and route overhead without pretending synthetic cakes work."""
    router = CompactSemanticRouter().eval()
    router.load_state_dict(load_file(str(router_path)), strict=True)
    cake_bytes = Path(cake_path).stat().st_size
    rows = []
    for size in CATALOG_SIZES:
        descriptors = []
        for index in range(size):
            domain = DOMAINS[index] if index < len(DOMAINS) else f"synthetic-{index:03d}"
            descriptors.append({
                "cake_id": f"{domain}-stress-v1",
                "domain": domain,
                "content_hash": "same-content-addressed-blob" if index else "real-python-blob",
                "real_trained": index == 0,
            })
        started = time.perf_counter_ns()
        registry = {row["cake_id"]: row for row in descriptors}
        domain_index = {row["domain"]: row["cake_id"] for row in descriptors}
        startup_ms = (time.perf_counter_ns() - started) / 1_000_000
        lookup_samples = []
        for index in range(1000):
            key = descriptors[index % size]["cake_id"] if size else "missing"
            before = time.perf_counter_ns()
            registry.get(key)
            lookup_samples.append((time.perf_counter_ns() - before) / 1_000_000)
        installed = set(domain_index) & set(DOMAINS)
        prompt = "Implement a bounded asynchronous worker pool in Python."
        route_samples = []
        selected = ()
        for _ in range(200):
            before = time.perf_counter_ns()
            route = router.route(prompt, installed=installed)
            route_samples.append((time.perf_counter_ns() - before) / 1_000_000)
            selected = route.selected
        descriptor_bytes = len(json.dumps(descriptors, sort_keys=True).encode("utf-8"))
        rows.append({
            "catalog_size": size,
            "real_trained_packages": min(size, 1),
            "synthetic_stress_descriptors": max(0, size - 1),
            "descriptor_bytes": descriptor_bytes,
            "content_addressed_blob_bytes": cake_bytes if size else 0,
            "startup_milliseconds": startup_ms,
            "registry_lookup_p50_milliseconds": statistics.median(lookup_samples),
            "registry_lookup_p95_milliseconds": sorted(lookup_samples)[949],
            "route_p50_milliseconds": statistics.median(route_samples),
            "route_p95_milliseconds": sorted(route_samples)[189],
            "selected": list(selected),
            "active_cakes": min(1, len(selected)),
            "active_package_bytes": cake_bytes if selected else 0,
        })
    baseline = next(row for row in rows if row["catalog_size"] == 5)
    largest = rows[-1]
    stress_pass = bool(
        largest["registry_lookup_p95_milliseconds"] < 1.0
        and largest["route_p95_milliseconds"] <= max(
            5.0, 2.0 * baseline["route_p95_milliseconds"]
        )
        and largest["active_cakes"] <= 1
    )
    result = {
        "format": "layercake-final-catalog-scaling/1",
        "status": "PASS" if stress_pass else "FAIL",
        "promotion_status": "INVALID_EVIDENCE",
        "promotion_reason": (
            "Only the Python package is a real trained cake; synthetic descriptors are permitted "
            "for management stress but cannot prove many-domain capability."
        ),
        "catalog_sizes": list(CATALOG_SIZES),
        "route_model_parameters": sum(parameter.numel() for parameter in router.parameters()),
        "rows": rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
