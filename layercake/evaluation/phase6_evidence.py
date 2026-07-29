"""Typed, fail-closed verification for Phase 6 orchestration evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping

from layercake.evaluation.phase5_evidence import validate_phase5_bundle
from layercake.moonshot_campaign import component_hashes


class Phase6EvidenceError(RuntimeError):
    """Raised when raw Phase 6 evidence cannot support promotion."""


CONTRACT = Path("moonshot/phase6_orchestration_preregistration.json")
CONTRACT_SHA256 = "30d04dd01630123C2861D93656D2B4E07D75DBA0B2C77FFEAD2717A0F7C6B48E".lower()
PROFILES = Path("moonshot/phase6_router_profiles.json")
PROFILES_SHA256 = "5397a0f28d145c15ee2aeaf13c021aaabb969b4226d16a2f66f776d2b48c3ec2"
FRAMEWORK = Path("results/moonshot/phase6/framework_freeze.json")
SUITE = Path("results/moonshot/phase6/final_routing_suite.jsonl")
ROUTING = Path("results/moonshot/phase6/raw_runs/routing_decisions.json")
FUNCTIONAL = Path("results/moonshot/phase6/raw_runs/functional_execution.json")
CATALOG = Path("results/moonshot/phase6/raw_runs/catalog_scaling.json")
TIMING = Path("results/moonshot/phase6/raw_runs/mixed_cpu_benchmark.json")
EXTERNAL = Path("results/moonshot/phase6/raw_runs/external_orchestration.json")
GATES = Path("results/moonshot/phase6/raw_runs/gate_observations.json")
PAYLOAD = Path("results/moonshot/phase6/certificate_payload.json")
CERTIFICATE = Path("results/moonshot/phase6/orchestration_certificate.json")
SEEDS = (10601, 10602, 10603)
PACKAGE_PATHS = {
    "python": Path(
        "artifacts/moonshot/phase4/release/"
        "python-token-plan-seed10141-direct-v1.0.0.cake"
    ),
    "sql": Path("artifacts/moonshot/phase5/release/sql-token-plan-v1.0.0.cake"),
    "regex": Path("artifacts/moonshot/phase5/release/regex-token-plan-v1.0.0.cake"),
}
PACKAGE_HASHES = {
    "python": "0585c79bfbea16b1c4165bf0030ba6985b8a8cdeab529cd7afe3f9c76c564ef7",
    "sql": "24efdb68885581318bee3a0b7c3cac0b0fe0b75eefa90070c601384a8bf5e105",
    "regex": "c336a552415b2d6161eb6696338086f89c3c3b6e25949e6cf0a94373e9690f15",
}
QWEN_DIGEST = "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67"


def _path(root: Path, relative: Path | str) -> Path:
    value = (root / relative).resolve()
    try:
        value.relative_to(root.resolve())
    except ValueError as error:
        raise Phase6EvidenceError(f"Phase 6 path escapes repository: {relative}") from error
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(root: Path, relative: Path | str) -> dict[str, Any]:
    path = _path(root, relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase6EvidenceError(f"cannot read Phase 6 evidence {relative}: {error}") from error
    if not isinstance(value, dict):
        raise Phase6EvidenceError(f"{relative} is not a JSON object")
    return value


def _canonical_sha(document: Mapping[str, Any]) -> str:
    payload = {
        key: value for key, value in document.items() if key != "evidence_sha256"
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_self_hash(document: Mapping[str, Any], relative: Path | str) -> None:
    if document.get("evidence_sha256") != _canonical_sha(document):
        raise Phase6EvidenceError(f"{relative} has a stale evidence hash")


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def phase6_evidence_files(root: Path) -> list[Path]:
    relatives = {
        CONTRACT,
        PROFILES,
        FRAMEWORK,
        SUITE,
        ROUTING,
        FUNCTIONAL,
        CATALOG,
        TIMING,
        EXTERNAL,
        GATES,
        PAYLOAD,
        CERTIFICATE,
        *PACKAGE_PATHS.values(),
        Path("moonshot/phase4-direct-token-plan-publisher.public.pem"),
        Path("moonshot/phase5-multidomain-publisher.public.pem"),
    }
    result_dir = _path(root, Path("results/moonshot/phase6"))
    if result_dir.is_dir():
        relatives.update(
            path.relative_to(root)
            for path in result_dir.rglob("*")
            if path.is_file()
            and path.name
            not in {
                "candidate.json",
                "candidate_verification.json",
                "release_certificate.json",
                "handoff.json",
                "seal.json",
            }
        )
    return [_path(root, relative) for relative in sorted(relatives)]


def _validate_framework(root: Path) -> dict[str, Any]:
    if _sha256(_path(root, CONTRACT)) != CONTRACT_SHA256:
        raise Phase6EvidenceError("Phase 6 preregistration changed")
    if _sha256(_path(root, PROFILES)) != PROFILES_SHA256:
        raise Phase6EvidenceError("Phase 6 routing profiles changed")
    framework = _read(root, FRAMEWORK)
    _validate_self_hash(framework, FRAMEWORK)
    if (
        framework.get("format") != "layercake-phase6-framework-freeze/1"
        or framework.get("status") != "FROZEN"
        or framework.get("contract_sha256") != CONTRACT_SHA256
        or framework.get("profiles_sha256") != PROFILES_SHA256
        or framework.get("suite_present_at_freeze") is not False
        or framework.get("raw_evidence_present_at_freeze") is not False
    ):
        raise Phase6EvidenceError("Phase 6 framework freeze is invalid")
    commit = framework.get("framework_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise Phase6EvidenceError("Phase 6 framework commit is malformed")
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if not exists:
        raise Phase6EvidenceError("Phase 6 framework commit does not exist")
    release = _read(root, Path("results/moonshot/phase6/release_certificate.json"))
    matrix = _read(root, Path("moonshot/invalidation_matrix.yaml"))
    expected_components = release.get("component_hashes")
    if not isinstance(expected_components, dict) or not expected_components:
        raise Phase6EvidenceError("Phase 6 release lacks its component snapshot")
    actual_components = component_hashes(root, matrix)
    invalidating = sorted(
        name
        for name, policy in matrix.get("components", {}).items()
        if policy.get("invalidates_from_phase", 0) <= 6
        and expected_components.get(name) != actual_components.get(name)
    )
    if invalidating:
        raise Phase6EvidenceError(
            f"Phase 6 dependent components changed after freeze: {invalidating}"
        )
    return framework


def _validate_packages(root: Path) -> None:
    for domain, relative in PACKAGE_PATHS.items():
        if _sha256(_path(root, relative)) != PACKAGE_HASHES[domain]:
            raise Phase6EvidenceError(f"sealed {domain} package changed")


def _validate_suite(root: Path) -> list[dict[str, Any]]:
    path = _path(root, SUITE)
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise Phase6EvidenceError(f"cannot read Phase 6 final suite: {error}") from error
    if not all(
        isinstance(row, dict)
        and row.get("format") == "layercake-phase6-routing-row/1"
        and row.get("seed") in SEEDS
        for row in rows
    ):
        raise Phase6EvidenceError("Phase 6 final suite contains invalid rows")
    expected_counts = {
        "top1": 360,
        "topk": 120,
        "core": 120,
        "adversarial_control": 60,
    }
    for seed in SEEDS:
        counts = {
            category: sum(
                row["seed"] == seed and row["category"] == category for row in rows
            )
            for category in expected_counts
        }
        if counts != expected_counts:
            raise Phase6EvidenceError(f"Phase 6 suite counts failed for seed {seed}: {counts}")
    if len({row["id"] for row in rows}) != len(rows):
        raise Phase6EvidenceError("Phase 6 suite identifiers are not unique")
    return rows


def _validate_routing(
    root: Path, suite: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, float]]:
    routing = _read(root, ROUTING)
    _validate_self_hash(routing, ROUTING)
    records = routing.get("records")
    if (
        routing.get("format") != "layercake-phase6-routing-decisions/1"
        or routing.get("status") != "RAW"
        or routing.get("profiles_sha256") != PROFILES_SHA256
        or routing.get("suite", {}).get("sha256") != _sha256(_path(root, SUITE))
        or not isinstance(records, list)
        or len(records) != len(suite)
    ):
        raise Phase6EvidenceError("Phase 6 routing evidence identity is invalid")
    suite_by_id = {row["id"]: row for row in suite}
    for record in records:
        source = suite_by_id.get(record.get("id"))
        if source is None or record.get("seed") != source["seed"]:
            raise Phase6EvidenceError("routing record is not bound to the final suite")
        expected = sorted(source["expected"])
        selected = sorted(record.get("selected", []))
        if (
            record.get("expected") != expected
            or record.get("category") != source["category"]
            or record.get("prompt_sha256")
            != hashlib.sha256(source["prompt"].encode("utf-8")).hexdigest()
            or record.get("correct") is not (selected == expected)
            or record.get("required_hits") != len(set(expected) & set(selected))
            or record.get("required_total") != len(expected)
            or record.get("false_activation") != int(not expected and bool(selected))
        ):
            raise Phase6EvidenceError(f"routing record is internally inconsistent: {record.get('id')}")
        if not isinstance(record.get("route_milliseconds"), (int, float)):
            raise Phase6EvidenceError("routing latency is missing")
    top1 = [row for row in records if row["category"] == "top1"]
    topk = [row for row in records if row["category"] == "topk"]
    negative = [
        row
        for row in records
        if row["category"] in {"core", "adversarial_control"}
    ]
    per_seed: dict[str, float] = {}
    for seed in SEEDS:
        seeded_top1 = [row for row in top1 if row["seed"] == seed]
        seeded_topk = [row for row in topk if row["seed"] == seed]
        seeded_negative = [row for row in negative if row["seed"] == seed]
        per_seed[f"top1_accuracy:{seed}"] = sum(
            row["correct"] for row in seeded_top1
        ) / len(seeded_top1)
        per_seed[f"topk_recall:{seed}"] = sum(
            row["required_hits"] for row in seeded_topk
        ) / sum(row["required_total"] for row in seeded_topk)
        per_seed[f"false_specialist_activation:{seed}"] = sum(
            row["false_activation"] for row in seeded_negative
        ) / len(seeded_negative)
    metrics = {
        "top1_accuracy": sum(row["correct"] for row in top1) / len(top1),
        "topk_recall": sum(row["required_hits"] for row in topk)
        / sum(row["required_total"] for row in topk),
        "false_specialist_activation": sum(
            row["false_activation"] for row in negative
        )
        / len(negative),
    }
    return routing, {**metrics, **per_seed}


def _validate_functional(root: Path) -> tuple[dict[str, Any], dict[str, float]]:
    document = _read(root, FUNCTIONAL)
    _validate_self_hash(document, FUNCTIONAL)
    records = document.get("records")
    compositions = document.get("compositions")
    if (
        document.get("format") != "layercake-phase6-functional-execution/1"
        or document.get("status") != "RAW"
        or document.get("package_hashes") != PACKAGE_HASHES
        or not isinstance(records, list)
        or not isinstance(compositions, list)
        or len(records) != 3 * 128 * 2
        or len(compositions) != 20
    ):
        raise Phase6EvidenceError("Phase 6 functional evidence identity is invalid")
    for domain in PACKAGE_HASHES:
        for mode in ("automatic_top1", "manual"):
            selected = [
                row
                for row in records
                if row.get("domain") == domain and row.get("mode") == mode
            ]
            if len(selected) != 128:
                raise Phase6EvidenceError(f"{domain}/{mode} lacks 128 functional rows")
    if any(
        row.get("functional_success") is not True
        or row.get("output_matches_frozen_response") is not True
        or row.get("inactive_forward_calls") != 0
        or row.get("selected") != [row.get("expected")]
        or (
            row.get("mode") == "manual"
            and row.get("automatic_output_equal") is not True
        )
        for row in records
    ):
        raise Phase6EvidenceError("functional retention or sparse execution failed")
    if any(
        row.get("functional_success") is not True
        or row.get("structured_output_count") != 2
        or row.get("inactive_forward_calls") != 0
        or row.get("latent_neural_fusion_claimed") is not False
        for row in compositions
    ):
        raise Phase6EvidenceError("structured multidomain composition failed")
    automatic = [row for row in records if row["mode"] == "automatic_top1"]
    manual = [row for row in records if row["mode"] == "manual"]
    return document, {
        "routed_functional_success_retention": sum(
            row["functional_success"] for row in automatic
        )
        / len(automatic),
        "manual_functional_success_retention": sum(
            row["functional_success"] for row in manual
        )
        / len(manual),
        "inactive_cake_forward_calls": float(
            sum(row["inactive_forward_calls"] for row in records)
            + sum(row["inactive_forward_calls"] for row in compositions)
        ),
        "sealed_package_hash_retention": 1.0,
    }


def _validate_catalog(root: Path) -> tuple[dict[str, Any], float]:
    document = _read(root, CATALOG)
    _validate_self_hash(document, CATALOG)
    records = document.get("records")
    if (
        document.get("format") != "layercake-phase6-catalog-scaling/1"
        or document.get("status") != "RAW"
        or document.get("claim_boundary", {}).get("real_promoted_capabilities") != 3
        or document.get("claim_boundary", {}).get(
            "management_descriptors_are_capabilities"
        )
        is not False
        or not isinstance(records, list)
        or [row.get("catalog_size") for row in records]
        != [3, 10, 25, 50, 100, 250, 500]
    ):
        raise Phase6EvidenceError("catalog evidence identity is invalid")
    if any(
        row.get("promoted_real_capabilities") != 3
        or row.get("installed_real_capabilities") != 3
        or row.get("management_only_descriptors") != row["catalog_size"] - 3
        or row.get("search_observations") != 100
        for row in records
    ):
        raise Phase6EvidenceError("catalog evidence relabels management descriptors")
    return document, float(max(row["catalog_size"] for row in records))


def _validate_timing(root: Path) -> tuple[dict[str, Any], dict[str, float]]:
    document = _read(root, TIMING)
    _validate_self_hash(document, TIMING)
    records = document.get("records")
    protocol = document.get("protocol", {})
    if (
        document.get("format") != "layercake-phase6-mixed-cpu-benchmark/1"
        or document.get("status") != "RAW"
        or not isinstance(records, list)
        or len(records) != 120
        or protocol.get("distinct_prompts") != 100
        or protocol.get("repeated_prompt_observations") != 20
        or protocol.get("qwen_num_gpu") != 0
        or protocol.get("qwen_cpu_threads") != 14
        or protocol.get("qwen_digest") != QWEN_DIGEST
    ):
        raise Phase6EvidenceError("mixed CPU benchmark identity is invalid")
    if len({row["prompt_sha256"] for row in records[:100]}) != 100:
        raise Phase6EvidenceError("mixed CPU benchmark lacks 100 distinct prompts")
    if any(
        row.get("domain") not in {"python", "sql", "regex"}
        or row.get("layercake", {}).get("functional_success") is not True
        or row.get("layercake", {}).get("selected")
        != [f"{row['domain']}-token-plan"]
        or row.get("qwen", {}).get("authoritative_generated_tokens", 0) <= 0
        or row.get("qwen", {}).get("generated_bytes", 0) <= 0
        or not _close(
            row.get("paired_throughput_ratio", 0.0),
            row["layercake"]["timing"]["bytes_per_second_total"]
            / row["qwen"]["timing"]["bytes_per_second_total"],
        )
        for row in records
    ):
        raise Phase6EvidenceError("mixed CPU raw observations are inconsistent")
    ratios = [float(row["paired_throughput_ratio"]) for row in records]
    route_median = statistics.median(
        row["layercake"]["timing"]["route_seconds"] for row in records
    )
    total_median = statistics.median(
        row["layercake"]["timing"]["total_latency_seconds"] for row in records
    )
    median_ratio = statistics.median(ratios)
    aggregates = document.get("aggregates", {})
    if (
        not _close(aggregates.get("median_paired_throughput_ratio", -1), median_ratio)
        or aggregates.get("layercake_functional_successes") != 100
    ):
        raise Phase6EvidenceError("mixed CPU aggregates are stale")
    return document, {
        "routing_warm_latency_fraction": route_median / total_median,
        "mixed_workload_cpu_transformer_ratio": median_ratio,
    }


def _validate_external(root: Path) -> tuple[dict[str, Any], float]:
    document = _read(root, EXTERNAL)
    _validate_self_hash(document, EXTERNAL)
    external = document.get("external", {})
    external_result = external.get("result", {})
    internal = document.get("internal_comparable", {})
    equivalent = (
        document.get("format") == "layercake-phase6-external-orchestration/1"
        and document.get("status") == "RAW"
        and document.get("returncode") == 0
        and external.get("status") == "PASS"
        and external_result.get("selected") == internal.get("selected")
        and external_result.get("output") == internal.get("output")
        and external_result.get("execution_path") == internal.get("execution_path")
    )
    if document.get("equivalent") is not equivalent or not equivalent:
        raise Phase6EvidenceError("external/internal orchestration equivalence failed")
    return document, 1.0


def _validate_derived(
    root: Path, metrics: Mapping[str, float], framework: Mapping[str, Any]
) -> None:
    gates = _read(root, GATES)
    payload = _read(root, PAYLOAD)
    certificate = _read(root, CERTIFICATE)
    for relative, document in (
        (GATES, gates),
        (PAYLOAD, payload),
        (CERTIFICATE, certificate),
    ):
        _validate_self_hash(document, relative)
    if (
        gates.get("format") != "layercake-phase6-gate-observations/1"
        or gates.get("source_commit") != framework["framework_commit"]
        or payload.get("format") != "layercake-phase6-certificate-payload/1"
        or payload.get("status") != "PASS"
        or payload.get("router", {}).get("profiles_sha256") != PROFILES_SHA256
        or certificate.get("format")
        != "layercake-phase6-orchestration-certificate/1"
        or certificate.get("status") != "PASS"
        or certificate.get("profiles_sha256") != PROFILES_SHA256
        or certificate.get("suite_sha256") != _sha256(_path(root, SUITE))
    ):
        raise Phase6EvidenceError("Phase 6 derived evidence identity is invalid")
    records = gates.get("records", [])
    if {row.get("seed") for row in records} != set(SEEDS):
        raise Phase6EvidenceError("gate observations lack three seeds")
    for gate_id, value in metrics.items():
        observations = [
            float(row["value"])
            for row in records
            if row.get("gate_id") == gate_id
        ]
        if len(observations) != 3 or any(not _close(item, value) for item in observations):
            raise Phase6EvidenceError(f"stale gate observation: {gate_id}")
    claims = {row["gate_id"]: row for row in payload.get("claims", [])}
    required = {
        "top1_accuracy",
        "topk_recall",
        "false_specialist_activation",
        "routing_warm_latency_fraction",
        "largest_catalog_size",
        "mixed_workload_cpu_transformer_ratio",
    }
    if set(claims) != required:
        raise Phase6EvidenceError("Phase 6 payload has missing or extra promoted claims")
    for gate_id in required:
        if not _close(claims[gate_id]["value"], metrics[gate_id]):
            raise Phase6EvidenceError(f"payload claim is stale: {gate_id}")
    for key, value in payload.get("additional_gates", {}).items():
        if key not in metrics or not _close(value, metrics[key]):
            raise Phase6EvidenceError(f"additional gate is stale: {key}")
    if certificate.get("metrics") != dict(metrics):
        raise Phase6EvidenceError("Phase 6 certificate metrics are stale")


def derive_phase6_metrics(root: Path) -> dict[str, Any]:
    framework = _validate_framework(root)
    _validate_packages(root)
    suite = _validate_suite(root)
    _, routing_metrics = _validate_routing(root, suite)
    _, functional_metrics = _validate_functional(root)
    _, largest_catalog = _validate_catalog(root)
    _, timing_metrics = _validate_timing(root)
    _, external_equivalence = _validate_external(root)
    validate_phase5_bundle(root, _path(root, "results/moonshot/phase5"))
    metrics = {
        "top1_accuracy": routing_metrics["top1_accuracy"],
        "topk_recall": routing_metrics["topk_recall"],
        "false_specialist_activation": routing_metrics[
            "false_specialist_activation"
        ],
        **timing_metrics,
        "largest_catalog_size": largest_catalog,
        **functional_metrics,
        "external_internal_equivalence": external_equivalence,
        "phase5_dependent_gate_retention": 1.0,
    }
    _validate_derived(root, metrics, framework)
    thresholds = _read(root, CONTRACT)["promotion_gates"]
    if not (
        metrics["top1_accuracy"] >= thresholds["top1_accuracy"]
        and metrics["topk_recall"] >= thresholds["topk_recall"]
        and metrics["false_specialist_activation"]
        <= thresholds["false_specialist_activation"]
        and metrics["routing_warm_latency_fraction"]
        <= thresholds["routing_warm_latency_fraction"]
        and metrics["largest_catalog_size"] >= thresholds["largest_catalog_size"]
        and metrics["mixed_workload_cpu_transformer_ratio"]
        >= thresholds["mixed_workload_cpu_transformer_ratio"]
        and metrics["routed_functional_success_retention"] == 1.0
        and metrics["manual_functional_success_retention"] == 1.0
        and metrics["inactive_cake_forward_calls"] == 0.0
        and metrics["sealed_package_hash_retention"] == 1.0
        and metrics["external_internal_equivalence"] == 1.0
        and metrics["phase5_dependent_gate_retention"] == 1.0
    ):
        raise Phase6EvidenceError(f"Phase 6 promotion gates failed: {metrics}")
    return {
        "status": "PASS",
        "framework_commit": framework["framework_commit"],
        "profiles_sha256": PROFILES_SHA256,
        "suite_sha256": _sha256(_path(root, SUITE)),
        "metrics": metrics,
        "seed_metrics": {
            key: value for key, value in routing_metrics.items() if ":" in key
        },
        "claim_boundaries": {
            "real_promoted_neural_capabilities": 3,
            "management_only_catalog_descriptors": 497,
            "latent_neural_fusion_claimed": False,
            "hundred_domain_quality_claimed": False,
        },
    }


def validate_phase6_bundle(root: Path, evidence_dir: Path) -> dict[str, Any]:
    expected = _path(root, "results/moonshot/phase6")
    if evidence_dir.resolve() != expected:
        raise Phase6EvidenceError("Phase 6 evidence directory is not canonical")
    return derive_phase6_metrics(root)
