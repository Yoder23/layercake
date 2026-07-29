"""Independent, fail-closed verification for the final Phase 8 release."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any, Mapping

from layercake.evaluation.phase2_r3_evidence import validate_phase2_r3_bundle
from layercake.evaluation.phase3_retirement_evidence import (
    validate_phase3_retirement_bundle,
)
from layercake.evaluation.phase4_evidence import validate_phase4_bundle
from layercake.evaluation.phase5_evidence import validate_phase5_bundle
from layercake.evaluation.phase6_evidence import validate_phase6_bundle
from layercake.evaluation.phase7_evidence import validate_phase7_bundle
from layercake.moonshot_campaign import component_hashes


class Phase8EvidenceError(RuntimeError):
    """Raised when the final independent proof does not survive verification."""


CONTRACT = Path(
    "moonshot/phase8_independent_verification_preregistration.json"
)
CONTRACT_SHA256 = (
    "5426d2ff19a7f40b8fd3588567796abe5199e7da721add3486584d9929c72830"
)
FRAMEWORK = Path("results/moonshot/phase8/framework_freeze.json")
SOURCE_AUDIT = Path("results/moonshot/phase8/source_audit.json")
ENVIRONMENT = Path(
    "results/moonshot/phase8/raw_runs/cleanroom_environment.json"
)
PERFORMANCE = Path(
    "results/moonshot/phase8/raw_runs/reproduction_performance.json"
)
DOMAINS = Path("results/moonshot/phase8/raw_runs/domain_retention.json")
LIFECYCLE = Path(
    "results/moonshot/phase8/raw_runs/lifecycle_portability.json"
)
ROUTING = Path("results/moonshot/phase8/raw_runs/routing_catalog.json")
ADVERSARIAL = Path(
    "results/moonshot/phase8/raw_runs/adversarial_falsification.json"
)
PRIOR = Path(
    "results/moonshot/phase8/raw_runs/prior_gate_recomputation.json"
)
DATA_HASHES = Path("results/moonshot/phase8/manifests/data_hashes.json")
CHECKPOINT_HASHES = Path(
    "results/moonshot/phase8/manifests/checkpoint_hashes.json"
)
PACKAGE_HASHES = Path(
    "results/moonshot/phase8/manifests/package_hashes.json"
)
SOURCE_SCAN = Path("results/moonshot/phase8/manifests/source_scan.json")
GATES = Path("results/moonshot/phase8/raw_runs/gate_observations.json")
PAYLOAD = Path("results/moonshot/phase8/certificate_payload.json")
CERTIFICATE = Path(
    "results/moonshot/phase8/independent_verification_certificate.json"
)
REPORT = Path("results/moonshot/phase8/release_report.md")
FINAL_CERTIFICATE = Path("results/moonshot/final/release_certificate.json")
FINAL_REPORT = Path("results/moonshot/final/release_report.md")
LEGACY_CERTIFICATE = Path(
    "results/moonshot/final/history/"
    "pre_gated_campaign_ec4d074a5740/release_certificate.json"
)
LEGACY_CERTIFICATE_SHA256 = (
    "ec4d074a57401f126f6140d0938502967b022b5081ff118ebe72b03ce8b4710e"
)
SEEDS = (10801, 10802, 10803)
PARENT_TAG = "layercake-moonshot-phase7"
PARENT_COMMIT = "62aa899d1f3b50eec067f3621ba8390271837c8b"
QWEN_DIGEST = (
    "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67"
)
ABI_HASH = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)
PACKAGES = {
    "python": (
        Path(
            "artifacts/moonshot/phase4/release/"
            "python-token-plan-seed10141-direct-v1.0.0.cake"
        ),
        "0585c79bfbea16b1c4165bf0030ba6985b8a8cdeab529cd7afe3f9c76c564ef7",
    ),
    "sql": (
        Path(
            "artifacts/moonshot/phase5/release/"
            "sql-token-plan-v1.0.0.cake"
        ),
        "24efdb68885581318bee3a0b7c3cac0b0fe0b75eefa90070c601384a8bf5e105",
    ),
    "regex": (
        Path(
            "artifacts/moonshot/phase5/release/"
            "regex-token-plan-v1.0.0.cake"
        ),
        "c336a552415b2d6161eb6696338086f89c3c3b6e25949e6cf0a94373e9690f15",
    ),
}
CHECKPOINTS = {
    "seed-9824": (
        Path(
            "artifacts/moonshot/phase2_shallow_sparse_pretrained/"
            "student2400-seed-9824/model.safetensors"
        ),
        "9e0e6b9add32b4c460f7b570a32584f380e59bf6d631e313ff813069d24e09e1",
    ),
    "seed-9825": (
        Path(
            "artifacts/moonshot/phase2_shallow_sparse_pretrained/"
            "student2400-seed-9825/model.safetensors"
        ),
        "81f78a2154353f0ea3c2a4ff685c3e4ffb91876328cd0075559a11bd0d1c2a01",
    ),
    "seed-9826": (
        Path(
            "artifacts/moonshot/phase2_shallow_sparse_pretrained/"
            "student2400-seed-9826/model.safetensors"
        ),
        "f2987c0629460f2050489dda07e3d660e80f48d3c19f1574d51477ce8bdcbf1d",
    ),
}
REQUIRED_ATTACK_CATEGORIES = {
    "raw_evidence_and_certificate_mutation",
    "package_payload_tampering",
    "signature_and_trust_confusion",
    "archive_structure_and_path_traversal",
    "abi_and_profile_mismatch",
    "registry_blob_corruption",
    "router_control_injection",
    "unsigned_or_permissioned_auto_activation",
    "inactive_cake_execution",
    "receiver_training_or_calibration",
    "core_or_package_mutation",
    "hidden_retrieval_templates_or_stored_answers",
    "hard_coded_benchmark_outputs_or_verifier_values",
    "data_split_overlap_or_prompt_duplication",
    "failed_seed_omission",
    "baseline_runtime_or_gpu_residency",
    "timing_and_token_accounting",
    "incompatible_memory_accounting",
    "private_key_or_secret_leakage",
    "one_lineage_and_stale_artifacts",
    "uninstall_reinstall_identity",
    "cross_host_semantic_retention",
    "dynamic_catalog_discovery",
    "claim_boundary_overreach",
}


def _path(root: Path, relative: Path | str) -> Path:
    value = (root / relative).resolve()
    try:
        value.relative_to(root.resolve())
    except ValueError as error:
        raise Phase8EvidenceError(
            f"Phase 8 path escapes repository: {relative}"
        ) from error
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(root: Path, relative: Path | str) -> dict[str, Any]:
    path = _path(root, relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase8EvidenceError(
            f"cannot read Phase 8 evidence {relative}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise Phase8EvidenceError(f"{relative} is not a JSON object")
    return value


def _canonical_sha(document: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in document.items()
        if key != "evidence_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _validate_self_hash(
    document: Mapping[str, Any], relative: Path | str
) -> None:
    if document.get("evidence_sha256") != _canonical_sha(document):
        raise Phase8EvidenceError(f"{relative} has a stale evidence hash")


def _close(left: float, right: float) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _bootstrap_mean(
    values: list[float], *, seed: int, samples: int = 5000
) -> list[float]:
    generator = random.Random(seed)
    estimates = [
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    estimates.sort()
    return [
        estimates[math.floor(0.025 * (samples - 1))],
        estimates[math.ceil(0.975 * (samples - 1))],
    ]


def phase8_evidence_files(root: Path) -> list[Path]:
    relatives = {
        CONTRACT,
        FRAMEWORK,
        SOURCE_AUDIT,
        ENVIRONMENT,
        PERFORMANCE,
        DOMAINS,
        LIFECYCLE,
        ROUTING,
        ADVERSARIAL,
        PRIOR,
        DATA_HASHES,
        CHECKPOINT_HASHES,
        PACKAGE_HASHES,
        SOURCE_SCAN,
        GATES,
        PAYLOAD,
        CERTIFICATE,
        REPORT,
        FINAL_CERTIFICATE,
        FINAL_REPORT,
        LEGACY_CERTIFICATE,
        *[value[0] for value in PACKAGES.values()],
    }
    phase = _path(root, "results/moonshot/phase8")
    if phase.is_dir():
        relatives.update(
            path.relative_to(root)
            for path in phase.rglob("*")
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
    final = _path(root, "results/moonshot/final")
    if final.is_dir():
        for directory in (
            "data_hashes",
            "checkpoint_hashes",
            "package_hashes",
            "training",
            "quality",
            "samples",
            "domains",
            "portability",
            "routing",
            "catalog_scaling",
            "cpu_vs_cpu",
            "gpu_vs_gpu",
            "cpu_vs_gpu",
            "mobile",
            "independent_verification",
            "raw_runs",
            "history/pre_gated_campaign_ec4d074a5740",
        ):
            base = final / directory
            if base.is_dir():
                relatives.update(
                    path.relative_to(root)
                    for path in base.rglob("*")
                    if path.is_file()
                )
    return [_path(root, relative) for relative in sorted(relatives)]


def _validate_framework(root: Path) -> dict[str, Any]:
    if _sha256(_path(root, CONTRACT)) != CONTRACT_SHA256:
        raise Phase8EvidenceError("Phase 8 preregistration changed")
    framework = _read(root, FRAMEWORK)
    audit = _read(root, SOURCE_AUDIT)
    _validate_self_hash(framework, FRAMEWORK)
    _validate_self_hash(audit, SOURCE_AUDIT)
    if (
        framework.get("format")
        != "layercake-phase8-framework-freeze/1"
        or framework.get("status") != "FROZEN"
        or framework.get("contract_sha256") != CONTRACT_SHA256
        or framework.get("parent_tag") != PARENT_TAG
        or framework.get("parent_commit") != PARENT_COMMIT
        or framework.get("phase8_evidence_present_at_freeze") is not False
    ):
        raise Phase8EvidenceError("Phase 8 framework freeze is invalid")
    commit = framework.get("framework_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise Phase8EvidenceError("Phase 8 framework commit is malformed")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode:
        raise Phase8EvidenceError("Phase 8 framework commit does not exist")
    if (
        audit.get("format") != "layercake-phase8-source-audit/1"
        or audit.get("status") != "PASS"
        or audit.get("framework_commit") != commit
        or audit.get("source_changes_after_freeze") is not False
        or audit.get("governed_source_sha256")
        != framework.get("governed_source_sha256")
        or audit.get("implementation_hashes")
        != framework.get("implementation_hashes")
    ):
        raise Phase8EvidenceError("Phase 8 source audit failed")
    matrix = _read(root, "moonshot/invalidation_matrix.yaml")
    current = component_hashes(root, matrix)
    changed = sorted(
        name
        for name, policy in matrix.get("components", {}).items()
        if policy.get("invalidates_from_phase", 0) <= 8
        and framework.get("component_hashes", {}).get(name)
        != current.get(name)
    )
    if changed:
        raise Phase8EvidenceError(
            f"sealed product components changed in Phase 8: {changed}"
        )
    return framework


def _validate_environment(root: Path) -> dict[str, Any]:
    document = _read(root, ENVIRONMENT)
    _validate_self_hash(document, ENVIRONMENT)
    commands = document.get("commands", {})
    if (
        document.get("format")
        != "layercake-phase8-cleanroom-environment/1"
        or document.get("status") != "PASS"
        or document.get("git", {}).get("head") != PARENT_COMMIT
        or document.get("git", {}).get("status_before") != ""
        or document.get("git", {}).get("status_after") != ""
        or document.get("git", {}).get("detached") is not True
        or commands.get("pip_check", {}).get("returncode") != 0
        or commands.get("pip_dry_run", {}).get("returncode") != 0
        or commands.get("pytest", {}).get("returncode") != 0
        or commands.get("pytest", {}).get("passed", 0) < 597
        or commands.get("verify_all", {}).get("returncode") != 0
        or commands.get("verify_all", {}).get(
            "completed_phases_valid"
        )
        is not True
    ):
        raise Phase8EvidenceError(
            "detached clean-room environment did not reproduce"
        )
    assets = document.get("external_release_assets", [])
    if len(assets) != 3:
        raise Phase8EvidenceError(
            "clean checkout lacks declared external checkpoint assets"
        )
    for row in assets:
        expected = CHECKPOINTS.get(row.get("id"))
        if (
            expected is None
            or row.get("path") != expected[0].as_posix()
            or row.get("sha256") != expected[1]
            or row.get("copied_and_rehashed") is not True
        ):
            raise Phase8EvidenceError(
                "external clean-room checkpoint identity is invalid"
            )
    return document


def _performance_metrics(
    document: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    records = document.get("records")
    protocol = document.get("protocol", {})
    if (
        document.get("format")
        != "layercake-phase8-fresh-performance/1"
        or document.get("status") != "RAW"
        or not isinstance(records, list)
        or len(records) != 120
        or protocol.get("distinct_prompts") != 100
        or protocol.get("repeated_observations") != 20
        or protocol.get("observations_per_system") != 120
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("qwen_digest") != QWEN_DIGEST
    ):
        raise Phase8EvidenceError(
            "fresh performance protocol identity is invalid"
        )
    if any(
        sum(row.get("seed") == seed for row in records) != 40
        for seed in SEEDS
    ):
        raise Phase8EvidenceError(
            "fresh performance lacks three 40-observation seeds"
        )
    if (
        len(
            {
                row.get("prompt_sha256")
                for row in records
                if row.get("trial") == 1
            }
        )
        != 100
        or sum(row.get("trial") == 2 for row in records) != 20
    ):
        raise Phase8EvidenceError(
            "fresh performance prompt depth is insufficient"
        )
    systems = (
        "layercake_cpu",
        "transformer_cpu",
        "layercake_gpu",
        "transformer_gpu",
    )
    ratios: dict[str, list[float]] = {
        "cpu_cpu_throughput_ratio": [],
        "cpu_cpu_latency_ratio": [],
        "gpu_gpu_throughput_ratio": [],
        "cpu_gpu_throughput_ratio": [],
        "cpu_gpu_latency_ratio": [],
    }
    distinct: list[Mapping[str, Any]] = []
    for row in records:
        for system in systems:
            result = row.get(system, {})
            timing = result.get("timing", {})
            if (
                result.get("generated_bytes", 0) <= 0
                or timing.get("total_latency_seconds", 0) <= 0
                or not _close(
                    timing["bytes_per_second_total"],
                    result["generated_bytes"]
                    / timing["total_latency_seconds"],
                )
            ):
                raise Phase8EvidenceError(
                    f"invalid fresh timing row {row.get('prompt_id')} {system}"
                )
            if system.startswith("transformer_") and result.get(
                "authoritative_generated_tokens", 0
            ) <= 0:
                raise Phase8EvidenceError(
                    "fresh transformer token accounting is absent"
                )
        if (
            row["layercake_cpu"].get("functional_success") is not True
            or row["layercake_gpu"].get("functional_success") is not True
            or row["layercake_cpu"].get("inactive_cake_forward_calls") != 0
            or row["layercake_gpu"].get("inactive_cake_forward_calls") != 0
            or row["layercake_cpu"].get("output_sha256")
            != row["layercake_gpu"].get("output_sha256")
        ):
            raise Phase8EvidenceError(
                "fresh LayerCake quality, sparsity, or device identity failed"
            )
        lc_cpu = row["layercake_cpu"]["timing"]
        tf_cpu = row["transformer_cpu"]["timing"]
        lc_gpu = row["layercake_gpu"]["timing"]
        tf_gpu = row["transformer_gpu"]["timing"]
        ratios["cpu_cpu_throughput_ratio"].append(
            lc_cpu["bytes_per_second_total"]
            / tf_cpu["bytes_per_second_total"]
        )
        ratios["cpu_cpu_latency_ratio"].append(
            lc_cpu["total_latency_seconds"]
            / tf_cpu["total_latency_seconds"]
        )
        ratios["gpu_gpu_throughput_ratio"].append(
            lc_gpu["bytes_per_second_total"]
            / tf_gpu["bytes_per_second_total"]
        )
        ratios["cpu_gpu_throughput_ratio"].append(
            lc_cpu["bytes_per_second_total"]
            / tf_gpu["bytes_per_second_total"]
        )
        ratios["cpu_gpu_latency_ratio"].append(
            lc_cpu["total_latency_seconds"]
            / tf_gpu["total_latency_seconds"]
        )
        if row.get("trial") == 1:
            distinct.append(row)
    cpu_delta = [
        float(row["layercake_cpu"]["functional_success"])
        - float(row["transformer_cpu"]["functional_success"])
        for row in distinct
    ]
    gpu_delta = [
        float(row["layercake_gpu"]["functional_success"])
        - float(row["transformer_gpu"]["functional_success"])
        for row in distinct
    ]
    cpu_ci = _bootstrap_mean(cpu_delta, seed=SEEDS[0])
    gpu_ci = _bootstrap_mean(gpu_delta, seed=SEEDS[1])
    metrics = {
        "cpu_cpu_throughput_ratio": statistics.median(
            ratios["cpu_cpu_throughput_ratio"]
        ),
        "cpu_cpu_median_latency_ratio": statistics.median(
            ratios["cpu_cpu_latency_ratio"]
        ),
        "gpu_gpu_throughput_ratio": statistics.median(
            ratios["gpu_gpu_throughput_ratio"]
        ),
        "cpu_gpu_throughput_ratio": statistics.median(
            ratios["cpu_gpu_throughput_ratio"]
        ),
        "cpu_gpu_median_latency_ratio": statistics.median(
            ratios["cpu_gpu_latency_ratio"]
        ),
    }
    quality = {
        "layercake_cpu_successes": sum(
            row["layercake_cpu"]["functional_success"] for row in distinct
        ),
        "transformer_cpu_successes": sum(
            row["transformer_cpu"]["functional_success"]
            for row in distinct
        ),
        "layercake_gpu_successes": sum(
            row["layercake_gpu"]["functional_success"] for row in distinct
        ),
        "transformer_gpu_successes": sum(
            row["transformer_gpu"]["functional_success"]
            for row in distinct
        ),
        "cpu_quality_delta_bootstrap_95ci": cpu_ci,
        "gpu_quality_delta_bootstrap_95ci": gpu_ci,
    }
    cold = document.get("cold_start", [])
    if (
        {row.get("system") for row in cold} != set(systems)
        or any(
            row.get("single_real_request") is not True
            or row.get("load_probe_request") is not False
            or row.get("timing", {}).get(
                "time_to_first_output_seconds", 0
            )
            <= 0
            or row.get("timing", {}).get(
                "total_latency_seconds", 0
            )
            <= 0
            for row in cold
        )
    ):
        raise Phase8EvidenceError("fresh cold-start evidence is invalid")
    memory = document.get("memory", {})
    if (
        memory.get("layercake_gpu_peak_allocated_bytes", 0) <= 0
        or memory.get("layercake_gpu_peak_reserved_bytes", 0) <= 0
        or memory.get("qwen_gpu_vram_bytes", 0) <= 0
        or memory.get("process_rss_bytes", 0) <= 0
    ):
        raise Phase8EvidenceError("fresh memory evidence is absent")
    return metrics, quality


def _validate_domains(root: Path) -> dict[str, Any]:
    document = _read(root, DOMAINS)
    _validate_self_hash(document, DOMAINS)
    records = document.get("records")
    core = document.get("core_only_abstentions")
    if (
        document.get("format")
        != "layercake-phase8-domain-retention/1"
        or document.get("status") != "RAW"
        or not isinstance(records, list)
        or len(records) != 384
        or not isinstance(core, list)
        or len(core) != 100
    ):
        raise Phase8EvidenceError(
            "fresh full-domain retention identity is invalid"
        )
    for domain in PACKAGES:
        if sum(row.get("domain") == domain for row in records) != 128:
            raise Phase8EvidenceError(
                f"fresh domain retention lacks 128 {domain} rows"
            )
    if not all(
        row.get("cpu_functional_success") is True
        and row.get("gpu_functional_success") is True
        and row.get("cpu_output_sha256")
        == row.get("gpu_output_sha256")
        and row.get("cpu_inactive_forward_calls") == 0
        and row.get("gpu_inactive_forward_calls") == 0
        for row in records
    ):
        raise Phase8EvidenceError(
            "fresh full-domain quality or device identity failed"
        )
    if any(
        row.get("selected") != []
        or row.get("execution_path") != "core_only"
        or row.get("cake_forward_calls") != 0
        for row in core
    ):
        raise Phase8EvidenceError(
            "fresh core-only requests activated a cake"
        )
    return {
        "cases_per_device": len(records),
        "cpu_successes": sum(
            row["cpu_functional_success"] for row in records
        ),
        "gpu_successes": sum(
            row["gpu_functional_success"] for row in records
        ),
        "device_identical_outputs": sum(
            row["cpu_output_sha256"] == row["gpu_output_sha256"]
            for row in records
        ),
        "core_only_abstentions": len(core),
    }


def _validate_lifecycle(root: Path) -> dict[str, Any]:
    document = _read(root, LIFECYCLE)
    _validate_self_hash(document, LIFECYCLE)
    hosts = document.get("hosts")
    if (
        document.get("format")
        != "layercake-phase8-lifecycle-portability/1"
        or document.get("status") != "PASS"
        or not isinstance(hosts, list)
        or {row.get("seed") for row in hosts} != set(SEEDS)
    ):
        raise Phase8EvidenceError(
            "fresh lifecycle/portability identity is invalid"
        )
    reference: dict[str, str] | None = None
    for host in hosts:
        if (
            host.get("receiver_training_examples") != 0
            or host.get("receiver_calibration_runs") != 0
            or host.get("core_hashes_before")
            != host.get("core_hashes_after")
            or host.get("installed_archive_hashes")
            != host.get("reinstalled_archive_hashes")
            or host.get("outputs_before")
            != host.get("outputs_after")
            or set(host.get("installed_archive_hashes", {}))
            != set(PACKAGES)
        ):
            raise Phase8EvidenceError(
                "fresh lifecycle changed core, package, or behavior"
            )
        expected_hashes = {
            domain: value[1] for domain, value in PACKAGES.items()
        }
        if host["installed_archive_hashes"] != expected_hashes:
            raise Phase8EvidenceError(
                "fresh lifecycle archive hashes are stale"
            )
        if reference is None:
            reference = host["outputs_after"]
        elif host["outputs_after"] != reference:
            raise Phase8EvidenceError(
                "fresh cross-host semantic outputs differ"
            )
    return {
        "hosts": len(hosts),
        "package_byte_identity": 1.0,
        "core_immutability": 1.0,
        "cross_host_semantic_retention": 1.0,
        "receiver_training_examples": 0,
        "receiver_calibration_runs": 0,
    }


def _validate_routing(root: Path) -> dict[str, Any]:
    document = _read(root, ROUTING)
    _validate_self_hash(document, ROUTING)
    records = document.get("records")
    catalog = document.get("catalog", {})
    if (
        document.get("format")
        != "layercake-phase8-routing-catalog/1"
        or document.get("status") != "PASS"
        or not isinstance(records, list)
        or len(records) != 1980
        or {row.get("seed") for row in records}
        != {10601, 10602, 10603}
        or not all(row.get("correct") for row in records)
        or catalog.get("largest_size") != 500
        or catalog.get("real_promoted_capabilities") != 3
        or catalog.get("management_only_descriptors") != 497
        or document.get("mode_probes", {}).get("all_pass") is not True
        or document.get("external_process", {}).get("equivalent")
        is not True
    ):
        raise Phase8EvidenceError(
            "fresh routing, catalog, or external orchestration failed"
        )
    return {
        "routing_rows": len(records),
        "routing_accuracy": 1.0,
        "catalog_size": 500,
        "mode_probes": 1.0,
        "external_internal_equivalence": 1.0,
    }


def _validate_manifests(root: Path) -> dict[str, Any]:
    data = _read(root, DATA_HASHES)
    checkpoints = _read(root, CHECKPOINT_HASHES)
    packages = _read(root, PACKAGE_HASHES)
    scan = _read(root, SOURCE_SCAN)
    for relative, document in (
        (DATA_HASHES, data),
        (CHECKPOINT_HASHES, checkpoints),
        (PACKAGE_HASHES, packages),
        (SOURCE_SCAN, scan),
    ):
        _validate_self_hash(document, relative)
        if document.get("status") != "PASS":
            raise Phase8EvidenceError(f"{relative} did not pass")
    if (
        data.get("split_prompt_overlap_count") != 0
        or data.get("duplicate_performance_prompt_count") != 0
        or data.get("all_hashes_match") is not True
    ):
        raise Phase8EvidenceError("data isolation/hash manifest failed")
    expected_checkpoints = {
        key: {"path": value[0].as_posix(), "sha256": value[1]}
        for key, value in CHECKPOINTS.items()
    }
    if checkpoints.get("checkpoints") != expected_checkpoints:
        raise Phase8EvidenceError("checkpoint manifest is stale")
    expected_packages = {
        key: {"path": value[0].as_posix(), "sha256": value[1]}
        for key, value in PACKAGES.items()
    }
    if (
        packages.get("packages") != expected_packages
        or packages.get("abi_sha256") != ABI_HASH
        or packages.get("qwen_digest") != QWEN_DIGEST
    ):
        raise Phase8EvidenceError("package/ABI manifest is stale")
    if (
        scan.get("committed_private_keys") != []
        or scan.get("executable_package_members") != []
        or scan.get("unresolved_runtime_backdoors") != []
        or scan.get("hard_coded_promoted_output_hashes") != []
    ):
        raise Phase8EvidenceError(
            "source/package security scan found an unresolved issue"
        )
    return {
        "data_integrity": 1.0,
        "checkpoint_identity": 1.0,
        "package_abi_identity": 1.0,
        "source_scan": 1.0,
    }


def _validate_prior(root: Path) -> dict[str, Any]:
    document = _read(root, PRIOR)
    _validate_self_hash(document, PRIOR)
    summaries = {
        "phase2": validate_phase2_r3_bundle(
            root, _path(root, "results/moonshot/phase2_recertification")
        ),
        "phase3": validate_phase3_retirement_bundle(
            root, _path(root, "results/moonshot/phase3")
        ),
        "phase4": validate_phase4_bundle(
            root, _path(root, "results/moonshot/phase4")
        ),
        "phase5": validate_phase5_bundle(
            root, _path(root, "results/moonshot/phase5")
        ),
        "phase6": validate_phase6_bundle(
            root, _path(root, "results/moonshot/phase6")
        ),
        "phase7": validate_phase7_bundle(
            root, _path(root, "results/moonshot/phase7")
        ),
    }
    if (
        document.get("format")
        != "layercake-phase8-prior-gate-recomputation/1"
        or document.get("status") != "PASS"
        or document.get("typed_summaries") != summaries
        or document.get("phases_0_through_7_sealed") is not True
        or document.get("all_contract_required_gates_retained") is not True
        or document.get("training_efficiency_claimed") is not False
        or document.get("phase3_retirement_retained") is not True
    ):
        raise Phase8EvidenceError(
            "prior required gates did not independently retain"
        )
    return {
        "prior_required_gate_retention": 1.0,
        "typed_summaries": summaries,
    }


def _validate_adversarial(root: Path) -> dict[str, Any]:
    document = _read(root, ADVERSARIAL)
    _validate_self_hash(document, ADVERSARIAL)
    records = document.get("records")
    if (
        document.get("format")
        != "layercake-phase8-adversarial-falsification/1"
        or document.get("status") != "PASS"
        or not isinstance(records, list)
        or len(records) < 24
        or len({row.get("attack_id") for row in records}) != len(records)
        or not REQUIRED_ATTACK_CATEGORIES
        <= {row.get("category") for row in records}
        or any(
            row.get("outcome")
            not in {"DETECTED", "INVARIANT_RETAINED"}
            or row.get("resolved") is not True
            for row in records
        )
        or document.get("unresolved_findings") != []
        or document.get("unresolved_by_severity")
        != {"critical": 0, "high": 0, "low": 0, "medium": 0}
    ):
        raise Phase8EvidenceError(
            "adversarial falsification is incomplete or unresolved"
        )
    return {
        "attacks": len(records),
        "categories": len({row["category"] for row in records}),
        "adversarial_falsification_findings_resolved": 1.0,
    }


def _validate_derived(
    root: Path,
    framework: Mapping[str, Any],
    metrics: Mapping[str, float],
    details: Mapping[str, Any],
) -> None:
    gates = _read(root, GATES)
    payload = _read(root, PAYLOAD)
    certificate = _read(root, CERTIFICATE)
    final = _read(root, FINAL_CERTIFICATE)
    for relative, document in (
        (GATES, gates),
        (PAYLOAD, payload),
        (CERTIFICATE, certificate),
        (FINAL_CERTIFICATE, final),
    ):
        _validate_self_hash(document, relative)
    if (
        gates.get("format")
        != "layercake-phase8-gate-observations/1"
        or gates.get("status") != "RAW_DERIVED"
        or gates.get("framework_commit")
        != framework["framework_commit"]
        or payload.get("format")
        != "layercake-phase8-certificate-payload/1"
        or payload.get("status") != "PASS"
        or certificate.get("format")
        != "layercake-phase8-independent-verification-certificate/1"
        or certificate.get("status") != "PROVEN"
        or certificate.get("metrics") != dict(metrics)
        or certificate.get("details") != dict(details)
        or final != certificate
    ):
        raise Phase8EvidenceError(
            "final derived evidence identity is invalid"
        )
    records = gates.get("records", [])
    if {row.get("seed") for row in records} != set(SEEDS):
        raise Phase8EvidenceError(
            "final gate observations lack three seeds"
        )
    for gate_id, value in metrics.items():
        selected = [
            float(row["value"])
            for row in records
            if row.get("gate_id") == gate_id
        ]
        if len(selected) != 3 or any(
            not _close(item, value) for item in selected
        ):
            raise Phase8EvidenceError(
                f"stale final gate observation: {gate_id}"
            )
    claims = {row["gate_id"]: row for row in payload.get("claims", [])}
    if set(claims) != set(metrics):
        raise Phase8EvidenceError(
            "final payload has missing or extra claims"
        )
    for gate_id, value in metrics.items():
        if not _close(claims[gate_id]["value"], value):
            raise Phase8EvidenceError(
                f"stale final payload claim: {gate_id}"
            )
    legacy = _path(root, LEGACY_CERTIFICATE)
    if (
        not legacy.is_file()
        or _sha256(legacy) != LEGACY_CERTIFICATE_SHA256
    ):
        raise Phase8EvidenceError(
            "pre-gated final certificate was not preserved"
        )
    if (
        not _path(root, REPORT).is_file()
        or not _path(root, FINAL_REPORT).is_file()
        or _sha256(_path(root, REPORT))
        != _sha256(_path(root, FINAL_REPORT))
    ):
        raise Phase8EvidenceError("final release report mirror is stale")


def derive_phase8_metrics(root: Path) -> dict[str, Any]:
    framework = _validate_framework(root)
    environment = _validate_environment(root)
    performance = _read(root, PERFORMANCE)
    _validate_self_hash(performance, PERFORMANCE)
    performance_metrics, quality = _performance_metrics(performance)
    domains = _validate_domains(root)
    lifecycle = _validate_lifecycle(root)
    routing = _validate_routing(root)
    manifests = _validate_manifests(root)
    prior = _validate_prior(root)
    adversarial = _validate_adversarial(root)
    thresholds = _read(root, CONTRACT)["reproduction_thresholds"]
    performance_pass = (
        performance_metrics["cpu_cpu_throughput_ratio"]
        >= thresholds["cpu_cpu_throughput_ratio"]["value"]
        and performance_metrics["cpu_cpu_median_latency_ratio"]
        <= thresholds["cpu_cpu_median_latency_ratio"]["value"]
        and performance_metrics["gpu_gpu_throughput_ratio"]
        > thresholds["gpu_gpu_throughput_ratio"]["value"]
        and performance_metrics["cpu_gpu_throughput_ratio"]
        >= thresholds["cpu_gpu_throughput_ratio"]["value"]
        and performance_metrics["cpu_gpu_median_latency_ratio"]
        <= thresholds["cpu_gpu_median_latency_ratio"]["value"]
        and quality["layercake_cpu_successes"] == 100
        and quality["layercake_gpu_successes"] == 100
        and quality["cpu_quality_delta_bootstrap_95ci"][0] > 0
        and quality["gpu_quality_delta_bootstrap_95ci"][0] > 0
        and domains["cpu_successes"] == 384
        and domains["gpu_successes"] == 384
        and domains["device_identical_outputs"] == 384
        and domains["core_only_abstentions"] == 100
    )
    metrics = {
        "prior_required_gate_retention": prior[
            "prior_required_gate_retention"
        ],
        "clean_room_reproduction": float(
            performance_pass
            and environment["status"] == "PASS"
            and lifecycle["package_byte_identity"] == 1.0
            and routing["routing_accuracy"] == 1.0
            and manifests["data_integrity"] == 1.0
        ),
        "adversarial_falsification_findings_resolved": adversarial[
            "adversarial_falsification_findings_resolved"
        ],
    }
    if any(value != 1.0 for value in metrics.values()):
        raise Phase8EvidenceError(
            f"final Phase 8 promotion gates failed: {metrics}"
        )
    details = {
        "performance": performance_metrics,
        "quality": quality,
        "domains": domains,
        "lifecycle": lifecycle,
        "routing": routing,
        "manifests": manifests,
        "adversarial": adversarial,
        "cleanroom_tests": environment["commands"]["pytest"]["passed"],
        "training": {
            "status": "RETIRED_BY_GOVERNANCE",
            "scientific_training_efficiency_passed": False,
            "product_gate": False,
        },
        "mobile": {
            "status": "NOT_RUN_NO_HARDWARE",
            "physical_mobile_performance_claimed": False,
        },
        "claim_boundaries": {
            "physical_mobile_performance_claimed": False,
            "gpu_training_dominance_claimed": False,
            "latent_neural_fusion_claimed": False,
            "external_human_or_laboratory_independence_claimed": False,
            "energy_dominance_claimed": False,
        },
    }
    _validate_derived(root, framework, metrics, details)
    return {
        "status": "PROVEN",
        "framework_commit": framework["framework_commit"],
        "metrics": metrics,
        "details": details,
        "seeds": list(SEEDS),
        "tag": "layercake-moonshot-final",
    }


def validate_phase8_bundle(
    root: Path, evidence_dir: Path
) -> dict[str, Any]:
    expected = _path(root, "results/moonshot/phase8")
    if evidence_dir.resolve() != expected:
        raise Phase8EvidenceError(
            "Phase 8 evidence directory is not canonical"
        )
    return derive_phase8_metrics(root)
