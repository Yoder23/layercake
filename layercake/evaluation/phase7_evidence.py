"""Typed, fail-closed verification for Phase 7 integrated performance."""

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
from layercake.evaluation.phase6_evidence import validate_phase6_bundle
from layercake.moonshot_campaign import component_hashes


class Phase7EvidenceError(RuntimeError):
    """Raised when Phase 7 evidence cannot support promotion."""


CONTRACT = Path("moonshot/phase7_integrated_performance_preregistration.json")
CONTRACT_SHA256 = "a70df7bbb62ed4869bcc9b9c62ed84635c80461b5837db40f93a3aa16588afa1"
PROFILES = Path("moonshot/phase6_router_profiles.json")
PROFILES_SHA256 = "5397a0f28d145c15ee2aeaf13c021aaabb969b4226d16a2f66f776d2b48c3ec2"
PHASE6_CPU = Path("results/moonshot/phase6/raw_runs/mixed_cpu_benchmark.json")
PHASE6_CPU_SHA256 = "5aac50d5259c4e8d43fddf57d9e8771ce890682b84c992d636c2f73b9efbb303"
PHASE6_FUNCTIONAL = Path(
    "results/moonshot/phase6/raw_runs/functional_execution.json"
)
FRAMEWORK = Path("results/moonshot/phase7/framework_freeze.json")
SOURCE_AUDIT = Path("results/moonshot/phase7/source_audit.json")
COLD = Path("results/moonshot/phase7/raw_runs/cold_start.json")
CPU_SUPPLEMENT = Path(
    "results/moonshot/phase7/raw_runs/cpu_layercake_timing_supplement.json"
)
GPU_PERFORMANCE = Path("results/moonshot/phase7/raw_runs/gpu_performance.json")
GPU_RETENTION = Path(
    "results/moonshot/phase7/raw_runs/gpu_domain_retention.json"
)
GENERAL_QUALITY = Path(
    "results/moonshot/phase7/raw_runs/general_quality_retention.json"
)
GATES = Path("results/moonshot/phase7/raw_runs/gate_observations.json")
PAYLOAD = Path("results/moonshot/phase7/certificate_payload.json")
CERTIFICATE = Path(
    "results/moonshot/phase7/integrated_performance_certificate.json"
)
SEEDS = (10701, 10702, 10703)
QWEN_DIGEST = "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67"
PACKAGE_PATHS = {
    "python": Path(
        "artifacts/moonshot/phase4/release/"
        "python-token-plan-seed10141-direct-v1.0.0.cake"
    ),
    "sql": Path(
        "artifacts/moonshot/phase5/release/sql-token-plan-v1.0.0.cake"
    ),
    "regex": Path(
        "artifacts/moonshot/phase5/release/regex-token-plan-v1.0.0.cake"
    ),
}
PACKAGE_HASHES = {
    "python": "0585c79bfbea16b1c4165bf0030ba6985b8a8cdeab529cd7afe3f9c76c564ef7",
    "sql": "24efdb68885581318bee3a0b7c3cac0b0fe0b75eefa90070c601384a8bf5e105",
    "regex": "c336a552415b2d6161eb6696338086f89c3c3b6e25949e6cf0a94373e9690f15",
}


def _path(root: Path, relative: Path | str) -> Path:
    value = (root / relative).resolve()
    try:
        value.relative_to(root.resolve())
    except ValueError as error:
        raise Phase7EvidenceError(
            f"Phase 7 path escapes repository: {relative}"
        ) from error
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(root: Path, relative: Path | str) -> dict[str, Any]:
    path = _path(root, relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase7EvidenceError(
            f"cannot read Phase 7 evidence {relative}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise Phase7EvidenceError(f"{relative} is not a JSON object")
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
        raise Phase7EvidenceError(f"{relative} has a stale evidence hash")


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


def phase7_evidence_files(root: Path) -> list[Path]:
    relatives = {
        CONTRACT,
        PROFILES,
        PHASE6_CPU,
        PHASE6_FUNCTIONAL,
        FRAMEWORK,
        SOURCE_AUDIT,
        COLD,
        CPU_SUPPLEMENT,
        GPU_PERFORMANCE,
        GPU_RETENTION,
        GENERAL_QUALITY,
        GATES,
        PAYLOAD,
        CERTIFICATE,
        *PACKAGE_PATHS.values(),
        Path(
            "results/moonshot/phase2_recertification/"
            "certificate_payload.json"
        ),
        Path(
            "results/moonshot/phase2_recertification/"
            "raw_runs/quality_seeds.json"
        ),
        Path(
            "results/moonshot/phase2_recertification/runtime_proofs.json"
        ),
        Path("results/moonshot/phase6/release_certificate.json"),
        Path("results/moonshot/phase6/seal.json"),
    }
    result_dir = _path(root, Path("results/moonshot/phase7"))
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
        raise Phase7EvidenceError("Phase 7 preregistration changed")
    if _sha256(_path(root, PROFILES)) != PROFILES_SHA256:
        raise Phase7EvidenceError("sealed router profiles changed")
    if _sha256(_path(root, PHASE6_CPU)) != PHASE6_CPU_SHA256:
        raise Phase7EvidenceError("sealed Phase 6 CPU evidence changed")
    for domain, relative in PACKAGE_PATHS.items():
        if _sha256(_path(root, relative)) != PACKAGE_HASHES[domain]:
            raise Phase7EvidenceError(f"sealed {domain} package changed")
    framework = _read(root, FRAMEWORK)
    audit = _read(root, SOURCE_AUDIT)
    _validate_self_hash(framework, FRAMEWORK)
    _validate_self_hash(audit, SOURCE_AUDIT)
    if (
        framework.get("format")
        != "layercake-phase7-framework-freeze/1"
        or framework.get("status") != "FROZEN"
        or framework.get("contract_sha256") != CONTRACT_SHA256
        or framework.get("profiles_sha256") != PROFILES_SHA256
        or framework.get("phase6_cpu_sha256") != PHASE6_CPU_SHA256
        or framework.get("raw_evidence_present_at_freeze") is not False
    ):
        raise Phase7EvidenceError("Phase 7 framework freeze is invalid")
    commit = framework.get("framework_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise Phase7EvidenceError("Phase 7 framework commit is malformed")
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if not exists:
        raise Phase7EvidenceError("Phase 7 framework commit does not exist")
    if (
        audit.get("format") != "layercake-phase7-source-audit/1"
        or audit.get("status") != "PASS"
        or audit.get("framework_commit") != commit
        or audit.get("source_changes_after_freeze") is not False
        or audit.get("governed_source_sha256")
        != framework.get("governed_source_sha256")
        or audit.get("implementation_hashes")
        != framework.get("implementation_hashes")
    ):
        raise Phase7EvidenceError("Phase 7 source audit failed")
    matrix = _read(root, Path("moonshot/invalidation_matrix.yaml"))
    actual_components = component_hashes(root, matrix)
    invalidating = sorted(
        name
        for name, policy in matrix.get("components", {}).items()
        if policy.get("invalidates_from_phase", 0) <= 7
        and framework.get("component_hashes", {}).get(name)
        != actual_components.get(name)
    )
    if invalidating:
        raise Phase7EvidenceError(
            "Phase 7 dependent components changed after freeze: "
            f"{invalidating}"
        )
    hardware = framework.get("hardware", {})
    if (
        hardware.get("gpu_name")
        != "NVIDIA GeForce RTX 3080 Laptop GPU"
        or hardware.get("cpu_physical") != 14
        or hardware.get("cpu_logical") != 20
        or int(hardware.get("gpu_total_memory_bytes", 0))
        < 15 * 1024**3
    ):
        raise Phase7EvidenceError(
            "declared Phase 7 hardware was not measured"
        )
    return framework


def _validate_cold(root: Path) -> dict[str, Any]:
    document = _read(root, COLD)
    _validate_self_hash(document, COLD)
    records = document.get("records")
    expected = {
        "layercake_cpu",
        "layercake_gpu",
        "transformer_cpu",
        "transformer_gpu",
    }
    if (
        document.get("format") != "layercake-phase7-cold-start/1"
        or document.get("status") != "RAW"
        or not isinstance(records, list)
        or {row.get("system") for row in records} != expected
    ):
        raise Phase7EvidenceError("cold-start evidence identity is invalid")
    protocol = document.get("protocol", {})
    if (
        protocol.get("single_real_request_per_system") is not True
        or protocol.get("transformer_load_probe_request") is not False
        or protocol.get("transformer_unload_control_before_request")
        is not True
    ):
        raise Phase7EvidenceError("cold-start protocol is invalid")
    for row in records:
        request = row.get("single_real_request") or row.get(
            "single_real_streaming_request"
        )
        timing = request.get("timing", {}) if isinstance(request, dict) else {}
        if (
            not request
            or request.get("cold") is not True
            or timing.get("time_to_first_output_seconds", 0) <= 0
            or timing.get("total_latency_seconds", 0) <= 0
        ):
            raise Phase7EvidenceError(
                "cold-start request lacks direct timing"
            )
        if row["system"].startswith("transformer_"):
            if row.get("unload_control", {}).get("done") is not True:
                raise Phase7EvidenceError(
                    "transformer cold run lacks unload control"
                )
            if request.get("authoritative_generated_tokens", 0) <= 0:
                raise Phase7EvidenceError(
                    "transformer cold run lacks token accounting"
                )
    return document


def _model_is_gpu_resident(document: Mapping[str, Any]) -> bool:
    return any(
        row.get("digest") == QWEN_DIGEST
        and int(row.get("size_vram", 0)) > 0
        for row in document.get("models", [])
    )


def _nearest_rank(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _system_descriptives(
    records: list[Mapping[str, Any]], system: str
) -> dict[str, Any]:
    observations = [row[system] for row in records]
    timing = [row["timing"] for row in observations]

    def quantiles(field: str) -> dict[str, float]:
        values = [float(row[field]) for row in timing]
        return {
            "p50": statistics.median(values),
            "p95_nearest_rank": _nearest_rank(values, 0.95),
            "p99_nearest_rank": _nearest_rank(values, 0.99),
        }

    result: dict[str, Any] = {
        "observations": len(observations),
        "total_latency_seconds": quantiles("total_latency_seconds"),
        "time_to_first_output_seconds": quantiles(
            "time_to_first_output_seconds"
        ),
        "bytes_per_second_total": quantiles("bytes_per_second_total"),
        "characters_per_second_total": quantiles(
            "characters_per_second_total"
        ),
        "correct_task_completions_per_wall_second": (
            sum(bool(row["functional_success"]) for row in observations)
            / sum(float(row["timing"]["total_latency_seconds"]) for row in observations)
        ),
    }
    if all("authoritative_generated_tokens" in row for row in observations):
        token_rates = [
            float(row["authoritative_generated_tokens"])
            / float(row["timing"]["total_latency_seconds"])
            for row in observations
        ]
        result["authoritative_tokens_per_second_total"] = {
            "p50": statistics.median(token_rates),
            "p95_nearest_rank": _nearest_rank(token_rates, 0.95),
            "p99_nearest_rank": _nearest_rank(token_rates, 0.99),
        }
    return result


def _validate_cpu_supplement(
    root: Path, cpu: Mapping[str, Any]
) -> dict[str, Any]:
    document = _read(root, CPU_SUPPLEMENT)
    _validate_self_hash(document, CPU_SUPPLEMENT)
    records = document.get("records")
    protocol = document.get("protocol", {})
    if (
        document.get("format")
        != "layercake-phase7-cpu-timing-supplement/1"
        or document.get("status") != "RAW"
        or not isinstance(records, list)
        or len(records) != 120
        or protocol.get("distinct_prompts") != 100
        or protocol.get("repeated_prompt_observations") != 20
        or protocol.get("observations") != 120
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("observations_per_seed") != 40
        or protocol.get("threads") != 14
    ):
        raise Phase7EvidenceError(
            "LayerCake CPU timing supplement identity is invalid"
        )
    if any(
        sum(row.get("seed") == seed for row in records) != 40
        for seed in SEEDS
    ):
        raise Phase7EvidenceError(
            "LayerCake CPU supplement lacks three 40-observation seeds"
        )
    cpu_keys = {
        (row["prompt_id"], row["trial"]): row
        for row in cpu.get("records", [])
    }
    supplement_keys = {
        (row["prompt_id"], row["trial"]): row for row in records
    }
    if len(cpu_keys) != 120 or set(cpu_keys) != set(supplement_keys):
        raise Phase7EvidenceError(
            "LayerCake CPU supplement prompt-trial pairing differs"
        )
    for key, row in supplement_keys.items():
        reference = cpu_keys[key]
        result = row.get("layercake_cpu", {})
        timing = result.get("timing", {})
        if (
            row.get("prompt_sha256") != reference.get("prompt_sha256")
            or row.get("domain") != reference.get("domain")
            or result.get("selected")
            != [f"{row.get('domain')}-token-plan"]
            or result.get("functional_success") is not True
            or result.get("inactive_cake_forward_calls") != 0
            or timing.get("time_to_first_output_seconds", 0) <= 0
            or timing.get("total_latency_seconds", 0) <= 0
            or timing.get("bytes_per_second_total", 0) <= 0
            or timing.get("characters_per_second_total", 0) <= 0
        ):
            raise Phase7EvidenceError(
                f"invalid LayerCake CPU supplement row: {key}"
            )
    memory = document.get("memory", {})
    if (
        len(memory.get("loaded_tensor_bytes_by_seed", [])) != 3
        or len(memory.get("process_rss_samples_bytes", [])) != 3
        or any(
            value <= 0
            for value in memory.get("loaded_tensor_bytes_by_seed", [])
        )
        or any(
            value <= 0
            for value in memory.get("process_rss_samples_bytes", [])
        )
    ):
        raise Phase7EvidenceError(
            "LayerCake CPU supplement memory evidence is missing"
        )
    expected = {"layercake_cpu": _system_descriptives(
        records, "layercake_cpu"
    )}
    if document.get("aggregates") != expected:
        raise Phase7EvidenceError(
            "LayerCake CPU supplement aggregates are stale"
        )
    return document


def _validate_performance(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    cpu = _read(root, PHASE6_CPU)
    gpu = _read(root, GPU_PERFORMANCE)
    _validate_self_hash(gpu, GPU_PERFORMANCE)
    records = gpu.get("records")
    protocol = gpu.get("protocol", {})
    if (
        gpu.get("format") != "layercake-phase7-gpu-performance/1"
        or gpu.get("status") != "RAW"
        or not isinstance(records, list)
        or len(records) != 120
        or protocol.get("distinct_prompts") != 100
        or protocol.get("repeated_prompt_observations") != 20
        or protocol.get("observations_per_system") != 120
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("layercake_precision") != "fp32"
        or protocol.get("transformer_num_gpu") != 99
        or protocol.get("transformer_digest") != QWEN_DIGEST
    ):
        raise Phase7EvidenceError(
            "GPU performance evidence identity is invalid"
        )
    if {row.get("seed") for row in records} != set(SEEDS) or any(
        sum(row["seed"] == seed for row in records) != 40
        for seed in SEEDS
    ):
        raise Phase7EvidenceError(
            "GPU performance lacks three 40-observation seeds"
        )
    if (
        len(
            {
                row["prompt_sha256"]
                for row in records
                if row.get("trial") == 1
            }
        )
        != 100
        or sum(row.get("trial") == 2 for row in records) != 20
    ):
        raise Phase7EvidenceError(
            "GPU performance lacks required prompt depth"
        )
    cpu_rows = {
        (row["prompt_id"], row["trial"]): row
        for row in cpu.get("records", [])
    }
    gpu_rows = {
        (row["prompt_id"], row["trial"]): row for row in records
    }
    if len(cpu_rows) != 120 or set(cpu_rows) != set(gpu_rows):
        raise Phase7EvidenceError("CPU/GPU prompt-trial pairing differs")
    for key in sorted(cpu_rows):
        left = cpu_rows[key]
        right = gpu_rows[key]
        domain = right.get("domain")
        layer = right.get("layercake_gpu", {})
        transformer = right.get("transformer_gpu", {})
        if (
            left.get("prompt_sha256") != right.get("prompt_sha256")
            or left.get("domain") != domain
            or domain not in {"python", "sql", "regex"}
            or layer.get("selected") != [f"{domain}-token-plan"]
            or layer.get("functional_success") is not True
            or layer.get("inactive_cake_forward_calls") != 0
            or layer.get("generated_bytes", 0) <= 0
            or transformer.get("authoritative_generated_tokens", 0) <= 0
            or transformer.get("generated_bytes", 0) <= 0
            or layer.get("timing", {}).get(
                "total_latency_seconds", 0
            )
            <= 0
            or transformer.get("timing", {}).get(
                "total_latency_seconds", 0
            )
            <= 0
        ):
            raise Phase7EvidenceError(
                f"invalid GPU performance row: {key}"
            )
    if not _model_is_gpu_resident(
        gpu.get("qwen_warm_model_report", {})
    ):
        raise Phase7EvidenceError("Qwen was not physically GPU resident")
    memory = gpu.get("memory", {})
    if (
        memory.get("qwen_size_vram_bytes", 0) <= 0
        or memory.get("layercake_peak_allocated_bytes", 0) <= 0
        or memory.get("layercake_peak_reserved_bytes", 0) <= 0
        or any(
            value <= 0
            for value in memory.get(
                "layercake_loaded_tensor_bytes_by_seed", []
            )
        )
    ):
        raise Phase7EvidenceError("GPU memory evidence is missing")
    expected_aggregates = {
        "layercake_gpu": _system_descriptives(
            records, "layercake_gpu"
        ),
        "transformer_gpu": _system_descriptives(
            records, "transformer_gpu"
        ),
    }
    if gpu.get("aggregates") != expected_aggregates:
        raise Phase7EvidenceError(
            "GPU descriptive aggregates are stale"
        )
    pairs: list[dict[str, Any]] = []
    for key in sorted(cpu_rows):
        c = cpu_rows[key]
        g = gpu_rows[key]
        pairs.append(
            {
                "prompt_id": key[0],
                "trial": key[1],
                "cpu_cpu_throughput_ratio": (
                    c["layercake"]["timing"]["bytes_per_second_total"]
                    / c["qwen"]["timing"]["bytes_per_second_total"]
                ),
                "cpu_cpu_latency_ratio": (
                    c["layercake"]["timing"]["total_latency_seconds"]
                    / c["qwen"]["timing"]["total_latency_seconds"]
                ),
                "gpu_gpu_throughput_ratio": (
                    g["layercake_gpu"]["timing"][
                        "bytes_per_second_total"
                    ]
                    / g["transformer_gpu"]["timing"][
                        "bytes_per_second_total"
                    ]
                ),
                "cpu_gpu_throughput_ratio": (
                    c["layercake"]["timing"]["bytes_per_second_total"]
                    / g["transformer_gpu"]["timing"][
                        "bytes_per_second_total"
                    ]
                ),
                "cpu_gpu_latency_ratio": (
                    c["layercake"]["timing"]["total_latency_seconds"]
                    / g["transformer_gpu"]["timing"][
                        "total_latency_seconds"
                    ]
                ),
            }
        )
    return cpu, gpu, pairs


def _validate_retention(root: Path) -> float:
    document = _read(root, GPU_RETENTION)
    _validate_self_hash(document, GPU_RETENTION)
    records = document.get("records")
    if (
        document.get("format")
        != "layercake-phase7-gpu-domain-retention/1"
        or document.get("status") != "RAW"
        or document.get("package_hashes") != PACKAGE_HASHES
        or not isinstance(records, list)
        or len(records) != 384
        or document.get("cpu_reference", {}).get(
            "successful_automatic_rows"
        )
        != 384
        or document.get("cpu_reference", {}).get("sha256")
        != _sha256(_path(root, PHASE6_FUNCTIONAL))
    ):
        raise Phase7EvidenceError(
            "GPU retention evidence identity is invalid"
        )
    for domain in PACKAGE_HASHES:
        if sum(
            row.get("domain") == domain for row in records
        ) != 128:
            raise Phase7EvidenceError(
                f"GPU retention lacks 128 {domain} rows"
            )
    if {row.get("seed") for row in records} != set(SEEDS):
        raise Phase7EvidenceError("GPU retention lacks three seeds")
    if not all(
        row.get("selected") == [f"{row.get('domain')}-token-plan"]
        and row.get("functional_success") is True
        and row.get("cpu_output_equal") is True
        and row.get("output_sha256")
        == row.get("cpu_reference_output_sha256")
        and row.get("inactive_forward_calls") == 0
        for row in records
    ):
        raise Phase7EvidenceError(
            "GPU domain success or CPU identity retention failed"
        )
    return 1.0


def _validate_general(root: Path) -> float:
    document = _read(root, GENERAL_QUALITY)
    _validate_self_hash(document, GENERAL_QUALITY)
    summary = validate_phase2_r3_bundle(
        root, _path(root, "results/moonshot/phase2_recertification")
    )
    abstentions = document.get("core_only_abstentions")
    if (
        document.get("format")
        != "layercake-phase7-general-quality-retention/1"
        or document.get("status") != "PASS"
        or document.get("phase2_typed_summary") != summary
        or document.get("actual_core_checkpoint_hashes")
        != document.get("expected_core_checkpoint_hashes")
        or document.get("general_quality_noninferior") is not True
        or not isinstance(abstentions, list)
        or len(abstentions) != 100
        or {row.get("seed") for row in abstentions} != set(SEEDS)
    ):
        raise Phase7EvidenceError(
            "general quality retention identity is invalid"
        )
    if any(
        row.get("selected") != []
        or row.get("execution_path") != "core_only"
        or row.get("inactive_forward_calls") != 0
        for row in abstentions
    ):
        raise Phase7EvidenceError(
            "core-only fallback activated a specialist"
        )
    return 1.0


def _quality_superiority(
    cpu: Mapping[str, Any], gpu: Mapping[str, Any]
) -> tuple[float, dict[str, Any]]:
    cpu_rows = {
        (row["prompt_id"], row["trial"]): row
        for row in cpu["records"]
    }
    gpu_rows = {
        (row["prompt_id"], row["trial"]): row
        for row in gpu["records"]
    }
    keys = sorted(key for key in cpu_rows if key[1] == 1)
    cpu_delta = [
        float(cpu_rows[key]["layercake"]["functional_success"])
        - float(cpu_rows[key]["qwen"]["functional_success"])
        for key in keys
    ]
    gpu_delta = [
        float(gpu_rows[key]["layercake_gpu"]["functional_success"])
        - float(
            gpu_rows[key]["transformer_gpu"]["functional_success"]
        )
        for key in keys
    ]
    cpu_ci = _bootstrap_mean(cpu_delta, seed=SEEDS[0])
    gpu_ci = _bootstrap_mean(gpu_delta, seed=SEEDS[1])
    details = {
        "cpu_layercake_successes": sum(
            cpu_rows[key]["layercake"]["functional_success"]
            for key in keys
        ),
        "cpu_transformer_successes": sum(
            cpu_rows[key]["qwen"]["functional_success"]
            for key in keys
        ),
        "gpu_layercake_successes": sum(
            gpu_rows[key]["layercake_gpu"]["functional_success"]
            for key in keys
        ),
        "gpu_transformer_successes": sum(
            gpu_rows[key]["transformer_gpu"]["functional_success"]
            for key in keys
        ),
        "cpu_mean_success_delta": statistics.fmean(cpu_delta),
        "gpu_mean_success_delta": statistics.fmean(gpu_delta),
        "cpu_paired_bootstrap_95ci": cpu_ci,
        "gpu_paired_bootstrap_95ci": gpu_ci,
        "paired_rows": len(keys),
    }
    passed = (
        details["cpu_mean_success_delta"] > 0
        and details["gpu_mean_success_delta"] > 0
        and cpu_ci[0] > 0
        and gpu_ci[0] > 0
    )
    return float(passed), details


def _metrics(
    pairs: list[dict[str, Any]],
    general: float,
    superior: float,
    retention: float,
) -> dict[str, float]:
    return {
        "cpu_cpu_throughput_ratio": statistics.median(
            row["cpu_cpu_throughput_ratio"] for row in pairs
        ),
        "cpu_cpu_median_latency_ratio": statistics.median(
            row["cpu_cpu_latency_ratio"] for row in pairs
        ),
        "gpu_gpu_throughput_ratio": statistics.median(
            row["gpu_gpu_throughput_ratio"] for row in pairs
        ),
        "cpu_gpu_throughput_ratio": statistics.median(
            row["cpu_gpu_throughput_ratio"] for row in pairs
        ),
        "cpu_gpu_median_latency_ratio": statistics.median(
            row["cpu_gpu_latency_ratio"] for row in pairs
        ),
        "general_quality_noninferior": general,
        "mixed_domain_quality_superior": superior,
        "promoted_domain_success_retention": retention,
    }


def _validate_derived(
    root: Path,
    framework: Mapping[str, Any],
    metrics: Mapping[str, float],
    quality: Mapping[str, Any],
    descriptive_performance: Mapping[str, Any],
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
        gates.get("format")
        != "layercake-phase7-gate-observations/1"
        or gates.get("status") != "RAW_DERIVED"
        or gates.get("source_commit")
        != framework["framework_commit"]
        or gates.get("quality_details") != quality
        or payload.get("format")
        != "layercake-phase7-certificate-payload/1"
        or payload.get("status") != "PASS"
        or payload.get("quality") != quality
        or payload.get("descriptive_performance")
        != descriptive_performance
        or certificate.get("format")
        != "layercake-phase7-integrated-performance-certificate/1"
        or certificate.get("status") != "PASS"
        or certificate.get("metrics") != dict(metrics)
        or certificate.get("quality") != quality
        or certificate.get("descriptive_performance")
        != descriptive_performance
        or certificate.get("phase6_cpu_sha256")
        != PHASE6_CPU_SHA256
        or certificate.get("profiles_sha256") != PROFILES_SHA256
    ):
        raise Phase7EvidenceError(
            "Phase 7 derived evidence identity is invalid"
        )
    records = gates.get("records", [])
    if {row.get("seed") for row in records} != set(SEEDS):
        raise Phase7EvidenceError(
            "Phase 7 gate observations lack three seeds"
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
            raise Phase7EvidenceError(
                f"stale Phase 7 gate observation: {gate_id}"
            )
    claims = {
        row["gate_id"]: row for row in payload.get("claims", [])
    }
    if set(claims) != set(metrics):
        raise Phase7EvidenceError(
            "Phase 7 payload has missing or extra claims"
        )
    for gate_id, value in metrics.items():
        if not _close(claims[gate_id]["value"], value):
            raise Phase7EvidenceError(
                f"stale Phase 7 claim: {gate_id}"
            )


def derive_phase7_metrics(root: Path) -> dict[str, Any]:
    framework = _validate_framework(root)
    validate_phase6_bundle(
        root, _path(root, "results/moonshot/phase6")
    )
    _validate_cold(root)
    cpu, gpu, pairs = _validate_performance(root)
    cpu_supplement = _validate_cpu_supplement(root, cpu)
    retention = _validate_retention(root)
    general = _validate_general(root)
    superior, quality = _quality_superiority(cpu, gpu)
    metrics = _metrics(pairs, general, superior, retention)
    descriptive_performance = {
        "layercake_cpu": cpu_supplement["aggregates"]["layercake_cpu"],
        "transformer_cpu": _system_descriptives(
            cpu["records"], "qwen"
        ),
        "layercake_gpu": gpu["aggregates"]["layercake_gpu"],
        "transformer_gpu": gpu["aggregates"]["transformer_gpu"],
        "source_policy": {
            "layercake_cpu": CPU_SUPPLEMENT.as_posix(),
            "transformer_cpu": PHASE6_CPU.as_posix(),
            "layercake_gpu": GPU_PERFORMANCE.as_posix(),
            "transformer_gpu": GPU_PERFORMANCE.as_posix(),
            "headline_cpu_gate_source": PHASE6_CPU.as_posix(),
        },
    }
    _validate_derived(
        root, framework, metrics, quality, descriptive_performance
    )
    thresholds = _read(root, CONTRACT)["promotion_thresholds"]
    if not (
        metrics["cpu_cpu_throughput_ratio"]
        >= thresholds["cpu_cpu_throughput_ratio"]
        and metrics["cpu_cpu_median_latency_ratio"]
        <= thresholds["cpu_cpu_median_latency_ratio"]
        and metrics["gpu_gpu_throughput_ratio"]
        > thresholds[
            "gpu_gpu_throughput_ratio_strictly_greater_than"
        ]
        and metrics["cpu_gpu_throughput_ratio"]
        >= thresholds["cpu_gpu_throughput_ratio"]
        and metrics["cpu_gpu_median_latency_ratio"]
        <= thresholds["cpu_gpu_median_latency_ratio"]
        and metrics["general_quality_noninferior"] == 1.0
        and metrics["mixed_domain_quality_superior"] == 1.0
        and metrics["promoted_domain_success_retention"] == 1.0
    ):
        raise Phase7EvidenceError(
            f"Phase 7 promotion gates failed: {metrics}"
        )
    return {
        "status": "PASS",
        "framework_commit": framework["framework_commit"],
        "metrics": metrics,
        "quality": quality,
        "descriptive_performance": descriptive_performance,
        "seeds": list(SEEDS),
        "hardware": framework["hardware"],
        "claim_boundaries": {
            "physical_mobile_hardware_claimed": False,
            "gpu_training_dominance_claimed": False,
            "latent_neural_fusion_claimed": False,
        },
    }


def validate_phase7_bundle(
    root: Path, evidence_dir: Path
) -> dict[str, Any]:
    expected = _path(root, "results/moonshot/phase7")
    if evidence_dir.resolve() != expected:
        raise Phase7EvidenceError(
            "Phase 7 evidence directory is not canonical"
        )
    return derive_phase7_metrics(root)
