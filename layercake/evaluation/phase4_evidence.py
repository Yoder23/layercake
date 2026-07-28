"""Typed, fail-closed verification for the Phase 4 direct neural cake."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any, Mapping

import torch

from layercake.cake.package import load_package
from layercake.portable_domain import state_dict_hash


class Phase4EvidenceError(RuntimeError):
    """Raised when Phase 4 raw evidence does not support promotion."""


DIRECT_ABI_VERSION = "lc-direct-neural-decoder/1"
DIRECT_ABI_SHA256 = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)
PACKAGE_SHA256 = (
    "0585c79bfbea16b1c4165bf0030ba6985b8a8cdeab529cd7afe3f9c76c564ef7"
)
ARTIFACT_SHA256 = (
    "211db7a97194234f4fbf2a04c99f99eea4698cc73bd486f901e7d12fb8af439e"
)
ARTIFACT_PAYLOAD_HASH = (
    "a70fcb62a2c24305ca0dad3929124ca6881363d26366c9374b96e7951b5cd49f"
)
QWEN_DIGEST = (
    "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67"
)
PHASE2_CORE_RATIO = 2.183032521299001

PACKAGE = Path(
    "artifacts/moonshot/phase4/release/"
    "python-token-plan-seed10141-direct-v1.0.0.cake"
)
ARTIFACT = Path(
    "artifacts/moonshot/phase4/candidates/"
    "python-portable-token-plan-seed10141.pt"
)
PUBLIC_KEY = Path("moonshot/phase4-direct-token-plan-publisher.public.pem")
TRANSFER = Path(
    "results/moonshot/phase4/direct_decoder_transfer_certificate.json"
)
BENCHMARK = Path(
    "results/moonshot/phase4/direct_decoder_cpu_product_benchmark.json"
)
BASELINE = Path(
    "results/moonshot/phase4/baseline_validation_seed9824.json"
)
GATE_OBSERVATIONS = Path(
    "results/moonshot/phase4/gate_observations.json"
)
PAYLOAD = Path(
    "results/moonshot/phase4/certificate_payload.json"
)
REGRESSION_SUMMARY = Path(
    "results/moonshot/phase4/regression_tests.json"
)
REGRESSION_JUNIT = Path(
    "results/moonshot/phase4/regression_tests.xml"
)


def _path(root: Path, relative: Path) -> Path:
    value = (root / relative).resolve()
    try:
        value.relative_to(root.resolve())
    except ValueError as error:
        raise Phase4EvidenceError(
            f"Phase 4 path escapes repository: {relative}"
        ) from error
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(root: Path, relative: Path) -> dict[str, Any]:
    path = _path(root, relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase4EvidenceError(
            f"cannot read Phase 4 evidence {relative}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise Phase4EvidenceError(f"{relative} is not an object")
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
    document: Mapping[str, Any], relative: Path
) -> None:
    observed = document.get("evidence_sha256")
    expected = _canonical_sha(document)
    if observed != expected:
        raise Phase4EvidenceError(
            f"{relative} evidence hash is stale: {observed} != {expected}"
        )


def _close(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=tolerance
    )


def _seed_paths(seed: int) -> tuple[Path, Path, Path]:
    base = f"results/moonshot/phase4/python-portable-token-plan-seed{seed}"
    return (
        Path(f"{base}-training.json"),
        Path(f"{base}-lexical.json"),
        Path(f"{base}-functional.json"),
    )


def phase4_evidence_files(root: Path) -> list[Path]:
    """Return only promoted Phase 4 evidence, excluding historical branches."""

    relatives: list[Path] = [
        BASELINE,
        TRANSFER,
        BENCHMARK,
        GATE_OBSERVATIONS,
        PAYLOAD,
        REGRESSION_SUMMARY,
        REGRESSION_JUNIT,
        Path(
            "results/moonshot/phase4/"
            "python-portable-token-plan-seed10141-test.json"
        ),
        PACKAGE,
        ARTIFACT,
        PUBLIC_KEY,
        Path("moonshot/phase4-direct-token-plan-trust-store.json"),
        Path("moonshot/phase4_canonical_direct_decoder_abi_v1.json"),
        Path("moonshot/phase4_direct_decoder_interface_amendment.json"),
        Path("moonshot/phase4_direct_decoder_release_preregistration.json"),
        Path("moonshot/phase4_direct_decoder_runtime_protocol.json"),
        Path("moonshot/phase4_cpu_cake_training_lock.json"),
    ]
    for seed in (10140, 10141, 10142):
        relatives.extend(_seed_paths(seed))
    return [_path(root, relative) for relative in relatives]


def _validate_training_seeds(root: Path) -> dict[str, Any]:
    rows = []
    for seed in (10140, 10141, 10142):
        training_path, lexical_path, functional_path = _seed_paths(seed)
        training = _read(root, training_path)
        lexical = _read(root, lexical_path)
        functional = _read(root, functional_path)
        for document, relative in (
            (training, training_path),
            (lexical, lexical_path),
            (functional, functional_path),
        ):
            _validate_self_hash(document, relative)
        if (
            training.get("format")
            != "layercake-phase4-portable-token-plan-training/1"
            or training.get("status") != "TRAINED"
            or training.get("seed") != seed
            or training.get("primary_device_name")
            != "NVIDIA GeForce RTX 3080 Laptop GPU"
            or training.get("precision") != "fp32"
            or training.get("cpu_fallback_smoke", {}).get("status")
            != "PASS"
        ):
            raise Phase4EvidenceError(
                f"seed {seed} training evidence is invalid"
            )
        if (
            lexical.get("status") != "PASS"
            or lexical.get("distinct_prompts") != 256
            or lexical.get("exact_prefix_successes") != 256
        ):
            raise Phase4EvidenceError(
                f"seed {seed} lexical evidence failed"
            )
        successes = functional.get("functional_successes")
        if (
            functional.get("status") != "PASS"
            or functional.get("distinct_prompts") != 64
            or not isinstance(successes, int)
            or successes < 52
        ):
            raise Phase4EvidenceError(
                f"seed {seed} functional evidence failed"
            )
        rows.append(
            {
                "seed": seed,
                "functional_successes": successes,
                "lexical_successes": 256,
                "gpu_wall_seconds": training["gpu_wall_seconds"],
                "peak_accelerator_memory_bytes": training[
                    "peak_accelerator_memory_bytes"
                ],
            }
        )
    return {"seeds": rows, "unique_seeds": 3}


def _validate_package_and_transfer(root: Path) -> dict[str, Any]:
    package_path = _path(root, PACKAGE)
    artifact_path = _path(root, ARTIFACT)
    if _sha256(package_path) != PACKAGE_SHA256:
        raise Phase4EvidenceError("direct cake archive changed")
    if _sha256(artifact_path) != ARTIFACT_SHA256:
        raise Phase4EvidenceError("selected training artifact changed")
    artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=True
    )
    if artifact.get("payload_hash") != ARTIFACT_PAYLOAD_HASH:
        raise Phase4EvidenceError("selected artifact payload hash changed")
    transfer = _read(root, TRANSFER)
    _validate_self_hash(transfer, TRANSFER)
    if (
        transfer.get("format")
        != "layercake-phase4-direct-decoder-transfer-certificate/1"
        or transfer.get("status") != "PASS"
        or transfer.get("package", {}).get("archive_sha256")
        != PACKAGE_SHA256
        or transfer.get("canonical_interface", {}).get("version")
        != DIRECT_ABI_VERSION
        or transfer.get("canonical_interface", {}).get("sha256")
        != DIRECT_ABI_SHA256
        or transfer.get("canonical_interface", {}).get("conforms")
        is not True
    ):
        raise Phase4EvidenceError(
            "direct-decoder transfer certificate is invalid"
        )
    key_id = transfer["package"]["key_id"]
    package = load_package(
        package_path,
        trust_store={key_id: _path(root, PUBLIC_KEY)},
    )
    if (
        not package.signed
        or package.manifest.abi_version != DIRECT_ABI_VERSION
        or package.manifest.abi_hash != DIRECT_ABI_SHA256
        or package.manifest.cake_type != "portable_decoder"
        or package.manifest.input_contract.get("mode")
        != "direct_selected_portable_decoder"
        or package.manifest.output_contract.get("composition")
        != "direct_selected_one_cake_no_router"
        or state_dict_hash(package.tensors) != ARTIFACT_PAYLOAD_HASH
    ):
        raise Phase4EvidenceError(
            "signed package does not contain the selected direct-ABI payload"
        )
    tracked_private = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            PACKAGE.with_suffix(".private.pem").as_posix(),
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if tracked_private:
        raise Phase4EvidenceError("private package key is committed")
    source_records = transfer.get("source", {}).get("records")
    receivers = transfer.get("receivers")
    if (
        not isinstance(source_records, list)
        or len(source_records) != 128
        or not all(row.get("functional_success") for row in source_records)
        or not isinstance(receivers, list)
        or len(receivers) != 3
    ):
        raise Phase4EvidenceError(
            "transfer source or receiver population is incomplete"
        )
    source_ids = {
        row["id"]
        for row in source_records
        if row.get("functional_success")
    }
    receiver_summaries = []
    for receiver in receivers:
        records = receiver.get("records")
        success_ids = {
            row["id"]
            for row in records
            if isinstance(row, dict) and row.get("functional_success")
        } if isinstance(records, list) else set()
        if (
            receiver.get("core_unchanged") is not True
            or receiver.get("archive_hash_equal") is not True
            or receiver.get("tensor_identity") is not True
            or receiver.get("output_identity") is not True
            or receiver.get("reinstall_output_identity") is not True
            or receiver.get("retention_rate") != 1.0
            or receiver.get("receiver_training_examples") != 0
            or receiver.get("receiver_calibration_runs") != 0
            or not source_ids.issubset(success_ids)
        ):
            raise Phase4EvidenceError(
                f"receiver {receiver.get('seed')} lost package identity "
                "or semantic behavior"
            )
        receiver_summaries.append(
            {
                "seed": receiver["seed"],
                "device": receiver["device"],
                "success_task_ids": sorted(success_ids),
                "receiver_training_examples": 0,
                "calibration_performed": False,
            }
        )
    devices = [row["device"] for row in receiver_summaries]
    if devices.count("cpu") != 2 or "cuda:0" not in devices:
        raise Phase4EvidenceError("CPU/CUDA receiver matrix changed")
    incremental = transfer.get("incremental_execution", {})
    if (
        incremental.get("status") != "PASS"
        or {
            row.get("device")
            for row in incremental.get("devices", [])
            if row.get("status") == "PASS"
            and row.get("completed_prefix_recomputation") is False
        }
        != {"cpu", "cuda:0"}
        or transfer.get("adversarial_package_test", {}).get("rejected")
        is not True
    ):
        raise Phase4EvidenceError(
            "incremental or adversarial package verification failed"
        )
    return {
        "archive_sha256": package.archive_hash,
        "tensor_payload_hash": package.manifest.tensor_payload_hash,
        "active_tensor_bytes": sum(
            value.numel() * value.element_size()
            for value in package.tensors.values()
        ),
        "source_success_task_ids": sorted(source_ids),
        "receivers": receiver_summaries,
    }


def _bootstrap_mean(
    values: list[float], *, samples: int = 10000
) -> list[float]:
    rng = random.Random(4404)
    means = []
    for _ in range(samples):
        means.append(
            statistics.fmean(
                values[rng.randrange(len(values))]
                for _ in range(len(values))
            )
        )
    means.sort()
    return [
        means[int(0.025 * samples)],
        means[int(0.975 * samples) - 1],
    ]


def _validate_benchmark(
    root: Path, package_summary: Mapping[str, Any]
) -> dict[str, float | int]:
    evidence = _read(root, BENCHMARK)
    _validate_self_hash(evidence, BENCHMARK)
    if (
        evidence.get("format")
        != "layercake-phase4-direct-decoder-cpu-product-benchmark/1"
        or evidence.get("status") != "PASS"
        or evidence.get("package", {}).get("archive_sha256")
        != package_summary["archive_sha256"]
        or evidence.get("package", {}).get("tensor_payload_hash")
        != package_summary["tensor_payload_hash"]
    ):
        raise Phase4EvidenceError("CPU benchmark identity is invalid")
    records = evidence.get("records")
    if not isinstance(records, list) or len(records) != 120:
        raise Phase4EvidenceError("CPU benchmark must have 120 paired rows")
    prompt_ids = [row.get("prompt_id") for row in records]
    if len(set(prompt_ids)) != 100:
        raise Phase4EvidenceError(
            "CPU benchmark must have 100 distinct prompts"
        )
    counts = {identifier: prompt_ids.count(identifier) for identifier in set(prompt_ids)}
    if (
        sum(value == 2 for value in counts.values()) != 20
        or any(value not in {1, 2} for value in counts.values())
    ):
        raise Phase4EvidenceError(
            "CPU benchmark repeat structure is invalid"
        )
    layercake_bps = [
        float(row["layercake"]["timing"]["bytes_per_second_total"])
        for row in records
    ]
    qwen_bps = [
        float(row["qwen"]["timing"]["bytes_per_second_total"])
        for row in records
    ]
    ratios = [
        layercake / qwen
        for layercake, qwen in zip(layercake_bps, qwen_bps)
    ]
    layercake_ttft = [
        float(row["layercake"]["timing"]["time_to_first_output_seconds"])
        for row in records
    ]
    qwen_ttft = [
        float(row["qwen"]["timing"]["time_to_first_output_seconds"])
        for row in records
    ]
    first_hundred = records[:100]
    layercake_successes = sum(
        bool(row["layercake"]["functional_success"])
        for row in first_hundred
    )
    qwen_successes = sum(
        bool(row["qwen"]["functional_success"])
        for row in first_hundred
    )
    before = evidence["core_sentinel_before"]
    after = evidence["core_sentinel_after"]
    if (
        [row["output_sha256"] for row in before["records"]]
        != [row["output_sha256"] for row in after["records"]]
    ):
        raise Phase4EvidenceError("sealed core output changed")
    retention = (
        float(after["median_bytes_per_second"])
        / float(before["median_bytes_per_second"])
    )
    bootstrap = _bootstrap_mean(ratios)
    recomputed = {
        "distinct_prompts": 100,
        "repeated_prompt_observations": 20,
        "observations_per_system": 120,
        "layercake_functional_successes": layercake_successes,
        "qwen_functional_successes": qwen_successes,
        "layercake_median_bytes_per_second": statistics.median(
            layercake_bps
        ),
        "qwen_median_bytes_per_second": statistics.median(qwen_bps),
        "median_paired_throughput_ratio": statistics.median(ratios),
        "mean_paired_throughput_ratio": statistics.fmean(ratios),
        "bootstrap_lower": bootstrap[0],
        "bootstrap_upper": bootstrap[1],
        "layercake_median_ttft_seconds": statistics.median(
            layercake_ttft
        ),
        "qwen_median_ttft_seconds": statistics.median(qwen_ttft),
        "median_ttft_ratio": (
            statistics.median(layercake_ttft)
            / statistics.median(qwen_ttft)
        ),
        "phase2_core_sentinel_retention": retention,
        "phase2_core_plus_inactive_cake_transformer_ratio": (
            PHASE2_CORE_RATIO * retention
        ),
    }
    aggregate = evidence.get("aggregates", {})
    comparisons = {
        "layercake_median_bytes_per_second": (
            "layercake_median_bytes_per_second"
        ),
        "qwen_median_bytes_per_second": "qwen_median_bytes_per_second",
        "median_paired_throughput_ratio": (
            "median_paired_throughput_ratio"
        ),
        "mean_paired_throughput_ratio": "mean_paired_throughput_ratio",
        "layercake_median_ttft_seconds": "layercake_median_ttft_seconds",
        "qwen_median_ttft_seconds": "qwen_median_ttft_seconds",
        "median_ttft_ratio": "median_ttft_ratio",
        "phase2_core_sentinel_retention": (
            "phase2_core_sentinel_retention"
        ),
        "phase2_core_plus_inactive_cake_transformer_ratio": (
            "phase2_core_plus_inactive_cake_transformer_ratio"
        ),
    }
    for observed_key, recomputed_key in comparisons.items():
        if not _close(
            aggregate.get(observed_key, float("nan")),
            recomputed[recomputed_key],
        ):
            raise Phase4EvidenceError(
                f"benchmark aggregate {observed_key} is not raw-derived"
            )
    reported_bootstrap = aggregate.get(
        "paired_mean_ratio_bootstrap_95ci"
    )
    if (
        not isinstance(reported_bootstrap, list)
        or len(reported_bootstrap) != 2
        or not all(
            _close(left, right)
            for left, right in zip(reported_bootstrap, bootstrap)
        )
    ):
        raise Phase4EvidenceError(
            "paired bootstrap interval is not raw-derived"
        )
    qwen_models = aggregate.get("qwen_model_report", {}).get("models", [])
    if (
        not qwen_models
        or qwen_models[0].get("digest") != QWEN_DIGEST
        or qwen_models[0].get("size_vram") != 0
    ):
        raise Phase4EvidenceError(
            "optimized transformer identity or CPU placement changed"
        )
    if (
        layercake_successes != 100
        or recomputed["median_paired_throughput_ratio"] < 2.0
        or bootstrap[0] < 2.0
        or recomputed["median_ttft_ratio"] > 1.0
        or package_summary["active_tensor_bytes"]
        >= int(qwen_models[0]["size"])
        or retention < 0.9
        or recomputed[
            "phase2_core_plus_inactive_cake_transformer_ratio"
        ]
        < 2.0
        or any(
            row["layercake"]["persistent_state"].get(
                "completed_prefix_recomputation"
            )
            is not False
            for row in records
        )
    ):
        raise Phase4EvidenceError("one or more CPU product gates failed")
    return recomputed


def derive_phase4_metrics(root: Path) -> dict[str, Any]:
    """Recompute all Phase 4 promotion values from raw evidence."""

    seed_summary = _validate_training_seeds(root)
    package_summary = _validate_package_and_transfer(root)
    benchmark = _validate_benchmark(root, package_summary)
    baseline = _read(root, BASELINE)
    _validate_self_hash(baseline, BASELINE)
    selected = _read(
        root,
        Path(
            "results/moonshot/phase4/"
            "python-portable-token-plan-seed10141-functional.json"
        ),
    )
    if (
        baseline.get("system") != "frozen_core"
        or baseline.get("distinct_prompts") != 64
        or selected.get("distinct_prompts") != 64
        or baseline.get("dataset_sha256") != selected.get("dataset_sha256")
    ):
        raise Phase4EvidenceError(
            "frozen-core and selected validation suites are not matched"
        )
    baseline_failures = int(baseline["functional_failures"])
    selected_failures = int(selected["functional_failures"])
    if baseline_failures <= 0:
        raise Phase4EvidenceError(
            "functional error ratio denominator is not positive"
        )
    receiver_training = sum(
        receiver["receiver_training_examples"]
        for receiver in package_summary["receivers"]
    )
    receiver_calibration = sum(
        int(receiver["calibration_performed"])
        for receiver in package_summary["receivers"]
    )
    metrics = {
        "functional_error_ratio": selected_failures / baseline_failures,
        "core_plus_cake_phase3_throughput_retention": benchmark[
            "phase2_core_sentinel_retention"
        ],
        "core_plus_cake_cpu_transformer_ratio": benchmark[
            "phase2_core_plus_inactive_cake_transformer_ratio"
        ],
        "english_core_parameters_changed": 0.0,
        "source_success_retention_ratio": 1.0,
        "receiver_training_examples": float(receiver_training),
        "receiver_calibration_runs": float(receiver_calibration),
        "identical_package_bytes": 1.0,
        "inactive_cake_compute": 0.0,
    }
    return {
        "metrics": metrics,
        "seeds": seed_summary,
        "package": package_summary,
        "benchmark": benchmark,
    }


def validate_phase4_bundle(root: Path, phase_dir: Path) -> dict[str, Any]:
    del phase_dir
    derived = derive_phase4_metrics(root)
    observations = _read(root, GATE_OBSERVATIONS)
    _validate_self_hash(observations, GATE_OBSERVATIONS)
    if observations.get("format") != "layercake-phase4-gate-observations/1":
        raise Phase4EvidenceError("Phase 4 gate observations are invalid")
    observed = {
        row.get("gate_id"): row.get("value")
        for row in observations.get("records", [])
        if isinstance(row, dict)
    }
    if set(observed) != set(derived["metrics"]):
        raise Phase4EvidenceError(
            "Phase 4 gate observation set is incomplete"
        )
    for gate_id, value in derived["metrics"].items():
        if not _close(observed[gate_id], value):
            raise Phase4EvidenceError(
                f"Phase 4 gate {gate_id} is not raw-derived"
            )
    payload = _read(root, PAYLOAD)
    if (
        payload.get("format")
        != "layercake-phase4-certificate-payload/1"
        or payload.get("phase") != 4
        or payload.get("status") != "EVIDENCE_READY"
        or payload.get("abi_hash") != DIRECT_ABI_SHA256
        or payload.get("package", {}).get("archive_sha256")
        != PACKAGE_SHA256
    ):
        raise Phase4EvidenceError("Phase 4 certificate payload is invalid")
    tests = _read(root, REGRESSION_SUMMARY)
    if (
        tests.get("status") != "PASS"
        or tests.get("failures") != 0
        or tests.get("errors") != 0
        or tests.get("tests", 0) <= 0
        or tests.get("junit_sha256")
        != _sha256(_path(root, REGRESSION_JUNIT))
    ):
        raise Phase4EvidenceError("Phase 4 regression suite is not green")
    return {
        "scope": "one_useful_lossless_portable_python_domain",
        "canonical_interface": DIRECT_ABI_VERSION,
        "canonical_interface_sha256": DIRECT_ABI_SHA256,
        "unique_training_seeds": derived["seeds"]["unique_seeds"],
        "seed_functional_successes": [
            row["functional_successes"]
            for row in derived["seeds"]["seeds"]
        ],
        "source_success_tasks": len(
            derived["package"]["source_success_task_ids"]
        ),
        "receiver_hosts": len(derived["package"]["receivers"]),
        "receiver_devices": [
            row["device"] for row in derived["package"]["receivers"]
        ],
        "functional_error_ratio": derived["metrics"][
            "functional_error_ratio"
        ],
        "median_paired_cpu_throughput_ratio": derived["benchmark"][
            "median_paired_throughput_ratio"
        ],
        "paired_mean_ratio_bootstrap_95ci": [
            derived["benchmark"]["bootstrap_lower"],
            derived["benchmark"]["bootstrap_upper"],
        ],
        "median_ttft_ratio": derived["benchmark"]["median_ttft_ratio"],
        "active_tensor_bytes": derived["package"][
            "active_tensor_bytes"
        ],
        "regression_tests": tests["tests"],
    }
