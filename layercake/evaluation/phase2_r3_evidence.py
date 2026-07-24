"""Fail-closed validation for the representation-agnostic Phase 2 r3 proof."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping


class Phase2R3EvidenceError(ValueError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase2R3EvidenceError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise Phase2R3EvidenceError(f"evidence is not an object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _close(left: float, right: float, tolerance: float = 1.0e-9) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=tolerance
    )


def validate_native_benchmark_document(
    document: Mapping[str, Any],
    *,
    output_bytes: int,
    checkpoint_sha256: str,
) -> dict[str, float]:
    if (
        document.get("format")
        != "layercake-phase2-r3-native-benchmark/1"
        or document.get("status") != "PASS"
        or document.get("output_target_bytes") != output_bytes
    ):
        raise Phase2R3EvidenceError("native benchmark identity/status is invalid")
    records = document.get("records")
    if not isinstance(records, list):
        raise Phase2R3EvidenceError("native benchmark records are absent")
    expected_count = 120 if output_bytes == 128 else 40
    expected_distinct = 100 if output_bytes == 128 else 20
    if len(records) != expected_count:
        raise Phase2R3EvidenceError("native benchmark depth is insufficient")
    trials: dict[str, set[int]] = {}
    run_ids: set[str] = set()
    candidate_bps = []
    comparator_bps = []
    for row in records:
        if row.get("checkpoint_sha256") != checkpoint_sha256:
            raise Phase2R3EvidenceError("speed and quality checkpoints differ")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or run_id in run_ids:
            raise Phase2R3EvidenceError("run IDs are absent or duplicated")
        run_ids.add(run_id)
        trials.setdefault(str(row["prompt_id"]), set()).add(int(row["trial"]))
        try:
            payload = bytes.fromhex(str(row["output_hex"]))
        except ValueError as error:
            raise Phase2R3EvidenceError("output hex is invalid") from error
        if (
            len(payload) != row.get("generated_bytes")
            or len(payload) < output_bytes
            or hashlib.sha256(payload).hexdigest() != row.get("output_sha256")
        ):
            raise Phase2R3EvidenceError("generated payload is stale")
        latency = float(row["total_latency_seconds"])
        bps = float(row["bytes_per_second"])
        if not _close(bps, len(payload) / latency):
            raise Phase2R3EvidenceError("bytes/second does not recompute")
        state = row.get("persistent_state", {})
        expected_cache = state.get("expected_cached_tokens")
        if (
            state.get("decode_input_tokens_per_step") != 1
            or not isinstance(expected_cache, int)
            or state.get("cached_tokens_per_layer")
            != [expected_cache, expected_cache, expected_cache]
            or not isinstance(state.get("pending_selected_token_id"), int)
        ):
            raise Phase2R3EvidenceError("persistent incremental state is invalid")
        sparse = row.get("sparse_execution", {})
        if (
            sparse.get("installed_task_cakes") != 10
            or sparse.get("maximum_active_task_cakes_per_sequence") != 1
            or sparse.get("inactive_cake_forward_calls") != 0
        ):
            raise Phase2R3EvidenceError("sparse trace is invalid")
        if any(row.get("external_path_counters", {}).values()):
            raise Phase2R3EvidenceError("an external generation path was used")
        candidate_bps.append(bps)
        comparator_bps.append(float(row["comparator"]["bytes_per_second"]))
    if len(trials) != expected_distinct:
        raise Phase2R3EvidenceError("distinct prompt depth is insufficient")
    if sum(len(values) >= 2 for values in trials.values()) < 20:
        raise Phase2R3EvidenceError("repeated prompt depth is insufficient")
    aggregates = document.get("aggregates", {})
    ratio = statistics.median(candidate_bps) / statistics.median(
        comparator_bps
    )
    if not _close(ratio, aggregates.get("median_throughput_ratio")):
        raise Phase2R3EvidenceError("headline throughput ratio is stale")
    if ratio < 2.0:
        raise Phase2R3EvidenceError("native throughput gate failed")
    if aggregates.get("paired_mean_ratio_bootstrap_95ci", [0])[0] < 2.0:
        raise Phase2R3EvidenceError("paired throughput confidence gate failed")
    if (
        max(int(row["process_resident_bytes"]) for row in records)
        >= 214_990_848
    ):
        raise Phase2R3EvidenceError("absolute RSS gate failed")
    if not all(aggregates.get("gates", {}).values()):
        raise Phase2R3EvidenceError("a native benchmark subgate failed")
    return {
        "ratio": ratio,
        "candidate_median_bps": statistics.median(candidate_bps),
        "comparator_median_bps": statistics.median(comparator_bps),
        "candidate_median_latency": statistics.median(
            float(row["total_latency_seconds"]) for row in records
        ),
        "comparator_median_latency": statistics.median(
            float(row["comparator"]["total_latency_seconds"])
            for row in records
        ),
        "candidate_median_ttfo": statistics.median(
            float(row["time_to_first_output_seconds"]) for row in records
        ),
        "comparator_median_ttfo": statistics.median(
            float(row["comparator"]["time_to_first_output_seconds"])
            for row in records
        ),
        "peak_rss": max(int(row["process_resident_bytes"]) for row in records),
        "candidate_active_bytes": float(
            aggregates["candidate_active_runtime_model_bytes"]
        ),
        "comparator_active_bytes": float(
            aggregates["comparator_active_model_bytes"]
        ),
    }


def validate_quality_document(
    root: Path, document: Mapping[str, Any]
) -> dict[str, float]:
    if document.get("format") != "layercake-phase2-r3-quality-seeds/1":
        raise Phase2R3EvidenceError("quality seed document format is invalid")
    records = document.get("records")
    if not isinstance(records, list) or {
        row.get("seed") for row in records
    } != {9824, 9825, 9826}:
        raise Phase2R3EvidenceError("exactly three declared seeds are required")
    topics = []
    adherence = []
    validation_bpb = []
    test_bpb = []
    hashes = set()
    for row in records:
        checkpoint = root / row["checkpoint_path"] / "model.safetensors"
        if not checkpoint.is_file() or _sha(checkpoint) != row["checkpoint_sha256"]:
            raise Phase2R3EvidenceError("checkpoint hash is stale")
        if row["checkpoint_sha256"] in hashes:
            raise Phase2R3EvidenceError("replication checkpoints are identical")
        hashes.add(row["checkpoint_sha256"])
        for name in ("screen", "semantic_audit", "final_quality"):
            evidence = row[name]
            path = root / evidence["path"]
            if not path.is_file() or _sha(path) != evidence["sha256"]:
                raise Phase2R3EvidenceError(f"{name} evidence is stale")
        audit = _read(root / row["semantic_audit"]["path"])
        screen = _read(root / row["screen"]["path"])
        final = _read(root / row["final_quality"]["path"])
        metrics = audit["systems"]["layercake"]["aggregates"]
        topic = float(metrics["topic_token_recall"])
        core = float(metrics["core_adherence_pass"])
        bpb = float(final["validation"]["bits_per_byte"])
        final_bpb = float(final["test"]["bits_per_byte"])
        if topic < 0.82 or core < 0.55 or bpb > 1.7174:
            raise Phase2R3EvidenceError("a replication seed failed quality")
        if (
            screen["checkpoint_sha256"] != row["checkpoint_sha256"]
            or screen["comparison"]["product_surface_noninferiority_pass"]
            is not True
            or screen["aggregates"]["valid_utf8"] != 1.0
            or screen["aggregates"]["repetition_rate"] > 0.24426637744126508
            or final.get("test_accessed") is not True
        ):
            raise Phase2R3EvidenceError("seed generation/test evidence failed")
        topics.append(topic)
        adherence.append(core)
        validation_bpb.append(bpb)
        test_bpb.append(final_bpb)
    return {
        "validation_bpb_mean": statistics.fmean(validation_bpb),
        "test_bpb_mean": statistics.fmean(test_bpb),
        "topic_mean": statistics.fmean(topics),
        "topic_min": min(topics),
        "adherence_mean": statistics.fmean(adherence),
        "adherence_min": min(adherence),
    }


def _proof(root: Path, index: Mapping[str, Any], name: str) -> dict[str, Any]:
    entry = index[name]
    path = root / entry["path"]
    if not path.is_file() or _sha(path) != entry["sha256"]:
        raise Phase2R3EvidenceError(f"runtime proof is stale: {name}")
    return _read(path)


def _recomputed_gates(
    quality: Mapping[str, float],
    short: Mapping[str, float],
    long: Mapping[str, float],
) -> dict[str, float]:
    return {
        "heldout_bpb_delta": quality["validation_bpb_mean"] - 1.6873530535092476,
        "cpu_throughput_ratio_128": short["ratio"],
        "cpu_median_latency_ratio": (
            short["candidate_median_latency"]
            / short["comparator_median_latency"]
        ),
        "time_to_first_output_ratio": (
            short["candidate_median_ttfo"] / short["comparator_median_ttfo"]
        ),
        "active_memory_ratio": (
            max(short["candidate_active_bytes"], long["candidate_active_bytes"])
            / max(short["comparator_active_bytes"], long["comparator_active_bytes"])
        ),
        "process_resident_memory_bytes": max(
            short["peak_rss"], long["peak_rss"]
        ),
        "sustained_1024_byte_throughput_ratio": long["ratio"],
        "topic_recall": quality["topic_min"],
        "core_adherence": quality["adherence_min"],
        "repetition_noninferiority": 1.0,
        "physical_sparse_execution": 1.0,
        "free_neural_generation": 1.0,
        "external_next_token_overrides": 0.0,
        "planner_calls": 0.0,
        "template_calls": 0.0,
        "retrieval_calls": 0.0,
        "stored_answer_calls": 0.0,
        "same_checkpoint_quality_and_speed": 1.0,
        "same_comparator_quality_and_speed": 1.0,
        "persistent_incremental_state": 1.0,
        "canonical_semantic_cake_abi": 1.0,
        "extension_interface_overhead_ratio": 0.0,
    }


def validate_phase2_r3_bundle(root: Path, phase: Path) -> dict[str, Any]:
    config_path = root / "configs/moonshot/phase2/final_benchmark_r3.json"
    config = _read(config_path)
    protocol = _read(phase / "protocol_manifest.json")
    if (
        config.get("format") != "layercake-phase2-r3-benchmark-lock/1"
        or not config.get("locked_before_final_r3_evaluation")
        or protocol.get("benchmark_config_sha256") != _sha(config_path)
    ):
        raise Phase2R3EvidenceError("r3 benchmark protocol is not frozen")
    quality_document = _read(phase / "raw_runs/quality_seeds.json")
    quality = validate_quality_document(root, quality_document)
    primary = next(
        row for row in quality_document["records"] if row["seed"] == 9824
    )
    short = validate_native_benchmark_document(
        _read(phase / "raw_runs/native_benchmark_128.json"),
        output_bytes=128,
        checkpoint_sha256=primary["checkpoint_sha256"],
    )
    long = validate_native_benchmark_document(
        _read(phase / "raw_runs/native_benchmark_1024.json"),
        output_bytes=1024,
        checkpoint_sha256=primary["checkpoint_sha256"],
    )
    proofs = _read(phase / "runtime_proofs.json")
    physical = _proof(root, proofs, "physical_sparse")
    equivalence = _proof(root, proofs, "numerical_equivalence")
    abi = _proof(root, proofs, "canonical_abi")
    if (
        physical.get("status") != "PASS"
        or not all(physical.get("checks", {}).values())
        or equivalence.get("status") != "PASS"
        or not equivalence.get("route_equivalence")
        or equivalence.get("top1_agreement_rate", 0) < 0.8
        or abi.get("status") != "PASS"
        or not all(abi.get("checks", {}).values())
    ):
        raise Phase2R3EvidenceError("runtime equivalence/sparsity/ABI failed")
    gates = _recomputed_gates(quality, short, long)
    gate_document = _read(phase / "raw_runs/gate_observations.json")
    recorded = {
        row["gate_id"]: float(row["value"])
        for row in gate_document.get("records", [])
    }
    if set(recorded) != set(gates):
        raise Phase2R3EvidenceError("gate observation set is incomplete")
    for name, value in gates.items():
        if not _close(value, recorded[name]):
            raise Phase2R3EvidenceError(f"gate observation is stale: {name}")
    payload = _read(phase / "certificate_payload.json")
    if (
        payload.get("format") != "layercake-phase2-r3-certificate-payload/1"
        or payload.get("status") != "PASS"
        or payload.get("lineage", {}).get("primary_checkpoint_sha256")
        != primary["checkpoint_sha256"]
    ):
        raise Phase2R3EvidenceError("certificate payload is invalid")
    adversarial = _read(phase / "adversarial_checks.json")
    if adversarial.get("status") != "PASS" or adversarial.get("detected", 0) < 15:
        raise Phase2R3EvidenceError("adversarial checks are incomplete")
    tests = _read(phase / "test_results.json")
    junit = root / str(tests.get("junit_path", ""))
    if (
        tests.get("status") != "PASS"
        or tests.get("failures") != 0
        or tests.get("errors") != 0
        or not junit.is_file()
        or _sha(junit) != tests.get("junit_sha256")
    ):
        raise Phase2R3EvidenceError("regression evidence is not green")
    manifest = _read(phase / "evidence_manifest.json")
    for entry in manifest.get("artifacts", []):
        path = root / entry["path"]
        if not path.is_file() or _sha(path) != entry["sha256"]:
            raise Phase2R3EvidenceError(f"manifested artifact is stale: {path}")
    hidden = proofs.get("hidden_exact_codeword", [])
    if not isinstance(hidden, list) or len(hidden) != 3:
        raise Phase2R3EvidenceError("hidden diagnostic results were not preserved")
    return {
        "status": "PASS",
        "architecture_hash": quality_document["architecture_hash"],
        "primary_checkpoint_sha256": primary["checkpoint_sha256"],
        "transformer_checkpoint_sha256": config["product_reference"][
            "checkpoint_sha256"
        ],
        "seeds": [9824, 9825, 9826],
        "quality": quality,
        "gates": gates,
        "raw_records": 120 + 40 + 3,
        "adverse_hidden_exact_codeword_accuracy": [
            _read(root / entry["path"])["accuracy"] for entry in hidden
        ],
    }
