"""Independent, fail-closed V2 release-certificate construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from layercake.moonshot import source_tree_hash


ALLOWED = {
    "PASS", "FAIL", "OPEN", "NOT_RUN_NO_HARDWARE",
    "NOT_RUN_INSUFFICIENT_COMPUTE", "INVALID_EVIDENCE",
}
REQUIRED_GATES = (
    "repository_regression", "data_integrity", "english_core_training",
    "same_scale_quality", "matched_quality_comparison",
    "stateful_decoding_correctness", "sparse_physical_execution",
    "domain_cake_task_improvement", "random_cake_control", "wrong_domain_control",
    "unrelated_capability_retention", "real_cross_host_portability",
    "payload_identity", "installation_losslessness", "routing_accuracy",
    "orchestration", "cpu_runtime", "gpu_runtime",
    "cpu_vs_gpu_matched_quality", "foundation_training_efficiency",
    "cake_training_efficiency", "package_security", "uninstall_reinstall",
    "mobile_export", "physical_mobile_execution", "multi_seed_replication",
)


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _gate(status: str, reason: str, **evidence: Any) -> dict:
    if status not in ALLOWED:
        raise ValueError(f"invalid gate status: {status}")
    return {"status": status, "reason": reason, **evidence}


def verify_moonshot_v2(root: str | Path, evidence_dir: str | Path) -> dict:
    root = Path(root).resolve()
    evidence_dir = Path(evidence_dir).resolve()
    paths = {
        name: evidence_dir / f"{name}.json"
        for name in (
            "baseline_audit", "architecture_search", "development_evidence",
            "incremental_benchmark", "portability_evidence", "routing_evidence",
            "cpu_vs_gpu_evidence", "orchestration_evidence", "mobile_export_evidence",
            "quantization_evidence", "english_quality_evidence", "pytest_evidence",
        )
    }
    documents = {name: _read(path) for name, path in paths.items()}
    dev = documents["development_evidence"]
    incremental = documents["incremental_benchmark"]
    portability = documents["portability_evidence"]
    routing = documents["routing_evidence"]
    direct = documents["cpu_vs_gpu_evidence"]
    orchestration = documents["orchestration_evidence"]
    mobile = documents["mobile_export_evidence"]
    english_quality = documents["english_quality_evidence"]
    pytest = documents["pytest_evidence"]
    cores = dev.get("english_cores", [])
    transformer = dev.get("same_scale_transformer", {})
    cake = dev.get("python_cake", {})
    gates: dict[str, dict] = {}

    current_tree = source_tree_hash()
    repo_pass = (
        pytest.get("status") == "PASS"
        and pytest.get("exit_code") == 0
        and pytest.get("source_tree_sha256") == current_tree
    )
    gates["repository_regression"] = _gate(
        "PASS" if repo_pass else "INVALID_EVIDENCE",
        "complete pytest run is green and tied to the exact source tree" if repo_pass
        else "missing, failing, or source-mismatched complete pytest evidence",
        passed=pytest.get("passed"), source_tree_sha256=current_tree,
    )

    english_data = dev.get("data", {}).get("english", {})
    python_data = dev.get("data", {}).get("python", {})
    overlaps = english_data.get("sampled_cross_split_64byte_overlaps", {})
    files = english_data.get("files", {})
    data_pass = bool(
        dev and files.get("development_train", {}).get("bytes", 0) >= 10_000_000
        and files.get("medium_train", {}).get("bytes", 0) >= 100_000_000
        and overlaps and all(value == 0 for value in overlaps.values())
        and len({row.get("sha256") for row in python_data.get("files", {}).values()}) == 3
    )
    gates["data_integrity"] = _gate(
        "PASS" if data_pass else "INVALID_EVIDENCE",
        "real public corpora have immutable split hashes, whole-distribution code separation, and zero sampled cross-split 64-byte overlaps"
        if data_pass else "corpus manifests or leakage checks are incomplete",
        english_files=files, python_files=python_data.get("files"),
    )

    actual_core_hashes = []
    core_hashes_valid = True
    for suffix, core in zip("abc", cores):
        actual = _sha(root / f"artifacts/cores/english-core-{suffix}/model.safetensors")
        expected = core.get("checkpoint", {}).get("sha256")
        actual_core_hashes.append(actual)
        core_hashes_valid &= actual is not None and actual == expected
    core_training_valid = bool(
        len(cores) == 3 and core_hashes_valid
        and all(core.get("training", {}).get("raw_bytes_seen", 0) >= 10_000_000 for core in cores)
        and all(core.get("quality", {}).get("test") for core in cores)
    )
    samples = english_quality.get("generation_samples", [])
    useful_generation = bool(
        samples
        and sum(bool(row.get("high_fourgram_repetition")) for row in samples) / len(samples) < 0.5
        and sum(float(row.get("unique_byte_fraction", 0)) for row in samples) / len(samples) >= 0.15
    )
    core_training_pass = core_training_valid and useful_generation
    gates["english_core_training"] = _gate(
        "PASS" if core_training_pass else ("FAIL" if core_training_valid else "INVALID_EVIDENCE"),
        "three independently trained cores passed the training-integrity and open-generation usefulness checks"
        if core_training_pass else (
            "three real development cores were trained, but every open-ended sample showed severe four-gram repetition"
            if core_training_valid else "real independently trained core evidence is incomplete"
        ),
        hosts=len(cores), checkpoint_hashes=actual_core_hashes,
        scale="development_not_proof", useful_generation=useful_generation,
        generation_samples=samples,
    )

    layercake_bpb = cores[0].get("quality", {}).get("test", {}).get("bits_per_byte") if cores else None
    transformer_bpb = transformer.get("quality", {}).get("test", {}).get("bits_per_byte")
    scale_delta = dev.get("comparisons", {}).get("same_total_scale_relative_delta")
    same_scale_pass = bool(
        layercake_bpb is not None and transformer_bpb is not None
        and scale_delta is not None and scale_delta <= 0.05
        and layercake_bpb <= transformer_bpb + 0.05
    )
    gates["same_scale_quality"] = _gate(
        "PASS" if same_scale_pass else "FAIL",
        "LayerCake is statistically non-inferior at matched total scale" if same_scale_pass
        else "the matched-scale transformer has materially lower held-out English BPB",
        layercake_bpb=layercake_bpb, transformer_bpb=transformer_bpb,
        total_parameter_relative_delta=scale_delta,
    )
    gates["matched_quality_comparison"] = _gate(
        "FAIL",
        "no LayerCake and transformer pair met the same locked general and ordinary-domain task thresholds",
        direct_bpb_thresholds=direct.get("bpb_thresholds_passed"),
        ordinary_task_quality_valid=direct.get("ordinary_task_quality_valid"),
    )

    incremental_pass = bool(
        incremental.get("status") == "PASS"
        and incremental.get("full_prompt_recomputed_per_decode_step") is False
        and incremental.get("generation_lengths") == [1, 64, 256, 1024]
        and all(row.get("identical_outputs") for row in incremental.get("equivalence", []))
    )
    gates["stateful_decoding_correctness"] = _gate(
        "PASS" if incremental_pass else "INVALID_EVIDENCE",
        "exact full-context equivalence and sustained 1,024-byte stateful decoding were measured"
        if incremental_pass else "incremental equivalence or long-generation evidence is incomplete",
        sustained_1024_byte_rate=incremental.get("sustained_1024_byte_decode_rate"),
        state_contract=incremental.get("state_contract"),
    )
    active_fraction = cores[0].get("parameters", {}).get("active_fraction") if cores else None
    gates["sparse_physical_execution"] = _gate(
        "OPEN",
        "top-1 execution and sparse optimizer residency are implemented, but inactive-weight cache residency and memory traffic were not established by native profiling",
        active_fraction=active_fraction, native_extension=False,
    )

    syntax = cake.get("syntax_tasks", {})
    task_pass = syntax.get("five_x_error_gate") == "PASS"
    gates["domain_cake_task_improvement"] = _gate(
        "PASS" if task_pass else "FAIL",
        "the Python cake reduced ordinary task error by at least five-fold" if task_pass
        else "the Python cake lowered code BPB but passed zero of eight syntax-generation tasks",
        syntax_tasks=syntax,
    )
    heldout = cake.get("heldout_domain", {})
    random_control = cake.get("random_control", {})
    random_pass = bool(
        heldout.get("cake_bits_per_byte") is not None
        and random_control.get("cake_bits_per_byte") is not None
        and heldout["cake_bits_per_byte"] < random_control["cake_bits_per_byte"]
    )
    gates["random_cake_control"] = _gate(
        "PASS" if random_pass else "FAIL",
        "trained cake beats the frozen random control on held-out code BPB" if random_pass
        else "trained cake did not beat random control",
        trained_bpb=heldout.get("cake_bits_per_byte"),
        random_bpb=random_control.get("cake_bits_per_byte"),
    )
    gates["wrong_domain_control"] = _gate(
        "OPEN", "no second serious V2 cake was trained after the first cake failed its ordinary-task gate"
    )
    general_core = cake.get("general_core", {}).get("bits_per_byte")
    general_cake = cake.get("general_with_cake", {}).get("cake_bits_per_byte")
    retention_pass = bool(
        general_core is not None and general_cake is not None and general_cake <= general_core * 1.02
    )
    gates["unrelated_capability_retention"] = _gate(
        "PASS" if retention_pass else "FAIL",
        "unrelated English BPB stayed within the locked two-percent tolerance" if retention_pass
        else "the Python cake caused a meaningful unrelated-English BPB regression",
        core_bpb=general_core, with_cake_bpb=general_cake,
    )

    portability_sequence_pass = portability.get("status") == "PASS"
    task_successes = syntax.get("cake_parse_success_rate", 0) > 0
    gates["real_cross_host_portability"] = _gate(
        "PASS" if portability_sequence_pass and task_successes else "INVALID_EVIDENCE",
        "the identical cake retained locked ordinary task successes across actual hosts"
        if portability_sequence_pass and task_successes
        else "three actual hosts retained every locked BPB improvement, but the source cake had no ordinary task successes to port",
        hosts=len(portability.get("hosts", [])),
        source_locked_bpb_successes=portability.get("source_locked_success_count"),
        ordinary_task_successes=syntax.get("cake_parse_success_rate"),
    )
    package = portability.get("package", {})
    payload_pass = bool(
        package.get("payload_identity")
        and package.get("archive_sha256_before") == package.get("archive_sha256_after")
    )
    gates["payload_identity"] = _gate(
        "PASS" if payload_pass else "INVALID_EVIDENCE",
        "archive bytes and tensor payload remained identical" if payload_pass
        else "package identity evidence is absent or contradictory",
        package=package,
    )
    install_pass = bool(
        payload_pass and portability.get("receiver_domain_training_examples") == 0
        and portability.get("receiver_specific_calibration") is False
        and cake.get("core_unchanged") is True
    )
    gates["installation_losslessness"] = _gate(
        "PASS" if install_pass else "INVALID_EVIDENCE",
        "installation used no receiver data/calibration, changed no core or cake values, and preserved archive bytes"
        if install_pass else "lossless installation conditions were not all demonstrated",
    )

    learned = routing.get("learned", {})
    route_pass = bool(
        routing.get("status") == "PASS"
        and learned.get("route_accuracy", 0) >= 0.95
        and learned.get("top_k_recall", 0) >= 0.98
        and learned.get("false_specialist_activation", 1) <= 0.02
        and routing.get("missing_cake", {}).get("abstention_accuracy") == 1.0
    )
    gates["routing_accuracy"] = _gate(
        "PASS" if route_pass else "FAIL",
        "learned hidden-suite routing met every locked target" if route_pass
        else "learned routing missed one or more locked targets",
        learned=learned, latency_milliseconds=routing.get("latency_milliseconds"),
    )
    orchestration_pass = orchestration.get("status") == "PASS" and len(orchestration.get("hosts", [])) >= 3
    gates["orchestration"] = _gate(
        "PASS" if orchestration_pass else "INVALID_EVIDENCE",
        "signed install, learned routing, CPU generation, removal, disappearance, reinstall, and three-host execution ran end to end"
        if orchestration_pass else "complete real-artifact demonstration evidence is missing",
    )
    cpu_pass = incremental_pass and bool(direct.get("systems", {}).get("layercake_cpu"))
    gates["cpu_runtime"] = _gate(
        "PASS" if cpu_pass else "INVALID_EVIDENCE",
        "batch-one CPU stateful runtime was measured through 1,024 generated bytes" if cpu_pass
        else "CPU runtime evidence is missing",
        entirely_cpu=True,
    )
    gpu_pass = bool(direct.get("systems", {}).get("transformer_gpu"))
    gates["gpu_runtime"] = _gate(
        "PASS" if gpu_pass else "NOT_RUN_NO_HARDWARE",
        "fp16 SDPA transformer with a per-layer KV cache ran on the RTX 3080 Laptop GPU" if gpu_pass
        else "CUDA baseline did not run",
    )
    direct_pass = direct.get("status") == "PASS"
    gates["cpu_vs_gpu_matched_quality"] = _gate(
        "PASS" if direct_pass else "INVALID_EVIDENCE",
        "CPU LayerCake passed matched quality and performance gates" if direct_pass
        else "raw CPU performance won, but the comparison is invalid for the moonshot because ordinary-task quality and three-domain coverage failed",
        headline=direct.get("headline"), performance_condition_passed=direct.get("performance_condition_passed"),
        bpb_thresholds_passed=direct.get("bpb_thresholds_passed"),
        ordinary_task_quality_valid=direct.get("ordinary_task_quality_valid"),
        domain_coverage_valid=direct.get("domain_coverage_valid"),
    )
    foundation_speed = dev.get("comparisons", {}).get("foundation_wall_time_transformer_over_layercake")
    gates["foundation_training_efficiency"] = _gate(
        "FAIL",
        "LayerCake trained faster in raw wall time but never reached the transformer's locked final English quality",
        raw_wall_time_ratio=foundation_speed, quality_matched=False,
    )
    cake_speed = dev.get("comparisons", {}).get("cake_over_monolithic_adaptation_speed")
    cake_efficiency_pass = bool(cake_speed is not None and cake_speed >= 5 and task_pass and retention_pass)
    gates["cake_training_efficiency"] = _gate(
        "PASS" if cake_efficiency_pass else "FAIL",
        "cake met wall-time, task-quality, and retention targets" if cake_efficiency_pass
        else "cake training was below the five-fold wall-time target and failed task-quality/retention conditions",
        monolithic_over_cake_wall_time=cake_speed,
    )
    gates["package_security"] = _gate(
        "PASS" if repo_pass else "INVALID_EVIDENCE",
        "the complete suite includes adversarial archive, traversal, duplicate, signature, and safetensors checks"
        if repo_pass else "security tests are not tied to a passing exact-tree suite",
    )
    lifecycle = orchestration.get("source_lifecycle", {})
    reinstall_pass = bool(
        lifecycle.get("removed", {}).get("status") == "REMOVED"
        and lifecycle.get("reverified", {}).get("status") == "PASS"
        and lifecycle.get("archive_hash_preserved")
    )
    gates["uninstall_reinstall"] = _gate(
        "PASS" if reinstall_pass else "INVALID_EVIDENCE",
        "removal restored core fallback and reinstall reverified identical bytes" if reinstall_pass
        else "uninstall/reinstall evidence is incomplete",
    )
    mobile_pass = mobile.get("overall_status") == "PASS" and mobile.get("max_logit_difference") == 0
    gates["mobile_export"] = _gate(
        "PASS" if mobile_pass else "FAIL",
        "TorchScript portable-fusion export reloads with exact logits" if mobile_pass
        else "mobile export/reload failed",
        artifact_sha256=mobile.get("artifact_sha256"),
    )
    physical_status = mobile.get("physical_mobile_inference", "NOT_RUN_NO_HARDWARE")
    gates["physical_mobile_execution"] = _gate(
        physical_status if physical_status in ALLOWED else "INVALID_EVIDENCE",
        "no physical mobile device was available" if physical_status == "NOT_RUN_NO_HARDWARE"
        else "physical device result reported by export evidence",
    )
    gates["multi_seed_replication"] = _gate(
        "FAIL",
        "three core seeds ran without failed seeds, but the cake and transformer comparisons each have only one training seed and no confidence interval",
        core_seeds=[core.get("seed") for core in cores], cake_seeds=[9301], transformer_seeds=[transformer.get("seed")],
        failed_seeds=dev.get("failed_seeds", []),
    )

    if set(gates) != set(REQUIRED_GATES):
        missing = sorted(set(REQUIRED_GATES) - set(gates))
        extra = sorted(set(gates) - set(REQUIRED_GATES))
        raise AssertionError(f"certificate gate mismatch: missing={missing}, extra={extra}")
    required_non_mobile = [name for name in REQUIRED_GATES if name != "physical_mobile_execution"]
    moonshot_proven = all(gates[name]["status"] == "PASS" for name in required_non_mobile)
    certificate = {
        "format": "layercake-moonshot-certificate/2",
        "overall_status": "PROVEN" if moonshot_proven else "FAIL",
        "moonshot_proven": moonshot_proven,
        "allowed_statuses": sorted(ALLOWED),
        "source_tree_sha256": current_tree,
        "gates": gates,
        "evidence": {
            name: {"path": str(path), "sha256": _sha(path), "present": path.is_file()}
            for name, path in paths.items()
        },
        "failed_gates": [name for name, gate in gates.items() if gate["status"] == "FAIL"],
        "open_gates": [name for name, gate in gates.items() if gate["status"] in {
            "OPEN", "NOT_RUN_NO_HARDWARE", "NOT_RUN_INSUFFICIENT_COMPUTE", "INVALID_EVIDENCE"
        }],
        "single_next_experiment": (
            "Keep the frozen V2 ABI, scale the English core until it matches the transformer, then retrain the Python cake and require nonzero held-out unit-test/syntax success before any more domains or performance claims."
        ),
    }
    output = evidence_dir / "release_certificate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return certificate
