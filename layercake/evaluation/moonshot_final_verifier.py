"""Independent fail-closed verifier for the forty-gate final mandate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from layercake.moonshot import source_tree_hash


ALLOWED = {
    "PASS", "FAIL", "OPEN", "INVALID_EVIDENCE",
    "NOT_RUN_NO_HARDWARE", "NOT_RUN_INSUFFICIENT_COMPUTE",
}
REQUIRED_GATES = (
    "repository_regression", "data_integrity", "benchmark_integrity",
    "english_core_quality", "english_generation_quality", "expert_utilization",
    "sparse_physical_execution", "stateful_decoding", "same_scale_comparison",
    "matched_quality_comparison", "python_capability", "second_domain", "third_domain",
    "random_cake_control", "wrong_domain_control", "shuffled_cake_control",
    "general_retention", "package_losslessness", "installation_losslessness",
    "mathematical_execution_losslessness", "semantic_losslessness",
    "same_seed_portability", "cross_seed_portability", "cross_size_portability",
    "uninstall_reinstall", "one_domain_mode", "multi_domain_mode", "routing_accuracy",
    "multidomain_composition", "catalog_size_scaling", "foundation_training_speed",
    "domain_training_speed", "cpu_versus_cpu_inference", "gpu_versus_gpu_inference",
    "cpu_versus_gpu_inference", "package_security", "mobile_export",
    "physical_mobile_execution", "multi_seed_replication", "overall_moonshot",
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
        raise ValueError(f"invalid final gate status: {status}")
    return {"status": status, "reason": reason, **evidence}


def verify_moonshot_final(root: str | Path, output_path: str | Path) -> dict:
    root = Path(root).resolve()
    final = root / "results/moonshot/final"
    v2 = root / "results/moonshot/v2"
    paths = {
        "audit": final / "baseline_audit.json",
        "pytest": final / "pytest_evidence.json",
        "medium_layercake": final / "foundation_medium.json",
        "medium_transformer": final / "transformer_medium.json",
        "adaptive_pilot": final / "foundation_adaptive_medium_pilot.json",
        "adaptive_pilot_run": root / "artifacts/final/adaptive-medium-pilot/routed_adaptive_5x5_top1_8e/seed-9811/metadata.json",
        "adaptive_routing": final / "foundation_adaptive_routing_search.json",
        "variable_patch": final / "foundation_variable_patch_search.json",
        "depth_search": final / "foundation_adaptive_depth_search.json",
        "context_search": final / "foundation_adaptive_context_search.json",
        "catalog": final / "catalog_scaling.json",
        "incremental": v2 / "incremental_benchmark.json",
        "portability": v2 / "portability_evidence.json",
        "routing": v2 / "routing_evidence.json",
        "orchestration": v2 / "orchestration_evidence.json",
        "direct": v2 / "cpu_vs_gpu_evidence.json",
        "generation": v2 / "english_quality_evidence.json",
        "mobile": root / "artifacts/mobile/python-fusion-v2.pt.manifest.json",
        "python_cake": root / "artifacts/cakes/python.evidence.json",
        "english_manifest": root / "data/moonshot/v2/wikitext103/manifest.json",
        "python_manifest": root / "data/moonshot/v2/python/manifest.json",
    }
    doc = {name: _read(path) for name, path in paths.items()}
    gates: dict[str, dict] = {}
    tree = source_tree_hash()

    pytest = doc["pytest"]
    repo_pass = bool(
        pytest.get("status") == "PASS"
        and pytest.get("exit_code") == 0
        and pytest.get("source_tree_sha256") == tree
    )
    gates["repository_regression"] = _gate(
        "PASS" if repo_pass else "INVALID_EVIDENCE",
        "The complete test suite passed against this exact source tree."
        if repo_pass else "Complete-suite evidence is absent, failing, or tied to another tree.",
        source_tree_sha256=tree, passed=pytest.get("passed"), warnings=pytest.get("warnings"),
    )

    english_manifest = doc["english_manifest"]
    python_manifest = doc["python_manifest"]
    english_files = english_manifest.get("files", {})
    overlaps = english_manifest.get("sampled_cross_split_64byte_overlaps", {})
    data_pass = bool(
        english_files.get("development_train", {}).get("bytes", 0) >= 10_000_000
        and english_files.get("medium_train", {}).get("bytes", 0) >= 100_000_000
        and overlaps and all(value == 0 for value in overlaps.values())
        and len(python_manifest.get("files", {})) == 3
    )
    gates["data_integrity"] = _gate(
        "PASS" if data_pass else "INVALID_EVIDENCE",
        "Immutable public-corpus hashes, split policy, licenses, and leakage sampling are present."
        if data_pass else "Required corpus integrity evidence is incomplete.",
        english_files=english_files, python_files=python_manifest.get("files"),
    )

    direct = doc["direct"]
    locked = direct.get("locked_specification", {})
    search_docs = [
        doc[name] for name in (
            "medium_layercake", "medium_transformer", "adaptive_pilot",
            "adaptive_routing", "variable_patch", "depth_search", "context_search",
        )
    ]
    benchmark_pass = bool(
        locked.get("frozen_before_execution") is True
        and locked.get("prompts") == 100
        and locked.get("repetitions") == 3
        and locked.get("generated_length_buckets_bytes") == [256, 384]
        and all(item.get("final_test_accessed", False) is False for item in search_docs if item)
    )
    gates["benchmark_integrity"] = _gate(
        "PASS" if benchmark_pass else "INVALID_EVIDENCE",
        "Search stayed validation-only and the realistic 100-prompt inference suite was frozen."
        if benchmark_pass else "Final-test isolation or locked benchmark metadata is incomplete.",
        prompt_count=locked.get("prompts"), raw_rows=len(direct.get("raw_rows", [])),
    )

    adaptive_run = doc["adaptive_pilot_run"]
    adaptive_bpb = adaptive_run.get("quality", {}).get("validation", {}).get("bits_per_byte")
    transformer_ci = doc["medium_transformer"].get(
        "transformer_validation_bpb_confidence_interval_95", {}
    )
    transformer_bpb = transformer_ci.get("mean")
    quality_pass = bool(
        adaptive_bpb is not None and transformer_bpb is not None
        and adaptive_bpb <= transformer_ci.get("upper", transformer_bpb) + 0.01
    )
    gates["english_core_quality"] = _gate(
        "PASS" if quality_pass else "FAIL",
        "The sparse adaptive core is non-inferior to the transformer."
        if quality_pass else "At 100M bytes the best sparse adaptive pilot remains about 0.15 BPB behind the transformer.",
        adaptive_validation_bpb=adaptive_bpb,
        transformer_validation_bpb_mean=transformer_bpb,
        transformer_validation_bpb_ci95=transformer_ci,
    )
    samples = doc["generation"].get("generation_samples", [])
    generation_pass = bool(
        samples and sum(bool(row.get("high_fourgram_repetition")) for row in samples) / len(samples) < 0.5
    )
    gates["english_generation_quality"] = _gate(
        "PASS" if generation_pass else "FAIL",
        "Open continuations passed the locked repetition diagnostic."
        if generation_pass else "Every measured open English continuation exhibited severe four-gram repetition.",
        measured_samples=len(samples), repeated_samples=sum(bool(row.get("high_fourgram_repetition")) for row in samples),
    )

    pilot_routing = adaptive_run.get("routing") or {}
    expert_pass = bool(
        pilot_routing.get("all_experts_meaningfully_trained")
        and not pilot_routing.get("router_collapsed", True)
        and pilot_routing.get("maximum_load_fraction", 1) <= 0.5
    )
    gates["expert_utilization"] = _gate(
        "PASS" if expert_pass else "FAIL",
        "All eight adaptive experts trained with balanced causal routing."
        if expert_pass else "Adaptive expert-use evidence is missing or collapsed.",
        routing=pilot_routing,
    )
    total_parameters = adaptive_run.get("parameters")
    active_parameters = adaptive_run.get("active_parameters")
    sparse_pass = bool(
        expert_pass and isinstance(total_parameters, int) and isinstance(active_parameters, int)
        and active_parameters < total_parameters
    )
    gates["sparse_physical_execution"] = _gate(
        "PASS" if sparse_pass else "INVALID_EVIDENCE",
        "Hard token dispatch evaluates selected experts only and measured active parameters are below installed parameters."
        if sparse_pass else "Physical sparse-dispatch evidence is incomplete.",
        total_parameters=total_parameters, active_parameters=active_parameters,
    )

    incremental = doc["incremental"]
    stateful_pass = bool(
        incremental.get("status") == "PASS"
        and incremental.get("full_prompt_recomputed_per_decode_step") is False
        and all(row.get("identical_outputs") for row in incremental.get("equivalence", []))
        and 1024 in incremental.get("generation_lengths", [])
    )
    gates["stateful_decoding"] = _gate(
        "PASS" if stateful_pass else "INVALID_EVIDENCE",
        "Persistent-state decoding is exactly equivalent through the measured 1,024-byte path."
        if stateful_pass else "Incremental equivalence evidence is incomplete.",
        sustained_rate=incremental.get("sustained_1024_byte_decode_rate"),
    )

    transformer_params = None
    runs = doc["medium_transformer"].get("runs", [])
    if runs:
        transformer_params = runs[0].get("parameters")
    relative_delta = (
        abs(total_parameters - transformer_params) / transformer_params
        if isinstance(total_parameters, int) and isinstance(transformer_params, int) else None
    )
    same_scale_pass = bool(relative_delta is not None and relative_delta <= 0.05 and quality_pass)
    gates["same_scale_comparison"] = _gate(
        "PASS" if same_scale_pass else "FAIL",
        "Same-scale quality passed." if same_scale_pass
        else "Total scale is within tolerance, but LayerCake quality is materially worse.",
        layercake_parameters=total_parameters, transformer_parameters=transformer_params,
        relative_delta=relative_delta,
    )
    gates["matched_quality_comparison"] = _gate(
        "FAIL", "No LayerCake checkpoint reached the transformer's frozen final-quality interval.",
        adaptive_bpb=adaptive_bpb, transformer_bpb=transformer_bpb,
    )

    python_eval = doc["python_cake"].get("evaluation", {})
    syntax = python_eval.get("syntax_tasks", {})
    python_pass = syntax.get("five_x_error_gate") == "PASS" and syntax.get("cake_parse_success_rate", 0) > 0
    gates["python_capability"] = _gate(
        "PASS" if python_pass else "FAIL",
        "The Python cake passed functional tasks with at least five-fold error reduction."
        if python_pass else "The real Python cake improved code BPB but passed 0/8 ordinary syntax tasks.",
        syntax_tasks=syntax,
    )
    gates["second_domain"] = _gate("OPEN", "No second real neural domain cake passed functional held-out tasks.")
    gates["third_domain"] = _gate("OPEN", "No third real neural domain cake was trained and functionally validated.")
    heldout = python_eval.get("heldout_domain", {})
    random_control = python_eval.get("random_control", {})
    shuffled = python_eval.get("shuffled_control", {})
    gates["random_cake_control"] = _gate(
        "INVALID_EVIDENCE",
        "The trained cake beats random on code BPB, but the correct cake has no functional successes.",
        trained_bpb=heldout.get("cake_bits_per_byte"), random_bpb=random_control.get("cake_bits_per_byte"),
    )
    gates["wrong_domain_control"] = _gate("OPEN", "A second serious cake does not exist, so a real wrong-domain control is impossible.")
    gates["shuffled_cake_control"] = _gate(
        "INVALID_EVIDENCE", "The shuffled tensor control was run for BPB, but functional control evidence is empty.",
        shuffled_bpb=shuffled.get("cake_bits_per_byte"),
    )
    core_general = python_eval.get("general_core", {}).get("bits_per_byte")
    cake_general = python_eval.get("general_with_cake", {}).get("cake_bits_per_byte")
    retention_pass = bool(core_general and cake_general and cake_general <= core_general * 1.02)
    gates["general_retention"] = _gate(
        "PASS" if retention_pass else "FAIL",
        "Unrelated English stayed within two percent." if retention_pass
        else "Forced Python-cake activation regressed unrelated English beyond the locked tolerance.",
        core_bpb=core_general, with_cake_bpb=cake_general,
    )

    portability = doc["portability"]
    package = portability.get("package", {})
    package_pass = bool(
        portability.get("status") == "PASS" and package.get("payload_identity")
        and package.get("archive_sha256_before") == package.get("archive_sha256_after")
    )
    gates["package_losslessness"] = _gate(
        "PASS" if package_pass else "INVALID_EVIDENCE",
        "Archive and authenticated tensor payload hashes are identical across installation."
        if package_pass else "Package-byte identity evidence is incomplete.", package=package,
    )
    install_pass = bool(
        package_pass and portability.get("receiver_domain_training_examples") == 0
        and portability.get("receiver_specific_calibration") is False
        and doc["python_cake"].get("core", {}).get("unchanged") is True
    )
    gates["installation_losslessness"] = _gate(
        "PASS" if install_pass else "INVALID_EVIDENCE",
        "Installation performs no training/calibration and changes neither core nor cake."
        if install_pass else "No-training/no-calibration installation evidence is incomplete.",
    )
    hosts = portability.get("hosts", [])
    exact_reinstall = bool(hosts) and all(
        row.get("generation_before", {}).get("cake_output_hex")
        == row.get("generation_after", {}).get("cake_output_hex")
        and row.get("canonical_reinstall_max_difference") == 0
        for row in hosts
    )
    gates["mathematical_execution_losslessness"] = _gate(
        "PASS" if exact_reinstall else "INVALID_EVIDENCE",
        "Declared deterministic reinstall paths produce bit-identical outputs and canonical tensors."
        if exact_reinstall else "Exact deterministic execution evidence is incomplete.",
        hosts=len(hosts),
    )
    semantic_source_success = syntax.get("cake_parse_success_rate", 0) > 0
    gates["semantic_losslessness"] = _gate(
        "PASS" if semantic_source_success and all(row.get("retention_rate") == 1 for row in hosts) else "INVALID_EVIDENCE",
        "All source functional successes were retained." if semantic_source_success
        else "BPB improvements transfer, but the source cake has zero ordinary functional successes; semantic losslessness is undefined.",
    )
    gates["same_seed_portability"] = _gate(
        "PASS" if package_pass and exact_reinstall else "INVALID_EVIDENCE",
        "Same-host remove/reinstall preserves exact archive, tensors, and deterministic output.",
    )
    cross_seed_rows = [row for row in hosts if row.get("host", {}).get("seed") in {9202}]
    cross_size_rows = [row for row in hosts if row.get("host", {}).get("seed") in {9203}]
    gates["cross_seed_portability"] = _gate(
        "INVALID_EVIDENCE",
        "The identical package improves BPB on an independent-seed host, but there are no source functional successes to retain.",
        hosts=len(cross_seed_rows),
    )
    gates["cross_size_portability"] = _gate(
        "INVALID_EVIDENCE",
        "The identical package improves BPB on a different-size host, but semantic capability portability is unproven.",
        hosts=len(cross_size_rows),
    )
    orchestration = doc["orchestration"]
    lifecycle = orchestration.get("source_lifecycle", {})
    lifecycle_pass = bool(
        lifecycle.get("removed", {}).get("status") == "REMOVED"
        and lifecycle.get("reverified", {}).get("status") == "PASS"
        and lifecycle.get("archive_hash_preserved")
        and lifecycle.get("route_after_remove", {}).get("core_fallback")
    )
    gates["uninstall_reinstall"] = _gate(
        "PASS" if lifecycle_pass else "INVALID_EVIDENCE",
        "Removal restores core fallback and reinstall restores the identical signed archive."
        if lifecycle_pass else "Lifecycle evidence is incomplete.",
    )
    direct_modes = [row.get("single_domain_generation", {}).get("trace", {}) for row in orchestration.get("hosts", [])]
    one_domain_pass = bool(direct_modes) and all(row.get("route_bypassed") for row in direct_modes)
    gates["one_domain_mode"] = _gate(
        "PASS" if one_domain_pass else "INVALID_EVIDENCE",
        "Three hosts executed a directly bound real cake with routing bypassed."
        if one_domain_pass else "Direct single-domain execution evidence is missing.",
    )
    gates["multi_domain_mode"] = _gate("OPEN", "Only one real functional package is installed; many-package execution is not demonstrated.")
    routing = doc["routing"]
    learned = routing.get("learned", {})
    route_numerically_passed = bool(
        routing.get("status") == "PASS" and learned.get("route_accuracy", 0) >= 0.95
        and learned.get("top_k_recall", 0) >= 0.98
        and learned.get("false_specialist_activation", 1) <= 0.02
    )
    gates["routing_accuracy"] = _gate(
        "INVALID_EVIDENCE" if route_numerically_passed else "FAIL",
        "The 318-example held-out template-family suite scores 100%, but only Python has a real cake and the suite is generated; it is not promotable human routing evidence.",
        metrics=learned, latency=routing.get("latency_milliseconds"),
    )
    gates["multidomain_composition"] = _gate("OPEN", "No request composed two real compatible neural cakes.")
    catalog = doc["catalog"]
    gates["catalog_size_scaling"] = _gate(
        catalog.get("promotion_status", "INVALID_EVIDENCE") if catalog else "INVALID_EVIDENCE",
        catalog.get("promotion_reason", "Catalog stress evidence is absent."),
        stress_status=catalog.get("status"), catalog_sizes=catalog.get("catalog_sizes"),
    )

    curves = adaptive_run.get("training", {}).get("curves", [])
    adaptive_to_2 = next((row.get("wall_seconds") for row in curves if row.get("validation", {}).get("bits_per_byte", 99) <= 2.0), None)
    transformer_comparisons = doc["medium_transformer"].get("time_to_quality", [])
    transformer_to_2 = next((row.get("transformer_median_seconds") for row in transformer_comparisons if row.get("threshold_bpb") == 2.0), None)
    gates["foundation_training_speed"] = _gate(
        "FAIL",
        "LayerCake reaches 2.00 BPB faster, but never reaches the transformer's locked final quality; matched-quality speed therefore fails.",
        adaptive_seconds_to_2_bpb=adaptive_to_2,
        transformer_seconds_to_2_bpb=transformer_to_2,
        final_quality_matched=False,
    )
    gates["domain_training_speed"] = _gate(
        "FAIL", "The cake trains quickly but fails functional quality and retention, so time-to-matched-domain-quality is unavailable.",
        cake_seconds=doc["python_cake"].get("cake", {}).get("training_seconds"),
    )
    gates["cpu_versus_cpu_inference"] = _gate(
        "NOT_RUN_INSUFFICIENT_COMPUTE",
        "A full optimized CPU/CPU suite was not rerun after the general and domain matched-quality prerequisites failed; raw speed cannot become valid evidence.",
    )
    gates["gpu_versus_gpu_inference"] = _gate(
        "NOT_RUN_INSUFFICIENT_COMPUTE",
        "A full optimized GPU/GPU suite was not run because no LayerCake checkpoint met the locked quality prerequisite.",
    )
    gates["cpu_versus_gpu_inference"] = _gate(
        "INVALID_EVIDENCE",
        "The realistic raw benchmark shows a 6.47x CPU LayerCake throughput advantage, but ordinary-task quality and three-domain coverage fail.",
        benchmark_status=direct.get("status"), headline=direct.get("headline"),
        ordinary_task_quality_valid=direct.get("ordinary_task_quality_valid"),
        domain_coverage_valid=direct.get("domain_coverage_valid"),
    )
    gates["package_security"] = _gate(
        "PASS" if repo_pass and package_pass else "INVALID_EVIDENCE",
        "Signed, authenticated, non-executable safetensors packages and adversarial installer tests pass."
        if repo_pass and package_pass else "Security evidence is not tied to the exact passing tree.",
    )
    mobile = doc["mobile"]
    mobile_pass = mobile.get("overall_status") == "PASS" and mobile.get("max_logit_difference") == 0
    gates["mobile_export"] = _gate(
        "PASS" if mobile_pass else "FAIL",
        "TorchScript export reloads with exactly identical logits." if mobile_pass
        else "ARM/mobile-compatible export smoke failed.",
        artifact_sha256=mobile.get("artifact_sha256"), artifact_bytes=mobile.get("artifact_bytes"),
    )
    physical = mobile.get("physical_mobile_inference", "NOT_RUN_NO_HARDWARE")
    gates["physical_mobile_execution"] = _gate(
        physical if physical in ALLOWED else "INVALID_EVIDENCE",
        "No physical mobile device was available; no physical-performance claim is made."
        if physical == "NOT_RUN_NO_HARDWARE" else "Physical-device status reported by export evidence.",
    )
    gates["multi_seed_replication"] = _gate(
        "FAIL",
        "The old recurrent core and transformer have three seeds, but the materially improved adaptive architecture has only one 100M-byte pilot and still fails quality.",
        adaptive_medium_seeds=[adaptive_run.get("seed")],
        recurrent_medium_seeds=[row.get("seed") for row in doc["medium_layercake"].get("runs", [])],
        transformer_medium_seeds=[row.get("seed") for row in doc["medium_transformer"].get("runs", [])],
    )

    required_before_overall = [name for name in REQUIRED_GATES if name not in {"overall_moonshot", "physical_mobile_execution"}]
    proven = all(gates[name]["status"] == "PASS" for name in required_before_overall)
    gates["overall_moonshot"] = _gate(
        "PASS" if proven else "FAIL",
        "Every required non-hardware gate passes." if proven
        else "The complete platform is not proven; quality, functional-domain, semantic-portability, many-domain, and matched-speed gates remain unresolved.",
    )
    if tuple(gates) != REQUIRED_GATES:
        missing = sorted(set(REQUIRED_GATES) - set(gates))
        extra = sorted(set(gates) - set(REQUIRED_GATES))
        raise AssertionError(f"final certificate gate mismatch: missing={missing}, extra={extra}")
    certificate = {
        "format": "layercake-moonshot-final-certificate/1",
        "overall_status": "PROVEN" if proven else "NOT_YET_PROVEN",
        "moonshot_proven": proven,
        "allowed_statuses": sorted(ALLOWED),
        "source_tree_sha256": tree,
        "gates": gates,
        "failed_gates": [name for name, gate in gates.items() if gate["status"] == "FAIL"],
        "open_or_invalid_gates": [name for name, gate in gates.items() if gate["status"] not in {"PASS", "FAIL"}],
        "evidence": {
            name: {"path": str(path), "sha256": _sha(path), "present": path.is_file()}
            for name, path in paths.items()
        },
        "measured_primary_blocker": (
            "The best sparse adaptive LayerCake reaches 2.00 BPB faster but ends at 1.8312 BPB "
            "versus the transformer's 1.6790 mean at 100M bytes; the only real cake has zero functional successes."
        ),
        "continuation_commands": [
            "C:\\Python310\\python.exe -m layercake.moonshot_final train-core",
            "C:\\Python310\\python.exe -m layercake.moonshot_final train-domains",
            "C:\\Python310\\python.exe -m layercake.moonshot_final verify",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return certificate
