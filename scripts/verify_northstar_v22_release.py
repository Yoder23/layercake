from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "breakthrough_equal"


FILES = {
    "schema_cpu": "northstar_v22_schema_patch_cpu.json",
    "schema_gpu": "northstar_v22_schema_patch_cuda_graph_gpu.json",
    "relevance_cpu": "northstar_v22_relevance_patch_cpu.json",
    "relevance_gpu": "northstar_v22_relevance_patch_cuda_graph_gpu.json",
    "schema_int8": "northstar_v22_schema_patch_dynamic_int8_cpu.json",
    "relevance_int8": "northstar_v22_relevance_patch_dynamic_int8_cpu.json",
    "resources_int8": "northstar_v22_deployment_resources_dynamic_int8_cpu.json",
    "transfer_cpu": "northstar_v22_lossless_transfer_15m_to_5m_cpu.json",
    "transfer_gpu": "northstar_v22_lossless_transfer_15m_to_5m_gpu.json",
    "tests": "northstar_v22_pytest_summary.json",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _heldout(document: dict[str, Any]) -> dict[str, Any]:
    return document["splits"]["heldout"]["summary"]


def _seen(document: dict[str, Any]) -> dict[str, Any]:
    return document["splits"]["seen"]["summary"]


def _phase_bytes(config_name: str, bytes_per_step: int) -> int:
    config = _load(ROOT / "configs" / config_name)
    training = config["training"]
    return (
        int(training["steps"]) - int(training["lr_step_offset"])
    ) * bytes_per_step


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed North Star v22 publication certificate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "northstar_v22_release_certificate.json",
    )
    args = parser.parse_args()

    paths = {name: RESULTS / filename for name, filename in FILES.items()}
    runtime_manifest_path = (
        ROOT / "artifacts" / "layercake_v22_patch_int8.manifest.json"
    )
    runtime_artifact_path = ROOT / "artifacts" / "layercake_v22_patch_int8.ts"
    release_paths = {
        **paths,
        "runtime_manifest": runtime_manifest_path,
        "runtime_artifact": runtime_artifact_path,
    }
    missing = [
        str(path.relative_to(ROOT))
        for path in release_paths.values()
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"missing required evidence: {missing}")
    evidence = {name: _load(path) for name, path in paths.items()}

    schema_cpu = evidence["schema_cpu"]
    schema_gpu = evidence["schema_gpu"]
    relevance_cpu = evidence["relevance_cpu"]
    relevance_gpu = evidence["relevance_gpu"]
    schema_int8 = evidence["schema_int8"]
    relevance_int8 = evidence["relevance_int8"]
    resources = evidence["resources_int8"]
    transfer_cpu = evidence["transfer_cpu"]
    transfer_gpu = evidence["transfer_gpu"]
    tests = evidence["tests"]
    runtime_manifest = _load(runtime_manifest_path)

    checkpoint_parameters = schema_cpu["checkpoint_parameters"]
    parameter_ratio = float(
        checkpoint_parameters["ratio_layercake_over_transformer"]
    )
    layer_training = schema_cpu["training"]["layercake"]
    transformer_training = schema_cpu["training"]["transformer"]
    layer_latest = layer_training["latest"]
    transformer_latest = transformer_training["latest"]
    layer_total_bytes = float(layer_latest["train_bytes"])
    transformer_total_bytes = float(
        transformer_latest.get(
            "cumulative_train_bytes",
            transformer_latest["train_bytes"],
        )
    )

    corrected_configs = [
        "northstar_v14_unified_grounding.json",
        "northstar_v15_paraphrase_grounding.json",
        "northstar_v16_role_copy_grounding.json",
        "northstar_v17_compositional_holdout.json",
        "northstar_v18_supervised_context_copy.json",
        "northstar_v21_semantic_pointer.json",
    ]
    layer_corrected_task_bytes = sum(
        _phase_bytes(name, 2048) for name in corrected_configs
    )
    mix_components = transformer_latest
    # The realized shares live in the top-level training metrics, but the
    # declared final mix is exactly 75% task data and the recorded corpus
    # artifacts show 75.0129%. Use the smaller declared share conservatively.
    transformer_task_bytes = float(transformer_latest["phase_train_bytes"]) * 0.75

    dense_documents = [schema_cpu, schema_gpu, relevance_cpu, relevance_gpu]
    int8_documents = [schema_int8, relevance_int8]
    cpu_documents = [schema_cpu, relevance_cpu, schema_int8, relevance_int8]
    gpu_documents = [schema_gpu, relevance_gpu]
    fair_neural = all(
        document.get("benchmark_mode") == "fair_neural"
        and not document.get("layercake_structured_schema_head")
        and not document.get("layercake_direct_domain_cache")
        for document in dense_documents + int8_documents
    )
    dense_exact = all(
        _heldout(document)["layercake"]["exact_json_accuracy"] == 1.0
        for document in dense_documents
    ) and all(
        _seen(document)["layercake"]["exact_json_accuracy"] == 1.0
        for document in [schema_cpu, schema_gpu]
    )
    int8_exact = all(
        _heldout(document)["layercake"]["exact_json_accuracy"] == 1.0
        for document in int8_documents
    ) and _seen(schema_int8)["layercake"]["exact_json_accuracy"] == 1.0
    quality_advantage = all(
        _heldout(document)["layercake"]["exact_json_accuracy"]
        > _heldout(document)["transformer"]["exact_json_accuracy"]
        for document in [schema_cpu, schema_gpu, relevance_cpu, relevance_gpu]
    )
    dense_speed_ratios = [
        float(_heldout(document)["mean_speed_ratio_layercake_over_transformer"])
        for document in dense_documents
    ] + [
        float(_seen(document)["mean_speed_ratio_layercake_over_transformer"])
        for document in [schema_cpu, schema_gpu]
    ]
    int8_speed_ratios = [
        float(_heldout(document)["mean_speed_ratio_layercake_over_transformer"])
        for document in int8_documents
    ] + [
        float(_seen(schema_int8)["mean_speed_ratio_layercake_over_transformer"])
    ]
    one_call = all(
        _heldout(document)["layercake"]["mean_estimated_generated_calls"] == 1.0
        for document in dense_documents + int8_documents
    )
    graph_setup_seconds = [
        float(schema_gpu["layercake_cuda_graph_runtime"]["setup_seconds"]),
        float(relevance_gpu["layercake_cuda_graph_runtime"]["setup_seconds"]),
    ]

    transfer_exact = all(
        item["status"] == "PASS"
        and item["contract"]["unchanged_decoder_payload"] is True
        and item["independent_decoder_instances"] is True
        and item["max_logit_diff"] == 0.0
        and item["ppl_ratio"] == 1.0
        and item["generation"]["equal"] is True
        for item in [transfer_cpu, transfer_gpu]
    )
    transfer_cross_backend_equal = (
        transfer_cpu["contract"]["payload_hash"]
        == transfer_gpu["contract"]["payload_hash"]
        and transfer_cpu["generation"]["sha256"]
        == transfer_gpu["generation"]["sha256"]
    )
    int8_artifact_ratios = [
        float(item["deployment_artifact_bytes"]["ratio_layercake_over_transformer"])
        for item in int8_documents
    ]
    execution_provenance = (
        all(
            document.get("device") == "cpu"
            and document.get("cpu_threads") == 1
            and document.get("environment", {}).get("device_type") == "cpu"
            and document.get("environment", {}).get("torch")
            and document.get("environment", {}).get("cpu")
            for document in cpu_documents
        )
        and all(
            document.get("device") == "cuda"
            and document.get("environment", {}).get("device_type") == "cuda"
            and document.get("environment", {}).get("cuda_available") is True
            and document.get("environment", {}).get("torch_cuda")
            and document.get("environment", {}).get("gpu")
            and document.get("environment", {}).get("gpu_compute_capability")
            for document in gpu_documents
        )
    )
    resource_gates = resources.get("required_gates", {})
    test_counts = tests.get("counts", {})

    gates = {
        "all_evidence_present": not missing,
        "fair_neural_no_structured_shortcuts": fair_neural,
        "equal_size_parameter_ratio_at_most_1_05": parameter_ratio <= 1.05,
        "transformer_total_exposure_at_least_layercake": (
            transformer_total_bytes >= layer_total_bytes
        ),
        "transformer_corrected_task_exposure_at_least_layercake": (
            transformer_task_bytes >= layer_corrected_task_bytes
        ),
        "layercake_general_bpb_better": (
            float(layer_training["eval_bpb"])
            < float(transformer_training["eval_bpb"])
        ),
        "layercake_dense_quality_100_percent": dense_exact,
        "layercake_heldout_quality_strictly_better": quality_advantage,
        "layercake_dense_cpu_gpu_speed_at_least_5x": min(dense_speed_ratios) >= 5.0,
        "layercake_one_neural_call_per_answer": one_call,
        "cuda_graph_setup_recorded": all(value > 0.0 for value in graph_setup_seconds),
        "cpu_gpu_execution_provenance_recorded": execution_provenance,
        "int8_quality_100_percent": int8_exact,
        "int8_speed_at_least_5x": min(int8_speed_ratios) >= 5.0,
        "int8_artifact_at_most_half_transformer": max(int8_artifact_ratios) <= 0.5,
        "isolated_deployment_resource_gate": (
            resources.get("status") == "PASS"
            and not resources.get("failed_required")
            and bool(resource_gates)
            and all(resource_gates.values())
        ),
        "lossless_15m_to_5m_transfer_cpu_gpu": transfer_exact,
        "lossless_transfer_cross_backend_generation_equal": transfer_cross_backend_equal,
        "full_regression_suite": (
            tests.get("status") == "PASS"
            and tests.get("exit_code") == 0
            and int(test_counts.get("tests", 0)) > 0
            and int(test_counts.get("failures", 0)) == 0
            and int(test_counts.get("errors", 0)) == 0
            and int(test_counts.get("skipped", 0)) == 0
        ),
        "packaged_runtime_validated": (
            runtime_manifest["status"] == "PASS"
            and runtime_manifest["validation"]["exact_json"]
            == runtime_manifest["validation"]["samples"]
            and runtime_manifest["artifact_sha256"]
            == _sha256(runtime_artifact_path)
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]

    certificate = {
        "status": "PASS" if not failed else "FAIL",
        "claim": (
            "At approximately 15M source parameters and with a strengthened "
            "equal-size tokenizer-transformer comparator given more total and "
            "corrected-task bytes, Layercake is exactly correct on the locked "
            "schema and compositional holdouts, exceeds 5x steady-state CPU "
            "and GPU answer speed, preserves that quality and speed advantage "
            "in a smaller INT8 patch-generation deployment, and transfers an "
            "unchanged portable domain payload bit-exactly from an independent "
            "15M host to an independent 5M host on CPU and GPU."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "layercake_parameters": checkpoint_parameters["layercake"],
            "transformer_parameters": checkpoint_parameters["transformer"],
            "parameter_ratio": parameter_ratio,
            "layercake_general_bpb": layer_training["eval_bpb"],
            "transformer_general_bpb": transformer_training["eval_bpb"],
            "layercake_total_train_bytes": layer_total_bytes,
            "transformer_total_train_bytes": transformer_total_bytes,
            "layercake_corrected_task_bytes": layer_corrected_task_bytes,
            "transformer_corrected_task_bytes_conservative": transformer_task_bytes,
            "dense_speed_ratios": dense_speed_ratios,
            "int8_speed_ratios": int8_speed_ratios,
            "int8_artifact_ratios": int8_artifact_ratios,
            "deployment": resources["metrics"],
            "cuda_graph_setup_seconds": graph_setup_seconds,
            "pytest": tests["counts"],
            "transfer_payload_hash": transfer_cpu["contract"]["payload_hash"],
            "transfer_generation_hash": transfer_cpu["generation"]["sha256"],
            "packaged_runtime_bytes": runtime_artifact_path.stat().st_size,
            "packaged_runtime_sha256": _sha256(runtime_artifact_path),
        },
        "evidence": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "limitations": {
            "real_phone_hardware_measured": False,
            "mobile_evidence_scope": (
                "INT8 artifact size, isolated x86 CPU peak memory, quality, "
                "and speed proxy; no Android/iOS ARM, NPU, battery, or thermal measurement."
            ),
            "deployment_scope": (
                "The 8.7MB runtime is the global autoregressive patch-generation "
                "path used by the locked tasks, not the full general byte-LM decoder."
            ),
            "task_scope": (
                "Schema/action JSON and compositional prompt grounding plus a "
                "separate general-language BPB evaluation; not universal LLM dominance."
            ),
            "cuda_graph_setup": (
                "One-time setup is recorded and excluded from steady-state latency."
            ),
            "training_scope": (
                "This release certificate does not claim faster full-core training. "
                "The separate fail-closed training audit currently reports OPEN."
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_benchmark": schema_cpu["environment"],
            "gpu_benchmark": schema_gpu["environment"],
        },
    }
    certificate["evidence"]["runtime_manifest"] = {
        "path": str(runtime_manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(runtime_manifest_path),
        "bytes": runtime_manifest_path.stat().st_size,
    }
    certificate["evidence"]["runtime_artifact"] = {
        "path": str(runtime_artifact_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(runtime_artifact_path),
        "bytes": runtime_artifact_path.stat().st_size,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
