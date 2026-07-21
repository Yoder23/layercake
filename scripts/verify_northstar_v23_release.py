"""Fail-closed publication certificate for the v23 routed-cake architecture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "breakthrough_equal"
FILES = {
    "migration": "northstar_v23_lossless_migration.json",
    "route_isolation": "northstar_v23_route_isolation.json",
    "training_speed": "northstar_v23_domain_cake_training_speed.json",
    "schema_cpu": "northstar_v23_schema_patch_cpu.json",
    "schema_gpu": "northstar_v23_schema_patch_cuda_graph_gpu.json",
    "relevance_cpu": "northstar_v23_relevance_patch_cpu.json",
    "relevance_gpu": "northstar_v23_relevance_patch_cuda_graph_gpu.json",
    "transfer_cpu": "northstar_v23_lossless_transfer_15m_to_5m_cpu.json",
    "transfer_gpu": "northstar_v23_lossless_transfer_15m_to_5m_gpu.json",
    "tests": "northstar_v23_pytest_summary.json",
}
V22_GENERATION = {
    "schema_cpu": "northstar_v22_schema_patch_cpu.json",
    "schema_gpu": "northstar_v22_schema_patch_cuda_graph_gpu.json",
    "relevance_cpu": "northstar_v22_relevance_patch_cpu.json",
    "relevance_gpu": "northstar_v22_relevance_patch_cuda_graph_gpu.json",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _heldout(document: dict[str, Any]) -> dict[str, Any]:
    return document["splits"]["heldout"]["summary"]


def _seen(document: dict[str, Any]) -> dict[str, Any]:
    return document["splits"]["seen"]["summary"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "northstar_v23_release_certificate.json",
    )
    args = parser.parse_args()
    paths = {name: RESULTS / filename for name, filename in FILES.items()}
    v22_paths = {
        name: RESULTS / filename for name, filename in V22_GENERATION.items()
    }
    missing = [
        str(path.relative_to(ROOT))
        for path in list(paths.values()) + list(v22_paths.values())
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"missing required evidence: {missing}")
    evidence = {name: _load(path) for name, path in paths.items()}
    v22 = {name: _load(path) for name, path in v22_paths.items()}

    generation_names = (
        "schema_cpu",
        "schema_gpu",
        "relevance_cpu",
        "relevance_gpu",
    )
    generation = [evidence[name] for name in generation_names]
    schema = [evidence["schema_cpu"], evidence["schema_gpu"]]
    exact_quality = all(
        _heldout(document)["layercake"]["exact_json_accuracy"] == 1.0
        for document in generation
    ) and all(
        _seen(document)["layercake"]["exact_json_accuracy"] == 1.0
        for document in schema
    )
    quality_advantage = all(
        _heldout(document)["layercake"]["exact_json_accuracy"]
        > _heldout(document)["transformer"]["exact_json_accuracy"]
        for document in generation
    )
    speed_ratios = [
        float(_heldout(document)["mean_speed_ratio_layercake_over_transformer"])
        for document in generation
    ] + [
        float(_seen(document)["mean_speed_ratio_layercake_over_transformer"])
        for document in schema
    ]
    speed_retention = {}
    for name in generation_names:
        old_bps = float(_heldout(v22[name])["layercake"]["mean_bytes_per_second"])
        new_bps = float(
            _heldout(evidence[name])["layercake"]["mean_bytes_per_second"]
        )
        speed_retention[name] = new_bps / old_bps

    checkpoint_parameters = evidence["schema_cpu"]["checkpoint_parameters"]
    parameter_ratio = float(
        checkpoint_parameters["ratio_layercake_over_transformer"]
    )
    training = evidence["training_speed"]
    training_devices = training["devices"]
    training_minimums = {
        device: float(
            training_devices[device]["ratios"][
                "minimum_training_throughput_layercake_over_transformer"
            ]
        )
        for device in ("cpu", "cuda")
    }
    training_medians = {
        device: float(
            training_devices[device]["ratios"][
                "median_training_throughput_layercake_over_transformer"
            ]
        )
        for device in ("cpu", "cuda")
    }
    transfer_cpu = evidence["transfer_cpu"]
    transfer_gpu = evidence["transfer_gpu"]
    transfer_exact = all(
        document["status"] == "PASS"
        and document["contract"]["unchanged_decoder_payload"] is True
        and document["max_logit_diff"] == 0.0
        and document["ppl_ratio"] == 1.0
        and document["generation"]["equal"] is True
        for document in (transfer_cpu, transfer_gpu)
    )
    tests = evidence["tests"]
    test_counts = tests.get("counts", {})
    migration = evidence["migration"]
    isolation = evidence["route_isolation"]
    gates = {
        "all_evidence_present": not missing,
        "migration_all_logits_and_generation_bit_exact": (
            migration["status"] == "PASS"
            and migration["verification"]["next_byte_logits_bit_exact"]
            and migration["verification"]["context_abi_bit_exact"]
            and migration["verification"]["patch_prediction_logits_bit_exact"]
            and migration["verification"]["generated_patch_bit_exact"]
            and migration["verification"]["next_byte_logits_max_abs_diff"] == 0.0
            and migration["verification"]["patch_prediction_max_abs_diff"] == 0.0
        ),
        "equal_size_parameter_ratio_at_most_1_05": parameter_ratio <= 1.05,
        "generation_quality_100_percent_cpu_gpu": exact_quality,
        "generation_quality_strictly_better_than_transformer": quality_advantage,
        "generation_speed_at_least_5x_cpu_gpu": min(speed_ratios) >= 5.0,
        "generation_throughput_retains_at_least_95_percent_of_v22": (
            min(speed_retention.values()) >= 0.95
        ),
        "selected_domain_cake_training_cpu_gpu_at_least_5x": (
            training["status"] == "PASS"
            and min(training_minimums.values()) >= 5.0
        ),
        "training_protocol_is_generation_aligned_sparse_route": (
            training["protocol"]["layercake_training_mode"]
            == "shared3_routed_tail_int8_foundation"
            and training_devices["cpu"]["gates"][
                "all_optimizer_layercake_parameters_receive_gradients"
            ]
            and training_devices["cuda"]["gates"][
                "all_optimizer_layercake_parameters_receive_gradients"
            ]
        ),
        "route_training_converged_and_generation_path_isolated": (
            isolation["status"] == "PASS"
            and isolation["route_zero_generation_path_bit_exact"] is True
        ),
        "lossless_transfer_cpu_gpu": transfer_exact,
        "lossless_transfer_cross_backend_equal": (
            transfer_cpu["contract"]["payload_hash"]
            == transfer_gpu["contract"]["payload_hash"]
            and transfer_cpu["generation"]["sha256"]
            == transfer_gpu["generation"]["sha256"]
        ),
        "full_regression_suite": (
            tests.get("status") == "PASS"
            and tests.get("exit_code") == 0
            and int(test_counts.get("tests", 0)) > 0
            and int(test_counts.get("failures", 0)) == 0
            and int(test_counts.get("errors", 0)) == 0
            and int(test_counts.get("skipped", 0)) == 0
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    certificate = {
        "schema_version": 1,
        "status": "PASS" if not failed else "FAIL",
        "claim": (
            "The migration-compatible v23 LayerCake preserves the complete v22 "
            "generation path bit-exactly, retains locked quality and throughput, "
            "trains one isolated generation-aligned domain cake at more than 5x "
            "the full equal-capacity transformer throughput on the measured CPU "
            "and GPU protocols, and preserves lossless portable-domain transfer."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "layercake_parameters": checkpoint_parameters["layercake"],
            "transformer_parameters": checkpoint_parameters["transformer"],
            "parameter_ratio": parameter_ratio,
            "generation_speed_ratios": speed_ratios,
            "generation_throughput_retention_over_v22": speed_retention,
            "training_minimum_ratios": training_minimums,
            "training_median_ratios": training_medians,
            "route_optimizer_parameters": isolation["training"][
                "optimizer_parameters"
            ],
            "route_training_loss_ratio": isolation["training"][
                "loss_ratio_final_over_first"
            ],
            "pytest": test_counts,
            "transfer_payload_hash": transfer_cpu["contract"]["payload_hash"],
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
            "training_scope": (
                "The >5x result is selected-domain-cake fine-tuning with a frozen "
                "foundation and portable decoder versus full transformer training. "
                "It is not full-foundation pretraining or time-to-quality evidence."
            ),
            "cpu_training_precision": (
                "The frozen CPU foundation uses dynamic INT8; the active cake and "
                "decoder gradient path remain float32."
            ),
            "legacy_decoder_scope": (
                "The v22-compatible next-byte local decoder reuses inactive routed "
                "experts. Domain-route isolation is certified for the deployed "
                "autoregressive generation path, not that compatibility decoder."
            ),
            "task_scope": (
                "Quality covers the locked schema/action and relevance holdouts plus "
                "bit-exact preservation of the previously measured general-BPB logits."
            ),
        },
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": certificate["status"], "failed": failed}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
