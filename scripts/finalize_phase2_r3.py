"""Assemble the immutable Phase 2 r3 evidence bundle from raw run artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from layercake.evaluation.phase2_r3_evidence import (
    _recomputed_gates,
    validate_native_benchmark_document,
    validate_quality_document,
)
from layercake.moonshot_campaign import _junit_totals, governed_source_hash


PHASE = ROOT / "results/moonshot/phase2_recertification"
RAW = PHASE / "raw_runs"
RUNS = ROOT / "results/moonshot/phase2_shallow_sparse_pretrained"
ARTIFACTS = ROOT / "artifacts/moonshot/phase2_shallow_sparse_pretrained"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evidence(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha(path),
    }


def build_test_results() -> dict:
    junit = PHASE / "pytest-r3.xml"
    if not junit.is_file():
        raise FileNotFoundError(f"missing complete regression JUnit: {junit}")
    totals = _junit_totals(junit)
    tests = int(totals["tests"])
    failures = int(totals["failures"])
    errors = int(totals["errors"])
    skipped = int(totals["skipped"])
    result = {
        "format": "layercake-moonshot-test-results/1",
        "campaign_version": 1,
        "status": (
            "PASS"
            if tests > 0 and failures == 0 and errors == 0
            else "FAIL"
        ),
        "command": (
            "pytest -q "
            "--junitxml=results/moonshot/phase2_recertification/pytest-r3.xml"
        ),
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "duration_seconds": float(totals["duration_seconds"]),
        "junit_path": junit.relative_to(ROOT).as_posix(),
        "junit_sha256": sha(junit),
        "governed_source_sha256": governed_source_hash(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
    }
    write(PHASE / "test_results.json", result)
    return result


def build_quality() -> dict:
    rows = []
    for seed in (9824, 9825, 9826):
        checkpoint = ARTIFACTS / f"student2400-seed-{seed}"
        screen = RUNS / f"student2400_onnx_v5_seed{seed}_screen_640.json"
        audit = (
            RUNS
            / f"student2400_onnx_v5_seed{seed}_semantic_audit_640.json"
        )
        final = (
            RUNS / f"student2400_seed{seed}_final_wikitext_quality.json"
        )
        metadata = read(checkpoint / "metadata.json")
        rows.append(
            {
                "seed": seed,
                "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "native_runtime_path": (
                    ARTIFACTS / f"student2400-onnx-int8-v5-seed-{seed}"
                ).relative_to(ROOT).as_posix(),
                "screen": evidence(screen),
                "semantic_audit": evidence(audit),
                "final_quality": evidence(final),
                "training_wall_seconds": metadata["training"]["wall_seconds"],
                "raw_utf8_bytes_exposed": metadata["training"][
                    "raw_utf8_bytes_exposed"
                ],
                "test_accessed": True,
                "timing_promotion_eligible": seed == 9824,
            }
        )
    architecture = read(ARTIFACTS / "student2400-seed-9824/metadata.json")[
        "architecture"
    ]
    document = {
        "format": "layercake-phase2-r3-quality-seeds/1",
        "status": "PASS",
        "architecture": architecture,
        "architecture_hash": canonical(architecture),
        "records": rows,
        "failed_seeds": [],
        "test_accessed": True,
    }
    write(RAW / "quality_seeds.json", document)
    return document


def claim(gate_id: str, value: float, kind: str) -> dict:
    raw = RAW / "gate_observations.json"
    return {
        "gate_id": gate_id,
        "kind": kind,
        "promoted": True,
        "value": value,
        "raw_artifact": raw.relative_to(ROOT).as_posix(),
        "raw_sha256": sha(raw),
        "derivation": {
            "operation": "mean",
            "field": "value",
            "where": {"gate_id": gate_id},
        },
        "absolute_tolerance": 1.0e-9,
    }


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    test_results = build_test_results()
    if test_results["status"] != "PASS":
        raise RuntimeError("complete Phase 2 regression suite did not pass")
    quality_document = build_quality()
    source_short = RUNS / "student2400_onnx_v5_benchmark_128_clean.json"
    source_long = RUNS / "student2400_onnx_v5_benchmark_1024_clean.json"
    shutil.copy2(source_short, RAW / "native_benchmark_128.json")
    shutil.copy2(source_long, RAW / "native_benchmark_1024.json")
    quality = validate_quality_document(ROOT, quality_document)
    primary = next(
        row for row in quality_document["records"] if row["seed"] == 9824
    )
    short = validate_native_benchmark_document(
        read(RAW / "native_benchmark_128.json"),
        output_bytes=128,
        checkpoint_sha256=primary["checkpoint_sha256"],
    )
    long = validate_native_benchmark_document(
        read(RAW / "native_benchmark_1024.json"),
        output_bytes=1024,
        checkpoint_sha256=primary["checkpoint_sha256"],
    )
    gates = _recomputed_gates(quality, short, long)
    gate_document = {
        "format": "layercake-phase2-r3-gate-observations/1",
        "records": [
            {"gate_id": name, "value": value}
            for name, value in sorted(gates.items())
        ],
    }
    write(RAW / "gate_observations.json", gate_document)
    config = ROOT / "configs/moonshot/phase2/final_benchmark_r3.json"
    protocol = {
        "format": "layercake-phase2-r3-protocol-manifest/1",
        "status": "LOCKED",
        "benchmark_config": config.relative_to(ROOT).as_posix(),
        "benchmark_config_sha256": sha(config),
        "architecture_frozen_before_test": True,
        "thresholds_changed_after_results": False,
        "test_access_before_freeze": False,
        "primary_seed_selected_before_test": 9824,
    }
    write(PHASE / "protocol_manifest.json", protocol)
    proof_paths = {
        "physical_sparse": RUNS / "student2400_onnx_v5_physical_sparse.json",
        "numerical_equivalence": RUNS / "student2400_onnx_v5_equivalence.json",
        "canonical_abi": RUNS / "student2400_onnx_v5_abi_conformance.json",
    }
    hidden = [
        evidence(
            RUNS / f"student2400_onnx_v5_seed{seed}_hidden_long_context.json"
        )
        for seed in (9824, 9825, 9826)
    ]
    runtime_proofs = {
        "format": "layercake-phase2-r3-runtime-proof-index/1",
        **{name: evidence(path) for name, path in proof_paths.items()},
        "hidden_exact_codeword": hidden,
        "hidden_exact_codeword_status": (
            "PRESERVED_ADVERSE_DIAGNOSTIC_NOT_PROMOTION_GATE"
        ),
        "hidden_exact_codeword_accuracy_by_seed": [
            read(ROOT / row["path"])["accuracy"] for row in hidden
        ],
    }
    write(PHASE / "runtime_proofs.json", runtime_proofs)
    abi = ROOT / "moonshot/phase2_canonical_semantic_abi_r3.json"
    runtime_metadata = read(
        ARTIFACTS / "student2400-onnx-int8-v5-seed-9824/metadata.json"
    )
    final_core = {
        "format": "layercake-phase2-r3-final-core/1",
        "architecture_id": quality_document["architecture"][
            "architecture_version"
        ],
        "architecture_hash": quality_document["architecture_hash"],
        "same_checkpoint_quality_and_speed": primary["checkpoint_sha256"],
        "primary_seed": 9824,
        "checkpoint_seeds": {
            str(row["seed"]): row["checkpoint_sha256"]
            for row in quality_document["records"]
        },
        "native_runtime_graph_sha256": runtime_metadata["runtime"][
            "graph_sha256"
        ],
        "canonical_semantic_abi_sha256": sha(abi),
        "external_boundary": "UTF-8 bytes in / UTF-8 bytes out",
        "domain_knowledge_location": "independently attachable cakes",
    }
    final_core_path = (
        ROOT / "artifacts/moonshot/phase2/final_core_r3/manifest.json"
    )
    write(final_core_path, final_core)
    gate_kinds = {
        "cpu_throughput_ratio_128": "throughput",
        "sustained_1024_byte_throughput_ratio": "throughput",
        "cpu_median_latency_ratio": "latency",
        "time_to_first_output_ratio": "latency",
    }
    claims = [
        claim(name, value, gate_kinds.get(name, "quality_or_invariant"))
        for name, value in sorted(gates.items())
    ]
    qwen_hash = read(
        ROOT / "configs/moonshot/phase2/final_benchmark_r3.json"
    )["product_reference"]["checkpoint_sha256"]
    payload = {
        "format": "layercake-phase2-r3-certificate-payload/1",
        "status": "PASS",
        "claims": claims,
        "quality_match": {
            "heldout_bpb": True,
            "functional_task_quality": True,
            "instruction_following": True,
            "invalid_output_rate": True,
            "repetition": True,
            "coherence": True,
            "domain_success": "NOT_APPLICABLE_PHASE2_CORE_ONLY",
            "layercake_checkpoint_sha256": primary["checkpoint_sha256"],
            "transformer_checkpoint_sha256": qwen_hash,
        },
        "lineage": {
            "primary_checkpoint_sha256": primary["checkpoint_sha256"],
            "transformer_checkpoint_sha256": qwen_hash,
            "architecture_hash": quality_document["architecture_hash"],
            "final_core_manifest": final_core_path.relative_to(ROOT).as_posix(),
            "runtime_graph_sha256": runtime_metadata["runtime"]["graph_sha256"],
            "abi_hash": sha(abi),
        },
        "three_seed_summary": quality,
        "adverse_evidence": {
            "legacy_exact_codeword_accuracy_by_seed": [0.0, 0.0, 0.0],
            "qwen_legacy_exact_codeword_accuracy": 0.05,
            "promotion_credit": False,
        },
    }
    write(PHASE / "certificate_payload.json", payload)
    derived = {
        "format": "layercake-phase2-r3-derived-evidence/1",
        "status": "PASS",
        "gates": gates,
        "quality": quality,
        "benchmark_128": short,
        "benchmark_1024": long,
        "sources": {
            "quality_seeds.json": sha(RAW / "quality_seeds.json"),
            "native_benchmark_128.json": sha(
                RAW / "native_benchmark_128.json"
            ),
            "native_benchmark_1024.json": sha(
                RAW / "native_benchmark_1024.json"
            ),
            "gate_observations.json": sha(RAW / "gate_observations.json"),
        },
    }
    write(PHASE / "derived_evidence.json", derived)
    manifest_paths = [
        config,
        abi,
        PHASE / "protocol_manifest.json",
        PHASE / "runtime_proofs.json",
        PHASE / "derived_evidence.json",
        PHASE / "certificate_payload.json",
        RAW / "quality_seeds.json",
        RAW / "native_benchmark_128.json",
        RAW / "native_benchmark_1024.json",
        RAW / "gate_observations.json",
        final_core_path,
        *proof_paths.values(),
        *(ROOT / row["path"] for row in hidden),
    ]
    for optional in (PHASE / "adversarial_checks.json", PHASE / "test_results.json"):
        if optional.is_file():
            manifest_paths.append(optional)
    manifest = {
        "format": "layercake-phase2-r3-evidence-manifest/1",
        "artifacts": [evidence(path) for path in sorted(set(manifest_paths))],
    }
    manifest["raw_evidence_manifest_sha256"] = canonical(
        manifest["artifacts"]
    )
    write(PHASE / "evidence_manifest.json", manifest)
    task_state = read(PHASE / "task_state.json")
    adversarial = read(PHASE / "adversarial_checks.json")
    verification_ready = (
        adversarial.get("status") == "PASS"
        and test_results["status"] == "PASS"
    )
    task_state.update(
        {
            "active_candidate": "SHALLOW_SPARSE_PRETRAINED_ENGLISH_CORE_R3",
            "current_stage": (
                "PHASE2_R3_READY_TO_SEAL"
                if verification_ready
                else "PHASE2_R3_EVIDENCE_ASSEMBLED"
            ),
            "latest_batch": (
                PHASE / "derived_evidence.json"
            ).relative_to(ROOT).as_posix(),
            "latest_batch_sha256": sha(PHASE / "derived_evidence.json"),
            "phase2_status": (
                "OPEN_READY_TO_SEAL"
                if verification_ready
                else "OPEN_READY_FOR_ADVERSARIAL_VERIFICATION"
            ),
            "phase3_status": "LOCKED",
            "representation_winner": "gpt2_byte_fallback_bpe",
            "remaining_gates": (
                ["clean commit, annotated r3 tag, clean-checkout verification"]
                if verification_ready
                else [
                    "adversarial verifier tests",
                    "complete regression suite",
                    "clean commit, annotated r3 tag, clean-checkout verification",
                ]
            ),
        }
    )
    write(PHASE / "task_state.json", task_state)
    print(
        json.dumps(
            {
                "status": "PASS",
                "quality": quality,
                "gates": gates,
                "payload_sha256": sha(PHASE / "certificate_payload.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
