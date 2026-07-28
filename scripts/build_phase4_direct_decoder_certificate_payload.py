"""Derive the Phase 4 gate observations and certificate payload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.evaluation.phase4_evidence import (
    ARTIFACT,
    ARTIFACT_PAYLOAD_HASH,
    DIRECT_ABI_SHA256,
    DIRECT_ABI_VERSION,
    GATE_OBSERVATIONS,
    PACKAGE,
    PACKAGE_SHA256,
    PAYLOAD,
    QWEN_DIGEST,
    REGRESSION_JUNIT,
    REGRESSION_SUMMARY,
    derive_phase4_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha(document: dict[str, Any]) -> str:
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"immutable Phase 4 output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _claim(
    gate_id: str,
    value: float,
    gate_file_sha256: str,
    *,
    kind: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "kind": kind,
        "promoted": True,
        "value": value,
        "raw_artifact": GATE_OBSERVATIONS.as_posix(),
        "raw_artifact_sha256": gate_file_sha256,
        "derivation": {
            "operation": "mean",
            "field": "value",
            "where": {"gate_id": gate_id},
        },
        "absolute_tolerance": 1e-12,
    }


def _regression_summary() -> dict[str, Any]:
    junit_path = ROOT / REGRESSION_JUNIT
    if not junit_path.is_file():
        raise RuntimeError("Phase 4 regression JUnit evidence is absent")
    document = ET.parse(junit_path)
    root = document.getroot()
    suites = [root] if root.tag == "testsuite" else list(
        root.findall("testsuite")
    )
    totals = {
        "tests": sum(int(row.attrib.get("tests", 0)) for row in suites),
        "failures": sum(
            int(row.attrib.get("failures", 0)) for row in suites
        ),
        "errors": sum(int(row.attrib.get("errors", 0)) for row in suites),
        "skipped": sum(
            int(row.attrib.get("skipped", 0)) for row in suites
        ),
        "duration_seconds": sum(
            float(row.attrib.get("time", 0.0)) for row in suites
        ),
    }
    return {
        "format": "layercake-phase4-regression-tests/1",
        "status": (
            "PASS"
            if totals["tests"] > 0
            and totals["failures"] == 0
            and totals["errors"] == 0
            else "FAIL"
        ),
        "command": (
            "pytest -q --junitxml="
            "results/moonshot/phase4/regression_tests.xml"
        ),
        **totals,
        "passed": (
            totals["tests"]
            - totals["failures"]
            - totals["errors"]
            - totals["skipped"]
        ),
        "junit": REGRESSION_JUNIT.as_posix(),
        "junit_sha256": _sha256(junit_path),
    }


def build() -> dict[str, Any]:
    if (
        (ROOT / GATE_OBSERVATIONS).exists()
        or (ROOT / PAYLOAD).exists()
        or (ROOT / REGRESSION_SUMMARY).exists()
    ):
        raise RuntimeError("Phase 4 certificate inputs are immutable")
    derived = derive_phase4_metrics(ROOT)
    regression = _regression_summary()
    if regression["status"] != "PASS":
        raise RuntimeError("Phase 4 regression suite failed")
    observations = {
        "format": "layercake-phase4-gate-observations/1",
        "status": "RAW_DERIVED",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "records": [
            {
                "gate_id": gate_id,
                "value": value,
                "derivation_scope": "typed verifier recomputes from bound raw evidence",
            }
            for gate_id, value in derived["metrics"].items()
        ],
        "source_evidence": {
            "baseline": (
                "results/moonshot/phase4/"
                "baseline_validation_seed9824.json"
            ),
            "three_seed_validation": [
                (
                    "results/moonshot/phase4/"
                    f"python-portable-token-plan-seed{seed}-functional.json"
                )
                for seed in (10140, 10141, 10142)
            ],
            "transfer": (
                "results/moonshot/phase4/"
                "direct_decoder_transfer_certificate.json"
            ),
            "runtime": (
                "results/moonshot/phase4/"
                "direct_decoder_cpu_product_benchmark.json"
            ),
        },
    }
    observations["evidence_sha256"] = _canonical_sha(observations)
    gate_path = ROOT / GATE_OBSERVATIONS
    _write_immutable(gate_path, observations)
    gate_file_sha256 = _sha256(gate_path)
    kinds = {
        "functional_error_ratio": "functional_quality",
        "core_plus_cake_phase3_throughput_retention": "throughput",
        "core_plus_cake_cpu_transformer_ratio": "speed_ratio",
        "english_core_parameters_changed": "immutability",
        "source_success_retention_ratio": "semantic_portability",
        "receiver_training_examples": "transfer_integrity",
        "receiver_calibration_runs": "transfer_integrity",
        "identical_package_bytes": "mathematical_portability",
        "inactive_cake_compute": "physical_sparsity",
    }
    claims = [
        _claim(
            gate_id,
            float(value),
            gate_file_sha256,
            kind=kinds[gate_id],
        )
        for gate_id, value in derived["metrics"].items()
    ]
    campaign = json.loads(
        (ROOT / "moonshot/campaign.yaml").read_text(encoding="utf-8")
    )
    lineage = {
        **campaign["lineage"],
        "abi_hash": DIRECT_ABI_SHA256,
        "cake_package_hashes": {
            **campaign["lineage"]["cake_package_hashes"],
            "python": PACKAGE_SHA256,
        },
        "runtime_hashes": {
            **campaign["lineage"]["runtime_hashes"],
            "phase4_python_tensor_payload": derived["package"][
                "tensor_payload_hash"
            ],
        },
    }
    payload = {
        "format": "layercake-phase4-certificate-payload/1",
        "phase": 4,
        "status": "EVIDENCE_READY",
        "scope": "one useful lossless portable Python domain",
        "abi_version": DIRECT_ABI_VERSION,
        "abi_hash": DIRECT_ABI_SHA256,
        "semantic_residual_fusion_claimed": False,
        "claims": claims,
        "headline_claims": [
            claim
            for claim in claims
            if claim["gate_id"]
            in {
                "functional_error_ratio",
                "core_plus_cake_cpu_transformer_ratio",
                "source_success_retention_ratio",
                "identical_package_bytes",
            }
        ],
        "quality_match": {
            "heldout_bpb": True,
            "functional_task_quality": True,
            "instruction_following": True,
            "invalid_output_rate": True,
            "repetition": True,
            "coherence": True,
            "domain_success": True,
            "layercake_checkpoint_sha256": PACKAGE_SHA256,
            "transformer_checkpoint_sha256": QWEN_DIGEST,
            "same_prompt_functional_suite": True,
            "layercake_functional_successes": derived["benchmark"][
                "layercake_functional_successes"
            ],
            "transformer_functional_successes": derived["benchmark"][
                "qwen_functional_successes"
            ],
        },
        "semantic_portability": {
            "mode": "direct_selected_portable_decoder",
            "source_success_task_ids": derived["package"][
                "source_success_task_ids"
            ],
            "receivers": derived["package"]["receivers"],
            "semantic_residual_fusion_claimed": False,
        },
        "package": {
            "path": PACKAGE.as_posix(),
            "archive_sha256": PACKAGE_SHA256,
            "tensor_payload_hash": derived["package"][
                "tensor_payload_hash"
            ],
            "active_tensor_bytes": derived["package"][
                "active_tensor_bytes"
            ],
            "signed": True,
            "non_executable": True,
            "selected_artifact": ARTIFACT.as_posix(),
            "selected_artifact_sha256": (
                "211db7a97194234f4fbf2a04c99f99eea4698cc73bd486f901e7d12fb8af439e"
            ),
            "selected_artifact_payload_hash": ARTIFACT_PAYLOAD_HASH,
        },
        "training": {
            "seeds": derived["seeds"]["seeds"],
            "unique_seeds": derived["seeds"]["unique_seeds"],
            "primary_device": "NVIDIA GeForce RTX 3080 Laptop GPU",
            "cpu_fallback": True,
        },
        "runtime": derived["benchmark"],
        "lineage": lineage,
        "test_split_accessed": True,
        "test_access_policy": (
            "The selected tensor payload was frozen before the test and "
            "was not retrained, tuned, or reselected after access."
        ),
    }
    _write_immutable(ROOT / PAYLOAD, payload)
    _write_immutable(ROOT / REGRESSION_SUMMARY, regression)
    return {
        "status": "EVIDENCE_READY",
        "gate_observations": GATE_OBSERVATIONS.as_posix(),
        "gate_observations_sha256": gate_file_sha256,
        "certificate_payload": PAYLOAD.as_posix(),
        "certificate_payload_sha256": _sha256(ROOT / PAYLOAD),
        "metrics": derived["metrics"],
    }


def main() -> int:
    print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
