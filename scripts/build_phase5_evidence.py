"""Derive Phase 5 selection, source audit, gates, and release payload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
import xml.etree.ElementTree as ET

import _common
from layercake.evaluation.phase5_evidence import (
    CERTIFICATE,
    CONTRACT_SHA256,
    DOMAINS,
    GATES,
    PAYLOAD,
    SEEDS,
    SELECTION,
    _training_path,
    _validation_path,
    derive_phase5_metrics,
)
from layercake.moonshot_campaign import governed_source_hash
from layercake.training.generic_domain import canonical_sha, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _rooted(path: Path | str) -> Path:
    value = Path(path)
    value = value if value.is_absolute() else ROOT / value
    value = value.resolve()
    try:
        value.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Phase 5 evidence path escapes repository") from error
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _read(path: Path | str) -> dict[str, Any]:
    value = json.loads(_rooted(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"Phase 5 evidence is immutable: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value["evidence_sha256"] = canonical_sha(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def freeze_framework(output_path: Path) -> dict[str, Any]:
    if _git("status", "--porcelain=v1"):
        raise RuntimeError(
            "framework freeze requires a clean committed worktree"
        )
    commit = _git("rev-parse", "HEAD")
    result = {
        "format": "layercake-phase5-framework-freeze/1",
        "status": "FROZEN",
        "framework_commit": commit,
        "framework_tree": _git("show", "-s", "--format=%T", commit),
        "framework_subject": _git("show", "-s", "--format=%s", commit),
        "contract_sha256": CONTRACT_SHA256,
        "governed_source_sha256": governed_source_hash(ROOT),
        "domain_configs_present_at_freeze": [
            path.relative_to(ROOT).as_posix()
            for path in (
                ROOT / "configs/moonshot/phase5/domains"
            ).glob("*.json")
        ] if (
            ROOT / "configs/moonshot/phase5/domains"
        ).is_dir() else [],
        "domain_datasets_present_at_freeze": [
            path.relative_to(ROOT).as_posix()
            for path in (
                ROOT / "data/moonshot/phase5"
            ).glob("*.jsonl")
        ] if (
            ROOT / "data/moonshot/phase5"
        ).is_dir() else [],
    }
    if (
        result["domain_configs_present_at_freeze"]
        or result["domain_datasets_present_at_freeze"]
    ):
        raise RuntimeError(
            "domain configs or datasets predate the framework freeze"
        )
    _write(output_path, result)
    return result


def select(output_path: Path) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        candidates = []
        for seed in SEEDS:
            training_path = _rooted(_training_path(domain, seed))
            validation_path = _rooted(_validation_path(domain, seed))
            training = _read(training_path)
            validation = _read(validation_path)
            invalid = sum(
                any(
                    check.get("status") == "INVALID_UTF8"
                    for check in row.get("checks", [])
                )
                for row in validation["records"]
            )
            candidates.append(
                {
                    "seed": seed,
                    "functional_successes": validation[
                        "functional_successes"
                    ],
                    "invalid_output_count": invalid,
                    "training_nll": training[
                        "best_action_negative_log_likelihood"
                    ],
                    "artifact": training["artifact"],
                    "validation_evidence": {
                        "path": _relative(validation_path),
                        "sha256": sha256_file(validation_path),
                    },
                }
            )
        selected = sorted(
            candidates,
            key=lambda row: (
                -row["functional_successes"],
                row["invalid_output_count"],
                row["training_nll"],
                row["seed"],
            ),
        )[0]
        domains[domain] = {
            "candidates": candidates,
            "selected": selected,
        }
    result = {
        "format": "layercake-phase5-seed-selection/1",
        "status": "PASS",
        "selection_rule": [
            "highest validation functional successes",
            "lowest validation invalid-output count",
            "lowest training negative log likelihood",
            "lowest numeric seed",
        ],
        "domains": domains,
        "test_accessed": False,
    }
    _write(output_path, result)
    return result


def audit_source(
    framework_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    framework = _read(framework_path)
    commit = framework["framework_commit"]
    committed = {
        line.replace("\\", "/")
        for line in _git(
            "diff", "--name-only", f"{commit}..HEAD", "--"
        ).splitlines()
        if line
    }
    working = {
        line.replace("\\", "/")
        for command in (
            ("diff", "--name-only", "--"),
            ("ls-files", "--others", "--exclude-standard"),
        )
        for line in _git(*command).splitlines()
        if line
    }
    changed = sorted(committed | working)

    def source_path(path: str) -> bool:
        return (
            path == "AGENTS.md"
            or path == "moonshot/ACTIVE_MISSION.md"
            or (
                path.endswith(".py")
                and path.split("/", 1)[0]
                in {"layercake", "scripts", "tests"}
            )
            or path
            in {
                "moonshot/claim_contract.yaml",
                "moonshot/invalidation_matrix.yaml",
                "moonshot/benchmark_contract.yaml",
                "moonshot/data_contract.yaml",
                "moonshot/security_contract.yaml",
            }
        )

    prohibited = [path for path in changed if source_path(path)]
    result = {
        "format": "layercake-phase5-authoring-source-audit/1",
        "status": "PASS" if not prohibited else "FAIL",
        "framework_commit": commit,
        "changed_paths_after_framework_freeze": changed,
        "domain_specific_source_edits": prohibited,
        "domain_specific_source_edit_count": len(prohibited),
        "audit_definition": (
            "No changes after framework freeze to LayerCake, script, or "
            "test Python source; AGENTS/ACTIVE_MISSION; or static contracts."
        ),
    }
    _write(output_path, result)
    return result


def record_tests(
    junit_path: Path,
    output_path: Path,
    command: str,
) -> dict[str, Any]:
    document = ET.parse(junit_path)
    root = document.getroot()
    suites = [root] if root.tag == "testsuite" else list(
        root.findall("testsuite")
    )
    values = {
        name: sum(
            int(suite.attrib.get(name, 0)) for suite in suites
        )
        for name in ("tests", "failures", "errors", "skipped")
    }
    result = {
        "format": "layercake-phase5-regression-tests/1",
        "status": (
            "PASS"
            if values["tests"] > 0
            and values["failures"] == 0
            and values["errors"] == 0
            else "FAIL"
        ),
        **values,
        "passed": (
            values["tests"]
            - values["failures"]
            - values["errors"]
            - values["skipped"]
        ),
        "command": command,
        "junit_path": _relative(junit_path),
        "junit_sha256": sha256_file(junit_path),
        "source_commit": _git("rev-parse", "HEAD"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_payload() -> dict[str, Any]:
    gate_path = _rooted(GATES)
    payload_path = _rooted(PAYLOAD)
    if gate_path.exists() or payload_path.exists():
        raise RuntimeError("Phase 5 release payload is immutable")
    derived = derive_phase5_metrics(ROOT)
    gate_document = {
        "format": "layercake-phase5-gate-observations/1",
        "status": "RAW_DERIVED",
        "source_commit": _git("rev-parse", "HEAD"),
        "records": [
            {
                "gate_id": gate,
                "value": value,
                "derivation_scope": (
                    "typed verifier recomputes from bound raw evidence"
                ),
            }
            for gate, value in derived["metrics"].items()
        ],
        "source_evidence": {
            "selection": SELECTION.as_posix(),
            "transfer": CERTIFICATE.as_posix(),
            "framework": (
                "results/moonshot/phase5/framework_freeze.json"
            ),
            "authoring_audit": (
                "results/moonshot/phase5/"
                "authoring_source_audit.json"
            ),
        },
    }
    _write(gate_path, gate_document)
    raw_hash = sha256_file(gate_path)
    kinds = {
        "real_neural_domain_count": "functional_quality",
        "minimum_domain_functional_error_reduction": (
            "functional_quality"
        ),
        "generic_authoring_requires_source_edits": (
            "generic_extensibility"
        ),
        "inactive_cake_effect": "physical_sparsity",
    }
    claims = [
        {
            "gate_id": gate,
            "kind": kinds[gate],
            "value": value,
            "promoted": True,
            "raw_artifact": GATES.as_posix(),
            "raw_sha256": raw_hash,
            "derivation": {
                "operation": "mean",
                "field": "value",
                "where": {"gate_id": gate},
            },
            "absolute_tolerance": 1e-12,
        }
        for gate, value in derived["metrics"].items()
    ]
    transfer = _read(CERTIFICATE)
    package_hashes = {
        "python": transfer["sealed_python"]["archive_sha256"],
        **{
            domain: transfer["packages"][domain][
                "archive_sha256"
            ]
            for domain in DOMAINS
        },
    }
    tensor_hashes = {
        domain: transfer["packages"][domain][
            "tensor_payload_hash"
        ]
        for domain in DOMAINS
    }
    payload = {
        "format": "layercake-phase5-certificate-payload/1",
        "status": "EVIDENCE_READY",
        "phase": 5,
        "abi_version": "lc-direct-neural-decoder/1",
        "abi_hash": (
            "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
        ),
        "claims": claims,
        "headline_claims": claims,
        "claim_boundary": {
            "manual_selection": True,
            "routing_claimed": False,
            "top_k_claimed": False,
            "fusion_or_composition_claimed": False,
            "catalog_scaling_claimed": False,
        },
        "domains": {
            "python": {
                "source": "sealed Phase 4 package and evidence",
                "package_sha256": package_hashes["python"],
            },
            **{
                domain: {
                    "selected_seed": derived["selection"][
                        "domains"
                    ][domain]["seed"],
                    "final_test": derived["quality"]["domains"][
                        domain
                    ],
                    "package_sha256": package_hashes[domain],
                }
                for domain in DOMAINS
            },
        },
        "lineage": {
            "abi_hash": (
                "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
            ),
            "cake_package_hashes": package_hashes,
            "data_hashes": {
                domain: derived["data"][domain][
                    "dataset_sha256"
                ]
                for domain in DOMAINS
            },
            "runtime_hashes": {
                **{
                    "phase5_sql_tensor_payload": tensor_hashes["sql"],
                    "phase5_regex_tensor_payload": tensor_hashes[
                        "regex"
                    ],
                }
            },
            "framework_commit": derived["framework"][
                "framework_commit"
            ],
        },
        "package": {
            "installed_count": 3,
            "archive_hashes": package_hashes,
            "receiver_hosts_per_new_domain": 3,
            "receiver_training_examples": 0,
            "receiver_calibration_runs": 0,
        },
        "test_accessed": True,
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    freeze = subcommands.add_parser("freeze-framework")
    freeze.add_argument("--output", type=Path, required=True)
    choose = subcommands.add_parser("select")
    choose.add_argument("--output", type=Path, required=True)
    audit = subcommands.add_parser("audit-source")
    audit.add_argument("--framework", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    tests = subcommands.add_parser("record-tests")
    tests.add_argument("--junit", type=Path, required=True)
    tests.add_argument("--output", type=Path, required=True)
    tests.add_argument("--test-command", required=True)
    subcommands.add_parser("build-payload")
    args = parser.parse_args()
    if args.command == "freeze-framework":
        result = freeze_framework(_rooted(args.output))
    elif args.command == "select":
        result = select(_rooted(args.output))
    elif args.command == "audit-source":
        result = audit_source(
            _rooted(args.framework), _rooted(args.output)
        )
    elif args.command == "record-tests":
        result = record_tests(
            _rooted(args.junit),
            _rooted(args.output),
            args.test_command,
        )
    else:
        result = build_payload()
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "framework_commit",
                    "domain_specific_source_edit_count",
                    "tests",
                    "evidence_sha256",
                )
                if key in result
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("status") in {
        "PASS",
        "FROZEN",
        "EVIDENCE_READY",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
