"""Typed, fail-closed verification for Phase 5 multi-domain evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from layercake.cake.package import load_package
from layercake.portable_domain import state_dict_hash


class Phase5EvidenceError(RuntimeError):
    """Raised when Phase 5 raw evidence cannot support promotion."""


CONTRACT = Path("moonshot/phase5_generic_multidomain_preregistration.json")
CONTRACT_SHA256 = (
    "49ae047d9a2a11e066404cd2944b43590d08ce20971928996b77e3c62b747597"
)
DIRECT_ABI_VERSION = "lc-direct-neural-decoder/1"
DIRECT_ABI_SHA256 = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)
PYTHON_PACKAGE = Path(
    "artifacts/moonshot/phase4/release/"
    "python-token-plan-seed10141-direct-v1.0.0.cake"
)
PYTHON_PACKAGE_SHA256 = (
    "0585c79bfbea16b1c4165bf0030ba6985b8a8cdeab529cd7afe3f9c76c564ef7"
)
PUBLIC_KEY = Path("moonshot/phase5-multidomain-publisher.public.pem")
FRAMEWORK = Path("results/moonshot/phase5/framework_freeze.json")
SOURCE_AUDIT = Path("results/moonshot/phase5/authoring_source_audit.json")
SELECTION = Path("results/moonshot/phase5/seed_selection.json")
CERTIFICATE = Path(
    "results/moonshot/phase5/multidomain_transfer_certificate.json"
)
GATES = Path("results/moonshot/phase5/raw_runs/gate_observations.json")
PAYLOAD = Path("results/moonshot/phase5/certificate_payload.json")
REGRESSION = Path("results/moonshot/phase5/regression_tests.json")
REGRESSION_JUNIT = Path("results/moonshot/phase5/regression_tests.xml")
SEEDS = (10150, 10151, 10152)
DOMAINS = ("sql", "regex")


def _path(root: Path, relative: Path | str) -> Path:
    value = (root / relative).resolve()
    try:
        value.relative_to(root.resolve())
    except ValueError as error:
        raise Phase5EvidenceError(
            f"Phase 5 path escapes repository: {relative}"
        ) from error
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(root: Path, relative: Path | str) -> dict[str, Any]:
    path = _path(root, relative)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase5EvidenceError(
            f"cannot read Phase 5 evidence {relative}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise Phase5EvidenceError(f"{relative} is not an object")
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
    document: Mapping[str, Any],
    relative: Path | str,
) -> None:
    if document.get("evidence_sha256") != _canonical_sha(document):
        raise Phase5EvidenceError(
            f"{relative} has a stale evidence hash"
        )


def _close(left: float, right: float) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _training_path(domain: str, seed: int) -> Path:
    return Path(
        f"results/moonshot/phase5/{domain}_seed{seed}_training.json"
    )


def _validation_path(domain: str, seed: int) -> Path:
    return Path(
        f"results/moonshot/phase5/{domain}_seed{seed}_validation.json"
    )


def _test_path(domain: str) -> Path:
    return Path(f"results/moonshot/phase5/{domain}_test.json")


def _baseline_path(domain: str) -> Path:
    return Path(
        f"results/moonshot/phase5/{domain}_frozen_core_validation.json"
    )


def _data_freeze_path(domain: str) -> Path:
    return Path(f"results/moonshot/phase5/{domain}_data_freeze.json")


def _config_path(domain: str) -> Path:
    return Path(f"configs/moonshot/phase5/domains/{domain}.json")


def _dataset_path(domain: str) -> Path:
    return Path(f"data/moonshot/phase5/{domain}_v1.jsonl")


def phase5_evidence_files(root: Path) -> list[Path]:
    relatives: set[Path] = {
        CONTRACT,
        PYTHON_PACKAGE,
        PUBLIC_KEY,
        FRAMEWORK,
        SOURCE_AUDIT,
        SELECTION,
        CERTIFICATE,
        GATES,
        PAYLOAD,
        REGRESSION,
        REGRESSION_JUNIT,
    }
    result_dir = _path(root, Path("results/moonshot/phase5"))
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
    moonshot = root / "moonshot"
    relatives.update(
        path.relative_to(root)
        for path in moonshot.glob("phase5*.json")
        if path.is_file()
    )
    for domain in DOMAINS:
        relatives.update(
            {
                _config_path(domain),
                _dataset_path(domain),
                Path(
                    f"artifacts/moonshot/phase5/release/"
                    f"{domain}-token-plan-v1.0.0.cake"
                ),
            }
        )
    selection_path = _path(root, SELECTION)
    if selection_path.is_file():
        selection = _read(root, SELECTION)
        for domain in DOMAINS:
            selected = selection.get("domains", {}).get(
                domain, {}
            ).get("selected", {})
            artifact = selected.get("artifact", {}).get("path")
            if isinstance(artifact, str):
                relatives.add(Path(artifact))
    return [
        _path(root, relative)
        for relative in sorted(relatives)
    ]


def _validate_framework_and_authoring(root: Path) -> dict[str, Any]:
    framework = _read(root, FRAMEWORK)
    audit = _read(root, SOURCE_AUDIT)
    _validate_self_hash(framework, FRAMEWORK)
    _validate_self_hash(audit, SOURCE_AUDIT)
    if (
        framework.get("format")
        != "layercake-phase5-framework-freeze/1"
        or framework.get("status") != "FROZEN"
        or framework.get("contract_sha256") != CONTRACT_SHA256
    ):
        raise Phase5EvidenceError("framework freeze is invalid")
    commit = framework.get("framework_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise Phase5EvidenceError("framework commit is invalid")
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if not exists:
        raise Phase5EvidenceError("framework commit does not exist")
    if (
        audit.get("format")
        != "layercake-phase5-authoring-source-audit/1"
        or audit.get("status") != "PASS"
        or audit.get("framework_commit") != commit
        or audit.get("domain_specific_source_edits") != []
        or audit.get("domain_specific_source_edit_count") != 0
    ):
        raise Phase5EvidenceError(
            "domain authoring required post-freeze source edits"
        )
    return {
        "framework_commit": commit,
        "domain_specific_source_edits": 0,
    }


def _validate_data(root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for domain in DOMAINS:
        freeze = _read(root, _data_freeze_path(domain))
        _validate_self_hash(freeze, _data_freeze_path(domain))
        config_path = _path(root, _config_path(domain))
        dataset_path = _path(root, _dataset_path(domain))
        if (
            freeze.get("format")
            != "layercake-phase5-generic-domain-data-freeze/1"
            or freeze.get("status") != "PASS"
            or freeze.get("domain_id") != domain
            or freeze.get("config", {}).get("sha256")
            != _sha256(config_path)
            or freeze.get("dataset", {}).get("sha256")
            != _sha256(dataset_path)
            or freeze.get("rows")
            != {"train": 1024, "validation": 128, "test": 128}
            or freeze.get("family_count", 0) < 8
            or freeze.get("split_copy_lexeme_overlap") is not False
        ):
            raise Phase5EvidenceError(
                f"{domain} data freeze is invalid"
            )
        summary[domain] = {
            "dataset_sha256": _sha256(dataset_path),
            "config_sha256": _sha256(config_path),
            "families": freeze["family_count"],
            "rows": freeze["rows"],
        }
    return summary


def _validate_seeds_and_selection(
    root: Path,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    raw_candidates: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAINS:
        candidates = []
        for seed in SEEDS:
            training_path = _training_path(domain, seed)
            validation_path = _validation_path(domain, seed)
            training = _read(root, training_path)
            validation = _read(root, validation_path)
            _validate_self_hash(training, training_path)
            _validate_self_hash(validation, validation_path)
            if (
                training.get("format")
                != "layercake-phase5-generic-domain-training/1"
                or training.get("status") != "TRAINED"
                or training.get("domain_id") != domain
                or training.get("seed") != seed
                or training.get("primary_device_name")
                != "NVIDIA GeForce RTX 3080 Laptop GPU"
                or training.get("precision") != "fp32"
                or training.get("optimizer_steps") != 2500
                or training.get("cpu_fallback_smoke", {}).get(
                    "status"
                )
                != "PASS"
                or training.get("validation_accessed") is not False
                or training.get("test_accessed") is not False
                or training.get("dataset", {}).get("sha256")
                != data[domain]["dataset_sha256"]
            ):
                raise Phase5EvidenceError(
                    f"{domain} seed {seed} training is invalid"
                )
            artifact = training.get("artifact", {})
            artifact_path = _path(root, artifact.get("path", ""))
            if (
                artifact.get("sha256") != _sha256(artifact_path)
                or not isinstance(artifact.get("payload_hash"), str)
            ):
                raise Phase5EvidenceError(
                    f"{domain} seed {seed} artifact is stale"
                )
            loaded = torch.load(
                artifact_path, map_location="cpu", weights_only=True
            )
            if loaded.get("payload_hash") != artifact["payload_hash"]:
                raise Phase5EvidenceError(
                    f"{domain} seed {seed} payload changed"
                )
            if (
                validation.get("format")
                != "layercake-phase5-generic-domain-functional/1"
                or validation.get("status") != "PASS"
                or validation.get("domain_id") != domain
                or validation.get("seed") != seed
                or validation.get("split") != "validation"
                or validation.get("distinct_prompts") != 128
                or validation.get("functional_successes", 0) < 122
                or validation.get("artifact", {}).get("sha256")
                != artifact["sha256"]
                or validation.get("dataset", {}).get("sha256")
                != data[domain]["dataset_sha256"]
                or validation.get("test_accessed") is not False
                or validation.get("autonomous_neural_generation")
                is not True
                or validation.get(
                    "templates_retrieval_stored_answers_or_output_rewrite"
                )
                is not False
            ):
                raise Phase5EvidenceError(
                    f"{domain} seed {seed} validation failed"
                )
            candidates.append(
                {
                    "seed": seed,
                    "functional_successes": validation[
                        "functional_successes"
                    ],
                    "invalid_output_count": sum(
                        any(
                            check.get("status") == "INVALID_UTF8"
                            for check in row.get("checks", [])
                        )
                        for row in validation["records"]
                    ),
                    "training_nll": training[
                        "best_action_negative_log_likelihood"
                    ],
                    "artifact": artifact,
                    "validation_evidence": {
                        "path": validation_path.as_posix(),
                        "sha256": _sha256(
                            _path(root, validation_path)
                        ),
                    },
                }
            )
        raw_candidates[domain] = candidates
    selection = _read(root, SELECTION)
    _validate_self_hash(selection, SELECTION)
    if (
        selection.get("format")
        != "layercake-phase5-seed-selection/1"
        or selection.get("status") != "PASS"
        or selection.get("test_accessed") is not False
    ):
        raise Phase5EvidenceError("seed selection record is invalid")
    selected_summary = {}
    for domain in DOMAINS:
        candidates = raw_candidates[domain]
        expected = sorted(
            candidates,
            key=lambda row: (
                -row["functional_successes"],
                row["invalid_output_count"],
                row["training_nll"],
                row["seed"],
            ),
        )[0]
        recorded = selection.get("domains", {}).get(domain)
        if (
            not isinstance(recorded, dict)
            or recorded.get("candidates") != candidates
            or recorded.get("selected") != expected
        ):
            raise Phase5EvidenceError(
                f"{domain} seed selection is not raw-derived"
            )
        selected_summary[domain] = expected
    return {
        "unique_seeds": len(SEEDS),
        "domains": selected_summary,
    }


def _validate_quality(
    root: Path,
    data: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    reductions = {"python": 100.0}
    details: dict[str, Any] = {
        "python": {
            "source": "sealed Phase 4 paired validation",
            "baseline_error_rate": 1.0,
            "cake_error_rate": 0.0,
            "error_reduction_percentage_points": 100.0,
        }
    }
    for domain in DOMAINS:
        baseline_path = _baseline_path(domain)
        baseline = _read(root, baseline_path)
        _validate_self_hash(baseline, baseline_path)
        selected = selection["domains"][domain]
        validation = _read(
            root,
            selected["validation_evidence"]["path"],
        )
        test_path = _test_path(domain)
        test = _read(root, test_path)
        _validate_self_hash(test, test_path)
        if (
            baseline.get("format")
            != "layercake-phase5-frozen-core-functional/1"
            or baseline.get("status") != "COMPLETE"
            or baseline.get("system")
            != "sealed_phase2_frozen_core"
            or baseline.get("domain_id") != domain
            or baseline.get("split") != "validation"
            or baseline.get("distinct_prompts") != 128
            or baseline.get("dataset", {}).get("sha256")
            != data[domain]["dataset_sha256"]
            or baseline.get("checkpoint_sha256")
            != "9e0e6b9add32b4c460f7b570a32584f380e59bf6d631e313ff813069d24e09e1"
            or baseline.get("test_accessed") is not False
        ):
            raise Phase5EvidenceError(
                f"{domain} frozen-core baseline is invalid"
            )
        if (
            test.get("format")
            != "layercake-phase5-generic-domain-functional/1"
            or test.get("status") != "PASS"
            or test.get("domain_id") != domain
            or test.get("seed") != selected["seed"]
            or test.get("split") != "test"
            or test.get("distinct_prompts") != 128
            or test.get("functional_successes", 0) < 122
            or test.get("artifact", {}).get("sha256")
            != selected["artifact"]["sha256"]
            or test.get("dataset", {}).get("sha256")
            != data[domain]["dataset_sha256"]
            or test.get("test_accessed") is not True
        ):
            raise Phase5EvidenceError(
                f"{domain} selected final test failed"
            )
        baseline_error = baseline["functional_failures"] / 128
        cake_error = validation["functional_failures"] / 128
        reduction = 100.0 * (baseline_error - cake_error)
        reductions[domain] = reduction
        details[domain] = {
            "baseline_error_rate": baseline_error,
            "cake_validation_error_rate": cake_error,
            "error_reduction_percentage_points": reduction,
            "final_test_successes": test["functional_successes"],
            "final_test_prompts": 128,
        }
    return {
        "minimum_error_reduction": min(reductions.values()),
        "domains": details,
    }


def _validate_transfer(root: Path) -> dict[str, Any]:
    certificate = _read(root, CERTIFICATE)
    _validate_self_hash(certificate, CERTIFICATE)
    if (
        certificate.get("format")
        != "layercake-phase5-multidomain-certificate/1"
        or certificate.get("status") != "PASS"
        or certificate.get("canonical_interface")
        != {
            "version": DIRECT_ABI_VERSION,
            "sha256": DIRECT_ABI_SHA256,
        }
        or certificate.get("sealed_python", {}).get(
            "archive_sha256"
        )
        != PYTHON_PACKAGE_SHA256
        or not all(certificate.get("gates", {}).values())
    ):
        raise Phase5EvidenceError(
            "multi-domain transfer certificate is invalid"
        )
    public_path = _path(
        root,
        certificate.get("publisher_public_key", {}).get("path", ""),
    )
    if (
        certificate["publisher_public_key"].get("sha256")
        != _sha256(public_path)
        or certificate["publisher_public_key"].get(
            "private_key_committed"
        )
        is not False
    ):
        raise Phase5EvidenceError("Phase 5 public-key binding is invalid")
    key_id = certificate["publisher_public_key"]["key_id"]
    package_summary = {}
    for domain in DOMAINS:
        item = certificate.get("packages", {}).get(domain)
        if not isinstance(item, dict):
            raise Phase5EvidenceError(
                f"{domain} package summary is missing"
            )
        package_path = _path(root, item.get("path", ""))
        if item.get("archive_sha256") != _sha256(package_path):
            raise Phase5EvidenceError(
                f"{domain} package archive changed"
            )
        package = load_package(
            package_path, trust_store={key_id: public_path}
        )
        if (
            not package.signed
            or package.manifest.abi_version != DIRECT_ABI_VERSION
            or package.manifest.abi_hash != DIRECT_ABI_SHA256
            or package.manifest.cake_type != "portable_decoder"
            or package.manifest.domains != (domain,)
            or package.manifest.output_contract.get("composition")
            != "direct_selected_one_cake_no_router"
            or package.manifest.tensor_payload_hash
            != item.get("tensor_payload_hash")
            or state_dict_hash(package.tensors)
            != item.get("artifact_payload_hash")
            or item.get("deterministic_rebuild_equal") is not True
            or item.get("tamper", {}).get("rejected") is not True
        ):
            raise Phase5EvidenceError(
                f"{domain} signed package is invalid"
            )
        receivers = certificate.get("receivers", {}).get(domain)
        if (
            not isinstance(receivers, list)
            or len(receivers) != 3
            or [row.get("device") for row in receivers]
            != ["cpu", "cpu", "cuda:0"]
        ):
            raise Phase5EvidenceError(
                f"{domain} receiver matrix is invalid"
            )
        for receiver in receivers:
            if (
                receiver.get("core_unchanged") is not True
                or receiver.get("archive_hash_equal") is not True
                or receiver.get("output_and_action_identity") is not True
                or receiver.get(
                    "reinstall_output_and_action_identity"
                )
                is not True
                or receiver.get("retention_rate") != 1.0
                or receiver.get("receiver_training_examples") != 0
                or receiver.get("receiver_calibration_runs") != 0
            ):
                raise Phase5EvidenceError(
                    f"{domain} receiver lost identity or behavior"
                )
        package_summary[domain] = {
            "archive_sha256": package.archive_hash,
            "tensor_payload_hash": (
                package.manifest.tensor_payload_hash
            ),
            "receiver_hosts": 3,
        }
    multidomain = certificate.get("multidomain_host", {})
    if (
        multidomain.get("status") != "PASS"
        or len(multidomain.get("installed_ids", [])) != 3
        or multidomain.get("selected_behavior_equal_to_solo")
        is not True
        or multidomain.get("inactive_cake_forward_calls") != 0
        or multidomain.get("manual_selection_only") is not True
        or multidomain.get("router_used") is not False
        or multidomain.get("fusion_or_composition_claimed") is not False
    ):
        raise Phase5EvidenceError(
            "multi-domain host is not physically sparse or claim-bounded"
        )
    return {
        "packages": package_summary,
        "inactive_cake_effect": 0.0,
        "installed_package_count": len(multidomain["installed_ids"]),
        "receiver_hosts_per_new_domain": 3,
    }


def derive_phase5_metrics(root: Path) -> dict[str, Any]:
    if _sha256(_path(root, CONTRACT)) != CONTRACT_SHA256:
        raise Phase5EvidenceError("Phase 5 contract changed")
    if _sha256(_path(root, PYTHON_PACKAGE)) != PYTHON_PACKAGE_SHA256:
        raise Phase5EvidenceError("sealed Python package changed")
    framework = _validate_framework_and_authoring(root)
    data = _validate_data(root)
    selection = _validate_seeds_and_selection(root, data)
    quality = _validate_quality(root, data, selection)
    transfer = _validate_transfer(root)
    metrics = {
        "real_neural_domain_count": 3.0,
        "minimum_domain_functional_error_reduction": quality[
            "minimum_error_reduction"
        ],
        "generic_authoring_requires_source_edits": float(
            framework["domain_specific_source_edits"]
        ),
        "inactive_cake_effect": transfer[
            "inactive_cake_effect"
        ],
    }
    return {
        "metrics": metrics,
        "framework": framework,
        "data": data,
        "selection": selection,
        "quality": quality,
        "transfer": transfer,
    }


def validate_phase5_bundle(
    root: Path,
    phase_dir: Path,
) -> dict[str, Any]:
    del phase_dir
    derived = derive_phase5_metrics(root)
    observations = _read(root, GATES)
    _validate_self_hash(observations, GATES)
    if (
        observations.get("format")
        != "layercake-phase5-gate-observations/1"
        or observations.get("status") != "RAW_DERIVED"
    ):
        raise Phase5EvidenceError("gate observations are invalid")
    observed = {
        row.get("gate_id"): row.get("value")
        for row in observations.get("records", [])
        if isinstance(row, dict)
    }
    if set(observed) != set(derived["metrics"]):
        raise Phase5EvidenceError(
            "gate observation set is incomplete"
        )
    for gate, value in derived["metrics"].items():
        if not _close(observed[gate], value):
            raise Phase5EvidenceError(
                f"gate {gate} is not raw-derived"
            )
    payload = _read(root, PAYLOAD)
    if (
        payload.get("format")
        != "layercake-phase5-certificate-payload/1"
        or payload.get("phase") != 5
        or payload.get("status") != "EVIDENCE_READY"
        or payload.get("abi_version") != DIRECT_ABI_VERSION
        or payload.get("abi_hash") != DIRECT_ABI_SHA256
        or payload.get("claim_boundary", {}).get("routing_claimed")
        is not False
        or payload.get("claim_boundary", {}).get(
            "fusion_or_composition_claimed"
        )
        is not False
    ):
        raise Phase5EvidenceError("certificate payload is invalid")
    tests = _read(root, REGRESSION)
    if (
        tests.get("status") != "PASS"
        or tests.get("failures") != 0
        or tests.get("errors") != 0
        or tests.get("tests", 0) <= 0
        or tests.get("junit_sha256")
        != _sha256(_path(root, REGRESSION_JUNIT))
    ):
        raise Phase5EvidenceError("regression suite is not green")
    return {
        "scope": "generic_manually_selected_multidomain_extensibility",
        "canonical_interface": DIRECT_ABI_VERSION,
        "real_neural_domains": 3,
        "domains": ["python", "sql", "regex"],
        "unique_training_seeds": derived["selection"][
            "unique_seeds"
        ],
        "selected_seeds": {
            domain: derived["selection"]["domains"][domain]["seed"]
            for domain in DOMAINS
        },
        "minimum_functional_error_reduction_percentage_points": (
            derived["quality"]["minimum_error_reduction"]
        ),
        "domain_specific_source_edits": 0,
        "installed_packages": derived["transfer"][
            "installed_package_count"
        ],
        "inactive_cake_forward_calls": 0,
        "receiver_hosts_per_new_domain": derived["transfer"][
            "receiver_hosts_per_new_domain"
        ],
        "routing_claimed": False,
        "fusion_or_composition_claimed": False,
        "regression_tests": tests["tests"],
    }
