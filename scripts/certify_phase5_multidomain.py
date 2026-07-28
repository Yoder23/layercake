"""Build, transfer, and sparsity-certify Phase 5 direct neural cakes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

import _common
from layercake.cake.installer import (
    CakeInstaller,
    HostCapabilities,
    InstallationError,
)
from layercake.cake.manifest import CakeManifest
from layercake.cake.package import (
    build_package,
    load_package,
    tensor_specs,
)
from layercake.cake.registry import CakeRegistry
from layercake.cake.signing import generate_keypair
from layercake.models.direct_cake_host import DirectCakeHost
from layercake.models.portable_decoder import (
    load_cake_module,
    portable_token_plan_manifest_architecture,
)
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import load_token_plan_artifact
from layercake.training.generic_domain import (
    canonical_sha,
    evaluate_generated,
    load_dataset,
    sha256_file,
)
from layercake.training.phase4_python_cake import _load_rows


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FORMAT = "layercake-phase5-multidomain-certification-protocol/1"
CONTRACT = ROOT / "moonshot/phase5_generic_multidomain_preregistration.json"
CONTRACT_SHA256 = (
    "49ae047d9a2a11e066404cd2944b43590d08ce20971928996b77e3c62b747597"
)
DIRECT_ABI_VERSION = "lc-direct-neural-decoder/1"
DIRECT_ABI_SHA256 = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)
PYTHON_PACKAGE_SHA256 = (
    "0585c79bfbea16b1c4165bf0030ba6985b8a8cdeab529cd7afe3f9c76c564ef7"
)
PYTHON_PACKAGE = (
    ROOT
    / "artifacts/moonshot/phase4/release/"
    "python-token-plan-seed10141-direct-v1.0.0.cake"
)
PYTHON_PUBLIC = (
    ROOT / "moonshot/phase4-direct-token-plan-publisher.public.pem"
)
PYTHON_DATASET = (
    ROOT / "data/moonshot/phase4/python_functional_v1.jsonl"
)
CORE_ROOT = (
    ROOT / "artifacts/moonshot/phase2_shallow_sparse_pretrained"
)
RECEIVER_SEEDS = (9824, 9825, 9826)


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Phase 5 path escapes repository") from error
    return path


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _key_id(package_path: Path) -> str:
    package = load_package(
        package_path,
        trust_store={
            _read(
                ROOT
                / "results/moonshot/phase4/"
                "direct_decoder_transfer_certificate.json"
            )["package"]["key_id"]: PYTHON_PUBLIC
        },
    )
    return str(package.manifest.signature["key_id"])


def _generate_model(
    model,
    prompt: str,
) -> dict[str, Any]:
    state = model.prefill_bytes(prompt + "\n")
    while not state.complete:
        model.decode_step(state)
    raw = model.tokenizer.decode_actions(
        state.generated_actions, state.source_lexemes
    )
    return {
        "output": raw,
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "actions": tuple(state.generated_actions),
    }


def _build_package(
    *,
    domain: str,
    artifact: Mapping[str, Any],
    test: Mapping[str, Any],
    dataset_path: Path,
    package_path: Path,
    private_key: bytes,
    key_id: str,
) -> tuple[Any, bool]:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in artifact["state_dict"].items()
    }
    manifest = CakeManifest(
        schema_version="1",
        cake_id=f"{domain}-token-plan",
        name=f"LayerCake {domain.upper()} Token Plan",
        description=(
            f"GPU-trained lossless directly selected {domain} capability"
        ),
        version="1.0.0",
        publisher={
            "id": "layercake-research",
            "name": "LayerCake Research",
            "key_id": key_id,
        },
        abi_version=DIRECT_ABI_VERSION,
        abi_hash=DIRECT_ABI_SHA256,
        cake_type="portable_decoder",
        input_contract={
            "mode": "direct_selected_portable_decoder",
            "external": "UTF-8 bytes",
            "canonical_semantic_abi_consumed": False,
        },
        output_contract={
            "mode": "autonomous_extended_vocabulary_actions",
            "external": "UTF-8 bytes",
            "composition": "direct_selected_one_cake_no_router",
        },
        architecture=portable_token_plan_manifest_architecture(
            artifact["spec"]
        ),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={
            "features": ["byte_input", "safe_tensors", "incremental"]
        },
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(state),
        package_hash="",
        training_data_provenance={
            "dataset": _relative(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "seed": artifact["training"]["seed"],
            "optimizer_steps": artifact["training"]["optimizer_steps"],
            "primary_device": artifact["training"][
                "primary_device_name"
            ],
            "receiver_data_used": False,
        },
        evaluation_evidence={
            "final_test_status": test["status"],
            "final_test_successes": test["functional_successes"],
            "final_test_prompts": test["distinct_prompts"],
            "final_test_evidence_sha256": test["evidence_sha256"],
            "claim_status": "PHASE5_MULTIDOMAIN_CANDIDATE",
        },
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id},
        domains=(domain,),
        keywords=(domain, "portable", "token-plan"),
        permissions=("local-inference",),
    )
    build_package(
        package_path, manifest, state, private_key=private_key
    )
    with tempfile.TemporaryDirectory(
        prefix=f"layercake-phase5-{domain}-rebuild-"
    ) as temporary:
        rebuilt = Path(temporary) / package_path.name
        build_package(
            rebuilt, manifest, state, private_key=private_key
        )
        deterministic = rebuilt.read_bytes() == package_path.read_bytes()
    package = load_package(
        package_path,
        trust_store={key_id: private_key_to_public(private_key)},
    )
    return package, deterministic


def private_key_to_public(private_key: bytes) -> bytes:
    """Derive the raw PEM public key without retaining private material."""

    from cryptography.hazmat.primitives import serialization

    loaded = serialization.load_pem_private_key(
        private_key, password=None
    )
    return loaded.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _tamper_rejected(
    package_path: Path,
    *,
    public: bytes,
    key_id: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="layercake-phase5-tamper-"
    ) as temporary:
        tampered = Path(temporary) / package_path.name
        raw = bytearray(package_path.read_bytes())
        raw[len(raw) // 2] ^= 1
        tampered.write_bytes(raw)
        installer = CakeInstaller(
            CakeRegistry(Path(temporary) / "registry"),
            HostCapabilities(
                DIRECT_ABI_VERSION,
                DIRECT_ABI_SHA256,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=frozenset(
                    {"byte_input", "safe_tensors", "incremental"}
                ),
            ),
            trust_store={key_id: public},
        )
        try:
            installer.inspect(tampered)
        except InstallationError as error:
            return {"rejected": True, "reason": str(error)}
    return {"rejected": False, "reason": None}


def _receiver(
    *,
    domain: str,
    package_path: Path,
    public: bytes,
    key_id: str,
    rows: list[dict[str, Any]],
    source: Mapping[str, Mapping[str, Any]],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    core_path = (
        CORE_ROOT
        / f"student2400-seed-{seed}"
        / "model.safetensors"
    )
    core_before = sha256_file(core_path)
    with tempfile.TemporaryDirectory(
        prefix=f"layercake-phase5-{domain}-receiver-{seed}-"
    ) as temporary:
        host = DirectCakeHost(
            Path(temporary) / "registry",
            abi_version=DIRECT_ABI_VERSION,
            abi_hash=DIRECT_ABI_SHA256,
            trust_store={key_id: public},
            device=device,
        )
        installed = host.install(package_path)
        verified = host.installer.verify(f"{domain}-token-plan")
        records = []
        for row in rows:
            result = host.generate(
                f"{domain}-token-plan", row["prompt"] + "\n"
            )
            passed, checks = evaluate_generated(result.output, row)
            records.append(
                {
                    "id": row["id"],
                    "output_sha256": hashlib.sha256(
                        result.output
                    ).hexdigest(),
                    "actions": list(result.actions),
                    "functional_success": passed,
                    "checks": checks,
                }
            )
        removed = host.remove(f"{domain}-token-plan")
        reinstalled = host.install(package_path)
        reinstall = [
            host.generate(
                f"{domain}-token-plan", row["prompt"] + "\n"
            )
            for row in rows
        ]
    core_after = sha256_file(core_path)
    source_ids = {
        identifier
        for identifier, record in source.items()
        if record["functional_success"]
    }
    retained = {
        record["id"]
        for record in records
        if record["functional_success"]
    }
    output_identity = all(
        record["output_sha256"]
        == source[record["id"]]["output_sha256"]
        and tuple(record["actions"])
        == tuple(source[record["id"]]["actions"])
        for record in records
    )
    reinstall_identity = all(
        hashlib.sha256(result.output).hexdigest()
        == records[index]["output_sha256"]
        and tuple(result.actions) == tuple(records[index]["actions"])
        for index, result in enumerate(reinstall)
    )
    return {
        "domain_id": domain,
        "seed": seed,
        "device": str(device),
        "core_hash_before": core_before,
        "core_hash_after": core_after,
        "core_unchanged": core_before == core_after,
        "install": installed,
        "verify": verified,
        "remove": removed,
        "reinstall": reinstalled,
        "archive_hash_equal": (
            installed["archive_hash"]
            == reinstalled["archive_hash"]
            == sha256_file(package_path)
        ),
        "output_and_action_identity": output_identity,
        "reinstall_output_and_action_identity": reinstall_identity,
        "source_successes": len(source_ids),
        "retained_source_successes": len(source_ids & retained),
        "retention_rate": len(source_ids & retained) / max(
            1, len(source_ids)
        ),
        "receiver_training_examples": 0,
        "receiver_calibration_runs": 0,
        "telemetry": host.telemetry(),
        "records": records,
    }


def _multidomain_host(
    *,
    packages: Mapping[str, Path],
    trust_store: Mapping[str, bytes | Path],
    rows: Mapping[str, list[dict[str, Any]]],
    source: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="layercake-phase5-multihost-"
    ) as temporary:
        host = DirectCakeHost(
            Path(temporary) / "registry",
            abi_version=DIRECT_ABI_VERSION,
            abi_hash=DIRECT_ABI_SHA256,
            trust_store=trust_store,
            device="cpu",
        )
        installs = [
            host.install(path) for path in packages.values()
        ]
        installed_ids = host.installed_ids()
        host.reset_telemetry()
        records = []
        for domain in ("python", "sql", "regex"):
            cake_id = (
                "python-token-plan"
                if domain == "python"
                else f"{domain}-token-plan"
            )
            for row in rows[domain][:16]:
                result = host.generate(
                    cake_id, row["prompt"] + "\n"
                )
                expected = source[domain][row["id"]]
                records.append(
                    {
                        "domain_id": domain,
                        "cake_id": cake_id,
                        "id": row["id"],
                        "output_sha256": hashlib.sha256(
                            result.output
                        ).hexdigest(),
                        "actions": list(result.actions),
                        "solo_output_equal": (
                            hashlib.sha256(
                                result.output
                            ).hexdigest()
                            == expected["output_sha256"]
                        ),
                        "solo_actions_equal": (
                            tuple(result.actions)
                            == tuple(expected["actions"])
                        ),
                    }
                )
        telemetry = host.telemetry()
    expected_calls = {
        "python-token-plan": 16,
        "sql-token-plan": 16,
        "regex-token-plan": 16,
    }
    prefill_correct = all(
        telemetry[cake_id]["prefill_calls"] == count
        for cake_id, count in expected_calls.items()
    )
    exact = all(
        row["solo_output_equal"] and row["solo_actions_equal"]
        for row in records
    )
    # Each per-request trace is also checked by taking a difference around a
    # single selected request. This proves nonselected counters do not move.
    isolation = []
    with tempfile.TemporaryDirectory(
        prefix="layercake-phase5-isolation-"
    ) as temporary:
        host = DirectCakeHost(
            Path(temporary) / "registry",
            abi_version=DIRECT_ABI_VERSION,
            abi_hash=DIRECT_ABI_SHA256,
            trust_store=trust_store,
            device="cpu",
        )
        for path in packages.values():
            host.install(path)
        for domain in ("python", "sql", "regex"):
            cake_id = (
                "python-token-plan"
                if domain == "python"
                else f"{domain}-token-plan"
            )
            before = host.telemetry()
            host.generate(cake_id, rows[domain][0]["prompt"] + "\n")
            after = host.telemetry()
            inactive_deltas = {
                identifier: {
                    key: after[identifier][key] - before[identifier][key]
                    for key in after[identifier]
                }
                for identifier in after
                if identifier != cake_id
            }
            isolation.append(
                {
                    "selected": cake_id,
                    "inactive_deltas": inactive_deltas,
                    "inactive_neural_forward_calls": sum(
                        values["prefill_calls"]
                        + values["decode_step_calls"]
                        for values in inactive_deltas.values()
                    ),
                }
            )
    return {
        "status": (
            "PASS"
            if len(installed_ids) == 3
            and exact
            and prefill_correct
            and all(
                row["inactive_neural_forward_calls"] == 0
                for row in isolation
            )
            else "FAIL"
        ),
        "manual_selection_only": True,
        "router_used": False,
        "fusion_or_composition_claimed": False,
        "installs": installs,
        "installed_ids": list(installed_ids),
        "selected_behavior_equal_to_solo": exact,
        "telemetry": telemetry,
        "selection_isolation": isolation,
        "inactive_cake_forward_calls": sum(
            row["inactive_neural_forward_calls"] for row in isolation
        ),
        "records": records,
    }


def certify(
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("Phase 5 certification output is immutable")
    protocol = _read(protocol_path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status")
        != "PREREGISTERED_BEFORE_PACKAGE_BUILD_AND_CERTIFICATION"
    ):
        raise ValueError("Phase 5 certification protocol is invalid")
    if sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise ValueError("Phase 5 contract changed")
    if protocol["parent_contract"] != {
        "path": _relative(CONTRACT),
        "sha256": CONTRACT_SHA256,
    }:
        raise ValueError("certification protocol parent mismatch")
    if sha256_file(PYTHON_PACKAGE) != PYTHON_PACKAGE_SHA256:
        raise ValueError("sealed Python package changed")
    private, public, key_id = generate_keypair()
    public_path = _rooted(protocol["publisher_public_key"])
    if public_path.exists():
        raise RuntimeError("Phase 5 public key output is immutable")
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(public)
    selected: dict[str, dict[str, Any]] = {}
    packages: dict[str, Path] = {"python": PYTHON_PACKAGE}
    package_summaries = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    source: dict[str, dict[str, dict[str, Any]]] = {}
    for domain in ("sql", "regex"):
        item = protocol["domains"][domain]
        artifact_path = _rooted(item["artifact"]["path"])
        test_path = _rooted(item["test_evidence"]["path"])
        dataset_path = _rooted(item["dataset"]["path"])
        for path, expected in (
            (artifact_path, item["artifact"]["sha256"]),
            (test_path, item["test_evidence"]["sha256"]),
            (dataset_path, item["dataset"]["sha256"]),
        ):
            if sha256_file(path) != expected:
                raise ValueError(
                    f"{domain} certification input hash mismatch"
                )
        test = _read(test_path)
        if (
            test["status"] != "PASS"
            or test["split"] != "test"
            or test["functional_successes"]
            < protocol["minimum_test_successes"]
        ):
            raise ValueError(f"{domain} final test is not promoted")
        artifact = torch.load(
            artifact_path, map_location="cpu", weights_only=True
        )
        if artifact["payload_hash"] != item["artifact"]["payload_hash"]:
            raise ValueError(f"{domain} payload hash mismatch")
        domain_rows = [
            row
            for row in load_dataset(dataset_path)
            if row["split"] == "test"
        ]
        _, _, source_model = load_token_plan_artifact(
            artifact, "cpu"
        )
        source_records: dict[str, dict[str, Any]] = {}
        for row in domain_rows:
            generated = _generate_model(source_model, row["prompt"])
            passed, checks = evaluate_generated(
                generated["output"], row
            )
            source_records[row["id"]] = {
                **generated,
                "functional_success": passed,
                "checks": checks,
            }
        if sum(
            item["functional_success"]
            for item in source_records.values()
        ) != test["functional_successes"]:
            raise ValueError(
                f"{domain} source re-evaluation differs from test evidence"
            )
        package_path = _rooted(item["package_path"])
        if package_path.exists():
            raise RuntimeError(f"{domain} package output is immutable")
        package, deterministic = _build_package(
            domain=domain,
            artifact=artifact,
            test=test,
            dataset_path=dataset_path,
            package_path=package_path,
            private_key=private,
            key_id=key_id,
        )
        packages[domain] = package_path
        rows[domain] = domain_rows
        source[domain] = source_records
        selected[domain] = {
            "artifact": artifact,
            "test": test,
        }
        package_summaries[domain] = {
            "path": _relative(package_path),
            "archive_sha256": package.archive_hash,
            "content_hash": package.manifest.package_hash,
            "tensor_payload_hash": (
                package.manifest.tensor_payload_hash
            ),
            "artifact_payload_hash": artifact["payload_hash"],
            "signed": package.signed,
            "key_id": key_id,
            "deterministic_rebuild_equal": deterministic,
            "bytes": package_path.stat().st_size,
            "state_dict_hash_equal": (
                state_dict_hash(package.tensors)
                == artifact["payload_hash"]
            ),
            "tamper": _tamper_rejected(
                package_path, public=public, key_id=key_id
            ),
        }
    python_key_id = _key_id(PYTHON_PACKAGE)
    python_package = load_package(
        PYTHON_PACKAGE,
        trust_store={python_key_id: PYTHON_PUBLIC},
    )
    python_model = load_cake_module(python_package).cpu().eval()
    python_rows = [
        row
        for row in _load_rows(PYTHON_DATASET)
        if row["split"] == "test"
    ]
    rows["python"] = python_rows
    source["python"] = {
        row["id"]: _generate_model(python_model, row["prompt"])
        for row in python_rows
    }
    trust_store: dict[str, bytes | Path] = {
        key_id: public,
        python_key_id: PYTHON_PUBLIC,
    }
    receivers = {
        domain: [
            _receiver(
                domain=domain,
                package_path=packages[domain],
                public=public,
                key_id=key_id,
                rows=rows[domain],
                source=source[domain],
                seed=seed,
                device=device,
            )
            for seed, device in zip(
                RECEIVER_SEEDS,
                (
                    torch.device("cpu"),
                    torch.device("cpu"),
                    torch.device("cuda:0"),
                ),
            )
        ]
        for domain in ("sql", "regex")
    }
    multidomain = _multidomain_host(
        packages=packages,
        trust_store=trust_store,
        rows=rows,
        source=source,
    )
    all_receiver_pass = all(
        receiver["core_unchanged"]
        and receiver["archive_hash_equal"]
        and receiver["output_and_action_identity"]
        and receiver["reinstall_output_and_action_identity"]
        and receiver["retention_rate"] == 1.0
        and receiver["receiver_training_examples"] == 0
        and receiver["receiver_calibration_runs"] == 0
        for values in receivers.values()
        for receiver in values
    )
    gates = {
        "sealed_python_package_unchanged": (
            sha256_file(PYTHON_PACKAGE) == PYTHON_PACKAGE_SHA256
        ),
        "two_new_signed_packages": (
            len(package_summaries) == 2
            and all(
                item["signed"]
                and item["deterministic_rebuild_equal"]
                and item["state_dict_hash_equal"]
                for item in package_summaries.values()
            )
        ),
        "tampered_packages_rejected": all(
            item["tamper"]["rejected"]
            for item in package_summaries.values()
        ),
        "three_receivers_per_new_domain": all(
            len(values) == 3 for values in receivers.values()
        ),
        "cpu_cpu_cuda_receiver_matrix": all(
            [row["device"] for row in values]
            == ["cpu", "cpu", "cuda:0"]
            for values in receivers.values()
        ),
        "complete_source_success_retention": all_receiver_pass,
        "receiver_learning_and_calibration_zero": all(
            row["receiver_training_examples"] == 0
            and row["receiver_calibration_runs"] == 0
            for values in receivers.values()
            for row in values
        ),
        "three_packages_coinstalled": (
            len(multidomain["installed_ids"]) == 3
        ),
        "selected_behavior_equal_solo": multidomain[
            "selected_behavior_equal_to_solo"
        ],
        "inactive_cake_forward_calls_zero": (
            multidomain["inactive_cake_forward_calls"] == 0
        ),
        "manual_selection_no_router_or_fusion_claim": (
            multidomain["manual_selection_only"]
            and not multidomain["router_used"]
            and not multidomain["fusion_or_composition_claimed"]
        ),
        "private_key_not_written": True,
    }
    output = {
        "format": "layercake-phase5-multidomain-certificate/1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": {
            "path": _relative(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "canonical_interface": {
            "version": DIRECT_ABI_VERSION,
            "sha256": DIRECT_ABI_SHA256,
        },
        "publisher_public_key": {
            "path": _relative(public_path),
            "sha256": sha256_file(public_path),
            "key_id": key_id,
            "private_key_committed": False,
        },
        "sealed_python": {
            "path": _relative(PYTHON_PACKAGE),
            "archive_sha256": sha256_file(PYTHON_PACKAGE),
            "key_id": python_key_id,
        },
        "packages": package_summaries,
        "receivers": receivers,
        "multidomain_host": multidomain,
        "gates": gates,
        "test_accessed": True,
    }
    output["evidence_sha256"] = canonical_sha(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = certify(
        _rooted(args.protocol), _rooted(args.output)
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "packages": result["packages"],
                "gates": result["gates"],
                "evidence_sha256": result["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
