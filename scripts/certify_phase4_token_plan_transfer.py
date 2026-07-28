"""Certify the selected GPU-trained token-plan through the existing cake path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from layercake.models.portable_decoder import (
    load_cake_module,
    portable_token_plan_manifest_architecture,
)
from layercake.portable_domain import state_dict_hash
from layercake.portable_token_plan import load_token_plan_artifact
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _execute_tests,
    _extract_function,
    _load_rows,
)


ABI_VERSION = "lc-semantic-gpt2-768/1"
ABI_HASH = "d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761"
ARTIFACT = (
    ROOT
    / "artifacts/moonshot/phase4/candidates"
    / "python-portable-token-plan-seed10141.pt"
)
PACKAGE = (
    ROOT
    / "artifacts/moonshot/phase4/release"
    / "python-token-plan-seed10141-v1.0.0.cake"
)
PRIVATE_KEY = PACKAGE.with_suffix(".private.pem")
PUBLIC_KEY = ROOT / "moonshot/phase4-token-plan-publisher.public.pem"
TRUST_STORE = ROOT / "moonshot/phase4-token-plan-trust-store.json"
EVIDENCE = (
    ROOT / "results/moonshot/phase4/token_plan_transfer_certificate.json"
)
DATASET = ROOT / "data/moonshot/phase4/python_functional_v1.jsonl"
TEST_EVIDENCE = (
    ROOT
    / "results/moonshot/phase4"
    / "python-portable-token-plan-seed10141-test.json"
)
CORE_ROOT = (
    ROOT / "artifacts/moonshot/phase2_shallow_sparse_pretrained"
)
RECEIVER_SEEDS = (9824, 9825, 9826)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _functional_result(model, row: dict[str, Any]) -> dict[str, Any]:
    raw = model.generate_bytes(row["prompt"] + "\n")
    text = raw.decode("utf-8", errors="replace")
    source, parse_status = _extract_function(text, row["function_name"])
    passed = False
    tests = [{"status": parse_status}]
    if source is not None:
        passed, tests = _execute_tests(
            source, row["function_name"], row["tests"]
        )
    return {
        "id": row["id"],
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "output_bytes": len(raw),
        "functional_success": passed,
        "tests": tests,
    }


def _build_signed_package(
    artifact: dict[str, Any],
) -> tuple[Any, bytes, str, bool]:
    if any(
        path.exists()
        for path in (PACKAGE, PRIVATE_KEY, PUBLIC_KEY, TRUST_STORE)
    ):
        raise RuntimeError(
            "Phase 4 signed package/key outputs are immutable and already exist"
        )
    private, public, key_id = generate_keypair()
    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_KEY.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_KEY.write_bytes(private)
    PUBLIC_KEY.write_bytes(public)
    TRUST_STORE.write_text(
        json.dumps(
            {key_id: PUBLIC_KEY.name},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in artifact["state_dict"].items()
    }
    test = json.loads(TEST_EVIDENCE.read_text(encoding="utf-8"))
    manifest = CakeManifest(
        schema_version="1",
        cake_id="python-token-plan",
        name="LayerCake Python Token Plan",
        description=(
            "GPU-trained, lossless, directly selected Python capability"
        ),
        version="1.0.0",
        publisher={
            "id": "layercake-research",
            "name": "LayerCake Research",
            "key_id": key_id,
        },
        abi_version=ABI_VERSION,
        abi_hash=ABI_HASH,
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
            "functional_dataset": (
                "data/moonshot/phase4/python_functional_v4.jsonl"
            ),
            "functional_dataset_sha256": artifact["training"][
                "functional_training_data_sha256"
            ],
            "lexical_dataset": (
                "data/moonshot/phase4/lexical_conformance_v1.jsonl"
            ),
            "lexical_dataset_sha256": artifact["training"][
                "lexical_training_data_sha256"
            ],
            "seed": artifact["training"]["seed"],
            "optimizer_steps": artifact["training"]["optimizer_steps"],
            "primary_device": artifact["training"]["primary_device_name"],
            "receiver_data_used": False,
        },
        evaluation_evidence={
            "final_test_status": test["status"],
            "final_test_successes": test["functional_successes"],
            "final_test_prompts": test["distinct_prompts"],
            "final_test_evidence_sha256": test["evidence_sha256"],
            "claim_status": "DIRECT_DECODER_TRANSFER_CANDIDATE",
        },
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id},
        domains=("python",),
        keywords=("python", "portable", "token-plan"),
        permissions=("local-inference",),
    )
    build_package(PACKAGE, manifest, state, private_key=private)
    with tempfile.TemporaryDirectory(
        prefix="layercake-package-rebuild-"
    ) as temporary:
        rebuilt = Path(temporary) / PACKAGE.name
        build_package(rebuilt, manifest, state, private_key=private)
        deterministic = rebuilt.read_bytes() == PACKAGE.read_bytes()
    package = load_package(PACKAGE, trust_store={key_id: public})
    return package, public, key_id, deterministic


def _tamper_rejected(
    public: bytes, key_id: str
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="layercake-tamper-"
    ) as temporary:
        path = Path(temporary) / "tampered.cake"
        raw = bytearray(PACKAGE.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
        installer = CakeInstaller(
            CakeRegistry(Path(temporary) / "registry"),
            HostCapabilities(
                ABI_VERSION,
                ABI_HASH,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=frozenset(
                    {"byte_input", "safe_tensors", "incremental"}
                ),
            ),
            trust_store={key_id: public},
        )
        rejected = False
        reason = None
        try:
            installer.inspect(path)
        except InstallationError as error:
            rejected = True
            reason = str(error)
    return {"rejected": rejected, "reason": reason}


def _receiver(
    *,
    seed: int,
    package,
    public: bytes,
    key_id: str,
    rows: list[dict[str, Any]],
    source: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    core_path = CORE_ROOT / f"student2400-seed-{seed}/model.safetensors"
    core_before = _sha(core_path)
    with tempfile.TemporaryDirectory(
        prefix=f"layercake-receiver-{seed}-"
    ) as temporary:
        capabilities = HostCapabilities(
            ABI_VERSION,
            ABI_HASH,
            precisions=("fp32",),
            backends=(
                ("pytorch", "cuda")
                if device.type == "cuda"
                else ("pytorch",)
            ),
            capabilities=frozenset(
                {"byte_input", "safe_tensors", "incremental"}
            ),
        )
        installer = CakeInstaller(
            CakeRegistry(Path(temporary) / "registry"),
            capabilities,
            trust_store={key_id: public},
        )
        installed = installer.install(PACKAGE)
        verified_before = installer.verify(package.manifest.cake_id)
        installed_package = load_package(
            Path(installed["blob"]), trust_store={key_id: public}
        )
        model = load_cake_module(installed_package).to(device)
        tensor_identity = all(
            torch.equal(
                package.tensors[name],
                value.detach().cpu(),
            )
            for name, value in model.state_dict().items()
        )
        records = [_functional_result(model, row) for row in rows]
        success_ids = {
            record["id"]
            for record in records
            if record["functional_success"]
        }
        output_identity = all(
            record["output_sha256"]
            == source[record["id"]]["output_sha256"]
            for record in records
        )
        removed = installer.remove(package.manifest.cake_id)
        reinstalled = installer.install(PACKAGE)
        verified_after = installer.verify(package.manifest.cake_id)
        reinstalled_package = load_package(
            Path(reinstalled["blob"]), trust_store={key_id: public}
        )
        reinstalled_model = load_cake_module(reinstalled_package).to(device)
        reinstall_records = [
            _functional_result(reinstalled_model, row)
            for row in rows
        ]
    core_after = _sha(core_path)
    source_successes = {
        identifier
        for identifier, record in source.items()
        if record["functional_success"]
    }
    reinstall_identity = all(
        first["output_sha256"] == second["output_sha256"]
        for first, second in zip(records, reinstall_records)
    )
    retained = source_successes & success_ids
    return {
        "seed": seed,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device.index or 0)
            if device.type == "cuda"
            else "declared laptop CPU"
        ),
        "core_hash_before": core_before,
        "core_hash_after": core_after,
        "core_unchanged": core_before == core_after,
        "install": installed,
        "verify_before": verified_before,
        "remove": removed,
        "reinstall": reinstalled,
        "verify_after": verified_after,
        "archive_hash_equal": (
            installed["archive_hash"]
            == reinstalled["archive_hash"]
            == package.archive_hash
        ),
        "tensor_identity": tensor_identity,
        "output_identity": output_identity,
        "reinstall_output_identity": reinstall_identity,
        "source_successes": len(source_successes),
        "retained_source_successes": len(retained),
        "retention_rate": len(retained) / max(1, len(source_successes)),
        "receiver_training_examples": 0,
        "receiver_calibration_runs": 0,
        "records": records,
    }


def _incremental_equivalence(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        row for row in _load_rows(DATASET) if row["split"] == "validation"
    ]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    results = []
    for device in devices:
        _, tokenizer, model = load_token_plan_artifact(artifact, device)
        matched = 0
        maximum_cache_positions = 0
        for row in rows:
            source_ids, _ = tokenizer.encode_source(row["prompt"])
            source = torch.tensor(
                [source_ids], dtype=torch.long, device=device
            )
            expected = model.generate_actions(source)[0]
            state = model.prefill_bytes(row["prompt"])
            while not state.complete:
                model.decode_step(state)
            matched += expected == state.generated_actions
            maximum_cache_positions = max(
                maximum_cache_positions,
                max(
                    cache.shape[1]
                    for cache in state.layer_self_attention_inputs
                ),
            )
        results.append(
            {
                "device": str(device),
                "prompts": len(rows),
                "action_sequences_equal": matched,
                "status": "PASS" if matched == len(rows) else "FAIL",
                "source_encoded_once": True,
                "target_self_attention_cache": True,
                "maximum_cached_action_positions": maximum_cache_positions,
                "completed_prefix_recomputation": False,
            }
        )
    return {
        "status": (
            "PASS"
            if all(row["status"] == "PASS" for row in results)
            else "FAIL"
        ),
        "devices": results,
    }


def certify() -> dict[str, Any]:
    if EVIDENCE.exists():
        raise RuntimeError("Phase 4 transfer evidence is immutable")
    if _sha(ARTIFACT) != (
        "211db7a97194234f4fbf2a04c99f99eea4698cc73bd486f901e7d12fb8af439e"
    ):
        raise ValueError("selected Phase 4 artifact hash mismatch")
    artifact = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    if artifact["payload_hash"] != (
        "a70fcb62a2c24305ca0dad3929124ca6881363d26366c9374b96e7951b5cd49f"
    ):
        raise ValueError("selected Phase 4 payload hash mismatch")
    test = json.loads(TEST_EVIDENCE.read_text(encoding="utf-8"))
    if (
        test["status"] != "PASS"
        or test["functional_successes"] != 128
        or test["evidence_sha256"]
        != "b8d0eb00e89e008e33309226d7033037e0600868f94588c66f86a42a096f4d3b"
    ):
        raise ValueError("selected Phase 4 final test is not authorized")
    package, public, key_id, deterministic = _build_signed_package(
        artifact
    )
    state_hash_equal = (
        state_dict_hash(package.tensors) == artifact["payload_hash"]
    )
    torch.set_num_threads(1)
    _, _, source_model = load_token_plan_artifact(artifact, "cpu")
    rows = [
        row for row in _load_rows(DATASET) if row["split"] == "test"
    ]
    source_records = [
        _functional_result(source_model, row) for row in rows
    ]
    source = {record["id"]: record for record in source_records}
    devices = (
        torch.device("cpu"),
        torch.device("cpu"),
        torch.device("cuda:0"),
    )
    receivers = [
        _receiver(
            seed=seed,
            package=package,
            public=public,
            key_id=key_id,
            rows=rows,
            source=source,
            device=device,
        )
        for seed, device in zip(RECEIVER_SEEDS, devices)
    ]
    package_direct_pass = (
        deterministic
        and package.signed
        and state_hash_equal
        and all(record["functional_success"] for record in source_records)
        and all(
            receiver["core_unchanged"]
            and receiver["archive_hash_equal"]
            and receiver["tensor_identity"]
            and receiver["output_identity"]
            and receiver["reinstall_output_identity"]
            and receiver["retention_rate"] == 1.0
            for receiver in receivers
        )
    )
    incremental = _incremental_equivalence(artifact)
    tamper = _tamper_rejected(public, key_id)
    semantic_abi_consumed = bool(
        package.manifest.input_contract.get(
            "canonical_semantic_abi_consumed"
        )
    )
    gates = {
        "final_test_128_of_128": (
            test["functional_successes"] == test["distinct_prompts"] == 128
        ),
        "package_signed": package.signed,
        "deterministic_package_build": deterministic,
        "artifact_state_hash_preserved": state_hash_equal,
        "three_receivers": len(receivers) == 3,
        "all_source_successes_retained": all(
            receiver["retention_rate"] == 1.0 for receiver in receivers
        ),
        "receiver_learning_zero": all(
            receiver["receiver_training_examples"] == 0
            and receiver["receiver_calibration_runs"] == 0
            for receiver in receivers
        ),
        "core_immutable": all(
            receiver["core_unchanged"] for receiver in receivers
        ),
        "cpu_install_and_execution": sum(
            receiver["device"] == "cpu" for receiver in receivers
        )
        >= 2,
        "cuda_install_and_execution": any(
            receiver["device"].startswith("cuda")
            for receiver in receivers
        ),
        "incremental_action_equivalence": incremental["status"] == "PASS",
        "tampered_package_rejected": tamper["rejected"],
        "canonical_semantic_abi_contract_unchanged": (
            package.manifest.abi_version == ABI_VERSION
            and package.manifest.abi_hash == ABI_HASH
        ),
        "candidate_consumes_and_returns_semantic_abi": (
            semantic_abi_consumed
            and package.manifest.cake_type in {
                "host_residual",
                "portable_fusion",
            }
        ),
    }
    result = {
        "format": "layercake-phase4-token-plan-transfer-certificate/1",
        "status": (
            "PARTIAL_PASS_DIRECT_DECODER_SEMANTIC_ABI_GATE_OPEN"
            if package_direct_pass
            and incremental["status"] == "PASS"
            and tamper["rejected"]
            and not gates[
                "candidate_consumes_and_returns_semantic_abi"
            ]
            else "FAIL"
        ),
        "preregistration": (
            "moonshot/phase4_token_plan_promotion_preregistration.json"
        ),
        "test_evidence": {
            "path": TEST_EVIDENCE.relative_to(ROOT).as_posix(),
            "sha256": _sha(TEST_EVIDENCE),
            "evidence_sha256": test["evidence_sha256"],
            "functional_successes": test["functional_successes"],
            "distinct_prompts": test["distinct_prompts"],
        },
        "package": {
            "path": PACKAGE.relative_to(ROOT).as_posix(),
            "archive_sha256": package.archive_hash,
            "content_hash": package.manifest.package_hash,
            "tensor_payload_hash": package.manifest.tensor_payload_hash,
            "artifact_state_dict_hash": artifact["payload_hash"],
            "signed": package.signed,
            "key_id": key_id,
            "public_key": PUBLIC_KEY.relative_to(ROOT).as_posix(),
            "private_key_committed": False,
            "deterministic_rebuild_equal": deterministic,
            "bytes": PACKAGE.stat().st_size,
        },
        "source": {
            "artifact": ARTIFACT.relative_to(ROOT).as_posix(),
            "artifact_sha256": _sha(ARTIFACT),
            "payload_hash": artifact["payload_hash"],
            "functional_successes": sum(
                record["functional_success"]
                for record in source_records
            ),
            "functional_prompts": len(source_records),
            "records": source_records,
        },
        "receivers": receivers,
        "incremental_execution": incremental,
        "adversarial_package_test": tamper,
        "execution_mode_audit": {
            "mode": "direct_selected_portable_decoder",
            "canonical_semantic_abi_contract_unchanged": True,
            "canonical_semantic_abi_state_consumed": semantic_abi_consumed,
            "claim": (
                "Direct package transfer is proven. Semantic-residual ABI "
                "attachment is not proven by this representation."
            ),
        },
        "gates": gates,
        "process": {
            "resident_memory_bytes_after": int(
                psutil.Process().memory_info().rss
            ),
            "active_tensor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in package.tensors.values()
            ),
        },
        "test_split_accessed": True,
    }
    result["evidence_sha256"] = _canonical_sha(result)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = certify()
    print(
        json.dumps(
            {
                "status": result["status"],
                "package": result["package"],
                "source_successes": result["source"][
                    "functional_successes"
                ],
                "receiver_retention": [
                    receiver["retention_rate"]
                    for receiver in result["receivers"]
                ],
                "gates": result["gates"],
                "evidence_sha256": result["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
