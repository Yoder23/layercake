"""Functional portability evaluation on real independently trained hosts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake.cake.package import load_package
from layercake.cake.registry import CakeRegistry
from layercake.models.portable_decoder import load_cake_module
from layercake.training.data import ByteCorpus, sha256_file
from layercake.training.foundation import load_core_checkpoint


@torch.inference_mode()
def _host_rows(core, cake, corpus, *, route: int, device: torch.device) -> dict:
    rows = []
    canonical_blocks = []
    for index, batch in enumerate(corpus.fixed_batches(
        batch_size=4, sequence_bytes=256, batches=8, device=device
    )):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        _, aux = core(inputs, route=route, return_aux=True)
        logits, _ = cake(aux["core_logits"], aux["canonical"], inputs)
        core_losses = F.cross_entropy(
            aux["core_logits"].transpose(1, 2), targets, reduction="none"
        ).mean(dim=1) / 0.6931471805599453
        cake_losses = F.cross_entropy(
            logits.transpose(1, 2), targets, reduction="none"
        ).mean(dim=1) / 0.6931471805599453
        canonical_blocks.append(aux["canonical"].float().cpu())
        for row_index, (core_bpb, cake_bpb) in enumerate(zip(core_losses, cake_losses)):
            rows.append({
                "id": f"{index}:{row_index}",
                "core_bpb": float(core_bpb),
                "cake_bpb": float(cake_bpb),
                "improvement_bpb": float(core_bpb - cake_bpb),
                "locked_success": bool(core_bpb - cake_bpb >= 0.10),
            })
    canonical = torch.cat(canonical_blocks, dim=0)
    return {
        "rows": rows,
        "core_bpb": sum(row["core_bpb"] for row in rows) / len(rows),
        "cake_bpb": sum(row["cake_bpb"] for row in rows) / len(rows),
        "locked_success_ids": [row["id"] for row in rows if row["locked_success"]],
        "canonical": canonical,
        "canonical_statistics": {
            "mean": float(canonical.mean()),
            "standard_deviation": float(canonical.std()),
            "mean_rms": float(canonical.square().mean(dim=-1).sqrt().mean()),
            "finite": bool(torch.isfinite(canonical).all()),
        },
    }


@torch.inference_mode()
def _generation(core, cake, *, route: int) -> dict:
    prompt = b"def bounded_map(function, values):\n    "
    core_state = core.prefill(prompt, route=route, capture_generated=True)
    _, core_state = core.decode_many(core_state, 128)
    cake_state = core.prefill(prompt, route=route, fusion_cake=cake, capture_generated=True)
    _, cake_state = core.decode_many(cake_state, 128, fusion_cake=cake)
    return {
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "core_output_hex": bytes(core_state.generated_bytes[0].cpu().tolist()).hex(),
        "cake_output_hex": bytes(cake_state.generated_bytes[0].cpu().tolist()).hex(),
    }


def verify_true_cross_host_portability(
    *,
    cake_path: str | Path,
    public_key_path: str | Path,
    host_dirs: list[str | Path],
    domain_test_path: str | Path,
    output_path: str | Path,
) -> dict:
    if len(host_dirs) < 3:
        raise ValueError("true portability requires at least three real host checkpoints")
    package_bytes_before = Path(cake_path).read_bytes()
    package = load_package(
        cake_path,
        trust_store={load_package_key_id(cake_path): public_key_path},
    )
    key_id = package.manifest.signature["key_id"]
    trust = {key_id: Path(public_key_path)}
    domain = ByteCorpus(domain_test_path)
    hosts = []
    source_successes: set[str] | None = None
    source_canonical = None
    for host_index, host_dir in enumerate(host_dirs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        core, metadata = load_core_checkpoint(host_dir, device=device)
        capabilities = HostCapabilities(
            abi_version=metadata["canonical_abi"]["version"],
            abi_hash=metadata["canonical_abi_hash"],
            precisions=("fp32", "fp16", "int8"),
            backends=("pytorch", "cuda"),
            capabilities=frozenset({"byte_input", "safe_tensors", "incremental"}),
        )
        with tempfile.TemporaryDirectory(prefix=f"layercake-host-{host_index}-") as registry_path:
            registry = CakeRegistry(registry_path)
            installer = CakeInstaller(registry, capabilities, trust_store=trust)
            install_result = installer.install(cake_path)
            verify_before = installer.verify(package.manifest.cake_id)
            installed_package = load_package(Path(install_result["blob"]), trust_store=trust)
            cake = load_cake_module(installed_package).to(device)
            scores_before = _host_rows(core, cake, domain, route=int(metadata["route"]), device=device)
            generation_before = _generation(core, cake, route=int(metadata["route"]))
            remove_result = installer.remove(package.manifest.cake_id)
            reinstall_result = installer.install(cake_path)
            verify_after = installer.verify(package.manifest.cake_id)
            reinstalled_package = load_package(Path(reinstall_result["blob"]), trust_store=trust)
            reinstalled_cake = load_cake_module(reinstalled_package).to(device)
            scores_after = _host_rows(
                core, reinstalled_cake, domain, route=int(metadata["route"]), device=device
            )
            generation_after = _generation(core, reinstalled_cake, route=int(metadata["route"]))
        successes = set(scores_before.pop("locked_success_ids"))
        after_successes = set(scores_after.pop("locked_success_ids"))
        canonical = scores_before.pop("canonical")
        after_canonical = scores_after.pop("canonical")
        if source_successes is None:
            source_successes = successes
            source_canonical = canonical
        retained = sorted(source_successes & successes)
        hosts.append({
            "host": {
                "path": str(Path(host_dir).resolve()),
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "architecture": metadata["architecture"],
                "seed": metadata["seed"],
                "test_quality": metadata["quality"]["test"],
            },
            "install": install_result,
            "verify_before": verify_before,
            "scores_before": scores_before,
            "generation_before": generation_before,
            "remove": remove_result,
            "reinstall": reinstall_result,
            "verify_after": verify_after,
            "scores_after": scores_after,
            "generation_after": generation_after,
            "locked_source_successes_retained": len(retained),
            "locked_source_successes_total": len(source_successes),
            "retention_rate": len(retained) / max(len(source_successes), 1),
            "own_successes_stable_after_reinstall": successes == after_successes,
            "canonical_anchor_max_difference_from_source": float(
                (source_canonical[..., source_canonical.shape[-1] // 2:] - canonical[..., canonical.shape[-1] // 2:]).abs().max()
            ),
            "canonical_full_mean_absolute_difference_from_source": float(
                (source_canonical - canonical).abs().mean()
            ),
            "canonical_reinstall_max_difference": float((canonical - after_canonical).abs().max()),
        })
    package_bytes_after = Path(cake_path).read_bytes()
    required_successes = len(source_successes or ()) > 0
    passed = (
        package_bytes_before == package_bytes_after
        and required_successes
        and all(host["retention_rate"] == 1.0 for host in hosts)
        and all(host["scores_before"]["cake_bpb"] < host["scores_before"]["core_bpb"] for host in hosts)
        and all(host["own_successes_stable_after_reinstall"] for host in hosts)
    )
    evidence = {
        "format": "layercake-true-portability/2",
        "status": "PASS" if passed else "FAIL",
        "functional_criterion_frozen_before_receivers": "retain every source-host sequence with >=0.10 BPB improvement",
        "source_locked_success_count": len(source_successes or ()),
        "receiver_domain_training_examples": 0,
        "receiver_specific_calibration": False,
        "package": {
            "path": str(Path(cake_path).resolve()),
            "archive_sha256_before": hashlib.sha256(package_bytes_before).hexdigest(),
            "archive_sha256_after": hashlib.sha256(package_bytes_after).hexdigest(),
            "payload_identity": package_bytes_before == package_bytes_after,
            "tensor_payload_hash": package.manifest.tensor_payload_hash,
            "package_hash": package.manifest.package_hash,
        },
        "domain_test_sha256": sha256_file(domain_test_path),
        "hosts": hosts,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def load_package_key_id(path: str | Path) -> str:
    import zipfile
    from layercake.cake.manifest import CakeManifest
    with zipfile.ZipFile(path, "r") as archive:
        manifest = CakeManifest.from_json(archive.read("manifest.json"))
    return str(manifest.signature["key_id"])
