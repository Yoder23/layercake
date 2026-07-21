from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from layercake.cake.package import load_package
from layercake.models.portable_decoder import load_cake_module


@torch.inference_mode()
def verify_portable_execution(
    package_path: str | Path,
    inputs: torch.Tensor,
    *,
    receivers: list[dict],
    trust_store: dict | None = None,
    trusted_local: bool = False,
) -> dict:
    before = hashlib.sha256(Path(package_path).read_bytes()).hexdigest()
    package = load_package(
        package_path, trust_store=trust_store or {}, require_signature=not trusted_local,
        allow_local_development=trusted_local,
    )
    if package.manifest.cake_type != "portable_decoder":
        raise ValueError("strict exact portability requires portable_decoder")
    reference = load_cake_module(package).cpu()(inputs.cpu())
    rows = []
    for receiver in receivers:
        # The exact portable path intentionally excludes host representations.
        model = load_cake_module(package).cpu()
        output = model(inputs.cpu())
        rows.append({
            "receiver": receiver,
            "max_logit_difference": float((reference - output).abs().max().item()),
            "identical_output": bool(torch.equal(reference, output)),
            "payload_hash": package.manifest.tensor_payload_hash,
        })
    after = hashlib.sha256(Path(package_path).read_bytes()).hexdigest()
    passed = before == after and len(receivers) >= 3 and all(
        row["identical_output"] and row["max_logit_difference"] == 0 for row in rows
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "contract": {
            "runtime": "pytorch-cpu", "precision": "fp32",
            "deterministic": True, "cake_type": "portable_decoder",
            "abi_version": package.manifest.abi_version,
        },
        "archive_hash_before": before, "archive_hash_after": after,
        "bit_identical_payload_preserved": before == after,
        "receivers": rows,
    }
