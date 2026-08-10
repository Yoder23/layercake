"""Construct verifier for the rank-192 basis-aligned progressive host v11."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch

from layercake.basis_aligned_progressive_core import BasisAlignedProgressiveCore
from layercake_extensions.basis_aligned_progressive_core import (
    BASIS_ALIGNED_PROGRESSIVE_ABI_SHA256,
    BASIS_ALIGNED_PROGRESSIVE_ABI_VERSION,
    BasisAlignedProgressiveCoreHost,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != "layercake-postrelease-basis-aligned-progressive-construct/1"
        or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY"
    ):
        raise RuntimeError("basis-aligned construct governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"basis-aligned binding changed: {relative}")
    path = root / "tests/models/test_basis_aligned_progressive_core.py"
    spec = importlib.util.spec_from_file_location("basis_aligned_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    with tempfile.TemporaryDirectory(prefix="lc-basis-aligned-v11-") as name:
        temp = Path(name)
        package, public, key_id, state_hash, tokenizer = fixture._package(temp / "source")
        archive_hash = sha(package)
        cpu = BasisAlignedProgressiveCoreHost(temp / "cpu", trust_store={key_id: public})
        active = cpu.activate(package)
        source_ids = tokenizer.encode_source("hello world")[0]
        state = cpu.prefill("hello world")
        first = torch.allclose(
            state.next_logits,
            cpu.module(torch.tensor([source_ids]))[:, -1],
            atol=1e-5,
            rtol=1e-5,
        )
        before = tuple(key.shape[2] for key in state.layer_keys)
        state.next_logits.zero_()
        state.next_logits[0, 4] = 1.0
        action, state = cpu.decode_step(state)
        second = torch.allclose(
            state.next_logits,
            cpu.module(torch.tensor([source_ids + [action]]))[:, -1],
            atol=1e-5,
            rtol=1e-5,
        )
        after = tuple(key.shape[2] for key in state.layer_keys)
        verified = cpu.verify()
        cpu.remove()
        reinstall = cpu.activate(package)
        cuda = None
        if torch.cuda.is_available():
            host = BasisAlignedProgressiveCoreHost(
                temp / "cuda", trust_store={key_id: public}, device="cuda"
            )
            cuda = host.activate(package)
        target_parameters = BasisAlignedProgressiveCore.parameter_count_for_config(
            fixed_vocab_size=32_015,
            full_width=3_072,
            bottleneck_width=192,
            replacement_layers=32,
            intermediate_size=768,
        )
        checks = {
            "abi_identity": BASIS_ALIGNED_PROGRESSIVE_ABI_VERSION == "lc-direct-neural-core/11"
            and BASIS_ALIGNED_PROGRESSIVE_ABI_SHA256 == protocol["interface_sha256"],
            "canonical_abi_hash": sha(root / protocol["canonical_abi"])
            == BASIS_ALIGNED_PROGRESSIVE_ABI_SHA256,
            "target_parameters": target_parameters == 291_382_272,
            "rank_192_basis_and_mean": cpu.module.bottleneck_width == 192
            and all(hasattr(layer, "mlp_residual_mean") for layer in cpu.module.layers),
            "no_injected_bos": state.source_ids.tolist() == [source_ids]
            and before == tuple([len(source_ids)] * len(before)),
            "first_prediction_identity": bool(first),
            "second_prediction_identity": bool(second),
            "persistent_cache": all(right == left + 1 for left, right in zip(before, after)),
            "cpu_state_identity": active["state_dict_hash"] == state_hash,
            "cpu_zero_learning": active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0,
            "cpu_verify": verified["status"] == "PASS",
            "lifecycle_identity": sha(package) == archive_hash
            and reinstall["archive_hash"] == active["archive_hash"],
            "cuda_available": torch.cuda.is_available(),
            "same_package_cuda": cuda is not None
            and cuda["archive_hash"] == active["archive_hash"]
            and cuda["state_dict_hash"] == state_hash,
            "zero_source_blocks": protocol["source_transformer_blocks"] == 0,
        }
    result = {
        "format": "layercake-postrelease-basis-aligned-progressive-construct-result/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": {"path": protocol_path.name, "sha256": sha(protocol_path)},
        "checks": checks,
        "target_parameters": target_parameters,
        "package": {"archive_hash": archive_hash, "state_dict_hash": state_hash},
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "external_artifact_used": False,
        "english_quality_tested": False,
        "performance_tested": False,
        "claim_boundary": "Generic rank-192 basis-aligned host construct only; no external acquisition, quality, or performance claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(
        (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol)
    output = root / args.output
    if args.command == "execute":
        if output.exists():
            raise RuntimeError("basis-aligned construct result immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result:
        raise RuntimeError("stored basis-aligned construct differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
