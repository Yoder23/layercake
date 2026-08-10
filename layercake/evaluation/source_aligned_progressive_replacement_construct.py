"""Construct verifier for the source-aligned progressive host v9."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch

from layercake.source_aligned_progressive_replacement_core import SourceAlignedProgressiveReplacementCore
from layercake_extensions.source_aligned_progressive_replacement_core import (
    SOURCE_ALIGNED_PROGRESSIVE_ABI_SHA256,
    SOURCE_ALIGNED_PROGRESSIVE_ABI_VERSION,
    SourceAlignedProgressiveReplacementCoreHost,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-source-aligned-progressive-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise RuntimeError("source-aligned progressive construct governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"source-aligned progressive binding changed: {relative}")
    path = root / "tests/models/test_source_aligned_progressive_replacement_core.py"
    spec = importlib.util.spec_from_file_location("source_aligned_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    with tempfile.TemporaryDirectory(prefix="lc-source-aligned-v9-") as name:
        temp = Path(name)
        package, public, key_id, state_hash, tokenizer = fixture._package(temp / "source")
        archive_hash = sha(package)
        cpu = SourceAlignedProgressiveReplacementCoreHost(temp / "cpu", trust_store={key_id: public})
        active = cpu.activate(package)
        source_ids = tokenizer.encode_source("hello world")[0]
        state = cpu.prefill("hello world")
        packed = torch.tensor([source_ids], dtype=torch.long)
        first = torch.allclose(state.next_logits, cpu.module(packed)[:, -1], atol=1e-5, rtol=1e-5)
        before = tuple(key.shape[2] for key in state.layer_keys)
        state.next_logits.zero_(); state.next_logits[0, 4] = 1.0
        action, state = cpu.decode_step(state)
        packed = torch.tensor([source_ids + [action]], dtype=torch.long)
        second = torch.allclose(state.next_logits, cpu.module(packed)[:, -1], atol=1e-5, rtol=1e-5)
        after = tuple(key.shape[2] for key in state.layer_keys)
        verified = cpu.verify()
        cpu.remove(); reinstall = cpu.activate(package)
        cuda = None
        if torch.cuda.is_available():
            host = SourceAlignedProgressiveReplacementCoreHost(temp / "cuda", trust_store={key_id: public}, device="cuda")
            cuda = host.activate(package)
        target_parameters = SourceAlignedProgressiveReplacementCore.parameter_count_for_config(
            fixed_vocab_size=32_015, full_width=3_072, bottleneck_width=192,
            replacement_layers=32, intermediate_size=768,
        )
        checks = {
            "abi_identity": SOURCE_ALIGNED_PROGRESSIVE_ABI_VERSION == "lc-direct-neural-core/9" and SOURCE_ALIGNED_PROGRESSIVE_ABI_SHA256 == protocol["interface_sha256"],
            "canonical_abi_hash": sha(root / protocol["canonical_abi"]) == SOURCE_ALIGNED_PROGRESSIVE_ABI_SHA256,
            "target_parameters": target_parameters == 253_535_232,
            "no_injected_bos": state.source_ids.tolist() == [source_ids] and before == tuple([len(source_ids)] * len(before)),
            "first_prediction_identity": bool(first),
            "second_prediction_identity": bool(second),
            "persistent_cache": all(right == left + 1 for left, right in zip(before, after)),
            "cpu_state_identity": active["state_dict_hash"] == state_hash,
            "cpu_zero_learning": active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0,
            "cpu_verify": verified["status"] == "PASS",
            "lifecycle_identity": sha(package) == archive_hash and reinstall["archive_hash"] == active["archive_hash"],
            "cuda_available": torch.cuda.is_available(),
            "same_package_cuda": cuda is not None and cuda["archive_hash"] == active["archive_hash"] and cuda["state_dict_hash"] == state_hash,
            "zero_source_blocks": protocol["source_transformer_blocks"] == 0,
        }
    result = {
        "format": "layercake-postrelease-source-aligned-progressive-construct-result/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": {"path": protocol_path.name, "sha256": sha(protocol_path)},
        "checks": checks,
        "target_parameters": target_parameters,
        "package": {"archive_hash": archive_hash, "state_dict_hash": state_hash},
        "receiver_training_steps": 0, "receiver_calibration_runs": 0,
        "external_artifact_used": False, "english_quality_tested": False, "performance_tested": False,
        "claim_boundary": "Generic source-aligned host construct only; no external acquisition, quality, or performance claim.",
    }
    result["evidence_sha256"] = hashlib.sha256((json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "verify")); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(argv); root = Path.cwd().resolve(); result = execute(root, root / args.protocol); output = root / args.output
    if args.command == "execute":
        if output.exists(): raise RuntimeError("source-aligned construct result immutable")
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result: raise RuntimeError("stored source-aligned construct differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2)); return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
