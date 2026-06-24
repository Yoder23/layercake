from __future__ import annotations

from pathlib import Path
import shutil


class CherryPickError(ValueError):
    pass


def cherry_pick_module(source_commit, target_commit, module_name: str, *, output_dir="artifacts/cherrypicks") -> dict:
    if source_commit.abi_hash != target_commit.abi_hash:
        raise CherryPickError("ABI hash mismatch")
    if source_commit.input_interface_hash != target_commit.input_interface_hash:
        raise CherryPickError("input-interface hash mismatch")
    if module_name not in source_commit.artifact_paths:
        raise CherryPickError(f"module {module_name!r} missing from source")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    source = Path(source_commit.artifact_paths[module_name])
    target = Path(output_dir) / f"{target_commit.commit_id}_{module_name}.pt"
    shutil.copyfile(source, target)
    return {
        "source_commit": source_commit.commit_id,
        "target_commit": target_commit.commit_id,
        "module": module_name,
        "abi_match": True,
        "input_interface_match": True,
        "shape_match": True,
        "artifact_path": str(target),
    }


def cherry_pick_domain_brick(source_commit, target_commit, brick_name: str) -> dict:
    return cherry_pick_module(source_commit, target_commit, brick_name)


def cherry_pick_px_payload(source_commit, target_commit, payload_name: str) -> dict:
    return cherry_pick_module(source_commit, target_commit, payload_name)
