from __future__ import annotations

from pathlib import Path
import shutil
import torch


class CherryPickError(ValueError):
    pass


def cherry_pick_module(source_commit, target_commit, module_name: str, *, output_dir="artifacts/cherrypicks") -> dict:
    if source_commit.abi_hash != target_commit.abi_hash:
        raise CherryPickError("ABI hash mismatch")
    if source_commit.input_interface_hash != target_commit.input_interface_hash:
        raise CherryPickError("input-interface hash mismatch")
    if module_name not in source_commit.artifact_paths:
        raise CherryPickError(f"module {module_name!r} missing from source")
    if module_name in target_commit.artifact_paths:
        shape_report = _compare_tensors(
            source_commit.artifact_paths[module_name],
            target_commit.artifact_paths[module_name],
        )
        if shape_report["checked"] and not shape_report["keys_match"]:
            raise CherryPickError("tensor key mismatch")
        if shape_report["checked"] and not shape_report["shapes_match"]:
            raise CherryPickError("tensor shape mismatch")
        if shape_report["checked"] and not shape_report["dtypes_match"]:
            raise CherryPickError("tensor dtype mismatch")
    else:
        shape_report = {
            "keys_match": None,
            "shapes_match": None,
            "dtypes_match": None,
            "checked": False,
        }
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
        "shape_match": shape_report["shapes_match"],
        "tensor_validation": shape_report,
        "artifact_path": str(target),
    }


def _compare_tensors(source_path: str, target_path: str) -> dict:
    try:
        source = torch.load(source_path, map_location="cpu")
        target = torch.load(target_path, map_location="cpu")
    except Exception as exc:
        return {
            "checked": False,
            "keys_match": None,
            "missing_in_target": [],
            "extra_in_target": [],
            "shapes_match": None,
            "shape_mismatches": [],
            "dtypes_match": None,
            "dtype_mismatches": [],
            "reason": f"not_torch_state_dict:{type(exc).__name__}",
        }
    if not isinstance(source, dict) or not isinstance(target, dict):
        return {
            "checked": False,
            "keys_match": None,
            "missing_in_target": [],
            "extra_in_target": [],
            "shapes_match": None,
            "shape_mismatches": [],
            "dtypes_match": None,
            "dtype_mismatches": [],
            "reason": "not_state_dict",
        }
    source_keys = set(source)
    target_keys = set(target)
    shared = sorted(source_keys & target_keys)
    shape_mismatches = [
        key for key in shared
        if tuple(source[key].shape) != tuple(target[key].shape)
    ]
    dtype_mismatches = [
        key for key in shared
        if source[key].dtype != target[key].dtype
    ]
    return {
        "checked": True,
        "keys_match": source_keys == target_keys,
        "missing_in_target": sorted(source_keys - target_keys),
        "extra_in_target": sorted(target_keys - source_keys),
        "shapes_match": not shape_mismatches,
        "shape_mismatches": shape_mismatches,
        "dtypes_match": not dtype_mismatches,
        "dtype_mismatches": dtype_mismatches,
    }


def cherry_pick_domain_brick(source_commit, target_commit, brick_name: str) -> dict:
    return cherry_pick_module(source_commit, target_commit, brick_name)


def cherry_pick_px_payload(source_commit, target_commit, payload_name: str) -> dict:
    return cherry_pick_module(source_commit, target_commit, payload_name)
