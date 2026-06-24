from __future__ import annotations


def diff_commits(a, b) -> dict:
    changed_modules = sorted(
        name
        for name in set(a.module_hashes) | set(b.module_hashes)
        if a.module_hashes.get(name) != b.module_hashes.get(name)
    )
    size_delta = {
        name: len(str(b.artifact_paths.get(name, ""))) - len(str(a.artifact_paths.get(name, "")))
        for name in set(a.artifact_paths) | set(b.artifact_paths)
    }
    return {
        "changed_modules": changed_modules,
        "changed_abi_hash": a.abi_hash != b.abi_hash,
        "changed_input_interface_hash": a.input_interface_hash != b.input_interface_hash,
        "changed_rubric": a.rubric_hash != b.rubric_hash,
        "gate_deltas": {},
        "benchmark_deltas": {},
        "artifact_size_deltas": size_delta,
    }
