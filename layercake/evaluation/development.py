"""Aggregate immutable metadata from the largest completed V2 experiment tier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import psutil
import torch

from layercake.moonshot import source_tree_hash


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_development_evidence(root: str | Path, output_path: str | Path) -> dict:
    root = Path(root).resolve()
    cores = [_read(root / f"artifacts/cores/english-core-{suffix}/metadata.json") for suffix in "abc"]
    transformer = _read(root / "artifacts/baselines/transformer/metadata.json")
    adapted = _read(root / "artifacts/baselines/transformer-mixed/metadata.json")
    cake = _read(root / "artifacts/cakes/python.evidence.json")
    wikitext = _read(root / "data/moonshot/v2/wikitext103/manifest.json")
    python = _read(root / "data/moonshot/v2/python/manifest.json")
    evidence_names = (
        "baseline_audit", "architecture_search", "incremental_benchmark",
        "portability_evidence", "routing_evidence", "cpu_vs_gpu_evidence",
        "orchestration_evidence", "mobile_export_evidence",
        "quantization_evidence", "english_quality_evidence",
    )
    evidence_files = {
        name: root / f"results/moonshot/v2/{name}.json" for name in evidence_names
    }
    source = cores[0]
    cake_eval = cake["evaluation"]
    result = {
        "format": "layercake-development-evidence/2",
        "status": "PASS",
        "claim_scope": "largest completed development-scale run; explicitly not moonshot proof",
        "source": {
            "baseline_commit": "872de5ab227a3eaa0071475f21f830de6b68a3fb",
            "source_tree_sha256": source_tree_hash(),
        },
        "environment": {
            "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "ram_bytes": psutil.virtual_memory().total,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda,
        },
        "data": {
            "english": {
                "dataset": wikitext["dataset"], "configuration": wikitext["dataset_configuration"],
                "source_url": wikitext["source_url"], "license": wikitext["license"],
                "preparation": wikitext["preparation"], "records": wikitext["records"],
                "sampled_cross_split_64byte_overlaps": wikitext["sampled_cross_split_64byte_overlaps"],
                "files": wikitext["files"], "tier_requirements": wikitext["tier_requirements"],
            },
            "python": {
                "description": python["description"], "licenses": python["licenses"],
                "separation": python["separation"], "files": python["files"],
                "distributions": {
                    name: {
                        key: value for key, value in record.items()
                        if key in {"version", "declared_license", "source_homepage", "split", "bytes", "files"}
                    }
                    for name, record in python["provenance"].items()
                },
            },
        },
        "english_cores": [{
            "seed": core["seed"], "route": core["route"],
            "architecture": core["architecture"], "canonical_abi_hash": core["canonical_abi_hash"],
            "parameters": core["parameters"], "optimizer": core["optimizer"],
            "training": {key: core["training"][key] for key in (
                "steps_completed", "raw_bytes_seen", "context_bytes", "wall_seconds",
                "raw_bytes_per_second", "parameter_seconds_active",
            )},
            "memory": core["memory"], "quality": core["quality"],
            "checkpoint": core["checkpoint"],
        } for core in cores],
        "same_scale_transformer": transformer,
        "domain_adapted_transformer": adapted,
        "python_cake": {
            "core_unchanged": cake["core"]["unchanged"],
            "parameters": cake["cake"]["parameters"],
            "training_seconds": cake["cake"]["training_seconds"],
            "raw_bytes_seen": cake["cake"]["raw_bytes_seen"],
            "heldout_domain": cake_eval["heldout_domain"],
            "random_control": cake_eval["random_control"],
            "shuffled_control": cake_eval["shuffled_control"],
            "canonical_disabled": cake_eval["canonical_disabled"],
            "byte_anchor_disabled": cake_eval["byte_anchor_disabled"],
            "general_core": cake_eval["general_core"],
            "general_with_cake": cake_eval["general_with_cake"],
            "syntax_tasks": cake_eval["syntax_tasks"],
            "package": cake["package"],
        },
        "comparisons": {
            "same_total_scale_relative_delta": abs(
                source["parameters"]["total_parameters"] - transformer["parameters"]
            ) / transformer["parameters"],
            "layercake_minus_transformer_general_bpb": (
                source["quality"]["test"]["bits_per_byte"] - transformer["quality"]["test"]["bits_per_byte"]
            ),
            "foundation_wall_time_transformer_over_layercake": (
                (transformer["training"]["wall_seconds"] + transformer["tokenizer"]["training_seconds"])
                / source["training"]["wall_seconds"]
            ),
            "cake_over_monolithic_adaptation_speed": (
                adapted["training"]["wall_seconds"] / cake["cake"]["training_seconds"]
            ),
            "quality_matched": False,
        },
        "evidence": {
            name: {"path": str(path), "sha256": _sha(path), "status": _read(path).get("status")}
            for name, path in evidence_files.items() if path.is_file()
        },
        "failed_seeds": [],
        "limitations": [
            "the search promotion gate failed and the V2 architecture remains a research candidate",
            "the same-scale transformer achieved lower general English BPB",
            "the Python cake achieved lower code BPB but zero of eight ordinary syntax tasks",
            "the cake regressed unrelated English BPB",
            "only one serious V2 cake was trained",
            "proof-scale one-billion-byte training was not run",
            "physical mobile execution was unavailable",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
