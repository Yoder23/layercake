from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import psutil
from safetensors.torch import load_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.portable_domain import state_dict_hash
from layercake.semantic_action_plan import (
    SemanticActionPlanResidual,
    build_semantic_action_plan_artifact,
    load_semantic_action_plan_artifact,
)
from layercake.training.phase4_python_cake import _canonical_sha
import scripts.repair_phase4_semantic_action_plan_lexical as evaluation
from scripts.train_phase4_semantic_action_plan import (
    ABI_SHA256,
    ABI_VERSION,
    _batch,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "moonshot"
    / "phase4_semantic_residual_blend_preregistration.json"
)
PREREGISTRATION_SHA256 = (
    "fc87932032f6478904ec209f8a29c86e4682d5f92468852c6f6f186de27e8e9a"
)
PARENT = (
    ROOT
    / "artifacts/moonshot/phase4/candidates"
    / "python-semantic-coordinate-replacement-seed10840.pt"
)
PARENT_SHA256 = (
    "7a5dd947e338fa7cdc561201b14124b9d1408343cfe928b6ae500132327e5a83"
)
PARENT_PAYLOAD_HASH = (
    "d2f9565ba239045cdbfaa49ba6a0e9eb619214f81c482640fbd485252c5e845f"
)
CACHE = (
    ROOT
    / "artifacts/moonshot/phase4/cache"
    / "seed9824-lexical-conformance-v1-train.safetensors"
)
CACHE_SHA256 = (
    "ee0488a5f407c1fbd1cb8383c8de5d27ee955042e05d9b1713b2b584051bcf79"
)
FRESH_PROBE = (
    ROOT
    / "data/moonshot/phase4"
    / "lexical_conformance_v2_fresh_probe.jsonl"
)
FRESH_PROBE_SHA256 = (
    "418a7a1ee3876c947278cf6bc3e8ec1e9cb61fbea3a11cded72c9448a928248e"
)
COORDINATE_BLEND = 0.73
CONVERSION_FORMAT = "layercake-phase4-semantic-residual-blend-conversion/1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _validate_inputs() -> None:
    observed = {
        "preregistration": _sha256(PREREGISTRATION),
        "parent": _sha256(PARENT),
        "cache": _sha256(CACHE),
        "fresh_probe": _sha256(FRESH_PROBE),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "parent": PARENT_SHA256,
        "cache": CACHE_SHA256,
        "fresh_probe": FRESH_PROBE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"locked residual-blend input mismatch: {observed} != {expected}"
        )


@torch.inference_mode()
def convert(args: argparse.Namespace) -> dict[str, Any]:
    output = _rooted(args.output)
    evidence_path = output.with_suffix(".json")
    if output.exists() or evidence_path.exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    _validate_inputs()
    device = _device(args.device)
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    parent, parent_artifact = load_semantic_action_plan_artifact(PARENT)
    if (
        parent_artifact["abi_version"] != ABI_VERSION
        or parent_artifact["abi_sha256"] != ABI_SHA256
    ):
        raise RuntimeError("parent artifact is bound to the wrong ABI")
    if parent_artifact["payload_hash"] != PARENT_PAYLOAD_HASH:
        raise RuntimeError("parent payload identity changed")
    config = parent.canonical_config()
    if (
        config["copy_coordinate_width"] <= 0
        or config["copy_coordinate_blend"] != 1.0
    ):
        raise RuntimeError("source is not an unblended coordinate artifact")
    config["copy_coordinate_blend"] = COORDINATE_BLEND
    model = SemanticActionPlanResidual(**config)
    model.load_state_dict(parent.state_dict(), strict=True)
    parent_tensor_hash = state_dict_hash(parent.state_dict())
    blended_tensor_hash = state_dict_hash(model.state_dict())
    if parent_tensor_hash != blended_tensor_hash:
        raise RuntimeError("residual-blend conversion changed a tensor")

    cached = load_file(str(CACHE), device="cpu")
    offsets = cached["row_offsets"].long().tolist()
    smoke_cpu = _batch(
        cached, offsets, [0], torch.device("cpu")
    )
    cpu_model = SemanticActionPlanResidual(**config).eval()
    cpu_model.load_state_dict(model.state_dict(), strict=True)
    cpu_result = cpu_model.training_forward(
        smoke_cpu["prompt_states"],
        smoke_cpu["current_states"],
        smoke_cpu["target_actions"],
        prompt_padding=smoke_cpu["prompt_padding"],
    )["residual"]
    if not torch.isfinite(cpu_result).all():
        raise RuntimeError("CPU fallback smoke produced non-finite residuals")

    model.to(device).eval()
    smoke_device = _batch(cached, offsets, [0], device)
    device_result = model.training_forward(
        smoke_device["prompt_states"],
        smoke_device["current_states"],
        smoke_device["target_actions"],
        prompt_padding=smoke_device["prompt_padding"],
    )["residual"].cpu()
    maximum_cpu_accelerator_difference = float(
        (cpu_result - device_result).abs().max()
    )
    mean_cpu_accelerator_difference = float(
        (cpu_result - device_result).abs().mean()
    )
    if not torch.allclose(
        cpu_result, device_result, atol=2e-3, rtol=2e-3
    ):
        raise RuntimeError(
            "CPU and accelerator residuals exceeded the declared "
            "2e-3 numerical tolerance: "
            f"maximum={maximum_cpu_accelerator_difference}"
        )
    wall = time.perf_counter() - started
    peak_rss = max(peak_rss, int(process.memory_info().rss))
    conversion = {
        "format": CONVERSION_FORMAT,
        "source_commit": _git_head(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "parent_artifact_sha256": PARENT_SHA256,
        "parent_payload_hash": PARENT_PAYLOAD_HASH,
        "coordinate_blend": COORDINATE_BLEND,
        "parent_linear_blend": 1.0 - COORDINATE_BLEND,
        "optimizer_steps": 0,
        "training_examples": 0,
        "new_runtime_parameters": 0,
        "new_matrix_operations": 0,
        "tensor_hash_before": parent_tensor_hash,
        "tensor_hash_after": blended_tensor_hash,
        "primary_device": str(device),
        "primary_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "declared laptop CPU"
        ),
        "gpu_wall_seconds": wall if device.type == "cuda" else 0.0,
        "cpu_wall_seconds": wall if device.type == "cpu" else 0.0,
        "peak_accelerator_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else 0
        ),
        "peak_process_resident_memory_bytes": peak_rss,
        "cpu_fallback_smoke": "PASS",
        "cpu_accelerator_numerical_equivalence": "PASS",
        "cpu_accelerator_absolute_tolerance": 0.002,
        "cpu_accelerator_relative_tolerance": 0.002,
        "maximum_cpu_accelerator_residual_difference": (
            maximum_cpu_accelerator_difference
        ),
        "mean_cpu_accelerator_residual_difference": (
            mean_cpu_accelerator_difference
        ),
    }
    artifact = build_semantic_action_plan_artifact(
        model.cpu(),
        abi_version=ABI_VERSION,
        abi_sha256=ABI_SHA256,
        training=conversion,
    )
    if artifact["payload_hash"] != PARENT_PAYLOAD_HASH:
        raise RuntimeError("converted artifact payload hash changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    evidence = {
        "format": CONVERSION_FORMAT,
        "status": "CONVERTED_WITHOUT_TENSOR_MUTATION",
        "protocol": PREREGISTRATION.relative_to(ROOT).as_posix(),
        "protocol_sha256": PREREGISTRATION_SHA256,
        "source_commit": conversion["source_commit"],
        "parent_artifact": PARENT.relative_to(ROOT).as_posix(),
        "parent_artifact_sha256": PARENT_SHA256,
        "artifact": output.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(output),
        "parent_payload_hash": PARENT_PAYLOAD_HASH,
        "payload_hash": artifact["payload_hash"],
        "spec_sha256": artifact["spec_sha256"],
        "abi_version": ABI_VERSION,
        "abi_sha256": ABI_SHA256,
        "architecture": model.canonical_config(),
        "coordinate_blend": COORDINATE_BLEND,
        "parent_linear_blend": 1.0 - COORDINATE_BLEND,
        "optimizer_steps": 0,
        "training_examples": 0,
        "new_runtime_parameters": 0,
        "new_matrix_operations": 0,
        "tensor_hash_before": parent_tensor_hash,
        "tensor_hash_after": blended_tensor_hash,
        "all_tensors_identical": parent_tensor_hash == blended_tensor_hash,
        "primary_device": conversion["primary_device"],
        "primary_device_name": conversion["primary_device_name"],
        "gpu_wall_seconds": conversion["gpu_wall_seconds"],
        "cpu_wall_seconds": conversion["cpu_wall_seconds"],
        "peak_accelerator_memory_bytes": conversion[
            "peak_accelerator_memory_bytes"
        ],
        "peak_process_resident_memory_bytes": peak_rss,
        "cpu_fallback_smoke": "PASS",
        "cpu_accelerator_numerical_equivalence": "PASS",
        "cpu_accelerator_absolute_tolerance": 0.002,
        "cpu_accelerator_relative_tolerance": 0.002,
        "maximum_cpu_accelerator_residual_difference": (
            maximum_cpu_accelerator_difference
        ),
        "mean_cpu_accelerator_residual_difference": (
            mean_cpu_accelerator_difference
        ),
        "fresh_probe_accessed": False,
        "python_validation_accessed": False,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _configure_evaluation() -> None:
    evaluation.PREREGISTRATION = PREREGISTRATION
    evaluation.LEXICAL_DATASET = FRESH_PROBE
    evaluation.LEXICAL_EVALUATION_SPLIT = "fresh_probe"
    evaluation.LEXICAL_EXPECTED_ROWS = 256
    evaluation.LEXICAL_MINIMUM_EXACT_RESPONSES = 231
    evaluation.LEXICAL_EVALUATION_FORMAT = (
        "layercake-phase4-semantic-residual-blend-fresh-lexical-evaluation/1"
    )
    evaluation.PYTHON_EVALUATION_FORMAT = (
        "layercake-phase4-semantic-residual-blend-python-evaluation/1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--device", default="cuda:0")
    lexical_parser = subparsers.add_parser("evaluate-fresh")
    lexical_parser.add_argument("--artifact", type=Path, required=True)
    lexical_parser.add_argument("--output", type=Path, required=True)
    lexical_parser.add_argument("--device", default="cuda:0")
    python_parser = subparsers.add_parser("evaluate-python")
    python_parser.add_argument("--artifact", type=Path, required=True)
    python_parser.add_argument("--lexical-evidence", type=Path, required=True)
    python_parser.add_argument("--output", type=Path, required=True)
    python_parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "convert":
        result = convert(args)
    else:
        _configure_evaluation()
        result = (
            evaluation.evaluate_lexical(args)
            if args.command == "evaluate-fresh"
            else evaluation.evaluate_python(args)
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
