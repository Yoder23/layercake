from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import psutil
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.portable_domain import state_dict_hash
from layercake.semantic_action_plan import (
    SemanticActionPlanResidual,
    build_semantic_action_plan_artifact,
    load_semantic_action_plan_artifact,
)
from layercake.training.phase2_shallow_sparse import load_student
from layercake.training.phase4_python_cake import _canonical_sha
import scripts.repair_phase4_semantic_action_plan_lexical as evaluation
from scripts.train_phase4_semantic_action_plan import (
    ABI_SHA256,
    ABI_VERSION,
    CHECKPOINT,
    FIXED_TOKEN_IDS,
    _batch,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "moonshot"
    / "phase4_semantic_coordinate_replacement_preregistration.json"
)
PARENT = (
    ROOT
    / "artifacts"
    / "moonshot"
    / "phase4"
    / "candidates"
    / "python-semantic-action-plan-lexical-repair-seed10440.pt"
)
CACHE = (
    ROOT
    / "artifacts"
    / "moonshot"
    / "phase4"
    / "cache"
    / "seed9824-lexical-conformance-v1-train.safetensors"
)
PARENT_SHA256 = (
    "a8211523110f8e69c851a65fefe63a5bb95ab487fda56980301fe28056215f64"
)
CACHE_SHA256 = (
    "ee0488a5f407c1fbd1cb8383c8de5d27ee955042e05d9b1713b2b584051bcf79"
)
COPY_COORDINATE_WIDTH = 512
COPY_COORDINATE_SCALE = 16.0
TRAINING_FORMAT = (
    "layercake-phase4-semantic-coordinate-replacement-training/1"
)
TRAINABLE_TENSORS = frozenset(
    {
        "copy_coordinate_norm.weight",
        "copy_coordinate_norm.bias",
        "copy_coordinate_input.weight",
        "copy_coordinate_input.bias",
        "copy_coordinate_output.weight",
    }
)


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


def _new_model() -> tuple[SemanticActionPlanResidual, dict[str, Any]]:
    parent, artifact = load_semantic_action_plan_artifact(PARENT)
    if (
        artifact["abi_version"] != ABI_VERSION
        or artifact["abi_sha256"] != ABI_SHA256
    ):
        raise RuntimeError("parent artifact is bound to the wrong ABI")
    config = parent.canonical_config()
    if config["copy_coordinate_width"] != 0:
        raise RuntimeError("parent already contains a coordinate codec")
    config["copy_coordinate_width"] = COPY_COORDINATE_WIDTH
    config["copy_coordinate_scale"] = COPY_COORDINATE_SCALE
    model = SemanticActionPlanResidual(**config)
    missing, unexpected = model.load_state_dict(
        parent.state_dict(), strict=False
    )
    if unexpected or set(missing) != TRAINABLE_TENSORS:
        raise RuntimeError(
            "coordinate codec initialization boundary mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    return model, artifact


def _subset_hash(
    model: SemanticActionPlanResidual, *, trainable: bool
) -> str:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if (name in TRAINABLE_TENSORS) is trainable
    }
    return state_dict_hash(state)


def _configure_trainable(
    model: SemanticActionPlanResidual,
) -> list[torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in TRAINABLE_TENSORS)
    observed = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if observed != TRAINABLE_TENSORS:
        raise RuntimeError(
            f"trainable tensor mismatch: {sorted(observed)}"
        )
    return [
        parameter
        for _, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = _rooted(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    if _sha256(PARENT) != PARENT_SHA256:
        raise RuntimeError("parent artifact identity changed")
    if _sha256(CACHE) != CACHE_SHA256:
        raise RuntimeError("training cache identity changed")
    device = _device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, parent_artifact = _new_model()
    frozen_hash_before = _subset_hash(model, trainable=False)
    trainable_parameters = _configure_trainable(model)
    trainable_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    if trainable_count != 788480:
        raise RuntimeError(
            f"coordinate codec parameter count changed: {trainable_count}"
        )
    model.to(device).train()
    core, _, core_metadata = load_student(CHECKPOINT)
    output_weight = core.output_weight.detach().float().to(device)
    normalized_weight = F.normalize(output_weight, dim=-1)
    del core
    cached = load_file(str(CACHE), device="cpu")
    offsets = cached["row_offsets"].long().tolist()
    row_count = len(offsets) - 1

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    started = time.perf_counter()
    curves = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(1, args.steps + 1):
        rows = [rng.randrange(row_count) for _ in range(args.batch_size)]
        batch = _batch(cached, offsets, rows, device)
        optimizer.zero_grad(set_to_none=True)
        result = model.training_forward(
            batch["prompt_states"],
            batch["current_states"],
            batch["target_actions"],
            prompt_padding=batch["prompt_padding"],
        )
        pointer_valid = batch["target_valid"] & batch[
            "target_actions"
        ].ge(model.fixed_action_count)
        if not pointer_valid.any():
            raise RuntimeError("lexical batch has no pointer units")
        coordinates = result["identity_coordinate"][pointer_valid]
        coordinate_targets = normalized_weight[
            batch["target_tokens"][pointer_valid]
        ]
        coordinate_loss = (
            1.0
            - F.cosine_similarity(
                coordinates, coordinate_targets, dim=-1
            ).mean()
        )
        logits = F.linear(result["adapted"][pointer_valid], output_weight)
        realization_loss = F.cross_entropy(
            logits, batch["target_tokens"][pointer_valid]
        )
        objective = coordinate_loss + realization_loss
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        objective_value = float(objective.detach())
        if objective_value < best_loss:
            best_loss = objective_value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if step == 1 or step % 50 == 0:
            record = {
                "step": step,
                "coordinate_cosine_loss": float(
                    coordinate_loss.detach()
                ),
                "bounded_replacement_cross_entropy": float(
                    realization_loss.detach()
                ),
                "selection_objective": objective_value,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(record)
            print(json.dumps(record), flush=True)
    if best_state is None:
        raise RuntimeError("training produced no coordinate codec")
    wall = time.perf_counter() - started
    model.load_state_dict(best_state, strict=True)
    model.eval()
    frozen_hash_after = _subset_hash(model, trainable=False)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("a frozen parent tensor changed")

    cpu_model = SemanticActionPlanResidual(**model.canonical_config())
    cpu_model.load_state_dict(model.state_dict(), strict=True)
    cpu_model.eval()
    smoke = _batch(cached, offsets, [0], torch.device("cpu"))
    with torch.inference_mode():
        smoke_result = cpu_model.training_forward(
            smoke["prompt_states"],
            smoke["current_states"],
            smoke["target_actions"],
            prompt_padding=smoke["prompt_padding"],
        )
    if not torch.isfinite(smoke_result["residual"]).all():
        raise RuntimeError("CPU fallback smoke produced non-finite residuals")

    peak_accelerator = (
        int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else 0
    )
    training = {
        "format": TRAINING_FORMAT,
        "source_commit": _git_head(),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "parent_artifact_sha256": PARENT_SHA256,
        "parent_payload_hash": parent_artifact["payload_hash"],
        "core_checkpoint_sha256": core_metadata["checkpoint"]["sha256"],
        "cache_sha256": CACHE_SHA256,
        "seed": args.seed,
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "primary_device": str(device),
        "primary_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "declared laptop CPU"
        ),
        "precision": "fp32",
        "gpu_wall_seconds": wall if device.type == "cuda" else 0.0,
        "cpu_wall_seconds": wall if device.type == "cpu" else 0.0,
        "peak_accelerator_memory_bytes": peak_accelerator,
        "peak_process_resident_memory_bytes": peak_rss,
        "trainable_tensor_names": sorted(TRAINABLE_TENSORS),
        "trainable_parameters": trainable_count,
        "total_parameters": model.parameter_count(),
        "active_parameter_seconds": trainable_count * wall,
        "best_selection_objective": best_loss,
        "frozen_parent_tensor_hash_before": frozen_hash_before,
        "frozen_parent_tensor_hash_after": frozen_hash_after,
        "learning_curves": curves,
    }
    artifact = build_semantic_action_plan_artifact(
        model.cpu(),
        abi_version=ABI_VERSION,
        abi_sha256=ABI_SHA256,
        training=training,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    evidence = {
        "format": TRAINING_FORMAT,
        "status": "TRAINED",
        "protocol": PREREGISTRATION.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(PREREGISTRATION),
        "source_commit": training["source_commit"],
        "seed": args.seed,
        "parent_artifact": PARENT.relative_to(ROOT).as_posix(),
        "parent_artifact_sha256": PARENT_SHA256,
        "parent_payload_hash": parent_artifact["payload_hash"],
        "artifact": output.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(output),
        "payload_hash": artifact["payload_hash"],
        "spec_sha256": artifact["spec_sha256"],
        "abi_version": ABI_VERSION,
        "abi_sha256": ABI_SHA256,
        "architecture": model.canonical_config(),
        "new_runtime_parameters": trainable_count,
        "total_parameters": model.parameter_count(),
        "trainable_parameters": trainable_count,
        "trainable_tensor_names": sorted(TRAINABLE_TENSORS),
        "core_parameters_changed": 0,
        "frozen_parent_tensor_hash_before": frozen_hash_before,
        "frozen_parent_tensor_hash_after": frozen_hash_after,
        "frozen_parent_tensors_identical": (
            frozen_hash_before == frozen_hash_after
        ),
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "primary_device": str(device),
        "primary_device_name": training["primary_device_name"],
        "precision": "fp32",
        "gpu_wall_seconds": training["gpu_wall_seconds"],
        "cpu_wall_seconds": training["cpu_wall_seconds"],
        "peak_accelerator_memory_bytes": peak_accelerator,
        "peak_process_resident_memory_bytes": peak_rss,
        "cpu_fallback_smoke": "PASS",
        "active_parameter_seconds": training["active_parameter_seconds"],
        "best_selection_objective": best_loss,
        "learning_curves": curves,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output.with_suffix(".json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _configure_evaluation() -> None:
    evaluation.PREREGISTRATION = PREREGISTRATION
    evaluation.LEXICAL_EVALUATION_FORMAT = (
        "layercake-phase4-semantic-coordinate-replacement-lexical-evaluation/1"
    )
    evaluation.PYTHON_EVALUATION_FORMAT = (
        "layercake-phase4-semantic-coordinate-replacement-python-evaluation/1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, default=10840)
    train_parser.add_argument("--steps", type=int, default=2000)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--device", default="cuda:0")
    lexical_parser = subparsers.add_parser("evaluate-lexical")
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
    if args.command == "train":
        result = train(args)
    else:
        _configure_evaluation()
        result = (
            evaluation.evaluate_lexical(args)
            if args.command == "evaluate-lexical"
            else evaluation.evaluate_python(args)
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
