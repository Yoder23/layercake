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
    _batch,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "moonshot"
    / "phase4_semantic_parallel_plan_preregistration.json"
)
PREREGISTRATION_SHA256 = (
    "79e90f7577e89f4eb7b2bf94d1890dcebcbe25437a3d7d3048be523cd0f7467b"
)
PARENT = (
    ROOT
    / "artifacts/moonshot/phase4/candidates"
    / "python-semantic-residual-blend-seed10840.pt"
)
PARENT_SHA256 = (
    "918c578f71e7b336f4d4a7da9a43cc74ab3db8e8266d459178b0d92a38520b72"
)
FUNCTIONAL_CACHE = (
    ROOT
    / "artifacts/moonshot/phase4/cache"
    / "seed9824-attentive-v4-exact-span-train.safetensors"
)
FUNCTIONAL_CACHE_SHA256 = (
    "4177a690c3f0b5d13ed3391eefa61add6da9e3a52ca4b038bd8953a073ba53cc"
)
LEXICAL_CACHE = (
    ROOT
    / "artifacts/moonshot/phase4/cache"
    / "seed9824-lexical-conformance-v1-train.safetensors"
)
LEXICAL_CACHE_SHA256 = (
    "ee0488a5f407c1fbd1cb8383c8de5d27ee955042e05d9b1713b2b584051bcf79"
)
LEXICAL_PROBE = (
    ROOT
    / "data/moonshot/phase4"
    / "lexical_conformance_v3_parallel_probe.jsonl"
)
LEXICAL_PROBE_SHA256 = (
    "3bbc8f4da5e4b540037074922ddb09a76b16f0cc427a46ecf1c317a9cebc194e"
)
FUNCTIONAL_PROBE = (
    ROOT
    / "data/moonshot/phase4/python_parallel_plan_probe_v1.jsonl"
)
FUNCTIONAL_PROBE_SHA256 = (
    "c46c894b18ba3460d6e46eb2271270d68db7d8bb84fcbbe740d5ceb1c2085131"
)
FROZEN_PREFIXES = (
    "input_norm.",
    "fixed_action_embedding.",
    "pointer_input.",
    "fixed_semantic_value.",
    "copy_semantic_value.",
    "copy_coordinate_",
    "copy_transition_",
    "plan_correction.",
)
TRAINING_FORMAT = "layercake-phase4-semantic-parallel-plan-training/1"


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


def _is_frozen(name: str) -> bool:
    return name.startswith(FROZEN_PREFIXES)


def _subset_hash(
    model: SemanticActionPlanResidual, *, frozen: bool
) -> str:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if _is_frozen(name) is frozen
    }
    if not state:
        raise RuntimeError("parallel-plan tensor subset is empty")
    return state_dict_hash(state)


def _validate_inputs() -> None:
    observed = {
        "preregistration": _sha256(PREREGISTRATION),
        "parent": _sha256(PARENT),
        "functional_cache": _sha256(FUNCTIONAL_CACHE),
        "lexical_cache": _sha256(LEXICAL_CACHE),
        "lexical_probe": _sha256(LEXICAL_PROBE),
        "functional_probe": _sha256(FUNCTIONAL_PROBE),
    }
    expected = {
        "preregistration": PREREGISTRATION_SHA256,
        "parent": PARENT_SHA256,
        "functional_cache": FUNCTIONAL_CACHE_SHA256,
        "lexical_cache": LEXICAL_CACHE_SHA256,
        "lexical_probe": LEXICAL_PROBE_SHA256,
        "functional_probe": FUNCTIONAL_PROBE_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"locked parallel-plan input mismatch: {observed} != {expected}"
        )


def _new_model() -> tuple[SemanticActionPlanResidual, dict[str, Any]]:
    parent, artifact = load_semantic_action_plan_artifact(PARENT)
    if (
        artifact["abi_version"] != ABI_VERSION
        or artifact["abi_sha256"] != ABI_SHA256
    ):
        raise RuntimeError("parent artifact is bound to the wrong ABI")
    config = parent.canonical_config()
    if config["parallel_plan"]:
        raise RuntimeError("parent already uses a parallel plan")
    if config["copy_coordinate_blend"] != 0.73:
        raise RuntimeError("parent lost the locked residual blend")
    config["parallel_plan"] = True
    model = SemanticActionPlanResidual(**config)
    model.load_state_dict(parent.state_dict(), strict=True)
    return model, artifact


def _configure_trainable(
    model: SemanticActionPlanResidual,
) -> list[torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not _is_frozen(name))
    trainable = [
        parameter
        for _, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("parallel plan has no trainable parameters")
    return trainable


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = _rooted(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    _validate_inputs()
    device = _device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, parent_artifact = _new_model()
    frozen_hash_before = _subset_hash(model, frozen=True)
    trainable_parameters = _configure_trainable(model)
    trainable_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    model.to(device).train()
    core, _, core_metadata = load_student(CHECKPOINT)
    output_weight = core.output_weight.detach().float().to(device)
    del core
    caches = (
        load_file(str(FUNCTIONAL_CACHE), device="cpu"),
        load_file(str(LEXICAL_CACHE), device="cpu"),
    )
    offsets = tuple(
        cache["row_offsets"].long().tolist() for cache in caches
    )
    row_counts = tuple(len(value) - 1 for value in offsets)
    if row_counts != (960, 4096):
        raise RuntimeError(f"training cache row counts changed: {row_counts}")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    started = time.perf_counter()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    curves = []
    exposure = [0, 0]
    for step in range(1, args.steps + 1):
        cache_index = (step + 1) % 2
        selected_rows = [
            rng.randrange(row_counts[cache_index])
            for _ in range(args.batch_size)
        ]
        exposure[cache_index] += len(selected_rows)
        batch = _batch(
            caches[cache_index],
            offsets[cache_index],
            selected_rows,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        result = model.training_forward(
            batch["prompt_states"],
            batch["current_states"],
            batch["target_actions"],
            prompt_padding=batch["prompt_padding"],
        )
        valid = batch["target_valid"]
        action_loss = F.nll_loss(
            result["action_log_probs"][valid],
            batch["target_actions"][valid],
        )
        logits = F.linear(result["adapted"][valid], output_weight)
        realization_loss = F.cross_entropy(
            logits, batch["target_tokens"][valid]
        )
        objective = action_loss + 0.25 * realization_loss
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        value = float(objective.detach())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if step == 1 or step % 50 == 0:
            record = {
                "step": step,
                "cache": (
                    "functional" if cache_index == 0 else "lexical"
                ),
                "action_cross_entropy": float(action_loss.detach()),
                "frozen_realization_cross_entropy": float(
                    realization_loss.detach()
                ),
                "selection_objective": value,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(record)
            print(json.dumps(record), flush=True)
    if best_state is None:
        raise RuntimeError("parallel plan produced no checkpoint")
    wall = time.perf_counter() - started
    model.load_state_dict(best_state, strict=True)
    model.eval()
    frozen_hash_after = _subset_hash(model, frozen=True)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("a frozen realization tensor changed")

    cpu_model = SemanticActionPlanResidual(
        **model.canonical_config()
    ).eval()
    cpu_model.load_state_dict(model.state_dict(), strict=True)
    smoke = _batch(
        caches[0], offsets[0], [0], torch.device("cpu")
    )
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
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "parent_artifact_sha256": PARENT_SHA256,
        "parent_payload_hash": parent_artifact["payload_hash"],
        "core_checkpoint_sha256": core_metadata["checkpoint"]["sha256"],
        "functional_cache_sha256": FUNCTIONAL_CACHE_SHA256,
        "lexical_cache_sha256": LEXICAL_CACHE_SHA256,
        "seed": args.seed,
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "functional_batch_rows_exposed": exposure[0],
        "lexical_batch_rows_exposed": exposure[1],
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
        "trainable_parameters": trainable_count,
        "total_parameters": model.parameter_count(),
        "active_parameter_seconds": trainable_count * wall,
        "frozen_tensor_hash_before": frozen_hash_before,
        "frozen_tensor_hash_after": frozen_hash_after,
        "best_selection_objective": best_loss,
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
        "protocol_sha256": PREREGISTRATION_SHA256,
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
        "trainable_parameters": trainable_count,
        "total_parameters": model.parameter_count(),
        "new_runtime_parameters": 0,
        "core_parameters_changed": 0,
        "frozen_tensor_hash_before": frozen_hash_before,
        "frozen_tensor_hash_after": frozen_hash_after,
        "frozen_realization_tensors_identical": (
            frozen_hash_before == frozen_hash_after
        ),
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "functional_batch_rows_exposed": exposure[0],
        "lexical_batch_rows_exposed": exposure[1],
        "primary_device": str(device),
        "primary_device_name": training["primary_device_name"],
        "precision": "fp32",
        "gpu_wall_seconds": training["gpu_wall_seconds"],
        "cpu_wall_seconds": training["cpu_wall_seconds"],
        "peak_accelerator_memory_bytes": peak_accelerator,
        "peak_process_resident_memory_bytes": peak_rss,
        "cpu_fallback_smoke": "PASS",
        "active_parameter_seconds": training[
            "active_parameter_seconds"
        ],
        "best_selection_objective": best_loss,
        "learning_curves": curves,
        "lexical_probe_accessed": False,
        "functional_probe_accessed": False,
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
    evaluation.LEXICAL_DATASET = LEXICAL_PROBE
    evaluation.LEXICAL_EVALUATION_SPLIT = "fresh_probe"
    evaluation.LEXICAL_EXPECTED_ROWS = 256
    evaluation.LEXICAL_MINIMUM_EXACT_RESPONSES = 231
    evaluation.LEXICAL_EVALUATION_FORMAT = (
        "layercake-phase4-semantic-parallel-plan-lexical-evaluation/1"
    )
    evaluation.PYTHON_DATASET = FUNCTIONAL_PROBE
    evaluation.PYTHON_EVALUATION_SPLIT = "parallel_probe"
    evaluation.PYTHON_EXPECTED_ROWS = 64
    evaluation.PYTHON_MINIMUM_FUNCTIONAL_SUCCESSES = 52
    evaluation.PYTHON_EVALUATION_FORMAT = (
        "layercake-phase4-semantic-parallel-plan-python-evaluation/1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, default=11040)
    train_parser.add_argument("--steps", type=int, default=2500)
    train_parser.add_argument("--batch-size", type=int, default=16)
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
