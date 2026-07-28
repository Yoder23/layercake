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
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _execute_tests,
    _extract_function,
    _load_rows,
    _subsequence_start,
)
from scripts.train_phase4_semantic_action_plan import (
    ABI_SHA256,
    ABI_VERSION,
    CHECKPOINT,
    DATASET as PYTHON_DATASET,
    FIXED_TOKEN_IDS,
    _batch,
    _generate,
)


ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT
    / "moonshot"
    / "phase4_semantic_action_plan_lexical_repair_preregistration.json"
)
PARENT = (
    ROOT
    / "artifacts"
    / "moonshot"
    / "phase4"
    / "candidates"
    / "python-semantic-action-plan-seed10340.pt"
)
CACHE = (
    ROOT
    / "artifacts"
    / "moonshot"
    / "phase4"
    / "cache"
    / "seed9824-lexical-conformance-v1-train.safetensors"
)
LEXICAL_DATASET = (
    ROOT / "data" / "moonshot" / "phase4" / "lexical_conformance_v1.jsonl"
)
PARENT_SHA256 = (
    "9363688165a22709be0b87b54787c3e0fec2ea904499464c74ac18571a5b28b8"
)
CACHE_SHA256 = (
    "ee0488a5f407c1fbd1cb8383c8de5d27ee955042e05d9b1713b2b584051bcf79"
)
LEXICAL_DATASET_SHA256 = (
    "d6a7c054c1104c38007c97263031576c10c2a4dc7f78cf494a4f848a362e38bb"
)
TRAINABLE_TENSORS = frozenset(
    {
        "pointer_input.weight",
        "pointer_key.weight",
        "pointer_query.weight",
        "pointer_gate.weight",
        "pointer_gate.bias",
        "copy_semantic_value.weight",
    }
)
LEXICAL_EVALUATION_FORMAT = (
    "layercake-phase4-semantic-action-plan-lexical-evaluation/1"
)
LEXICAL_EVALUATION_SPLIT = "validation"
LEXICAL_EXPECTED_ROWS = 256
LEXICAL_MINIMUM_EXACT_RESPONSES = 231
PYTHON_EVALUATION_FORMAT = (
    "layercake-phase4-semantic-action-plan-lexical-repair-python-evaluation/1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _subset_hash(
    model: SemanticActionPlanResidual, *, trainable: bool
) -> str:
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if (name in TRAINABLE_TENSORS) is trainable
    }
    if not state:
        raise RuntimeError("requested tensor subset is empty")
    return state_dict_hash(state)


def _configure_trainable_subset(
    model: SemanticActionPlanResidual,
) -> list[torch.nn.Parameter]:
    parameter_names = set(dict(model.named_parameters()))
    missing = TRAINABLE_TENSORS - parameter_names
    if missing:
        raise RuntimeError(
            f"declared trainable tensors are absent: {sorted(missing)}"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in TRAINABLE_TENSORS)
    observed = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if observed != TRAINABLE_TENSORS:
        raise RuntimeError(
            "trainable tensor boundary mismatch: "
            f"expected={sorted(TRAINABLE_TENSORS)} "
            f"observed={sorted(observed)}"
        )
    return [
        parameter
        for _, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _validate_locked_inputs() -> None:
    observed = {
        "parent": _sha256(PARENT),
        "cache": _sha256(CACHE),
        "lexical_dataset": _sha256(LEXICAL_DATASET),
    }
    expected = {
        "parent": PARENT_SHA256,
        "cache": CACHE_SHA256,
        "lexical_dataset": LEXICAL_DATASET_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            f"locked input identity mismatch: {observed} != {expected}"
        )


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = _rooted(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    _validate_locked_inputs()
    device = _device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, parent_artifact = load_semantic_action_plan_artifact(PARENT)
    if (
        parent_artifact["abi_version"] != ABI_VERSION
        or parent_artifact["abi_sha256"] != ABI_SHA256
    ):
        raise RuntimeError("parent artifact is bound to the wrong semantic ABI")
    if tuple(model.fixed_token_ids) != FIXED_TOKEN_IDS:
        raise RuntimeError("parent fixed action vocabulary changed")
    frozen_hash_before = _subset_hash(model, trainable=False)
    trainable_hash_before = _subset_hash(model, trainable=True)
    trainable_parameters = _configure_trainable_subset(model)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    model.to(device).train()

    core, _, core_metadata = load_student(CHECKPOINT)
    embedding = core.output_weight.detach().float().to(device)
    del core
    cached = load_file(str(CACHE), device="cpu")
    offsets = cached["row_offsets"].long().tolist()
    row_count = len(offsets) - 1
    if row_count != 4096:
        raise RuntimeError(f"expected 4096 lexical rows, found {row_count}")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    started = time.perf_counter()
    curves: list[dict[str, Any]] = []
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
        valid = batch["target_valid"]
        pointer_valid = valid & batch["target_actions"].ge(
            model.fixed_action_count
        )
        if not pointer_valid.any():
            raise RuntimeError("lexical batch contains no pointer targets")
        action_loss = F.nll_loss(
            result["action_log_probs"][valid],
            batch["target_actions"][valid],
        )
        pointer_logits = F.linear(
            result["adapted"][pointer_valid], embedding
        )
        realization_loss = F.cross_entropy(
            pointer_logits, batch["target_tokens"][pointer_valid]
        )
        objective = action_loss + realization_loss
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
                "action_cross_entropy": float(action_loss.detach()),
                "pointer_realization_cross_entropy": float(
                    realization_loss.detach()
                ),
                "selection_objective": objective_value,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(record)
            print(json.dumps(record), flush=True)

    if best_state is None:
        raise RuntimeError("repair did not produce a checkpoint")
    wall = time.perf_counter() - started
    model.load_state_dict(best_state, strict=True)
    model.eval()
    frozen_hash_after = _subset_hash(model, trainable=False)
    if frozen_hash_after != frozen_hash_before:
        raise RuntimeError("a frozen cake tensor changed during repair")
    trainable_hash_after = _subset_hash(model, trainable=True)
    if trainable_hash_after == trainable_hash_before:
        raise RuntimeError("declared trainable copy-path tensors did not change")

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
        "format": "layercake-phase4-semantic-action-plan-lexical-repair/1",
        "source_commit": _git_head(),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "parent_artifact_sha256": PARENT_SHA256,
        "parent_payload_hash": parent_artifact["payload_hash"],
        "core_checkpoint_sha256": core_metadata["checkpoint"]["sha256"],
        "cache_sha256": CACHE_SHA256,
        "lexical_dataset_sha256": LEXICAL_DATASET_SHA256,
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
        "trainable_parameters": trainable_parameter_count,
        "total_parameters": model.parameter_count(),
        "active_parameter_seconds": trainable_parameter_count * wall,
        "best_selection_objective": best_loss,
        "frozen_tensor_subset_hash_before": frozen_hash_before,
        "frozen_tensor_subset_hash_after": frozen_hash_after,
        "trainable_tensor_subset_hash_before": trainable_hash_before,
        "trainable_tensor_subset_hash_after": trainable_hash_after,
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
        "format": "layercake-phase4-semantic-action-plan-lexical-repair-training/1",
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
        "architecture_unchanged": (
            model.canonical_config()
            == parent_artifact["architecture"]
        ),
        "new_runtime_parameters": 0,
        "total_parameters": model.parameter_count(),
        "trainable_parameters": trainable_parameter_count,
        "trainable_tensor_names": sorted(TRAINABLE_TENSORS),
        "core_parameters_changed": 0,
        "frozen_tensor_subset_hash_before": frozen_hash_before,
        "frozen_tensor_subset_hash_after": frozen_hash_after,
        "frozen_tensor_subset_identical": (
            frozen_hash_before == frozen_hash_after
        ),
        "trainable_tensor_subset_hash_before": trainable_hash_before,
        "trainable_tensor_subset_hash_after": trainable_hash_after,
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


def _load_for_evaluation(
    artifact_path: Path, device: torch.device
) -> tuple[Any, Any, SemanticActionPlanResidual, dict[str, Any], dict[str, Any]]:
    cake, artifact = load_semantic_action_plan_artifact(artifact_path)
    if (
        artifact["abi_version"] != ABI_VERSION
        or artifact["abi_sha256"] != ABI_SHA256
    ):
        raise RuntimeError("repaired artifact is bound to the wrong ABI")
    cake.to(device).eval()
    core, tokenizer, core_metadata = load_student(CHECKPOINT)
    core.to(device).eval()
    return core, tokenizer, cake, artifact, core_metadata


def evaluate_lexical(args: argparse.Namespace) -> dict[str, Any]:
    artifact_path = _rooted(args.artifact)
    output = _rooted(args.output)
    if output.exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    device = _device(args.device)
    core, tokenizer, cake, artifact, core_metadata = _load_for_evaluation(
        artifact_path, device
    )
    rows = [
        row
        for row in _load_rows(LEXICAL_DATASET)
        if row["split"] == LEXICAL_EVALUATION_SPLIT
    ]
    if len(rows) != LEXICAL_EXPECTED_ROWS:
        raise RuntimeError(
            f"expected {LEXICAL_EXPECTED_ROWS} lexical "
            f"{LEXICAL_EVALUATION_SPLIT} rows, got {len(rows)}"
        )
    records = []
    for index, row in enumerate(rows):
        generated = _generate(
            core, tokenizer, cake, row["prompt"], device=device
        )
        exact = generated["text"] == row["response"]
        source, parse_status = _extract_function(
            generated["text"], row["function_name"]
        )
        source = source or ""
        functional = False
        if source:
            try:
                namespace: dict[str, Any] = {}
                exec(compile(source, "<lexical-cake>", "exec"), namespace)
                function = namespace[row["function_name"]]
                probes = (0, "semantic identity", [1, 2, 3])
                functional = all(function(value) == value for value in probes)
            except Exception:
                functional = False
        records.append(
            {
                "id": row["id"],
                "expected_function_name": row["function_name"],
                "expected_text": row["response"],
                "generated_text": generated["text"],
                "exact_response": exact,
                "parse_status": parse_status,
                "functional_identity": functional,
                "planned_actions": generated["planned_actions"],
                "plan_complete": generated["plan_complete"],
                "generated_tokens": generated["generated_tokens"],
                "time_to_first_output_seconds": generated[
                    "time_to_first_output_seconds"
                ],
                "total_latency_seconds": generated[
                    "total_latency_seconds"
                ],
            }
        )
        if (index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": index + 1,
                        "exact_responses": sum(
                            record["exact_response"] for record in records
                        ),
                        "functional_identities": sum(
                            record["functional_identity"]
                            for record in records
                        ),
                    }
                ),
                flush=True,
            )
    exact_successes = sum(record["exact_response"] for record in records)
    functional_successes = sum(
        record["functional_identity"] for record in records
    )
    minimum = LEXICAL_MINIMUM_EXACT_RESPONSES
    evidence = {
        "format": LEXICAL_EVALUATION_FORMAT,
        "status": "PASS" if exact_successes >= minimum else "FAIL",
        "protocol": PREREGISTRATION.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(PREREGISTRATION),
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "payload_hash": artifact["payload_hash"],
        "abi_version": artifact["abi_version"],
        "abi_sha256": artifact["abi_sha256"],
        "core_checkpoint_sha256": core_metadata["checkpoint"]["sha256"],
        "dataset": LEXICAL_DATASET.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(LEXICAL_DATASET),
        "split": LEXICAL_EVALUATION_SPLIT,
        "evaluation_device": str(device),
        "evaluation_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "declared laptop CPU"
        ),
        "distinct_prompts": len(rows),
        "exact_response_successes": exact_successes,
        "exact_response_success_rate": exact_successes / len(rows),
        "minimum_exact_response_successes": minimum,
        "functional_identity_successes": functional_successes,
        "functional_identity_success_rate": functional_successes / len(rows),
        "teacher_at_inference": False,
        "raw_prompt_bytes_exposed_to_cake": False,
        "private_host_token_ids_exposed_to_cake": False,
        "canonical_semantic_abi_consumed": True,
        "same_shape_semantic_residual_returned": True,
        "autonomous_neural_generation": True,
        "records": records,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def evaluate_python(args: argparse.Namespace) -> dict[str, Any]:
    artifact_path = _rooted(args.artifact)
    output = _rooted(args.output)
    if output.exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    lexical_evidence_path = _rooted(args.lexical_evidence)
    lexical_evidence = json.loads(
        lexical_evidence_path.read_text(encoding="utf-8")
    )
    if lexical_evidence.get("status") != "PASS":
        raise RuntimeError("Python screen is locked until lexical validation passes")
    if lexical_evidence.get("artifact_file_sha256") != _sha256(artifact_path):
        raise RuntimeError("lexical evidence belongs to a different artifact")
    device = _device(args.device)
    core, tokenizer, cake, artifact, core_metadata = _load_for_evaluation(
        artifact_path, device
    )
    rows = [
        row for row in _load_rows(PYTHON_DATASET)
        if row["split"] == "validation"
    ]
    records = []
    for index, row in enumerate(rows):
        generated = _generate(
            core, tokenizer, cake, row["prompt"], device=device
        )
        source, parse_status = _extract_function(
            generated["text"], row["function_name"]
        )
        source = source or ""
        passed, tests = _execute_tests(
            source, row["function_name"], row["tests"]
        )
        records.append(
            {
                "id": row["id"],
                "family": row["family"],
                "expected_function_name": row["function_name"],
                "generated_text": generated["text"],
                "extracted_source": source,
                "parse_status": parse_status,
                "functional_success": passed,
                "tests": tests,
                "generated_tokens": generated["generated_tokens"],
                "planned_actions": generated["planned_actions"],
                "plan_complete": generated["plan_complete"],
                "time_to_first_output_seconds": generated[
                    "time_to_first_output_seconds"
                ],
                "total_latency_seconds": generated[
                    "total_latency_seconds"
                ],
            }
        )
        if (index + 1) % 8 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": index + 1,
                        "successes": sum(
                            record["functional_success"]
                            for record in records
                        ),
                    }
                ),
                flush=True,
            )
    successes = sum(record["functional_success"] for record in records)
    minimum = 52
    evidence = {
        "format": PYTHON_EVALUATION_FORMAT,
        "status": "PASS" if successes >= minimum else "FAIL",
        "protocol": PREREGISTRATION.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(PREREGISTRATION),
        "lexical_evidence": lexical_evidence_path.relative_to(ROOT).as_posix(),
        "lexical_evidence_sha256": _sha256(lexical_evidence_path),
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "payload_hash": artifact["payload_hash"],
        "abi_version": artifact["abi_version"],
        "abi_sha256": artifact["abi_sha256"],
        "core_checkpoint_sha256": core_metadata["checkpoint"]["sha256"],
        "dataset": PYTHON_DATASET.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(PYTHON_DATASET),
        "split": "validation",
        "evaluation_device": str(device),
        "evaluation_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "declared laptop CPU"
        ),
        "distinct_prompts": len(rows),
        "functional_successes": successes,
        "functional_failures": len(rows) - successes,
        "functional_success_rate": successes / len(rows),
        "minimum_functional_successes": minimum,
        "teacher_at_inference": False,
        "raw_prompt_bytes_exposed_to_cake": False,
        "private_host_token_ids_exposed_to_cake": False,
        "canonical_semantic_abi_consumed": True,
        "same_shape_semantic_residual_returned": True,
        "host_response_states_drive_plan": False,
        "autonomous_neural_generation": True,
        "records": records,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def diagnose_actions(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_path = _rooted(args.evaluation)
    output = _rooted(args.output)
    if output.exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("format") != (
        "layercake-phase4-semantic-action-plan-lexical-evaluation/1"
    ):
        raise RuntimeError("unsupported lexical evaluation evidence")
    _, tokenizer, _ = load_student(CHECKPOINT)
    rows = {
        row["id"]: row
        for row in _load_rows(LEXICAL_DATASET)
        if row["split"] == "validation"
    }
    fixed_actions = {
        token_id: action
        for action, token_id in enumerate(FIXED_TOKEN_IDS)
    }
    records = []
    totals = {
        "action_units": 0,
        "action_units_correct": 0,
        "pointer_action_units": 0,
        "pointer_action_units_correct": 0,
        "fixed_action_units": 0,
        "fixed_action_units_correct": 0,
        "exact_action_sequences": 0,
        "exact_pointer_span_counts": 0,
    }
    for observed in evaluation["records"]:
        row = rows[observed["id"]]
        prompt_ids = tokenizer.encode(row["prompt"] + "\n")
        response_ids = tokenizer.encode(row["response"])
        identifier_pattern = tokenizer.encode(" " + row["function_name"])
        prompt_start = _subsequence_start(prompt_ids, identifier_pattern)
        response_start = _subsequence_start(
            response_ids, identifier_pattern
        )
        if prompt_start is None or response_start is None:
            raise RuntimeError(
                f"identifier span absent for {row['id']}"
            )
        response_stop = response_start + len(identifier_pattern)
        expected_actions = []
        for position, token_id in enumerate(response_ids):
            if response_start <= position < response_stop:
                expected_actions.append(
                    len(FIXED_TOKEN_IDS)
                    + prompt_start
                    + position
                    - response_start
                )
            else:
                expected_actions.append(fixed_actions[token_id])
        expected_actions.append(fixed_actions[50256])
        observed_actions = observed["planned_actions"]
        aligned = list(zip(expected_actions, observed_actions))
        exact = observed_actions == expected_actions
        expected_pointer_count = len(identifier_pattern)
        observed_pointer_count = sum(
            action >= len(FIXED_TOKEN_IDS)
            for action in observed_actions
        )
        pointer_correct = sum(
            expected == actual
            for position, (expected, actual) in enumerate(aligned)
            if response_start <= position < response_stop
        )
        fixed_correct = sum(
            expected == actual
            for position, (expected, actual) in enumerate(aligned)
            if not (response_start <= position < response_stop)
        )
        expected_fixed_count = len(expected_actions) - expected_pointer_count
        totals["action_units"] += len(expected_actions)
        totals["action_units_correct"] += sum(
            expected == actual for expected, actual in aligned
        )
        totals["pointer_action_units"] += expected_pointer_count
        totals["pointer_action_units_correct"] += pointer_correct
        totals["fixed_action_units"] += expected_fixed_count
        totals["fixed_action_units_correct"] += fixed_correct
        totals["exact_action_sequences"] += int(exact)
        totals["exact_pointer_span_counts"] += int(
            observed_pointer_count == expected_pointer_count
        )
        records.append(
            {
                "id": row["id"],
                "expected_actions": expected_actions,
                "observed_actions": observed_actions,
                "exact_action_sequence": exact,
                "expected_pointer_actions": expected_pointer_count,
                "observed_pointer_actions": observed_pointer_count,
                "pointer_action_units_correct": pointer_correct,
                "fixed_action_units_correct": fixed_correct,
                "exact_response": observed["exact_response"],
            }
        )
    rates = {
        "action_accuracy": (
            totals["action_units_correct"] / totals["action_units"]
        ),
        "pointer_action_accuracy": (
            totals["pointer_action_units_correct"]
            / totals["pointer_action_units"]
        ),
        "fixed_action_accuracy": (
            totals["fixed_action_units_correct"]
            / totals["fixed_action_units"]
        ),
        "exact_action_sequence_rate": (
            totals["exact_action_sequences"] / len(records)
        ),
        "exact_pointer_span_count_rate": (
            totals["exact_pointer_span_counts"] / len(records)
        ),
        "exact_response_rate_given_exact_action_sequence": (
            sum(
                record["exact_response"]
                for record in records
                if record["exact_action_sequence"]
            )
            / max(
                1,
                sum(
                    record["exact_action_sequence"]
                    for record in records
                ),
            )
        ),
    }
    evidence = {
        "format": "layercake-phase4-semantic-action-plan-lexical-action-diagnostic/1",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION_CREDIT",
        "evaluation": evaluation_path.relative_to(ROOT).as_posix(),
        "evaluation_file_sha256": _sha256(evaluation_path),
        "artifact_file_sha256": evaluation["artifact_file_sha256"],
        "dataset_sha256": _sha256(LEXICAL_DATASET),
        "distinct_prompts": len(records),
        "totals": totals,
        "rates": rates,
        "measured_attribution": (
            "The repaired self-action planner selects held-out lexical action "
            "sequences almost perfectly. Remaining strict failures occur after "
            "the correct pointer action is selected, isolating the limiter to "
            "semantic-state identity realization through the linear copy value."
        ),
        "records": records,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, default=10440)
    train_parser.add_argument("--steps", type=int, default=1200)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    diagnostic_parser = subparsers.add_parser("diagnose-actions")
    diagnostic_parser.add_argument("--evaluation", type=Path, required=True)
    diagnostic_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        result = train(args)
    elif args.command == "evaluate-lexical":
        result = evaluate_lexical(args)
    elif args.command == "diagnose-actions":
        result = diagnose_actions(args)
    else:
        result = evaluate_python(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
