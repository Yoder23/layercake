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

from layercake.semantic_token_plan import (
    SemanticTokenPlanResidual,
    build_semantic_token_plan_artifact,
    load_semantic_token_plan_artifact,
)
from layercake.training.phase2_shallow_sparse import load_student
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _execute_tests,
    _extract_function,
    _load_rows,
)


ROOT = Path(__file__).resolve().parents[1]
ABI_VERSION = "lc-semantic-gpt2-768/1"
ABI_SHA256 = "d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761"
PREREGISTRATION = (
    ROOT / "moonshot/phase4_semantic_token_plan_preregistration.json"
)
AMENDMENT = (
    ROOT / "moonshot/phase4_semantic_token_plan_cache_amendment.json"
)
CHECKPOINT = (
    ROOT
    / "artifacts/moonshot/phase2_shallow_sparse_pretrained"
    / "student2400-seed-9824"
)
CACHE = (
    ROOT
    / "artifacts/moonshot/phase4/cache"
    / "seed9824-attentive-v4-exact-span-train.safetensors"
)
DATASET = ROOT / "data/moonshot/phase4/python_functional_v1.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _row_tensors(
    cached: dict[str, torch.Tensor],
    offsets: list[int],
    row: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start, stop = offsets[row], offsets[row + 1]
    states = cached["semantic_states"][start:stop]
    targets = cached["target_ids"][start:stop].long()
    mask = cached["response_mask"][start:stop].bool()
    labels = cached["lexical_copy_labels"][start:stop].long()
    response_start = int(torch.nonzero(mask, as_tuple=False)[0].item())
    return (
        states[: response_start + 1],
        states[response_start:],
        targets[response_start:],
        labels[response_start:],
    )


def _batch(
    cached: dict[str, torch.Tensor],
    offsets: list[int],
    rows: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    values = [_row_tensors(cached, offsets, row) for row in rows]
    source_length = max(value[0].shape[0] for value in values)
    target_length = max(value[1].shape[0] for value in values)
    prompt_states = torch.zeros(
        len(rows), source_length, 768, dtype=torch.float32, device=device
    )
    response_states = torch.zeros(
        len(rows), target_length, 768, dtype=torch.float32, device=device
    )
    prompt_padding = torch.ones(
        len(rows), source_length, dtype=torch.bool, device=device
    )
    response_valid = torch.zeros(
        len(rows), target_length, dtype=torch.bool, device=device
    )
    targets = torch.zeros(
        len(rows), target_length, dtype=torch.long, device=device
    )
    pointer_labels = torch.full(
        (len(rows), target_length), -100, dtype=torch.long, device=device
    )
    for index, (source, response, target, labels) in enumerate(values):
        source_count = source.shape[0]
        target_count = response.shape[0]
        prompt_states[index, :source_count] = source.float().to(device)
        response_states[index, :target_count] = response.float().to(device)
        prompt_padding[index, :source_count] = False
        response_valid[index, :target_count] = True
        targets[index, :target_count] = target.to(device)
        pointer_labels[index, :target_count] = labels.to(device)
    return {
        "prompt_states": prompt_states,
        "response_states": response_states,
        "prompt_padding": prompt_padding,
        "response_valid": response_valid,
        "targets": targets,
        "pointer_labels": pointer_labels,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    args.output = _rooted(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output.exists():
        raise RuntimeError(f"immutable output already exists: {args.output}")
    torch.manual_seed(args.seed)
    random_generator = random.Random(args.seed)
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    core, _, core_metadata = load_student(CHECKPOINT)
    embedding = core.output_weight.detach().float().to(device)
    del core
    cached = load_file(str(CACHE), device="cpu")
    offsets = cached["row_offsets"].long().tolist()
    row_count = len(offsets) - 1
    model = SemanticTokenPlanResidual(dropout=0.1).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    started = time.perf_counter()
    best_loss = float("inf")
    best_state = None
    curves: list[dict[str, Any]] = []
    pointer_weight = 0.5
    copy_value_weight = 0.5
    copy_gate_weight = 0.25
    stability_weight = 0.002
    for step in range(1, args.steps + 1):
        rows = [
            random_generator.randrange(row_count)
            for _ in range(args.batch_size)
        ]
        batch = _batch(cached, offsets, rows, device)
        optimizer.zero_grad(set_to_none=True)
        result = model.training_forward(
            batch["prompt_states"],
            batch["response_states"],
            prompt_padding=batch["prompt_padding"],
        )
        valid = batch["response_valid"]
        targets = batch["targets"]
        logits = F.linear(result["adapted"][valid], embedding)
        language_loss = F.cross_entropy(logits, targets[valid])
        pointer_valid = valid & batch["pointer_labels"].ge(0)
        pointer_loss = F.cross_entropy(
            result["pointer_scores"][pointer_valid],
            batch["pointer_labels"][pointer_valid],
        )
        batch_indexes, response_positions = torch.nonzero(
            pointer_valid, as_tuple=True
        )
        source_positions = batch["pointer_labels"][pointer_valid]
        selected_source = batch["prompt_states"][
            batch_indexes, source_positions
        ]
        selected_current = batch["response_states"][
            batch_indexes, response_positions
        ]
        copy_residual = model.max_residual * torch.tanh(
            model.copy_value(model.input_norm(selected_source))
        )
        copy_logits = F.linear(selected_current + copy_residual, embedding)
        copy_value_loss = F.cross_entropy(
            copy_logits, targets[pointer_valid]
        )
        gate_loss = F.binary_cross_entropy_with_logits(
            result["gate_logits"][valid],
            pointer_valid[valid].float(),
        )
        residual = result["residual"][valid]
        stability = residual.square().mean() / (
            batch["response_states"][valid].square().mean().clamp_min(1e-6)
        )
        objective = (
            language_loss
            + pointer_weight * pointer_loss
            + copy_value_weight * copy_value_loss
            + copy_gate_weight * gate_loss
            + stability_weight * stability
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        objective_value = float(objective.detach())
        if objective_value < best_loss:
            best_loss = objective_value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if step == 1 or step % 100 == 0:
            record = {
                "step": step,
                "full_vocabulary_cross_entropy": float(
                    language_loss.detach()
                ),
                "pointer_cross_entropy": float(pointer_loss.detach()),
                "copy_value_cross_entropy": float(copy_value_loss.detach()),
                "copy_gate_binary_cross_entropy": float(gate_loss.detach()),
                "stability_ratio": float(stability.detach()),
                "selection_objective": objective_value,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(record)
            print(json.dumps(record), flush=True)
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    smoke_batch = _batch(cached, offsets, [0], torch.device("cpu"))
    cpu_model = SemanticTokenPlanResidual(dropout=0.1)
    cpu_model.load_state_dict(best_state)
    cpu_model.eval()
    with torch.inference_mode():
        smoke = cpu_model.training_forward(
            smoke_batch["prompt_states"],
            smoke_batch["response_states"],
            prompt_padding=smoke_batch["prompt_padding"],
        )
    if not torch.isfinite(smoke["residual"]).all():
        raise RuntimeError("CPU fallback smoke produced non-finite residuals")
    wall = time.perf_counter() - started
    peak_accelerator = (
        int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else 0
    )
    training = {
        "seed": args.seed,
        "source_commit": _git_head(),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "cache_amendment_sha256": _sha256(AMENDMENT),
        "core_checkpoint_sha256": core_metadata["checkpoint"]["sha256"],
        "cache_sha256": _sha256(CACHE),
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_weights": {
            "pointer": pointer_weight,
            "copy_value": copy_value_weight,
            "copy_gate": copy_gate_weight,
            "stability": stability_weight,
        },
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
        "raw_utf8_training_bytes_exposed": 224000,
        "model_visible_nonpadding_units": 73825,
        "active_parameter_seconds_to_quality": (
            model.parameter_count() * wall
        ),
        "best_selection_objective": best_loss,
        "learning_curves": curves,
    }
    artifact = build_semantic_token_plan_artifact(
        model.cpu(),
        abi_version=ABI_VERSION,
        abi_sha256=ABI_SHA256,
        training=training,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    evidence = {
        "format": "layercake-phase4-semantic-token-plan-training/1",
        "status": "TRAINED",
        "protocol": PREREGISTRATION.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(PREREGISTRATION),
        "cache_amendment": AMENDMENT.relative_to(ROOT).as_posix(),
        "cache_amendment_sha256": _sha256(AMENDMENT),
        "source_commit": training["source_commit"],
        "seed": args.seed,
        "artifact": args.output.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(args.output),
        "payload_hash": artifact["payload_hash"],
        "spec_sha256": artifact["spec_sha256"],
        "abi_version": ABI_VERSION,
        "abi_sha256": ABI_SHA256,
        "architecture": model.canonical_config(),
        "trainable_parameters": model.parameter_count(),
        "core_parameters_changed": 0,
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "primary_device": str(device),
        "primary_device_name": training["primary_device_name"],
        "precision": "fp32",
        "gpu_wall_seconds": wall if device.type == "cuda" else 0.0,
        "cpu_wall_seconds": wall if device.type == "cpu" else 0.0,
        "peak_accelerator_memory_bytes": peak_accelerator,
        "peak_process_resident_memory_bytes": peak_rss,
        "cpu_fallback_smoke": "PASS",
        "active_parameter_seconds_to_quality": (
            model.parameter_count() * wall
        ),
        "loss_weights": training["loss_weights"],
        "best_selection_objective": best_loss,
        "learning_curves": curves,
        "test_split_accessed": false,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path = args.output.with_suffix(".json")
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


@torch.inference_mode()
def _generate(
    core,
    tokenizer,
    cake: SemanticTokenPlanResidual,
    prompt: str,
    *,
    device: torch.device,
    maximum_tokens: int = 192,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt + "\n")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    result = core(
        input_ids,
        prompt_lengths=torch.tensor(
            [len(prompt_ids)], dtype=torch.long, device=device
        ),
        use_cache=True,
    )
    residual, cake_state = cake.prefill(result["hidden"])
    next_logits = F.linear(
        result["hidden"][:, -1] + residual, core.output_weight
    )
    past_key_values = result["past_key_values"]
    task_routes = result["task_routes"]
    generated: list[int] = []
    first_output = None
    for _ in range(maximum_tokens):
        token = next_logits.argmax(dim=-1)
        generated.append(int(token.item()))
        if first_output is None:
            first_output = time.perf_counter()
        text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if (
            token.item() == tokenizer.eos_token_id
            or "\n\n" in text
            or (text.count("\n") >= 2 and text.endswith("\n"))
        ):
            break
        result = core(
            token[:, None],
            task_routes=task_routes,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = result["past_key_values"]
        residual, cake_state = cake.step(
            result["hidden"][:, -1], cake_state
        )
        next_logits = F.linear(
            result["hidden"][:, -1] + residual, core.output_weight
        )
    ended = time.perf_counter()
    return {
        "text": tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "time_to_first_output_seconds": (
            (first_output or ended) - started
        ),
        "total_latency_seconds": ended - started,
        "generated_tokens": len(generated),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    args.artifact = _rooted(args.artifact)
    args.output = _rooted(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cake, artifact = load_semantic_token_plan_artifact(
        args.artifact, map_location="cpu"
    )
    cake.to(device).eval()
    core, tokenizer, metadata = load_student(CHECKPOINT)
    core.to(device).eval()
    rows = [
        row for row in _load_rows(DATASET) if row["split"] == args.split
    ]
    records = []
    for index, row in enumerate(rows):
        generated = _generate(
            core,
            tokenizer,
            cake,
            row["prompt"],
            device=device,
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
                "time_to_first_output_seconds": generated[
                    "time_to_first_output_seconds"
                ],
                "total_latency_seconds": generated[
                    "total_latency_seconds"
                ],
                "generated_tokens": generated["generated_tokens"],
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
    minimum = 52 if args.split == "validation" else len(rows)
    evidence = {
        "format": "layercake-phase4-semantic-token-plan-functional-evaluation/1",
        "status": "PASS" if successes >= minimum else "FAIL",
        "protocol": PREREGISTRATION.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(PREREGISTRATION),
        "cache_amendment": AMENDMENT.relative_to(ROOT).as_posix(),
        "cache_amendment_sha256": _sha256(AMENDMENT),
        "artifact": args.artifact.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(args.artifact),
        "payload_hash": artifact["payload_hash"],
        "abi_version": artifact["abi_version"],
        "abi_sha256": artifact["abi_sha256"],
        "core_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(DATASET),
        "split": args.split,
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
        "teacher_at_inference": false,
        "raw_prompt_bytes_exposed_to_cake": false,
        "private_host_token_ids_exposed_to_cake": false,
        "canonical_semantic_abi_consumed": true,
        "same_shape_semantic_residual_returned": true,
        "autonomous_neural_generation": true,
        "records": records,
        "test_split_accessed": args.split == "test",
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--steps", type=int, default=3000)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--device", default="cuda:0")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--artifact", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    evaluate_parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        result = train(args)
    else:
        result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
