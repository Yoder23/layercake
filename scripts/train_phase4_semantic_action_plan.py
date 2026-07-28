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
)


ROOT = Path(__file__).resolve().parents[1]
ABI_VERSION = "lc-semantic-gpt2-768/1"
ABI_SHA256 = "d024de52144a2d797d0501acb7deb55575ffca7e33f72900beff599cf0a97761"
PREREGISTRATION = (
    ROOT / "moonshot/phase4_semantic_action_plan_preregistration.json"
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
EOS_TOKEN_ID = 50256
NON_IDENTIFIER_TOKEN_IDS = (
    7, 8, 11, 12, 13, 15, 16, 17, 20, 25, 28, 58, 60, 64, 72, 76,
    77, 92, 198, 220, 257, 271, 273, 275, 280, 287, 318, 329, 352,
    361, 362, 366, 378, 407, 437, 477, 493, 532, 611, 657, 685, 796,
    826, 828, 949, 1029, 1136, 1149, 1174, 1220, 1255, 1271, 1279,
    1288, 1298, 1343, 1352, 1391, 1441, 1448, 1635, 1875, 1877,
    1911, 1988, 1994, 2160, 2302, 2352, 2420, 2472, 2496, 2546,
    2599, 2628, 2659, 2837, 3419, 3506, 3509, 3609, 3712, 3815,
    4008, 4064, 4113, 4299, 4808, 4943, 5239, 5470, 5912, 6045,
    6376, 6407, 6624, 6795, 6975, 7839, 8094, 8367, 9127, 9319,
    9464, 9630, 9853, 10352, 10641, 11677, 12429, 15437, 15853,
    16855, 17618, 17635, 18896, 21037, 22179, 22446, 23243, 23350,
    23814, 23884, 24432, 27056, 27160, 27444, 28955, 28968, 30629,
    31457, 33295, 35312, 36311, 39279, 47715, 48185,
)
FIXED_TOKEN_IDS = tuple(sorted((*NON_IDENTIFIER_TOKEN_IDS, EOS_TOKEN_ID)))


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
    fixed_actions: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start, stop = offsets[row], offsets[row + 1]
    states = cached["semantic_states"][start:stop]
    targets = cached["target_ids"][start:stop].long()
    mask = cached["response_mask"][start:stop].bool()
    labels = cached["lexical_copy_labels"][start:stop].long()
    response_start = int(torch.nonzero(mask, as_tuple=False)[0].item())
    prompt_states = states[: response_start + 1]
    current_states = states[response_start:].float()
    target_tokens = targets[response_start:]
    pointer_labels = labels[response_start:]
    actions = []
    for token, pointer in zip(
        target_tokens.tolist(), pointer_labels.tolist(), strict=True
    ):
        if pointer >= 0:
            actions.append(len(FIXED_TOKEN_IDS) + pointer)
        else:
            actions.append(fixed_actions[token])
    actions.append(fixed_actions[EOS_TOKEN_ID])
    current_states = torch.cat(
        (current_states, current_states[-1:].clone()), dim=0
    )
    target_tokens = torch.cat(
        (target_tokens, torch.tensor([EOS_TOKEN_ID], dtype=torch.long))
    )
    return (
        prompt_states,
        current_states,
        target_tokens,
        torch.tensor(actions, dtype=torch.long),
    )


def _batch(
    cached: dict[str, torch.Tensor],
    offsets: list[int],
    rows: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fixed_actions = {
        token_id: action
        for action, token_id in enumerate(FIXED_TOKEN_IDS)
    }
    values = [
        _row_tensors(cached, offsets, row, fixed_actions) for row in rows
    ]
    source_length = max(value[0].shape[0] for value in values)
    target_length = max(value[1].shape[0] for value in values)
    prompt_states = torch.zeros(
        len(rows), source_length, 768, dtype=torch.float32, device=device
    )
    current_states = torch.zeros(
        len(rows), target_length, 768, dtype=torch.float32, device=device
    )
    prompt_padding = torch.ones(
        len(rows), source_length, dtype=torch.bool, device=device
    )
    target_valid = torch.zeros(
        len(rows), target_length, dtype=torch.bool, device=device
    )
    target_tokens = torch.zeros(
        len(rows), target_length, dtype=torch.long, device=device
    )
    target_actions = torch.zeros(
        len(rows), target_length, dtype=torch.long, device=device
    )
    for index, (source, current, tokens, actions) in enumerate(values):
        source_count = source.shape[0]
        target_count = current.shape[0]
        prompt_states[index, :source_count] = source.float().to(device)
        current_states[index, :target_count] = current.float().to(device)
        prompt_padding[index, :source_count] = False
        target_valid[index, :target_count] = True
        target_tokens[index, :target_count] = tokens.to(device)
        target_actions[index, :target_count] = actions.to(device)
    return {
        "prompt_states": prompt_states,
        "current_states": current_states,
        "prompt_padding": prompt_padding,
        "target_valid": target_valid,
        "target_tokens": target_tokens,
        "target_actions": target_actions,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    args.output = _rooted(args.output)
    if args.output.exists():
        raise RuntimeError(f"immutable output already exists: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
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
    model = SemanticActionPlanResidual(
        fixed_token_ids=FIXED_TOKEN_IDS,
        eos_token_id=EOS_TOKEN_ID,
        dropout=0.1,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    action_weight = 1.0
    realization_weight = 1.0
    stability_weight = 0.002
    started = time.perf_counter()
    best_loss = float("inf")
    best_state = None
    curves = []
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
        action_loss = F.nll_loss(
            result["action_log_probs"][valid],
            batch["target_actions"][valid],
        )
        logits = F.linear(result["adapted"][valid], embedding)
        realization_loss = F.cross_entropy(
            logits, batch["target_tokens"][valid]
        )
        residual = result["residual"][valid]
        stability = residual.square().mean() / (
            batch["current_states"][valid].square().mean().clamp_min(1e-6)
        )
        objective = (
            action_weight * action_loss
            + realization_weight * realization_loss
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
                "action_cross_entropy": float(action_loss.detach()),
                "full_vocabulary_realization_cross_entropy": float(
                    realization_loss.detach()
                ),
                "stability_ratio": float(stability.detach()),
                "selection_objective": objective_value,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(record)
            print(json.dumps(record), flush=True)
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    wall = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    cpu_model = SemanticActionPlanResidual(
        fixed_token_ids=FIXED_TOKEN_IDS,
        eos_token_id=EOS_TOKEN_ID,
        dropout=0.1,
    )
    cpu_model.load_state_dict(best_state)
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
        "seed": args.seed,
        "source_commit": _git_head(),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "core_checkpoint_sha256": core_metadata["checkpoint"]["sha256"],
        "cache_sha256": _sha256(CACHE),
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_weights": {
            "action": action_weight,
            "realization": realization_weight,
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
    artifact = build_semantic_action_plan_artifact(
        model.cpu(),
        abi_version=ABI_VERSION,
        abi_sha256=ABI_SHA256,
        training=training,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    evidence = {
        "format": "layercake-phase4-semantic-action-plan-training/1",
        "status": "TRAINED",
        "protocol": PREREGISTRATION.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(PREREGISTRATION),
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
        "primary_device": str(device),
        "primary_device_name": training["primary_device_name"],
        "precision": "fp32",
        "gpu_wall_seconds": training["gpu_wall_seconds"],
        "cpu_wall_seconds": training["cpu_wall_seconds"],
        "peak_accelerator_memory_bytes": peak_accelerator,
        "peak_process_resident_memory_bytes": peak_rss,
        "cpu_fallback_smoke": "PASS",
        "active_parameter_seconds_to_quality": training[
            "active_parameter_seconds_to_quality"
        ],
        "loss_weights": training["loss_weights"],
        "best_selection_objective": best_loss,
        "learning_curves": curves,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    args.output.with_suffix(".json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


@torch.inference_mode()
def _generate(
    core,
    tokenizer,
    cake: SemanticActionPlanResidual,
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
    residual, cake_state, _ = cake.prefill(result["hidden"])
    next_logits = F.linear(
        result["hidden"][:, -1] + residual, core.output_weight
    )
    past_key_values = result["past_key_values"]
    task_routes = result["task_routes"]
    generated = []
    first_output = None
    for output_index in range(maximum_tokens):
        token = next_logits.argmax(dim=-1)
        generated.append(int(token.item()))
        if first_output is None:
            first_output = time.perf_counter()
        if (
            token.item() == tokenizer.eos_token_id
            or output_index + 1 >= maximum_tokens
        ):
            break
        result = core(
            token[:, None],
            task_routes=task_routes,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = result["past_key_values"]
        if cake_state.complete:
            break
        residual, cake_state, _ = cake.step(
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
        "generated_tokens": len(generated),
        "planned_actions": cake_state.planned_actions,
        "plan_complete": cake_state.complete,
        "time_to_first_output_seconds": (
            (first_output or ended) - started
        ),
        "total_latency_seconds": ended - started,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    args.artifact = _rooted(args.artifact)
    args.output = _rooted(args.output)
    device = torch.device(args.device)
    cake, artifact = load_semantic_action_plan_artifact(
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
    minimum = 52 if args.split == "validation" else len(rows)
    evidence = {
        "format": "layercake-phase4-semantic-action-plan-evaluation/1",
        "status": "PASS" if successes >= minimum else "FAIL",
        "protocol": PREREGISTRATION.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(PREREGISTRATION),
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
        "teacher_at_inference": False,
        "raw_prompt_bytes_exposed_to_cake": False,
        "private_host_token_ids_exposed_to_cake": False,
        "canonical_semantic_abi_consumed": True,
        "same_shape_semantic_residual_returned": True,
        "host_response_states_drive_plan": False,
        "autonomous_neural_generation": True,
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
    result = train(args) if args.command == "train" else evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
