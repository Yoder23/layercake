from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.semantic_action_plan import (
    load_semantic_action_plan_artifact,
)
from layercake.training.phase2_shallow_sparse import load_student
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _load_rows,
    _subsequence_start,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    ROOT
    / "artifacts/moonshot/phase2_shallow_sparse_pretrained"
    / "student2400-seed-9824"
)
DATASET = ROOT / "data/moonshot/phase4/python_functional_v1.jsonl"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected(
    tokenizer,
    row: dict,
    fixed_actions: dict[int, int],
    fixed_count: int,
    eos_token_id: int,
) -> tuple[list[int], list[int], list[int], list[int], int]:
    prompt_ids = tokenizer.encode(row["prompt"] + "\n")
    response_ids = tokenizer.encode(row["response"])
    pattern = tokenizer.encode(" " + row["function_name"])
    prompt_start = _subsequence_start(prompt_ids, pattern)
    response_start = _subsequence_start(response_ids, pattern)
    if prompt_start is None or response_start is None:
        pattern = tokenizer.encode(row["function_name"])
        prompt_start = _subsequence_start(prompt_ids, pattern)
        response_start = _subsequence_start(response_ids, pattern)
    if prompt_start is None or response_start is None:
        raise RuntimeError(f"identifier not found for {row['id']}")
    pointer_targets = {
        response_start + offset: prompt_start + offset
        for offset in range(len(pattern))
    }
    actions = []
    for index, token in enumerate(response_ids):
        if index in pointer_targets:
            actions.append(fixed_count + pointer_targets[index])
        else:
            actions.append(fixed_actions[token])
    actions.append(fixed_actions[eos_token_id])
    return prompt_ids, response_ids, actions, pattern, prompt_start


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    artifact_path = (
        args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    )
    evaluation_path = (
        args.evaluation
        if args.evaluation.is_absolute()
        else ROOT / args.evaluation
    )
    output_path = (
        args.output if args.output.is_absolute() else ROOT / args.output
    )
    device = torch.device(args.device)
    cake, artifact = load_semantic_action_plan_artifact(artifact_path)
    cake.to(device).eval()
    core, tokenizer, metadata = load_student(CHECKPOINT)
    core.to(device).eval()
    embedding = core.output_weight
    fixed_actions = cake.token_to_fixed_action()
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation_records = {
        record["id"]: record for record in evaluation["records"]
    }
    rows = [
        row for row in _load_rows(DATASET) if row["split"] == "validation"
    ]
    records = []
    totals = {
        "autonomous_action_units": 0,
        "autonomous_action_units_correct": 0,
        "autonomous_pointer_units": 0,
        "autonomous_pointer_units_correct": 0,
        "autonomous_fixed_units": 0,
        "autonomous_fixed_units_correct": 0,
        "autonomous_exact_action_sequences": 0,
        "teacher_forced_action_units": 0,
        "teacher_forced_action_units_correct": 0,
        "teacher_forced_realization_units": 0,
        "teacher_forced_realization_units_correct": 0,
        "teacher_forced_pointer_units": 0,
        "teacher_forced_pointer_realization_correct": 0,
        "previous_state_pointer_realization_correct": 0,
    }
    for row in rows:
        (
            prompt_ids,
            response_ids,
            expected_actions,
            identifier_pattern,
            prompt_identifier_start,
        ) = _expected(
            tokenizer,
            row,
            fixed_actions,
            cake.fixed_action_count,
            cake.eos_token_id,
        )
        autonomous = evaluation_records[row["id"]]["planned_actions"]
        compared = min(len(autonomous), len(expected_actions))
        action_correct = sum(
            autonomous[index] == expected_actions[index]
            for index in range(compared)
        )
        pointer_indexes = [
            index
            for index, action in enumerate(expected_actions)
            if action >= cake.fixed_action_count
        ]
        fixed_indexes = [
            index
            for index, action in enumerate(expected_actions)
            if action < cake.fixed_action_count
        ]
        autonomous_pointer_correct = sum(
            index < len(autonomous)
            and autonomous[index] == expected_actions[index]
            for index in pointer_indexes
        )
        autonomous_fixed_correct = sum(
            index < len(autonomous)
            and autonomous[index] == expected_actions[index]
            for index in fixed_indexes
        )
        exact = autonomous == expected_actions
        totals["autonomous_action_units"] += len(expected_actions)
        totals["autonomous_action_units_correct"] += action_correct
        totals["autonomous_pointer_units"] += len(pointer_indexes)
        totals["autonomous_pointer_units_correct"] += (
            autonomous_pointer_correct
        )
        totals["autonomous_fixed_units"] += len(fixed_indexes)
        totals["autonomous_fixed_units_correct"] += autonomous_fixed_correct
        totals["autonomous_exact_action_sequences"] += int(exact)

        ids = torch.tensor(
            [prompt_ids + response_ids], dtype=torch.long, device=device
        )
        prompt_result = core(
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            prompt_lengths=torch.tensor(
                [len(prompt_ids)], dtype=torch.long, device=device
            ),
        )
        result = core(ids, task_routes=prompt_result["task_routes"])
        states = result["hidden"][:, :-1]
        response_start = len(prompt_ids) - 1
        prompt_states = states[:, : response_start + 1]
        current_states = states[:, response_start:]
        current_states = torch.cat(
            (current_states, current_states[:, -1:].clone()), dim=1
        )
        target_tokens = torch.tensor(
            [response_ids + [cake.eos_token_id]],
            dtype=torch.long,
            device=device,
        )
        target_actions = torch.tensor(
            [expected_actions], dtype=torch.long, device=device
        )
        forward = cake.training_forward(
            prompt_states, current_states, target_actions
        )
        predicted_actions = forward["action_log_probs"].argmax(dim=-1)
        realized_tokens = F.linear(
            forward["adapted"], embedding
        ).argmax(dim=-1)
        teacher_action_correct = int(
            predicted_actions.eq(target_actions).sum()
        )
        realization_correct = int(
            realized_tokens.eq(target_tokens).sum()
        )
        pointer_tensor = torch.tensor(
            pointer_indexes, dtype=torch.long, device=device
        )
        pointer_realization_correct = int(
            realized_tokens[0, pointer_tensor]
            .eq(target_tokens[0, pointer_tensor])
            .sum()
        )
        decoded_pointer = forward["decoded"][0, pointer_tensor]
        current_pointer = current_states[0, pointer_tensor]
        source_positions = (
            target_actions[0, pointer_tensor] - cake.fixed_action_count
        )
        previous_positions = (source_positions - 1).clamp_min(0)
        previous_source = prompt_states[0, previous_positions]
        previous_value = cake.copy_semantic_value(
            cake.input_norm(previous_source)
        )
        previous_correction = cake.plan_correction(
            cake.output_norm(decoded_pointer)
        )
        previous_residual = cake.max_residual * torch.tanh(
            previous_value + previous_correction
        )
        previous_tokens = F.linear(
            current_pointer + previous_residual, embedding
        ).argmax(dim=-1)
        previous_correct = int(
            previous_tokens.eq(target_tokens[0, pointer_tensor]).sum()
        )
        totals["teacher_forced_action_units"] += len(expected_actions)
        totals["teacher_forced_action_units_correct"] += (
            teacher_action_correct
        )
        totals["teacher_forced_realization_units"] += len(expected_actions)
        totals["teacher_forced_realization_units_correct"] += (
            realization_correct
        )
        totals["teacher_forced_pointer_units"] += len(pointer_indexes)
        totals["teacher_forced_pointer_realization_correct"] += (
            pointer_realization_correct
        )
        totals["previous_state_pointer_realization_correct"] += (
            previous_correct
        )
        records.append(
            {
                "id": row["id"],
                "expected_action_units": len(expected_actions),
                "autonomous_action_units_correct": action_correct,
                "autonomous_exact_action_sequence": exact,
                "autonomous_pointer_units_correct": (
                    autonomous_pointer_correct
                ),
                "pointer_units": len(pointer_indexes),
                "teacher_forced_action_units_correct": (
                    teacher_action_correct
                ),
                "teacher_forced_realization_units_correct": (
                    realization_correct
                ),
                "teacher_forced_pointer_realization_correct": (
                    pointer_realization_correct
                ),
                "previous_state_pointer_realization_correct": (
                    previous_correct
                ),
                "identifier_token_units": len(identifier_pattern),
                "prompt_identifier_start": prompt_identifier_start,
            }
        )
    rates = {
        "autonomous_action_accuracy": (
            totals["autonomous_action_units_correct"]
            / totals["autonomous_action_units"]
        ),
        "autonomous_pointer_action_accuracy": (
            totals["autonomous_pointer_units_correct"]
            / totals["autonomous_pointer_units"]
        ),
        "autonomous_fixed_action_accuracy": (
            totals["autonomous_fixed_units_correct"]
            / totals["autonomous_fixed_units"]
        ),
        "autonomous_exact_action_sequence_rate": (
            totals["autonomous_exact_action_sequences"] / len(rows)
        ),
        "teacher_forced_action_accuracy": (
            totals["teacher_forced_action_units_correct"]
            / totals["teacher_forced_action_units"]
        ),
        "teacher_forced_realization_accuracy": (
            totals["teacher_forced_realization_units_correct"]
            / totals["teacher_forced_realization_units"]
        ),
        "teacher_forced_pointer_realization_accuracy": (
            totals["teacher_forced_pointer_realization_correct"]
            / totals["teacher_forced_pointer_units"]
        ),
        "previous_state_pointer_realization_accuracy": (
            totals["previous_state_pointer_realization_correct"]
            / totals["teacher_forced_pointer_units"]
        ),
    }
    evidence = {
        "format": "layercake-phase4-semantic-action-plan-diagnostic/1",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION_CREDIT",
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha(artifact_path),
        "payload_hash": artifact["payload_hash"],
        "evaluation": evaluation_path.relative_to(ROOT).as_posix(),
        "evaluation_sha256": _sha(evaluation_path),
        "core_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "dataset_sha256": _sha(DATASET),
        "distinct_prompts": len(rows),
        "totals": totals,
        "rates": rates,
        "records": records,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
