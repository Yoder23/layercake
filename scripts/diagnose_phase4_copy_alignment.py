from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import _common
from layercake.domain_runtime import AttentiveHostResidualCake
from layercake.training.phase4_python_cake import (
    _causal_copy_labels,
    _load_rows,
    load_student,
)


def _subsequence_start(values: list[int], pattern: list[int]) -> int | None:
    for start in range(len(values) - len(pattern) + 1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher-forced decomposition of Phase 4 lexical copying."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cake-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--maximum-response-tokens", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer, metadata = load_student(args.checkpoint)
    tensors = load_file(str(args.cake_checkpoint), device="cpu")
    hidden_width = int(tensors["input.weight"].shape[0])
    cake = AttentiveHostResidualCake(
        d_abi=768,
        hidden_width=hidden_width,
        layers=len(
            {
                name.split(".")[1]
                for name in tensors
                if name.startswith("blocks.")
            }
        ),
        heads=6,
        expansion=int(
            tensors["blocks.0.feedforward.0.weight"].shape[0] / hidden_width
        ),
        copy_width=int(tensors["copy_query.weight"].shape[0]),
        copy_value_projection=(
            "copy_value.weight" in tensors
            or "copy_transition_value.weight" in tensors
        ),
        selective_copy="copy_gate.weight" in tensors,
        transition_copy="copy_transition_value.weight" in tensors,
    )
    cake.load_state_dict(tensors, strict=True)
    cake.eval()
    embedding = model.output_weight.detach().float()
    rows = [
        row for row in _load_rows(args.dataset) if row["split"] == args.split
    ]

    records = []
    pointer_correct = 0
    value_correct = 0
    attended_value_correct = 0
    copyable_units = 0
    identifier_pointer_correct = 0
    identifier_value_correct = 0
    identifier_attended_correct = 0
    identifier_units = 0
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"] + "\n")
        response_ids = tokenizer.encode(row["response"])[
            : args.maximum_response_tokens
        ]
        prompt_result = model(
            torch.tensor([prompt_ids], dtype=torch.long),
            prompt_lengths=torch.tensor([len(prompt_ids)]),
            use_cache=False,
        )
        ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
        result = model(ids, task_routes=prompt_result["task_routes"])
        states = result["hidden"][:, :-1].float()
        targets = ids[:, 1:]
        valid = torch.ones_like(targets, dtype=torch.bool)
        response_mask = torch.zeros_like(valid)
        response_mask[:, len(prompt_ids) - 1 :] = True
        labels = _causal_copy_labels(targets, valid, response_mask)
        copyable = labels >= 0
        count = int(copyable.sum())
        if not count:
            continue

        scores = cake.copy_scores(states)
        predicted_positions = scores[copyable].argmax(dim=-1)
        true_positions = labels[copyable]
        row_pointer_correct = int(
            (predicted_positions == true_positions).sum()
        )
        batch_indexes, _ = torch.nonzero(copyable, as_tuple=True)
        true_values = cake.project_copy_positions(
            states,
            batch_indexes,
            true_positions,
        )
        value_predictions = F.linear(true_values, embedding).argmax(dim=-1)
        expected = targets[copyable]
        row_value_correct = int((value_predictions == expected).sum())

        attended = cake._copy_context(states)
        assert attended is not None
        attended_predictions = F.linear(
            attended[copyable], embedding
        ).argmax(dim=-1)
        row_attended_correct = int(
            (attended_predictions == expected).sum()
        )

        identifier_pattern = tokenizer.encode(" " + row["function_name"])
        prompt_identifier_start = _subsequence_start(
            prompt_ids, identifier_pattern
        )
        response_identifier_start = _subsequence_start(
            response_ids, identifier_pattern
        )
        if (
            prompt_identifier_start is None
            or response_identifier_start is None
        ):
            identifier_pattern = tokenizer.encode(row["function_name"])
            prompt_identifier_start = _subsequence_start(
                prompt_ids, identifier_pattern
            )
            response_identifier_start = _subsequence_start(
                response_ids, identifier_pattern
            )
        if (
            prompt_identifier_start is None
            or response_identifier_start is None
        ):
            raise RuntimeError(
                f"function identifier tokens not found for {row['id']}"
            )
        identifier_positions = torch.tensor(
            [
                len(prompt_ids)
                - 1
                + response_identifier_start
                + offset
                for offset in range(len(identifier_pattern))
            ],
            dtype=torch.long,
        )
        identifier_count = int(identifier_positions.numel())
        identifier_true_positions = torch.tensor(
            [
                prompt_identifier_start + offset
                for offset in range(identifier_count)
            ],
            dtype=torch.long,
        )
        identifier_pointer_predictions = scores[
            0, identifier_positions
        ].argmax(dim=-1)
        row_identifier_pointer = int(
            (
                identifier_pointer_predictions
                == identifier_true_positions
            ).sum()
        )
        identifier_values = cake.project_copy_positions(
            states,
            torch.zeros_like(identifier_true_positions),
            identifier_true_positions,
        )
        identifier_expected = targets[0, identifier_positions]
        row_identifier_value = int(
            (
                F.linear(identifier_values, embedding).argmax(dim=-1)
                == identifier_expected
            ).sum()
        )
        row_identifier_attended = int(
            (
                F.linear(
                    attended[0, identifier_positions], embedding
                ).argmax(dim=-1)
                == identifier_expected
            ).sum()
        )
        copyable_units += count
        pointer_correct += row_pointer_correct
        value_correct += row_value_correct
        attended_value_correct += row_attended_correct
        identifier_units += identifier_count
        identifier_pointer_correct += row_identifier_pointer
        identifier_value_correct += row_identifier_value
        identifier_attended_correct += row_identifier_attended
        records.append(
            {
                "id": row["id"],
                "copyable_response_units": count,
                "pointer_top1_correct": row_pointer_correct,
                "true_source_value_top1_correct": row_value_correct,
                "attended_value_top1_correct": row_attended_correct,
                "copyable_identifier_units": identifier_count,
                "identifier_pointer_top1_correct": row_identifier_pointer,
                "identifier_true_source_value_top1_correct": (
                    row_identifier_value
                ),
                "identifier_attended_value_top1_correct": (
                    row_identifier_attended
                ),
            }
        )

    result = {
        "format": "layercake-phase4-copy-alignment-diagnostic/2",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION_CREDIT",
        "identifier_label_protocol": (
            "exact monotonic prompt/response function-identifier spans"
        ),
        "split": args.split,
        "distinct_prompts": len(rows),
        "core_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "cake_checkpoint_sha256": hashlib.sha256(
            args.cake_checkpoint.read_bytes()
        ).hexdigest(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "copyable_response_units": copyable_units,
        "pointer_top1_accuracy": pointer_correct / max(1, copyable_units),
        "true_source_value_top1_accuracy": value_correct
        / max(1, copyable_units),
        "attended_value_top1_accuracy": attended_value_correct
        / max(1, copyable_units),
        "copyable_identifier_units": identifier_units,
        "identifier_pointer_top1_accuracy": identifier_pointer_correct
        / max(1, identifier_units),
        "identifier_true_source_value_top1_accuracy": identifier_value_correct
        / max(1, identifier_units),
        "identifier_attended_value_top1_accuracy": identifier_attended_correct
        / max(1, identifier_units),
        "records": records,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
