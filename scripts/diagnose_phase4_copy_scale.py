from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

import _common
from layercake.domain_runtime import AttentiveHostResidualCake
from layercake.training.phase4_python_cake import (
    _generate_code,
    _load_rows,
    load_student,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Non-promotable bounded diagnostic for a Phase 4 copy path."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cake-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--validation-index", type=int, default=0)
    parser.add_argument("--maximum-tokens", type=int, default=48)
    parser.add_argument(
        "--scale",
        type=float,
        action="append",
        default=[],
        help="Effective tanh(copy_alpha) value; may be repeated.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scales = args.scale or [-0.5, -0.1, 0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95]
    if any(not -1.0 < scale < 1.0 for scale in scales):
        raise ValueError("every effective scale must be strictly between -1 and 1")
    model, tokenizer, metadata = load_student(args.checkpoint)
    tensors = load_file(str(args.cake_checkpoint), device="cpu")
    cake = AttentiveHostResidualCake(
        d_abi=768,
        hidden_width=int(tensors["input.weight"].shape[0]),
        layers=len(
            {
                name.split(".")[1]
                for name in tensors
                if name.startswith("blocks.")
            }
        ),
        heads=6,
        expansion=int(
            tensors["blocks.0.feedforward.0.weight"].shape[0]
            / tensors["input.weight"].shape[0]
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
    rows = [
        row for row in _load_rows(args.dataset) if row["split"] == "validation"
    ]
    row = rows[args.validation_index]
    observations = []
    for scale in scales:
        with torch.no_grad():
            cake.copy_alpha.fill_(
                float(torch.atanh(torch.tensor(scale, dtype=torch.float32)))
            )
        generated = _generate_code(
            model,
            tokenizer,
            row["prompt"],
            cake=cake,
            maximum_tokens=args.maximum_tokens,
        )
        observations.append(
            {
                "effective_copy_scale": scale,
                "expected_function": row["function_name"],
                "generated_text": generated["text"],
                "requested_identifier_reproduced": (
                    row["function_name"] in generated["text"]
                ),
            }
        )
    result = {
        "format": "layercake-phase4-copy-scale-diagnostic/1",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION_CREDIT",
        "core_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "cake_checkpoint": str(args.cake_checkpoint),
        "split": "validation",
        "validation_index": args.validation_index,
        "observations": observations,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
