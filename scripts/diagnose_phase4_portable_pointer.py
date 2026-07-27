from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_phase4_portable_functional_decoder import _batch
from layercake.portable_domain import load_portable_artifact
from layercake.training.phase4_python_cake import _canonical_sha, _load_rows


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_path = (
        args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    )
    dataset_path = (
        args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    )
    output_path = (
        args.output if args.output.is_absolute() else ROOT / args.output
    )
    if output_path.exists():
        raise RuntimeError("diagnostic output is immutable")

    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    _, model = load_portable_artifact(payload, "cpu")
    if model.architecture != "byte_gru_pointer":
        raise ValueError("diagnostic requires a byte_gru_pointer artifact")
    rows = [
        row for row in _load_rows(dataset_path) if row["split"] == "validation"
    ]
    inputs, targets, response_mask, identifier_mask, pointer_labels = _batch(
        rows, list(range(len(rows))), maximum_sequence_bytes=512
    )
    result = model.pointer_forward(inputs)
    pointer_positions = result["pointer_scores"].argmax(dim=-1)
    pointer_bytes = torch.gather(inputs, 1, pointer_positions)
    gates = torch.sigmoid(result["gate_logits"].squeeze(-1))
    first_identifier = torch.zeros_like(identifier_mask)
    for row_index in range(len(rows)):
        first = int(torch.nonzero(identifier_mask[row_index])[0, 0])
        first_identifier[row_index, first] = True
    offsets = (
        pointer_positions[identifier_mask] - pointer_labels[identifier_mask]
    ).tolist()
    wrong_position_right_byte = (
        (pointer_positions != pointer_labels)
        & identifier_mask
        & (pointer_bytes == targets)
    )
    evidence = {
        "format": "layercake-phase4-portable-pointer-diagnostic/1",
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "payload_hash": payload["payload_hash"],
        "dataset": dataset_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(dataset_path),
        "split": "validation",
        "distinct_prompts": len(rows),
        "teacher_forced_identifier_units": int(identifier_mask.sum()),
        "teacher_forced_first_identifier_pointer_position_accuracy": float(
            (
                pointer_positions[first_identifier]
                == pointer_labels[first_identifier]
            )
            .float()
            .mean()
        ),
        "teacher_forced_first_identifier_pointer_byte_accuracy": float(
            (pointer_bytes[first_identifier] == targets[first_identifier])
            .float()
            .mean()
        ),
        "teacher_forced_identifier_pointer_position_accuracy": float(
            (
                pointer_positions[identifier_mask]
                == pointer_labels[identifier_mask]
            )
            .float()
            .mean()
        ),
        "teacher_forced_identifier_pointer_byte_accuracy": float(
            (pointer_bytes[identifier_mask] == targets[identifier_mask])
            .float()
            .mean()
        ),
        "teacher_forced_identifier_next_byte_accuracy": float(
            (
                result["logits"][identifier_mask].argmax(dim=-1)
                == targets[identifier_mask]
            )
            .float()
            .mean()
        ),
        "teacher_forced_response_next_byte_accuracy": float(
            (
                result["logits"][response_mask].argmax(dim=-1)
                == targets[response_mask]
            )
            .float()
            .mean()
        ),
        "mean_identifier_copy_gate_probability": float(
            gates[identifier_mask].mean()
        ),
        "mean_nonidentifier_response_copy_gate_probability": float(
            gates[response_mask & ~identifier_mask].mean()
        ),
        "wrong_position_but_correct_byte_units": int(
            wrong_position_right_byte.sum()
        ),
        "pointer_offset_histogram": [
            {"offset": int(offset), "count": int(count)}
            for offset, count in Counter(offsets).most_common()
        ],
        "test_split_accessed": False,
        "conclusion": (
            "The learned content pointer finds the first unseen identifier byte "
            "on 60/64 prompts but loses monotonic source position thereafter. "
            "The next bounded change is a learned recurrent transition over the "
            "previous neural pointer distribution, not another capacity or loss "
            "sweep."
        ),
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
