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
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.training.phase2_shallow_sparse import load_student
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _load_rows,
    _subsequence_start,
)
from scripts.train_phase4_semantic_action_plan import CHECKPOINT


ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    ROOT / "data" / "moonshot" / "phase4" / "lexical_conformance_v1.jsonl"
)
TRAIN_CACHE = (
    ROOT
    / "artifacts"
    / "moonshot"
    / "phase4"
    / "cache"
    / "seed9824-lexical-conformance-v1-train.safetensors"
)
DATASET_SHA256 = (
    "d6a7c054c1104c38007c97263031576c10c2a4dc7f78cf494a4f848a362e38bb"
)
TRAIN_CACHE_SHA256 = (
    "ee0488a5f407c1fbd1cb8383c8de5d27ee955042e05d9b1713b2b584051bcf79"
)
CORE_SHA256 = (
    "9e0e6b9add32b4c460f7b570a32584f380e59bf6d631e313ff813069d24e09e1"
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


@torch.inference_mode()
def cache_validation(args: argparse.Namespace) -> dict[str, Any]:
    output = _rooted(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    if _sha256(DATASET) != DATASET_SHA256:
        raise RuntimeError("lexical dataset identity changed")
    device = _device(args.device)
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    core, tokenizer, metadata = load_student(CHECKPOINT)
    if metadata["checkpoint"]["sha256"] != CORE_SHA256:
        raise RuntimeError("core checkpoint identity changed")
    core.to(device).eval()
    rows = [
        row for row in _load_rows(DATASET)
        if row["split"] == "validation"
    ]
    selected_states = []
    previous_states = []
    current_states = []
    target_ids = []
    row_indices = []
    unit_offsets = [0]
    for row_index, row in enumerate(rows):
        prompt_ids = tokenizer.encode(row["prompt"] + "\n")
        response_ids = tokenizer.encode(row["response"])
        prompt_tensor = torch.tensor(
            [prompt_ids], dtype=torch.long, device=device
        )
        prompt_result = core(
            prompt_tensor,
            prompt_lengths=torch.tensor(
                [len(prompt_ids)], dtype=torch.long, device=device
            ),
        )
        route = prompt_result["task_routes"]
        sequence = torch.tensor(
            [prompt_ids + response_ids],
            dtype=torch.long,
            device=device,
        )
        result = core(sequence, task_routes=route)
        states = result["hidden"][0, :-1]
        pattern = tokenizer.encode(" " + row["function_name"])
        prompt_start = _subsequence_start(prompt_ids, pattern)
        response_start = _subsequence_start(response_ids, pattern)
        if prompt_start is None or response_start is None:
            raise RuntimeError(
                f"identifier span absent for {row['id']}"
            )
        for offset, token_id in enumerate(pattern):
            source_position = prompt_start + offset
            response_position = (
                len(prompt_ids) - 1 + response_start + offset
            )
            selected_states.append(
                states[source_position].half().cpu()
            )
            previous_states.append(
                states[max(0, source_position - 1)].half().cpu()
            )
            current_states.append(
                states[response_position].half().cpu()
            )
            target_ids.append(token_id)
            row_indices.append(row_index)
        unit_offsets.append(len(target_ids))
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if (row_index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "cached_rows": row_index + 1,
                        "identity_units": len(target_ids),
                        "wall_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    tensors = {
        "selected_states": torch.stack(selected_states).contiguous(),
        "previous_states": torch.stack(previous_states).contiguous(),
        "current_states": torch.stack(current_states).contiguous(),
        "target_ids": torch.tensor(target_ids, dtype=torch.int64),
        "row_indices": torch.tensor(row_indices, dtype=torch.int64),
        "unit_offsets": torch.tensor(unit_offsets, dtype=torch.int64),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output))
    wall = time.perf_counter() - started
    evidence = {
        "format": "layercake-phase4-semantic-identity-validation-cache/1",
        "status": "DIAGNOSTIC_ONLY_NO_TRAINING_OR_PROMOTION_CREDIT",
        "source_commit": _git_head(),
        "dataset": DATASET.relative_to(ROOT).as_posix(),
        "dataset_sha256": DATASET_SHA256,
        "split": "validation",
        "rows": len(rows),
        "identity_units": len(target_ids),
        "unique_target_ids": len(set(target_ids)),
        "state_width": 768,
        "state_dtype": "float16",
        "core_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "core_parameters_changed": 0,
        "device": str(device),
        "device_name": (
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
        "cache": output.relative_to(ROOT).as_posix(),
        "cache_sha256": _sha256(output),
        "cache_bytes": output.stat().st_size,
        "used_as_optimizer_training_data": False,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output.with_suffix(".json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def _training_identity_units(
    cached: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    states = cached["semantic_states"].float()
    targets = cached["target_ids"].long()
    labels = cached["lexical_copy_labels"].long()
    offsets = cached["row_offsets"].long().tolist()
    selected = []
    previous = []
    current = []
    target_ids = []
    for row in range(len(offsets) - 1):
        start, stop = offsets[row], offsets[row + 1]
        row_labels = labels[start:stop]
        positions = torch.nonzero(
            row_labels.ge(0), as_tuple=False
        ).flatten()
        if not positions.numel():
            continue
        source_positions = row_labels[positions]
        selected.append(states[start + source_positions])
        previous.append(
            states[start + (source_positions - 1).clamp_min(0)]
        )
        current.append(states[start + positions])
        target_ids.append(targets[start + positions])
    return {
        "selected_states": torch.cat(selected),
        "previous_states": torch.cat(previous),
        "current_states": torch.cat(current),
        "target_ids": torch.cat(target_ids),
    }


def _ridge(
    source: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x = source.to(device=device, dtype=torch.float32)
    y = target.to(device=device, dtype=torch.float32)
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    x = x - x_mean
    y = y - y_mean
    covariance = x.T @ x
    ridge = 1e-3 * (
        covariance.diagonal().mean().clamp_min(1e-8)
    )
    mapping = torch.linalg.solve(
        covariance
        + ridge * torch.eye(
            covariance.shape[0],
            device=device,
            dtype=covariance.dtype,
        ),
        x.T @ y,
    )
    return {
        "x_mean": x_mean,
        "y_mean": y_mean,
        "mapping": mapping,
        "ridge": ridge,
    }


def _predict(
    fit: dict[str, torch.Tensor],
    source: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x = source.to(device=device, dtype=torch.float32)
    return (
        (x - fit["x_mean"]) @ fit["mapping"] + fit["y_mean"]
    )


def _top1(
    states: torch.Tensor,
    targets: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    normalize_states: bool = False,
    normalize_weight: bool = False,
    batch_size: int = 128,
) -> tuple[int, list[int]]:
    weight = (
        F.normalize(output_weight, dim=-1)
        if normalize_weight
        else output_weight
    )
    predictions = []
    for start in range(0, states.shape[0], batch_size):
        batch = states[start : start + batch_size]
        if normalize_states:
            batch = F.normalize(batch, dim=-1)
        predictions.extend((batch @ weight.T).argmax(dim=-1).tolist())
    correct = sum(
        prediction == target
        for prediction, target in zip(
            predictions, targets.tolist(), strict=True
        )
    )
    return correct, predictions


def _replacement_states(
    predicted_coordinates: torch.Tensor,
    current_states: torch.Tensor,
    scale: float,
    max_residual: float = 8.0,
) -> torch.Tensor:
    desired = scale * F.normalize(predicted_coordinates, dim=-1)
    current = current_states.to(
        device=desired.device, dtype=desired.dtype
    )
    residual = max_residual * torch.tanh(
        (desired - current) / max_residual
    )
    return current + residual


@torch.inference_mode()
def audit(args: argparse.Namespace) -> dict[str, Any]:
    validation_cache = _rooted(args.validation_cache)
    output = _rooted(args.output)
    if output.exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    if _sha256(TRAIN_CACHE) != TRAIN_CACHE_SHA256:
        raise RuntimeError("training cache identity changed")
    device = _device(args.device)
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    core, _, metadata = load_student(CHECKPOINT)
    if metadata["checkpoint"]["sha256"] != CORE_SHA256:
        raise RuntimeError("core identity changed")
    output_weight = core.output_weight.detach().float().to(device)
    normalized_weight = F.normalize(output_weight, dim=-1)
    del core
    train = _training_identity_units(
        load_file(str(TRAIN_CACHE), device="cpu")
    )
    validation = load_file(str(validation_cache), device="cpu")
    train_targets = train["target_ids"].long()
    validation_targets = validation["target_ids"].long()
    train_coordinates = normalized_weight[train_targets.to(device)].cpu()

    scale_sample_count = min(4096, train_targets.shape[0])
    generator = torch.Generator().manual_seed(10740)
    scale_sample = torch.randperm(
        train_targets.shape[0], generator=generator
    )[:scale_sample_count]
    candidate_scales = (1.0, 2.0, 4.0, 8.0, 16.0)
    scale_records = []
    for scale in candidate_scales:
        states = _replacement_states(
            train_coordinates[scale_sample].to(device),
            train["current_states"][scale_sample],
            scale,
        )
        correct, _ = _top1(
            states,
            train_targets[scale_sample],
            output_weight,
        )
        scale_records.append(
            {
                "scale": scale,
                "correct": correct,
                "units": scale_sample_count,
                "accuracy": correct / scale_sample_count,
            }
        )
    selected_scale = min(
        scale_records,
        key=lambda record: (-record["correct"], record["scale"]),
    )["scale"]

    selected_fit = _ridge(
        train["selected_states"], train_coordinates, device
    )
    transition_fit = _ridge(
        train["selected_states"] - train["previous_states"],
        train_coordinates,
        device,
    )
    validation_coordinates = normalized_weight[
        validation_targets.to(device)
    ]
    oracle_states = _replacement_states(
        validation_coordinates,
        validation["current_states"],
        selected_scale,
    )
    oracle_correct, oracle_predictions = _top1(
        oracle_states,
        validation_targets,
        output_weight,
    )
    selected_prediction = _predict(
        selected_fit, validation["selected_states"], device
    )
    transition_prediction = _predict(
        transition_fit,
        validation["selected_states"] - validation["previous_states"],
        device,
    )
    selected_coordinate_correct, selected_coordinate_predictions = _top1(
        selected_prediction,
        validation_targets,
        output_weight,
        normalize_states=True,
        normalize_weight=True,
    )
    transition_coordinate_correct, transition_coordinate_predictions = _top1(
        transition_prediction,
        validation_targets,
        output_weight,
        normalize_states=True,
        normalize_weight=True,
    )
    selected_states = _replacement_states(
        selected_prediction,
        validation["current_states"],
        selected_scale,
    )
    transition_states = _replacement_states(
        transition_prediction,
        validation["current_states"],
        selected_scale,
    )
    selected_replacement_correct, selected_replacement_predictions = _top1(
        selected_states,
        validation_targets,
        output_weight,
    )
    (
        transition_replacement_correct,
        transition_replacement_predictions,
    ) = _top1(
        transition_states,
        validation_targets,
        output_weight,
    )
    unit_count = validation_targets.shape[0]
    unique_targets = sorted(set(validation_targets.tolist()))
    raw_self_correct, _ = _top1(
        output_weight[unique_targets],
        torch.tensor(unique_targets, dtype=torch.long),
        output_weight,
    )
    normalized_self_correct, _ = _top1(
        normalized_weight[unique_targets],
        torch.tensor(unique_targets, dtype=torch.long),
        normalized_weight,
    )
    train_target_set = set(train_targets.tolist())
    heldout_only_mask = [
        int(target) not in train_target_set
        for target in validation_targets.tolist()
    ]
    heldout_only_units = sum(heldout_only_mask)

    def heldout_correct(predictions: list[int]) -> int:
        return sum(
            include and prediction == int(target)
            for include, prediction, target in zip(
                heldout_only_mask,
                predictions,
                validation_targets.tolist(),
                strict=True,
            )
        )

    def exact_rows(predictions: list[int]) -> int:
        by_row: dict[int, list[bool]] = {}
        for row_index, prediction, target in zip(
            validation["row_indices"].tolist(),
            predictions,
            validation_targets.tolist(),
            strict=True,
        ):
            by_row.setdefault(int(row_index), []).append(
                prediction == int(target)
            )
        return sum(all(values) for values in by_row.values())

    row_count = int(validation["unit_offsets"].numel() - 1)
    wall = time.perf_counter() - started
    records = [
        {
            "unit": index,
            "row_index": int(validation["row_indices"][index]),
            "target_id": int(validation_targets[index]),
            "oracle_prediction": oracle_predictions[index],
            "selected_coordinate_prediction": (
                selected_coordinate_predictions[index]
            ),
            "transition_coordinate_prediction": (
                transition_coordinate_predictions[index]
            ),
            "selected_replacement_prediction": (
                selected_replacement_predictions[index]
            ),
            "transition_replacement_prediction": (
                transition_replacement_predictions[index]
            ),
        }
        for index in range(unit_count)
    ]
    evidence = {
        "format": "layercake-phase4-semantic-identity-recoverability-audit/1",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION_CREDIT",
        "source_commit": _git_head(),
        "core_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "train_cache": TRAIN_CACHE.relative_to(ROOT).as_posix(),
        "train_cache_sha256": TRAIN_CACHE_SHA256,
        "validation_cache": validation_cache.relative_to(ROOT).as_posix(),
        "validation_cache_sha256": _sha256(validation_cache),
        "fit_split": "train",
        "evaluation_split": "validation",
        "fit_identity_units": int(train_targets.shape[0]),
        "evaluation_identity_units": int(unit_count),
        "evaluation_unique_target_ids": len(unique_targets),
        "target_id_sets_disjoint": not bool(
            train_target_set & set(validation_targets.tolist())
        ),
        "heldout_only_identity_units": heldout_only_units,
        "ridge_rule": {
            "formula": "lambda = 1e-3 * mean diagonal of centered X transpose X",
            "selected_state_lambda": float(selected_fit["ridge"]),
            "transition_state_lambda": float(transition_fit["ridge"]),
            "validation_used_to_choose_lambda": False,
        },
        "bounded_state_replacement": {
            "maximum_residual": 8.0,
            "candidate_scales": list(candidate_scales),
            "scale_fit_sample_seed": 10740,
            "scale_fit_sample_units": scale_sample_count,
            "training_scale_records": scale_records,
            "selected_scale": selected_scale,
            "validation_used_to_choose_scale": False,
        },
        "heldout_tied_coordinate_self_recovery": {
            "raw_correct": raw_self_correct,
            "normalized_correct": normalized_self_correct,
            "unique_units": len(unique_targets),
            "normalized_accuracy": normalized_self_correct
            / len(unique_targets),
        },
        "heldout_oracle_state_replacement": {
            "correct": oracle_correct,
            "units": unit_count,
            "accuracy": oracle_correct / unit_count,
            "heldout_only_correct": heldout_correct(oracle_predictions),
            "heldout_only_units": heldout_only_units,
            "heldout_only_accuracy": (
                heldout_correct(oracle_predictions) / heldout_only_units
            ),
            "exact_rows": exact_rows(oracle_predictions),
            "rows": row_count,
            "exact_row_rate": exact_rows(oracle_predictions) / row_count,
        },
        "heldout_affine_selected_state_coordinate_recovery": {
            "correct": selected_coordinate_correct,
            "units": unit_count,
            "accuracy": selected_coordinate_correct / unit_count,
            "heldout_only_correct": heldout_correct(
                selected_coordinate_predictions
            ),
            "heldout_only_units": heldout_only_units,
            "heldout_only_accuracy": (
                heldout_correct(selected_coordinate_predictions)
                / heldout_only_units
            ),
            "exact_rows": exact_rows(selected_coordinate_predictions),
            "rows": row_count,
            "exact_row_rate": (
                exact_rows(selected_coordinate_predictions) / row_count
            ),
        },
        "heldout_affine_transition_coordinate_recovery": {
            "correct": transition_coordinate_correct,
            "units": unit_count,
            "accuracy": transition_coordinate_correct / unit_count,
            "heldout_only_correct": heldout_correct(
                transition_coordinate_predictions
            ),
            "heldout_only_units": heldout_only_units,
            "heldout_only_accuracy": (
                heldout_correct(transition_coordinate_predictions)
                / heldout_only_units
            ),
            "exact_rows": exact_rows(transition_coordinate_predictions),
            "rows": row_count,
            "exact_row_rate": (
                exact_rows(transition_coordinate_predictions) / row_count
            ),
        },
        "heldout_affine_selected_state_bounded_replacement": {
            "correct": selected_replacement_correct,
            "units": unit_count,
            "accuracy": selected_replacement_correct / unit_count,
            "heldout_only_correct": heldout_correct(
                selected_replacement_predictions
            ),
            "heldout_only_units": heldout_only_units,
            "heldout_only_accuracy": (
                heldout_correct(selected_replacement_predictions)
                / heldout_only_units
            ),
            "exact_rows": exact_rows(selected_replacement_predictions),
            "rows": row_count,
            "exact_row_rate": (
                exact_rows(selected_replacement_predictions) / row_count
            ),
        },
        "heldout_affine_transition_bounded_replacement": {
            "correct": transition_replacement_correct,
            "units": unit_count,
            "accuracy": transition_replacement_correct / unit_count,
            "heldout_only_correct": heldout_correct(
                transition_replacement_predictions
            ),
            "heldout_only_units": heldout_only_units,
            "heldout_only_accuracy": (
                heldout_correct(transition_replacement_predictions)
                / heldout_only_units
            ),
            "exact_rows": exact_rows(transition_replacement_predictions),
            "rows": row_count,
            "exact_row_rate": (
                exact_rows(transition_replacement_predictions) / row_count
            ),
        },
        "device": str(device),
        "device_name": (
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
        "peak_process_resident_memory_bytes": max(
            peak_rss, int(process.memory_info().rss)
        ),
        "records": records,
        "validation_used_for_optimizer_training": False,
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
    cache_parser = subparsers.add_parser("cache-validation")
    cache_parser.add_argument("--output", type=Path, required=True)
    cache_parser.add_argument("--device", default="cuda:0")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument(
        "--validation-cache", type=Path, required=True
    )
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = (
        cache_validation(args)
        if args.command == "cache-validation"
        else audit(args)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
