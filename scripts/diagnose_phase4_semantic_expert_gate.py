from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.semantic_action_plan import load_semantic_action_plan_artifact
from layercake.training.phase2_shallow_sparse import load_student
from layercake.training.phase4_python_cake import _canonical_sha
from scripts.train_phase4_semantic_action_plan import CHECKPOINT


ROOT = Path(__file__).resolve().parents[1]
COORDINATE = (
    ROOT
    / "artifacts/moonshot/phase4/candidates"
    / "python-semantic-coordinate-replacement-seed10840.pt"
)
PARENT = (
    ROOT
    / "artifacts/moonshot/phase4/candidates"
    / "python-semantic-action-plan-lexical-repair-seed10440.pt"
)
TRANSITION = (
    ROOT
    / "artifacts/moonshot/phase4/candidates"
    / "python-semantic-transition-only-seed10640.pt"
)
CACHE = (
    ROOT
    / "artifacts/moonshot/phase4/cache"
    / "seed9824-lexical-conformance-v1-validation-identity.safetensors"
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


def _top1(
    states: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return (states @ weight.T).argmax(dim=-1).eq(targets)


def _exact_rows(
    correct: torch.Tensor, offsets: list[int]
) -> tuple[int, int]:
    exact = sum(
        bool(correct[start:stop].all())
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    )
    return exact, len(offsets) - 1


def _threshold_candidates(values: torch.Tensor) -> list[float]:
    ordered = torch.unique(values.detach().cpu()).sort().values
    if not ordered.numel():
        return [0.0]
    candidates = [
        float(ordered[0]) - 1e-6,
        float(ordered[-1]) + 1e-6,
    ]
    if ordered.numel() > 1:
        midpoints = (ordered[:-1] + ordered[1:]) / 2
        candidates.extend(float(value) for value in midpoints)
    return candidates


def _fit_threshold(
    values: torch.Tensor,
    coordinate_correct: torch.Tensor,
    alternate_correct: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for direction in ("coordinate_if_ge", "coordinate_if_lt"):
        for threshold in _threshold_candidates(values[mask]):
            choose_coordinate = (
                values.ge(threshold)
                if direction == "coordinate_if_ge"
                else values.lt(threshold)
            )
            correct = torch.where(
                choose_coordinate, coordinate_correct, alternate_correct
            )
            score = int(correct[mask].sum())
            candidate = {
                "direction": direction,
                "threshold": threshold,
                "fit_correct": score,
                "fit_units": int(mask.sum()),
            }
            if best is None or (
                candidate["fit_correct"],
                -abs(candidate["threshold"]),
                candidate["direction"],
            ) > (
                best["fit_correct"],
                -abs(best["threshold"]),
                best["direction"],
            ):
                best = candidate
    if best is None:
        raise RuntimeError("threshold fit received no units")
    return best


def _apply_threshold(
    values: torch.Tensor,
    coordinate_correct: torch.Tensor,
    alternate_correct: torch.Tensor,
    rule: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    choose_coordinate = (
        values.ge(rule["threshold"])
        if rule["direction"] == "coordinate_if_ge"
        else values.lt(rule["threshold"])
    )
    return (
        torch.where(
            choose_coordinate, coordinate_correct, alternate_correct
        ),
        choose_coordinate,
    )


@torch.inference_mode()
def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    output = _rooted(args.output)
    if output.exists():
        raise RuntimeError(f"immutable output already exists: {output}")
    device = _device(args.device)
    coordinate, coordinate_artifact = (
        load_semantic_action_plan_artifact(COORDINATE)
    )
    parent, parent_artifact = load_semantic_action_plan_artifact(PARENT)
    transition, transition_artifact = (
        load_semantic_action_plan_artifact(TRANSITION)
    )
    coordinate.to(device).eval()
    parent.to(device).eval()
    transition.to(device).eval()
    core, _, metadata = load_student(CHECKPOINT)
    weight = core.output_weight.detach().float().to(device)
    del core
    cached = load_file(str(CACHE), device="cpu")
    selected = cached["selected_states"].float().to(device)
    previous = cached["previous_states"].float().to(device)
    current = cached["current_states"].float().to(device)
    targets = cached["target_ids"].long().to(device)
    rows = cached["row_indices"].long().to(device)
    offsets = cached["unit_offsets"].long().tolist()

    coordinate_raw = coordinate.copy_coordinate_output(
        F.gelu(
            coordinate.copy_coordinate_input(
                coordinate.copy_coordinate_norm(selected)
            )
        )
    )
    coordinate_desired = (
        coordinate.copy_coordinate_scale
        * F.normalize(coordinate_raw, dim=-1)
    )
    coordinate_states = current + coordinate.max_residual * torch.tanh(
        (coordinate_desired - current) / coordinate.max_residual
    )

    parent_value = parent.copy_semantic_value(
        parent.input_norm(selected)
    )
    parent_states = current + parent.max_residual * torch.tanh(parent_value)

    delta = transition.copy_transition_norm(selected - previous)
    transition_value = transition.copy_transition_output(
        F.gelu(transition.copy_transition_input(delta))
    )
    transition_states = current + transition.max_residual * torch.tanh(
        transition_value
    )

    coordinate_correct = _top1(coordinate_states, targets, weight)
    parent_correct = _top1(parent_states, targets, weight)
    transition_correct = _top1(transition_states, targets, weight)
    features = {
        "coordinate_selected_cosine": F.cosine_similarity(
            coordinate_raw, selected, dim=-1
        ),
        "coordinate_parent_cosine": F.cosine_similarity(
            coordinate_raw, parent_value, dim=-1
        ),
        "coordinate_transition_cosine": F.cosine_similarity(
            coordinate_raw, transition_value, dim=-1
        ),
        "coordinate_raw_norm": coordinate_raw.norm(dim=-1),
        "parent_value_norm": parent_value.norm(dim=-1),
        "transition_value_norm": transition_value.norm(dim=-1),
        "selected_state_norm": selected.norm(dim=-1),
        "coordinate_parent_norm_ratio": coordinate_raw.norm(dim=-1)
        / parent_value.norm(dim=-1).clamp_min(1e-8),
    }
    alternates = {
        "parent": parent_correct,
        "transition": transition_correct,
    }
    alternate_states = {
        "parent": parent_states,
        "transition": transition_states,
    }
    row_count = len(offsets) - 1
    folds = rows.remainder(args.folds)
    gate_records = []
    for feature_name, values in features.items():
        for alternate_name, alternate_correct in alternates.items():
            cross_validated = torch.zeros_like(coordinate_correct)
            chosen_coordinate = torch.zeros_like(coordinate_correct)
            fold_rules = []
            for fold in range(args.folds):
                fit_mask = folds.ne(fold)
                test_mask = folds.eq(fold)
                rule = _fit_threshold(
                    values,
                    coordinate_correct,
                    alternate_correct,
                    fit_mask,
                )
                correct, choices = _apply_threshold(
                    values,
                    coordinate_correct,
                    alternate_correct,
                    rule,
                )
                cross_validated[test_mask] = correct[test_mask]
                chosen_coordinate[test_mask] = choices[test_mask]
                fold_rules.append(
                    {
                        **rule,
                        "fold": fold,
                        "test_correct": int(correct[test_mask].sum()),
                        "test_units": int(test_mask.sum()),
                    }
                )
            full_mask = torch.ones_like(coordinate_correct)
            full_rule = _fit_threshold(
                values,
                coordinate_correct,
                alternate_correct,
                full_mask,
            )
            full_correct, full_choices = _apply_threshold(
                values,
                coordinate_correct,
                alternate_correct,
                full_rule,
            )
            cv_rows, _ = _exact_rows(cross_validated, offsets)
            full_rows, _ = _exact_rows(full_correct, offsets)
            gate_records.append(
                {
                    "feature": feature_name,
                    "alternate": alternate_name,
                    "cross_validated_correct_units": int(
                        cross_validated.sum()
                    ),
                    "cross_validated_units": int(
                        cross_validated.numel()
                    ),
                    "cross_validated_exact_rows": cv_rows,
                    "rows": row_count,
                    "cross_validated_coordinate_choices": int(
                        chosen_coordinate.sum()
                    ),
                    "full_fit_rule": full_rule,
                    "full_fit_correct_units": int(full_correct.sum()),
                    "full_fit_exact_rows": full_rows,
                    "full_fit_coordinate_choices": int(
                        full_choices.sum()
                    ),
                    "fold_rules": fold_rules,
                }
            )
    gate_records.sort(
        key=lambda record: (
            -record["cross_validated_exact_rows"],
            -record["cross_validated_correct_units"],
            record["feature"],
            record["alternate"],
        )
    )
    blend_records = []
    alphas = torch.linspace(0.0, 1.0, 101, device=device)
    for alternate_name, alternate_state in alternate_states.items():
        correct_by_alpha = torch.stack(
            [
                _top1(
                    alpha * coordinate_states
                    + (1.0 - alpha) * alternate_state,
                    targets,
                    weight,
                )
                for alpha in alphas
            ]
        )
        cross_validated = torch.zeros_like(coordinate_correct)
        fold_rules = []
        for fold in range(args.folds):
            fit_mask = folds.ne(fold)
            test_mask = folds.eq(fold)
            fit_scores = correct_by_alpha[:, fit_mask].sum(dim=1)
            best_score = int(fit_scores.max())
            eligible = torch.nonzero(
                fit_scores.eq(best_score), as_tuple=False
            ).flatten()
            selected_index = int(eligible[-1])
            cross_validated[test_mask] = correct_by_alpha[
                selected_index, test_mask
            ]
            fold_rules.append(
                {
                    "fold": fold,
                    "coordinate_alpha": float(alphas[selected_index]),
                    "fit_correct": best_score,
                    "fit_units": int(fit_mask.sum()),
                    "test_correct": int(
                        correct_by_alpha[selected_index, test_mask].sum()
                    ),
                    "test_units": int(test_mask.sum()),
                }
            )
        full_scores = correct_by_alpha.sum(dim=1)
        best_score = int(full_scores.max())
        eligible = torch.nonzero(
            full_scores.eq(best_score), as_tuple=False
        ).flatten()
        selected_index = int(eligible[-1])
        cv_rows, _ = _exact_rows(cross_validated, offsets)
        full_rows, _ = _exact_rows(
            correct_by_alpha[selected_index], offsets
        )
        blend_records.append(
            {
                "alternate": alternate_name,
                "cross_validated_correct_units": int(
                    cross_validated.sum()
                ),
                "cross_validated_units": int(
                    cross_validated.numel()
                ),
                "cross_validated_exact_rows": cv_rows,
                "rows": row_count,
                "full_fit_coordinate_alpha": float(
                    alphas[selected_index]
                ),
                "full_fit_correct_units": best_score,
                "full_fit_exact_rows": full_rows,
                "fold_rules": fold_rules,
            }
        )
    blend_records.sort(
        key=lambda record: (
            -record["cross_validated_exact_rows"],
            -record["cross_validated_correct_units"],
            record["alternate"],
        )
    )

    def expert_record(
        name: str, correct: torch.Tensor
    ) -> dict[str, Any]:
        exact, rows_total = _exact_rows(correct, offsets)
        return {
            "expert": name,
            "correct_units": int(correct.sum()),
            "units": int(correct.numel()),
            "exact_identity_rows": exact,
            "rows": rows_total,
        }

    union_parent = coordinate_correct | parent_correct
    union_transition = coordinate_correct | transition_correct
    evidence = {
        "format": "layercake-phase4-semantic-expert-gate-diagnostic/1",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION_CREDIT",
        "source_commit": _git_head(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "declared laptop CPU"
        ),
        "core_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "coordinate_artifact_sha256": _sha256(COORDINATE),
        "coordinate_payload_hash": coordinate_artifact["payload_hash"],
        "parent_artifact_sha256": _sha256(PARENT),
        "parent_payload_hash": parent_artifact["payload_hash"],
        "transition_artifact_sha256": _sha256(TRANSITION),
        "transition_payload_hash": transition_artifact["payload_hash"],
        "calibration_cache": CACHE.relative_to(ROOT).as_posix(),
        "calibration_cache_sha256": _sha256(CACHE),
        "calibration_rows": row_count,
        "calibration_identity_units": int(targets.numel()),
        "teacher_forced_current_states": True,
        "plan_correction_excluded_from_expert_proxy": True,
        "folds": args.folds,
        "experts": [
            expert_record("coordinate", coordinate_correct),
            expert_record("parent_linear", parent_correct),
            expert_record("transition_only", transition_correct),
            expert_record("coordinate_or_parent_oracle", union_parent),
            expert_record(
                "coordinate_or_transition_oracle", union_transition
            ),
        ],
        "threshold_gates": gate_records,
        "recommended_gate": gate_records[0],
        "constant_residual_blends": blend_records,
        "recommended_constant_blend": blend_records[0],
        "fresh_probe_accessed": False,
        "python_validation_accessed": False,
        "promotion_credit": 0,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = diagnose(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
