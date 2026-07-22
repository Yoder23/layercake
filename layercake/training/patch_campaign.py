"""Locked variable-patch representation campaign for the final moonshot audit.

This campaign is deliberately validation-only.  It compares the frozen routed
foundation with dense variable-patch controls at comparable total scale.  The
controls answer one narrow question: is fixed byte patching the measured source
of the English quality gap?  They are diagnostics, not promotable sparse cores.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from layercake.causal_byte_models import (
    CausalAdaptiveBytePatchLM,
    CausalVariableBytePatchLM,
)
from .data import ByteCorpus, sha256_file
from .foundation import train_english_core


def _canonical_hash(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_model(candidate: dict) -> torch.nn.Module:
    kind = str(candidate["kind"])
    arguments = copy.deepcopy(candidate.get("model", {}))
    if kind == "variable":
        return CausalVariableBytePatchLM(**arguments)
    if kind == "adaptive_two_four":
        return CausalAdaptiveBytePatchLM(**arguments)
    raise ValueError(f"unsupported patch candidate kind: {kind}")


def _forward(model: torch.nn.Module, inputs: torch.Tensor):
    original_length = inputs.shape[1]
    if isinstance(model, CausalAdaptiveBytePatchLM):
        # The adaptive implementation consumes 2/4-byte patches and fixed local
        # windows.  Right padding is causally invisible to retained logits.
        multiple = math.lcm(4, int(model.local_window))
        pad = (-original_length) % multiple
        if pad:
            inputs = F.pad(inputs, (0, pad))
    logits, _, metadata = model(inputs)
    return logits[:, :original_length], metadata


def _real_patch_count(metadata: dict, input_length: int) -> int:
    patch_ids = metadata["patch_ids"][:, :input_length]
    return int((patch_ids.max(dim=1).values + 1).sum())


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    corpus: ByteCorpus,
    *,
    batch_size: int,
    sequence_bytes: int,
    batches: int,
    device: torch.device,
) -> dict:
    model.eval()
    loss_sum = 0.0
    correct = 0
    byte_count = 0
    patch_count = 0
    for row in corpus.fixed_batches(
        batch_size=batch_size,
        sequence_bytes=sequence_bytes,
        batches=batches,
        device=device,
    ):
        inputs, targets = row[:, :-1], row[:, 1:]
        logits, metadata = _forward(model, inputs)
        loss_sum += float(F.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten(), reduction="sum"
        ))
        correct += int((logits.argmax(-1) == targets).sum())
        byte_count += targets.numel()
        patch_count += _real_patch_count(metadata, inputs.shape[1])
    model.train()
    nats = loss_sum / max(1, byte_count)
    return {
        "bits_per_byte": nats / 0.6931471805599453,
        "byte_accuracy": correct / max(1, byte_count),
        "evaluated_bytes": byte_count,
        "patches": patch_count,
        "mean_bytes_per_patch": byte_count / max(1, patch_count),
    }


def _train_patch_candidate(
    candidate: dict,
    config: dict,
    *,
    seed: int,
    output: Path,
) -> dict:
    run_fingerprint = _canonical_hash({
        "candidate": candidate,
        "seed": seed,
        "device": config.get("device"),
        "precision": config.get("precision"),
        "training": config["training"],
        "evaluation": config["evaluation"],
        "data": config["data"],
    })
    existing_path = output / "metadata.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        historical_reported_bytes = (
            int(config["training"]["steps"])
            * int(config["training"]["batch_size"])
            * (int(config["training"]["sequence_bytes"]) - 1)
        )
        actual_bytes = (
            int(config["training"]["steps"])
            * int(config["training"]["batch_size"])
            * int(config["training"]["sequence_bytes"])
        )
        if (
            existing.get("status") == "PASS"
            and existing.get("candidate") == candidate
            and int(existing.get("seed", -1)) == seed
            and existing.get("run_fingerprint", run_fingerprint) == run_fingerprint
            and int(existing.get("training", {}).get("raw_bytes_seen", -1))
            in {historical_reported_bytes, actual_bytes}
            and not existing.get("quality", {}).get("test_accessed", True)
        ):
            if existing["training"]["raw_bytes_seen"] != actual_bytes:
                existing["training"]["historical_underreported_raw_bytes_seen"] = (
                    existing["training"]["raw_bytes_seen"]
                )
                existing["training"]["raw_bytes_seen"] = actual_bytes
                existing_path.write_text(
                    json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8"
                )
            return existing
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    requested = str(config.get("device", "cuda"))
    device = torch.device(
        "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
    )
    precision = str(config.get("precision", "fp32"))
    model = _build_model(candidate).to(device)
    train = ByteCorpus(config["data"]["train"])
    selection = ByteCorpus(config["data"]["architecture_selection"])
    validation = ByteCorpus(config["data"]["validation"])
    training = config["training"]
    evaluation = config["evaluation"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and precision == "fp16"
        else nullcontext()
    )
    steps = int(training["steps"])
    batch_size = int(training["batch_size"])
    sequence_bytes = int(training["sequence_bytes"])
    interval = int(training["evaluation_interval"])
    curves = []
    peak_memory = 0
    routed = getattr(model, "routed", None)
    routing_assignments = (
        torch.zeros(routed.expert_count, dtype=torch.long, device=device)
        if routed is not None else None
    )
    routing_entropy = 0.0
    routing_observations = 0
    expert_gradient_steps = (
        [0 for _ in range(routed.expert_count)] if routed is not None else []
    )
    resume_enabled = bool(training.get("resumable", False))
    resume_metadata_path = output / "resume.json"
    resume_model_path = output / "resume-model.safetensors"
    resume_optimizer_path = output / "resume-optimizer.pt"
    start_step = 0
    previous_wall_seconds = 0.0
    output.mkdir(parents=True, exist_ok=True)
    if resume_enabled and resume_metadata_path.is_file():
        resume = json.loads(resume_metadata_path.read_text(encoding="utf-8"))
        if resume.get("run_fingerprint") != run_fingerprint:
            raise ValueError("adaptive resume checkpoint belongs to a different locked run")
        model.load_state_dict(load_file(str(resume_model_path), device=str(device)), strict=True)
        training_state = torch.load(
            resume_optimizer_path, map_location=device, weights_only=True
        )
        optimizer.load_state_dict(training_state["optimizer"])
        scaler.load_state_dict(training_state["scaler"])
        start_step = int(resume["step"])
        previous_wall_seconds = float(resume["wall_seconds"])
        curves = list(resume.get("curves", []))
        peak_memory = int(resume.get("peak_memory", 0))
        if routed is not None:
            routing_assignments += torch.tensor(
                resume.get("routing_assignments", [0] * routed.expert_count),
                dtype=torch.long,
                device=device,
            )
            routing_entropy = float(resume.get("routing_entropy", 0.0))
            routing_observations = int(resume.get("routing_observations", 0))
            expert_gradient_steps = list(resume.get(
                "expert_gradient_steps", expert_gradient_steps
            ))
    started = time.perf_counter()
    for step, row in enumerate(train.batches(
        batch_size=batch_size,
        sequence_bytes=sequence_bytes,
        steps=steps,
        seed=seed,
        device=device,
    ), start=1):
        if step <= start_step:
            continue
        inputs, targets = row[:, :-1], row[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits, metadata = _forward(model, inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            routing = metadata.get("routing")
            if routing is not None:
                loss = loss + float(training.get("routing_balance_weight", 0.02)) * routing["balance_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if routed is not None:
            for expert_index, expert in enumerate(routed.experts):
                if any(parameter.grad is not None for parameter in expert.parameters()):
                    expert_gradient_steps[expert_index] += 1
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training.get("gradient_clip", 1.0))
        )
        scaler.step(optimizer)
        scaler.update()
        if metadata.get("routing") is not None:
            routing_assignments += metadata["routing"]["assignment_counts"].detach().long()
            routing_entropy += float(metadata["routing"]["normalized_entropy"].detach())
            routing_observations += 1
        if device.type == "cuda":
            peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
        if step == 1 or step % interval == 0 or step == steps:
            diagnostic = _evaluate(
                model,
                validation,
                batch_size=int(evaluation["batch_size"]),
                sequence_bytes=int(evaluation["sequence_bytes"]),
                batches=int(evaluation["batches"]),
                device=device,
            )
            curves.append({
                "step": step,
                "training_loss": float(loss.detach()),
                "validation": diagnostic,
                "wall_seconds": previous_wall_seconds + time.perf_counter() - started,
                "training_mean_bytes_per_patch": (
                    inputs.numel() / max(1, _real_patch_count(metadata, inputs.shape[1]))
                ),
            })
            if resume_enabled:
                model_temp = output / "resume-model.tmp.safetensors"
                optimizer_temp = output / "resume-optimizer.tmp.pt"
                metadata_temp = output / "resume.tmp.json"
                save_file(
                    {
                        name: value.detach().cpu().contiguous()
                        for name, value in model.state_dict().items()
                    },
                    str(model_temp),
                )
                torch.save(
                    {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
                    optimizer_temp,
                )
                metadata_temp.write_text(json.dumps({
                    "format": "layercake-adaptive-core-resume/1",
                    "run_fingerprint": run_fingerprint,
                    "step": step,
                    "wall_seconds": curves[-1]["wall_seconds"],
                    "curves": curves,
                    "peak_memory": peak_memory,
                    "routing_assignments": (
                        routing_assignments.cpu().tolist()
                        if routing_assignments is not None else None
                    ),
                    "routing_entropy": routing_entropy,
                    "routing_observations": routing_observations,
                    "expert_gradient_steps": expert_gradient_steps,
                }, indent=2, sort_keys=True), encoding="utf-8")
                model_temp.replace(resume_model_path)
                optimizer_temp.replace(resume_optimizer_path)
                metadata_temp.replace(resume_metadata_path)
    wall_seconds = previous_wall_seconds + time.perf_counter() - started
    selection_metrics = _evaluate(
        model,
        selection,
        batch_size=int(evaluation["batch_size"]),
        sequence_bytes=int(evaluation["sequence_bytes"]),
        batches=int(evaluation["batches"]),
        device=device,
    )
    validation_metrics = _evaluate(
        model,
        validation,
        batch_size=int(evaluation["batch_size"]),
        sequence_bytes=int(evaluation["sequence_bytes"]),
        batches=int(evaluation["batches"]),
        device=device,
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(checkpoint),
    )
    if routed is None:
        active_parameters = sum(parameter.numel() for parameter in model.parameters())
        routing_report = None
    else:
        expert_sizes = [sum(parameter.numel() for parameter in expert.parameters()) for expert in routed.experts]
        inactive_expert_parameters = sum(expert_sizes)
        active_experts = 2 if routed.mode in {"learned_top2", "expert_choice"} else 1
        active_parameters = (
            sum(parameter.numel() for parameter in model.parameters())
            - inactive_expert_parameters
            + sum(sorted(expert_sizes, reverse=True)[:active_experts])
        )
        assignment_total = int(routing_assignments.sum())
        utilization = (
            (routing_assignments.double() / max(1, assignment_total)).cpu().tolist()
        )
        minimum_steps = max(1, int(0.01 * steps))
        routing_report = {
            "mode": routed.mode,
            "causal_patch_routing": True,
            "assignment_counts": routing_assignments.cpu().tolist(),
            "utilization": utilization,
            "maximum_load_fraction": max(utilization),
            "experts_used": sum(value > 0 for value in utilization),
            "expert_gradient_steps": expert_gradient_steps,
            "all_experts_meaningfully_trained": all(
                value >= minimum_steps for value in expert_gradient_steps
            ),
            "mean_normalized_entropy": routing_entropy / max(1, routing_observations),
            "router_collapsed": max(utilization) > float(training.get("router_collapse_threshold", 0.5)),
        }
    metadata = {
        "format": "layercake-variable-patch-run/1",
        "status": "PASS",
        "candidate": candidate,
        "seed": seed,
        "run_fingerprint": run_fingerprint,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "active_parameters": active_parameters,
        "routing": routing_report,
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "sequence_bytes": sequence_bytes,
            "raw_bytes_seen": steps * batch_size * sequence_bytes,
            "wall_seconds": wall_seconds,
            "curves": curves,
            "resumable": resume_enabled,
            "resumed_from_step": start_step,
            "resume_metadata": str(resume_metadata_path.resolve()) if resume_enabled else None,
        },
        "quality": {
            "architecture_selection": selection_metrics,
            "validation": validation_metrics,
            "test_accessed": False,
        },
        "memory": {"cuda_peak_allocated_bytes": peak_memory},
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def run_variable_patch_campaign(
    config_path: str | Path,
    output_path: str | Path,
    *,
    artifact_root: str | Path = "artifacts/final/variable-patch-search",
) -> dict:
    """Run the predeclared reference and variable-patch candidates."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = list(config["candidates"])
    seeds = [int(value) for value in config["seeds"]]
    budget = config["budget"]
    if len(candidates) > int(budget["maximum_candidates"]):
        raise ValueError("candidate budget exceeded")
    if len(candidates) * len(seeds) > int(budget["maximum_runs"]):
        raise ValueError("run budget exceeded")
    if bool(config["evaluation"].get("evaluate_test", False)):
        raise ValueError("patch search may not access the final test split")
    artifacts = Path(artifact_root)
    locked_root = artifacts / "locked-configs"
    locked_root.mkdir(parents=True, exist_ok=True)
    campaign_started = time.perf_counter()
    runs = []
    for candidate in candidates:
        for seed in seeds:
            if time.perf_counter() - campaign_started > float(budget["maximum_wall_seconds"]):
                raise RuntimeError("patch campaign exhausted its predeclared wall-time budget")
            run_root = artifacts / candidate["name"] / f"seed-{seed}"
            try:
                if candidate["kind"] == "foundation_reference":
                    locked = {
                        "format": "layercake-core-training-config/2",
                        "scale_status": "variable_patch_validation_only",
                        "seed": seed,
                        "device": config["device"],
                        "precision": config["precision"],
                        "model": copy.deepcopy(candidate["model"]),
                        "training": copy.deepcopy(config["training"]),
                        "evaluation": copy.deepcopy(config["evaluation"]),
                        "data": copy.deepcopy(config["data"]),
                    }
                    locked["training"]["route"] = candidate.get("route", "learned")
                    locked_path = locked_root / (
                        f"{candidate['name']}-seed-{seed}-{_canonical_hash(locked)[:12]}.json"
                    )
                    if not locked_path.is_file():
                        locked_path.write_text(
                            json.dumps(locked, indent=2, sort_keys=True), encoding="utf-8"
                        )
                    evidence = train_english_core(locked_path, run_root)
                    row = {
                        "candidate": candidate["name"],
                        "kind": candidate["kind"],
                        "seed": seed,
                        "status": evidence["status"],
                        "parameters": evidence["parameters"]["total_parameters"],
                        "active_parameters": evidence["parameters"]["active_parameters"],
                        "selection_bpb": evidence["quality"]["architecture_selection"]["bits_per_byte"],
                        "validation_bpb": evidence["quality"]["validation"]["bits_per_byte"],
                        "wall_seconds": evidence["training"]["wall_seconds"],
                        "raw_bytes_seen": evidence["training"]["raw_bytes_seen"],
                        "test_accessed": evidence["quality"]["test_accessed"],
                        "artifact": str(run_root.resolve()),
                    }
                else:
                    evidence = _train_patch_candidate(
                        candidate, config, seed=seed, output=run_root
                    )
                    row = {
                        "candidate": candidate["name"],
                        "kind": candidate["kind"],
                        "seed": seed,
                        "status": evidence["status"],
                        "parameters": evidence["parameters"],
                        "active_parameters": evidence.get("active_parameters", evidence["parameters"]),
                        "selection_bpb": evidence["quality"]["architecture_selection"]["bits_per_byte"],
                        "validation_bpb": evidence["quality"]["validation"]["bits_per_byte"],
                        "mean_bytes_per_patch": evidence["quality"]["architecture_selection"]["mean_bytes_per_patch"],
                        "routing": evidence.get("routing"),
                        "wall_seconds": evidence["training"]["wall_seconds"],
                        "raw_bytes_seen": evidence["training"]["raw_bytes_seen"],
                        "test_accessed": False,
                        "artifact": str(run_root.resolve()),
                    }
            except Exception as exc:
                row = {
                    "candidate": candidate["name"],
                    "kind": candidate["kind"],
                    "seed": seed,
                    "status": "FAIL",
                    "failure": f"{type(exc).__name__}: {exc}",
                    "test_accessed": False,
                }
            runs.append(row)
    reference_name = str(config["reference_candidate"])
    reference = next(
        row for row in runs
        if row["candidate"] == reference_name and row["status"] == "PASS"
    )
    improvement = float(config["promotion"]["minimum_bpb_improvement"])
    for row in runs:
        routing = row.get("routing")
        routing_eligible = (
            routing is None
            or (
                (
                    not config["promotion"].get("requires_all_experts_meaningfully_trained", False)
                    or routing["all_experts_meaningfully_trained"]
                )
                and routing["maximum_load_fraction"]
                <= float(config["promotion"].get("maximum_router_load_fraction", 1.0))
                and not routing["router_collapsed"]
            )
        )
        row["eligible"] = (
            row["status"] == "PASS"
            and row["kind"] != "foundation_reference"
            and row["candidate"] != reference_name
            and not row["test_accessed"]
            and row["selection_bpb"] <= reference["selection_bpb"] - improvement
            and routing_eligible
        )
    selected_rows = [row for row in runs if row.get("eligible")]
    selected = min(selected_rows, key=lambda row: row["selection_bpb"]) if selected_rows else None
    result = {
        "format": "layercake-final-variable-patch-campaign/1",
        "status": "PASS" if all(row["status"] == "PASS" for row in runs) else "FAIL",
        "scientific_conclusion": (
            "variable_patch_representation_improves_locked_reference"
            if selected else "variable_patch_representation_does_not_close_measured_gap"
        ),
        "selection_split_only": True,
        "final_test_accessed": False,
        "source_config": str(config_path.resolve()),
        "source_config_sha256": sha256_file(config_path),
        "reference_candidate": reference_name,
        "reference_selection_bpb": reference["selection_bpb"],
        "minimum_predeclared_improvement": improvement,
        "selected_candidate": selected,
        "runs": runs,
        "failed_runs": [row for row in runs if row["status"] != "PASS"],
        "data": {
            name: {
                "path": str(Path(path).resolve()),
                "bytes": Path(path).stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in config["data"].items()
        },
        "wall_seconds": time.perf_counter() - campaign_started,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
