"""Reproducible from-scratch English-core training."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from layercake.models.foundation_v2 import FoundationV2Config, LayerCakeFoundationV2
from .data import ByteCorpus, sha256_file
from .sparse_optimizer import optimizer_state_report, sparse_adamw


def _config(path: str | Path) -> dict:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        import yaml
        value = yaml.safe_load(raw)
    if "inherits" in value:
        base_path = Path(value["inherits"])
        if not base_path.is_absolute():
            # Config inheritance is repository-relative by contract.
            base_path = Path.cwd() / base_path
        base = _config(base_path)
        for dotted, override in value.get("required_overrides", {}).items():
            target = base
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = override
        base["scale_status"] = value.get("scale_status", base.get("scale_status"))
        base["seed"] = value.get("seed", base.get("seed"))
        return base
    return value


@torch.inference_mode()
def evaluate_core(
    model: LayerCakeFoundationV2,
    corpus: ByteCorpus,
    *,
    batch_size: int,
    sequence_bytes: int,
    batches: int,
    device: torch.device,
    route: int,
) -> dict:
    model.eval()
    losses = []
    correct = 0
    count = 0
    confidences = []
    calibration_errors = []
    for row in corpus.fixed_batches(
        batch_size=batch_size, sequence_bytes=sequence_bytes, batches=batches, device=device
    ):
        inputs, targets = row[:, :-1], row[:, 1:]
        logits = model(inputs, route=route)
        losses.append(F.cross_entropy(logits.flatten(0, 1), targets.flatten()).item())
        probabilities = torch.softmax(logits.float(), dim=-1)
        confidence, prediction = probabilities.max(dim=-1)
        match = prediction == targets
        correct += int(match.sum())
        count += targets.numel()
        confidences.append(float(confidence.mean()))
        calibration_errors.append(float((confidence - match.float()).abs().mean()))
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {
        "bits_per_byte": mean_loss / 0.6931471805599453,
        "byte_accuracy": correct / count,
        "mean_confidence": sum(confidences) / len(confidences),
        "mean_absolute_calibration_error": sum(calibration_errors) / len(calibration_errors),
        "evaluated_bytes": count,
    }


def train_english_core(config_path: str | Path, output_dir: str | Path) -> dict:
    config_path = Path(config_path)
    config = _config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    requested = str(config.get("device", "cuda"))
    device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
    precision = str(config.get("precision", "fp32"))
    model = LayerCakeFoundationV2(FoundationV2Config(**config["model"])).to(device)
    train_corpus = ByteCorpus(config["data"]["train"])
    validation_corpus = ByteCorpus(config["data"]["validation"])
    test_corpus = ByteCorpus(config["data"]["test"])
    architecture_corpus = ByteCorpus(config["data"]["architecture_selection"])
    optimizer = sparse_adamw(
        model,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and precision == "fp16" else nullcontext()
    )
    steps = int(config["training"]["steps"])
    batch_size = int(config["training"]["batch_size"])
    sequence_bytes = int(config["training"]["sequence_bytes"])
    eval_interval = int(config["training"]["evaluation_interval"])
    evaluation_batches = int(config["evaluation"]["batches"])
    configured_route = config["training"].get("route", seed % model.config.routed_experts)
    route = -1 if configured_route in {None, "learned", "auto"} else int(configured_route)
    model.set_route(route)
    resume_metadata_path = output / "resume.json"
    resume_model_path = output / "resume-model.safetensors"
    resume_optimizer_path = output / "resume-optimizer.pt"
    initial_experts_path = output / "initial-experts.safetensors"
    resume_enabled = bool(config["training"].get("resumable", True))
    start_step = 0
    previous_wall_seconds = 0.0
    curves = []
    resume_metadata: dict = {}
    if resume_enabled and resume_metadata_path.is_file():
        resume_metadata = json.loads(resume_metadata_path.read_text(encoding="utf-8"))
        if resume_metadata.get("config_sha256") != sha256_file(config_path):
            raise ValueError("resume checkpoint belongs to a different locked configuration")
        model.load_state_dict(load_file(str(resume_model_path), device=str(device)), strict=True)
        training_state = torch.load(
            resume_optimizer_path, map_location=device, weights_only=True
        )
        optimizer.load_state_dict(training_state["optimizer"])
        scaler.load_state_dict(training_state["scaler"])
        start_step = int(resume_metadata["step"])
        previous_wall_seconds = float(resume_metadata["wall_seconds"])
        curves = list(resume_metadata.get("curves", []))
    failed = None
    peak_memory = 0
    if initial_experts_path.is_file():
        packed_initial = load_file(str(initial_experts_path))
        expert_initial = [
            {
                name: packed_initial[f"expert.{index}.{name}"].clone()
                for name in expert.state_dict()
            }
            for index, expert in enumerate(model.experts.experts)
        ]
    else:
        expert_initial = [
            {name: value.detach().cpu().clone() for name, value in expert.state_dict().items()}
            for expert in model.experts.experts
        ]
        save_file({
            f"expert.{index}.{name}": value.contiguous()
            for index, expert in enumerate(expert_initial)
            for name, value in expert.items()
        }, str(initial_experts_path))
    routing_assignments = torch.zeros(model.config.routed_experts, device=device, dtype=torch.long)
    routing_importance = torch.zeros(model.config.routed_experts, device=device, dtype=torch.float64)
    routing_entropy = torch.zeros((), device=device, dtype=torch.float64)
    routing_observations = 0
    expert_gradient_steps = [0 for _ in range(model.config.routed_experts)]
    active_gradient_parameter_sum = 0
    active_gradient_observations = 0
    if start_step:
        routing_assignments += torch.tensor(
            resume_metadata.get("routing_assignments", [0] * model.config.routed_experts),
            device=device, dtype=torch.long,
        )
        routing_importance += torch.tensor(
            resume_metadata.get("routing_importance", [0.0] * model.config.routed_experts),
            device=device, dtype=torch.float64,
        )
        routing_entropy += float(resume_metadata.get("routing_entropy", 0.0))
        routing_observations = int(resume_metadata.get("routing_observations", 0))
        expert_gradient_steps = list(resume_metadata.get(
            "expert_gradient_steps", expert_gradient_steps
        ))
        active_gradient_parameter_sum = int(resume_metadata.get(
            "active_gradient_parameter_sum", 0
        ))
        active_gradient_observations = int(resume_metadata.get(
            "active_gradient_observations", 0
        ))
        peak_memory = int(resume_metadata.get("peak_memory", 0))
    started = time.perf_counter()
    for step, batch in enumerate(train_corpus.batches(
        batch_size=batch_size, sequence_bytes=sequence_bytes, seed=seed,
        steps=steps, device=device,
    ), start=1):
        if step <= start_step:
            continue
        inputs, targets = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits, aux = model(inputs, route=route, return_aux=True)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss = loss + float(config["training"].get("routing_balance_weight", 0.01)) * aux["routing_balance_loss"]
            future_losses = []
            for raw_horizon, future_logits in aux.get("future_logits", {}).items():
                horizon = int(raw_horizon)
                usable = inputs.shape[1] - horizon + 1
                if usable > 0:
                    future_losses.append(F.cross_entropy(
                        future_logits[:, :usable].flatten(0, 1),
                        batch[:, horizon:horizon + usable].flatten(),
                    ))
            if future_losses:
                loss = loss + float(config["training"].get("auxiliary_loss_weight", 0.2)) * torch.stack(future_losses).mean()
            if aux.get("routing") is not None:
                entropy_penalty = 1.0 - aux["routing"]["normalized_entropy"]
                loss = loss + float(config["training"].get("routing_entropy_weight", 0.0)) * entropy_penalty
        if not torch.isfinite(loss):
            failed = {"step": step, "reason": "non-finite loss"}
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        active_gradient_parameter_sum += sum(
            parameter.numel() for parameter in model.parameters() if parameter.grad is not None
        )
        active_gradient_observations += 1
        for expert_index, expert in enumerate(model.experts.experts):
            if any(parameter.grad is not None for parameter in expert.parameters()):
                expert_gradient_steps[expert_index] += 1
        if aux.get("routing") is not None:
            routing_assignments += aux["routing"]["assignment_counts"].detach().to(torch.long)
            routing_importance += aux["routing"]["importance"].detach().to(torch.float64)
            routing_entropy += aux["routing"]["normalized_entropy"].detach().to(torch.float64)
            routing_observations += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("gradient_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
        if step == 1 or step % eval_interval == 0 or step == steps:
            validation = evaluate_core(
                model, validation_corpus, batch_size=int(config["evaluation"]["batch_size"]),
                sequence_bytes=int(config["evaluation"]["sequence_bytes"]),
                batches=evaluation_batches, device=device, route=route,
            )
            curves.append({
                "step": step,
                "raw_bytes_seen": step * batch_size * sequence_bytes,
                "training_loss": float(loss.detach()),
                "validation": validation,
                "wall_seconds": previous_wall_seconds + time.perf_counter() - started,
            })
            if resume_enabled:
                model_temp = output / "resume-model.tmp.safetensors"
                optimizer_temp = output / "resume-optimizer.tmp.pt"
                metadata_temp = output / "resume.tmp.json"
                save_file({
                    name: value.detach().cpu().contiguous()
                    for name, value in model.state_dict().items()
                }, str(model_temp))
                torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, optimizer_temp)
                metadata_temp.write_text(json.dumps({
                    "format": "layercake-core-resume/1",
                    "config_sha256": sha256_file(config_path),
                    "step": step,
                    "wall_seconds": curves[-1]["wall_seconds"],
                    "curves": curves,
                    "routing_assignments": routing_assignments.cpu().tolist(),
                    "routing_importance": routing_importance.cpu().tolist(),
                    "routing_entropy": float(routing_entropy.cpu()),
                    "routing_observations": routing_observations,
                    "expert_gradient_steps": expert_gradient_steps,
                    "active_gradient_parameter_sum": active_gradient_parameter_sum,
                    "active_gradient_observations": active_gradient_observations,
                    "peak_memory": peak_memory,
                }, indent=2, sort_keys=True), encoding="utf-8")
                model_temp.replace(resume_model_path)
                optimizer_temp.replace(resume_optimizer_path)
                metadata_temp.replace(resume_metadata_path)
    training_seconds = previous_wall_seconds + time.perf_counter() - started
    architecture_selection = evaluate_core(
        model, architecture_corpus, batch_size=int(config["evaluation"]["batch_size"]),
        sequence_bytes=int(config["evaluation"]["sequence_bytes"]), batches=evaluation_batches,
        device=device, route=route,
    )
    validation = evaluate_core(
        model, validation_corpus, batch_size=int(config["evaluation"]["batch_size"]),
        sequence_bytes=int(config["evaluation"]["sequence_bytes"]), batches=evaluation_batches,
        device=device, route=route,
    )
    evaluate_test = bool(config["evaluation"].get("evaluate_test", False))
    test = evaluate_core(
        model, test_corpus, batch_size=int(config["evaluation"]["batch_size"]),
        sequence_bytes=int(config["evaluation"]["sequence_bytes"]), batches=evaluation_batches,
        device=device, route=route,
    ) if evaluate_test else None

    tensor_path = output / "model.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(tensor_path))
    report = model.parameter_report(route)
    assignment_total = int(routing_assignments.sum().cpu())
    utilization = (
        (routing_assignments.double() / max(1, assignment_total)).cpu().tolist()
        if assignment_total else [0.0 for _ in range(model.config.routed_experts)]
    )
    mean_importance = (
        (routing_importance / max(1, routing_observations)).cpu().tolist()
        if routing_observations else [0.0 for _ in range(model.config.routed_experts)]
    )
    expert_delta_l2 = []
    for expert, initial in zip(model.experts.experts, expert_initial):
        squared = 0.0
        for name, value in expert.state_dict().items():
            delta = value.detach().cpu().float() - initial[name].float()
            squared += float(torch.sum(delta * delta))
        expert_delta_l2.append(squared ** 0.5)
    minimum_gradient_steps = max(1, int(0.01 * max(1, curves[-1]["step"] if curves else 0)))
    meaningful = [
        delta > 1e-5 and updates >= minimum_gradient_steps
        for delta, updates in zip(expert_delta_l2, expert_gradient_steps)
    ]
    collapse_threshold = float(config["training"].get(
        "router_collapse_threshold", max(0.5, 4.0 / model.config.routed_experts)
    ))
    mean_active_gradient_parameters = active_gradient_parameter_sum / max(1, active_gradient_observations)
    process = psutil.Process()
    evidence = {
        "format": "layercake-english-core/2",
        "status": "FAIL" if failed else "PASS",
        "failure": failed,
        "seed": seed,
        "route": route,
        "architecture": model.config.canonical_dict(),
        "canonical_abi": model.canonical.config.contract(),
        "canonical_abi_hash": model.canonical.config.abi_hash(),
        "parameters": report,
        "optimizer": optimizer_state_report(optimizer),
        "routing": {
            "mode": model.config.routing_mode,
            "causal_patch_routing": True,
            "route_override": None if route < 0 else route,
            "assignment_counts": routing_assignments.cpu().tolist(),
            "utilization": utilization,
            "mean_router_importance": mean_importance,
            "mean_normalized_entropy": float((routing_entropy / max(1, routing_observations)).cpu()),
            "maximum_load_fraction": max(utilization),
            "experts_used": sum(value > 0 for value in utilization),
            "expert_gradient_steps": expert_gradient_steps,
            "expert_delta_l2": expert_delta_l2,
            "meaningfully_trained": meaningful,
            "all_experts_meaningfully_trained": all(meaningful),
            "collapse_threshold": collapse_threshold,
            "router_collapsed": max(utilization) > collapse_threshold,
            "mean_parameters_with_grad_per_step": mean_active_gradient_parameters,
            "mean_fraction_with_grad_per_step": mean_active_gradient_parameters / report["total_parameters"],
        },
        "data": {
            name: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}
            for name, path in config["data"].items()
        },
        "training": {
            "steps_completed": curves[-1]["step"] if curves else 0,
            "configured_steps": steps,
            "batch_size": batch_size,
            "context_bytes": sequence_bytes,
            "raw_bytes_seen": (curves[-1]["step"] if curves else 0) * batch_size * sequence_bytes,
            "wall_seconds": training_seconds,
            "raw_bytes_per_second": ((curves[-1]["step"] if curves else 0) * batch_size * sequence_bytes) / max(training_seconds, 1e-9),
            "parameter_seconds_active": report["active_parameters_per_training_item"] * training_seconds,
            "parameter_seconds_total_installed": report["total_parameters"] * training_seconds,
            "curves": curves,
            "resumable": resume_enabled,
            "resumed_from_step": start_step,
            "resume_metadata": str(resume_metadata_path.resolve()) if resume_enabled else None,
        },
        "quality": {
            "architecture_selection": architecture_selection,
            "validation": validation,
            "test": test,
            "test_accessed": evaluate_test,
        },
        "memory": {
            "cuda_peak_allocated_bytes": int(peak_memory),
            "process_rss_bytes": int(process.memory_info().rss),
        },
        "checkpoint": {
            "tensors": str(tensor_path.resolve()),
            "sha256": sha256_file(tensor_path),
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def load_core_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[LayerCakeFoundationV2, dict]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    model = LayerCakeFoundationV2(FoundationV2Config(**metadata["architecture"]))
    model.load_state_dict(load_file(str(root / "model.safetensors"), device=str(device)), strict=True)
    model.to(device).eval()
    return model, metadata
