from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any

import psutil
import torch
import torch.nn.functional as F

import _common
from layercake.portable_domain import (
    LayerCakeRuntime,
    PortableDomainDecoder,
    PortableDomainSpec,
    build_portable_artifact,
    load_portable_artifact,
    state_dict_hash,
)
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _execute_tests,
    _extract_function,
    _load_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(
    rows: list[dict[str, Any]],
    indexes: list[int],
    *,
    maximum_sequence_bytes: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    sequences = []
    response_starts = []
    identifier_spans = []
    pointer_spans = []
    for index in indexes:
        row = rows[index]
        prompt = (row["prompt"] + "\n").encode("utf-8")
        response = row["response"].encode("utf-8")
        identifier = row["function_name"].encode("utf-8")
        sequence = list((prompt + response)[:maximum_sequence_bytes])
        if len(sequence) < 2 or len(prompt) >= len(sequence):
            raise ValueError("functional row has no trainable response bytes")
        sequences.append(sequence)
        response_starts.append(len(prompt) - 1)
        identifier_start = len(prompt) - 1 + len(b"def ")
        identifier_spans.append(
            (identifier_start, identifier_start + len(identifier))
        )
        prompt_identifier_start = prompt.find(identifier)
        if prompt_identifier_start < 0:
            raise ValueError("function identifier is absent from its prompt")
        pointer_spans.append(
            (prompt_identifier_start, len(identifier))
        )
    width = max(len(sequence) - 1 for sequence in sequences)
    inputs = torch.zeros((len(sequences), width), dtype=torch.long)
    targets = torch.zeros((len(sequences), width), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    identifier_mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    pointer_labels = torch.full(
        (len(sequences), width), -100, dtype=torch.long
    )
    for batch_index, (
        sequence,
        response_start,
        identifier_span,
        pointer_span,
    ) in enumerate(
        zip(
            sequences,
            response_starts,
            identifier_spans,
            pointer_spans,
        )
    ):
        length = len(sequence) - 1
        inputs[batch_index, :length] = torch.tensor(sequence[:-1])
        targets[batch_index, :length] = torch.tensor(sequence[1:])
        mask[batch_index, response_start:length] = True
        identifier_start, identifier_stop = identifier_span
        identifier_mask[
            batch_index,
            identifier_start : min(identifier_stop, length),
        ] = True
        prompt_identifier_start, identifier_length = pointer_span
        usable = min(
            identifier_length,
            max(0, length - identifier_start),
        )
        pointer_labels[
            batch_index,
            identifier_start : identifier_start + usable,
        ] = torch.arange(
            prompt_identifier_start,
            prompt_identifier_start + usable,
        )
    return inputs, targets, mask, identifier_mask, pointer_labels


def train(
    protocol_path: Path,
    artifact_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "PREREGISTERED_BEFORE_TRAINING":
        raise ValueError("functional decoder run is not preregistered")
    settings = protocol["bounded_screen"]
    architecture = protocol["architecture"]
    data_contract = protocol["training_data"]
    data_path = ROOT / data_contract["path"]
    if _sha256(data_path) != data_contract["sha256"]:
        raise ValueError("training dataset hash mismatch")
    if artifact_path.exists() or evidence_path.exists():
        raise RuntimeError("functional decoder outputs are immutable")
    rows = [row for row in _load_rows(data_path) if row["split"] == "train"]
    if len(rows) != data_contract["rows"]:
        raise ValueError("training row count mismatch")

    seed = int(settings["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(int(settings["torch_threads"]))
    model = PortableDomainDecoder(
        feature_width=int(architecture["feature_width"]),
        hidden_width=int(architecture["hidden_width"]),
        architecture=str(architecture["architecture"]),
        embedding_width=int(architecture["embedding_width"]),
    )
    if model.parameter_count() != architecture["parameters"]:
        raise ValueError("preregistered parameter count mismatch")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    history = []
    best_loss = float("inf")
    best_state = None
    steps = int(settings["optimizer_steps"])
    batch_size = int(settings["batch_size"])
    order = list(range(len(rows)))
    cursor = len(order)
    generator = random.Random(seed + 1)
    model.train()
    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            generator.shuffle(order)
            cursor = 0
        indexes = order[cursor : cursor + batch_size]
        cursor += batch_size
        inputs, targets, mask, _, _ = _batch(
            rows,
            indexes,
            maximum_sequence_bytes=int(settings["maximum_sequence_bytes"]),
        )
        logits = model(inputs)
        loss = F.cross_entropy(logits[mask], targets[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        value = float(loss.item())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if step == 1 or step % 100 == 0 or step == steps:
            with torch.no_grad():
                accuracy = float(
                    (logits[mask].argmax(dim=-1) == targets[mask])
                    .float()
                    .mean()
                    .item()
                )
            record = {
                "step": step,
                "response_cross_entropy": value,
                "response_byte_accuracy": accuracy,
                "gradient_norm_before_clip": float(gradient_norm),
                "cpu_wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    wall = time.perf_counter() - started
    artifact = build_portable_artifact(
        model,
        PortableDomainSpec(
            domain_id="python",
            feature_width=int(architecture["feature_width"]),
            hidden_width=int(architecture["hidden_width"]),
            architecture=str(architecture["architecture"]),
            embedding_width=int(architecture["embedding_width"]),
        ),
        training={
            "protocol": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": _sha256(protocol_path),
            "seed": seed,
            "optimizer_steps": steps,
            "training_rows": len(rows),
            "training_data_sha256": _sha256(data_path),
            "objective": settings["objective"],
            "best_response_cross_entropy": best_loss,
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    evidence = {
        "format": "layercake-phase4-portable-functional-training/1",
        "status": "TRAINED",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "device": "cpu",
        "torch_threads": int(settings["torch_threads"]),
        "seed": seed,
        "architecture": architecture,
        "training_data": data_contract,
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "best_response_cross_entropy": best_loss,
        "cpu_wall_seconds": wall,
        "peak_process_resident_memory_bytes": peak_rss,
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
        "history": history,
        "validation_split_accessed": False,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def continue_identifier(
    protocol_path: Path,
    initial_artifact_path: Path,
    artifact_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "PREREGISTERED_BEFORE_CONTINUATION":
        raise ValueError("identifier repair is not preregistered")
    parent = protocol["parent_artifact"]
    if _sha256(initial_artifact_path) != parent["file_sha256"]:
        raise ValueError("identifier repair parent artifact hash mismatch")
    if artifact_path.exists() or evidence_path.exists():
        raise RuntimeError("identifier repair outputs are immutable")
    data_contract = json.loads(
        (
            ROOT
            / "moonshot"
            / "phase4_portable_decoder_functional_preregistration.json"
        ).read_text(encoding="utf-8")
    )["training_data"]
    data_path = ROOT / data_contract["path"]
    if _sha256(data_path) != data_contract["sha256"]:
        raise ValueError("identifier repair training dataset hash mismatch")
    rows = [row for row in _load_rows(data_path) if row["split"] == "train"]
    settings = protocol["bounded_continuation"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(int(settings["torch_threads"]))
    initial = torch.load(
        initial_artifact_path, map_location="cpu", weights_only=True
    )
    spec, model = load_portable_artifact(initial, "cpu")
    if initial["payload_hash"] != parent["payload_hash"]:
        raise ValueError("identifier repair parent payload hash mismatch")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    history = []
    best_loss = float("inf")
    best_state = None
    steps = int(settings["optimizer_steps"])
    batch_size = int(settings["batch_size"])
    order = list(range(len(rows)))
    cursor = len(order)
    generator = random.Random(seed + 1)
    model.train()
    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            generator.shuffle(order)
            cursor = 0
        indexes = order[cursor : cursor + batch_size]
        cursor += batch_size
        (
            inputs,
            targets,
            response_mask,
            identifier_mask,
            _,
        ) = _batch(
            rows, indexes, maximum_sequence_bytes=512
        )
        logits = model(inputs)
        response_loss = F.cross_entropy(
            logits[response_mask], targets[response_mask]
        )
        identifier_loss = F.cross_entropy(
            logits[identifier_mask], targets[identifier_mask]
        )
        loss = response_loss + 4.0 * identifier_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        value = float(loss.item())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if step == 1 or step % 100 == 0 or step == steps:
            record = {
                "step": step,
                "combined_objective": value,
                "response_cross_entropy": float(response_loss.item()),
                "identifier_cross_entropy": float(identifier_loss.item()),
                "identifier_byte_accuracy": float(
                    (
                        logits[identifier_mask].argmax(dim=-1)
                        == targets[identifier_mask]
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "gradient_norm_before_clip": float(gradient_norm),
                "cpu_wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    if best_state is None:
        raise RuntimeError("identifier continuation produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    wall = time.perf_counter() - started
    artifact = build_portable_artifact(
        model,
        spec,
        training={
            **initial.get("training", {}),
            "identifier_repair_protocol": protocol_path.relative_to(
                ROOT
            ).as_posix(),
            "identifier_repair_protocol_sha256": _sha256(protocol_path),
            "identifier_repair_seed": seed,
            "identifier_repair_steps": steps,
            "best_identifier_repair_objective": best_loss,
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    evidence = {
        "format": "layercake-phase4-portable-identifier-repair-training/1",
        "status": "TRAINED",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "parent_artifact": initial_artifact_path.relative_to(ROOT).as_posix(),
        "parent_artifact_file_sha256": _sha256(initial_artifact_path),
        "parent_payload_hash": initial["payload_hash"],
        "device": "cpu",
        "torch_threads": int(settings["torch_threads"]),
        "seed": seed,
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "best_combined_objective": best_loss,
        "cpu_wall_seconds": wall,
        "peak_process_resident_memory_bytes": peak_rss,
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
        "history": history,
        "validation_split_accessed_during_continuation": False,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def continue_pointer(
    protocol_path: Path,
    initial_artifact_path: Path,
    artifact_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "PREREGISTERED_BEFORE_POINTER_TRAINING":
        raise ValueError("pointer-generator repair is not preregistered")
    parent = protocol["parent_artifact"]
    if _sha256(initial_artifact_path) != parent["file_sha256"]:
        raise ValueError("pointer repair parent artifact hash mismatch")
    if artifact_path.exists() or evidence_path.exists():
        raise RuntimeError("pointer repair outputs are immutable")
    data_contract = protocol["training_data"]
    data_path = ROOT / data_contract["path"]
    if _sha256(data_path) != data_contract["sha256"]:
        raise ValueError("pointer repair training dataset hash mismatch")
    rows = [row for row in _load_rows(data_path) if row["split"] == "train"]
    settings = protocol["bounded_run"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(int(settings["torch_threads"]))
    initial = torch.load(
        initial_artifact_path, map_location="cpu", weights_only=True
    )
    parent_spec, parent_model = load_portable_artifact(initial, "cpu")
    if initial["payload_hash"] != parent["payload_hash"]:
        raise ValueError("pointer repair parent payload hash mismatch")
    pointer_width = int(protocol["architecture"]["pointer_width"])
    model = PortableDomainDecoder(
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer",
        embedding_width=parent_spec.embedding_width,
        pointer_width=pointer_width,
    )
    missing, unexpected = model.load_state_dict(
        parent_model.state_dict(), strict=False
    )
    expected_missing = {
        "copy_query.weight",
        "copy_key.weight",
        "copy_gate.weight",
        "copy_gate.bias",
    }
    if set(missing) != expected_missing or unexpected:
        raise ValueError("pointer parent state is structurally incompatible")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("copy_"))
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    history = []
    best_loss = float("inf")
    best_state = None
    steps = int(settings["optimizer_steps"])
    batch_size = int(settings["batch_size"])
    order = list(range(len(rows)))
    cursor = len(order)
    generator = random.Random(seed + 1)
    model.train()
    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            generator.shuffle(order)
            cursor = 0
        indexes = order[cursor : cursor + batch_size]
        cursor += batch_size
        (
            inputs,
            targets,
            response_mask,
            identifier_mask,
            pointer_labels,
        ) = _batch(rows, indexes, maximum_sequence_bytes=512)
        result = model.pointer_forward(inputs)
        response_loss = F.nll_loss(
            result["logits"][response_mask],
            targets[response_mask],
        )
        alignment_loss = F.cross_entropy(
            result["pointer_scores"][identifier_mask],
            pointer_labels[identifier_mask],
        )
        gate_targets = identifier_mask[response_mask].float()
        gate_loss = F.binary_cross_entropy_with_logits(
            result["gate_logits"].squeeze(-1)[response_mask],
            gate_targets,
        )
        loss = response_loss + alignment_loss + gate_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        value = float(loss.item())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if step == 1 or step % 100 == 0 or step == steps:
            pointer_top1 = result["pointer_scores"][
                identifier_mask
            ].argmax(dim=-1)
            record = {
                "step": step,
                "combined_objective": value,
                "response_nll": float(response_loss.item()),
                "pointer_alignment_cross_entropy": float(
                    alignment_loss.item()
                ),
                "pointer_position_accuracy": float(
                    (
                        pointer_top1
                        == pointer_labels[identifier_mask]
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "gate_binary_cross_entropy": float(gate_loss.item()),
                "gradient_norm_before_clip": float(gradient_norm),
                "cpu_wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    if best_state is None:
        raise RuntimeError("pointer continuation produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    wall = time.perf_counter() - started
    spec = PortableDomainSpec(
        domain_id=parent_spec.domain_id,
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer",
        embedding_width=parent_spec.embedding_width,
        pointer_width=pointer_width,
    )
    artifact = build_portable_artifact(
        model,
        spec,
        training={
            **initial.get("training", {}),
            "pointer_repair_protocol": protocol_path.relative_to(
                ROOT
            ).as_posix(),
            "pointer_repair_protocol_sha256": _sha256(protocol_path),
            "pointer_repair_seed": seed,
            "pointer_repair_steps": steps,
            "best_pointer_repair_objective": best_loss,
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    evidence = {
        "format": "layercake-phase4-portable-pointer-training/1",
        "status": "TRAINED",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "parent_artifact": initial_artifact_path.relative_to(ROOT).as_posix(),
        "parent_artifact_file_sha256": _sha256(initial_artifact_path),
        "parent_payload_hash": initial["payload_hash"],
        "device": "cpu",
        "torch_threads": int(settings["torch_threads"]),
        "seed": seed,
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable
        ),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "best_combined_objective": best_loss,
        "cpu_wall_seconds": wall,
        "peak_process_resident_memory_bytes": peak_rss,
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
        "history": history,
        "validation_split_accessed_during_training": False,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def continue_transition_pointer(
    protocol_path: Path,
    initial_artifact_path: Path,
    artifact_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["status"]
        != "PREREGISTERED_BEFORE_IMPLEMENTATION_AND_TRAINING"
    ):
        raise ValueError("transition-pointer repair is not preregistered")
    parent = protocol["parent_artifact"]
    if _sha256(initial_artifact_path) != parent["file_sha256"]:
        raise ValueError("transition-pointer parent artifact hash mismatch")
    if artifact_path.exists() or evidence_path.exists():
        raise RuntimeError("transition-pointer outputs are immutable")
    data_contract = protocol["training_data"]
    data_path = ROOT / data_contract["path"]
    if _sha256(data_path) != data_contract["sha256"]:
        raise ValueError("transition-pointer training dataset hash mismatch")
    rows = [row for row in _load_rows(data_path) if row["split"] == "train"]
    if len(rows) != data_contract["rows"]:
        raise ValueError("transition-pointer training row count mismatch")
    settings = protocol["bounded_run"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(int(settings["torch_threads"]))
    initial = torch.load(
        initial_artifact_path, map_location="cpu", weights_only=True
    )
    parent_spec, parent_model = load_portable_artifact(initial, "cpu")
    if initial["payload_hash"] != parent["payload_hash"]:
        raise ValueError("transition-pointer parent payload hash mismatch")
    model = PortableDomainDecoder(
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer_transition",
        embedding_width=parent_spec.embedding_width,
        pointer_width=parent_spec.pointer_width,
    )
    missing, unexpected = model.load_state_dict(
        parent_model.state_dict(), strict=False
    )
    expected_missing = {
        "copy_transition_logits",
        "copy_transition_gate.weight",
        "copy_transition_gate.bias",
    }
    if set(missing) != expected_missing or unexpected:
        raise ValueError(
            "transition-pointer parent state is structurally incompatible"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("copy_transition_"))
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    expected_trainable = int(
        protocol["single_bounded_change"]["trainable_parameters"]
    )
    if trainable_count != expected_trainable:
        raise ValueError("transition-pointer parameter count mismatch")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    history = []
    best_loss = float("inf")
    best_state = None
    steps = int(settings["optimizer_steps"])
    batch_size = int(settings["batch_size"])
    order = list(range(len(rows)))
    cursor = len(order)
    generator = random.Random(seed + 1)
    model.train()
    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            generator.shuffle(order)
            cursor = 0
        indexes = order[cursor : cursor + batch_size]
        cursor += batch_size
        (
            inputs,
            targets,
            response_mask,
            identifier_mask,
            pointer_labels,
        ) = _batch(rows, indexes, maximum_sequence_bytes=512)
        result = model.pointer_forward(inputs)
        response_loss = F.nll_loss(
            result["logits"][response_mask],
            targets[response_mask],
        )
        alignment_loss = F.nll_loss(
            result["pointer_scores"][identifier_mask],
            pointer_labels[identifier_mask],
        )
        previous_identifier = torch.zeros_like(identifier_mask)
        previous_identifier[:, 1:] = identifier_mask[:, :-1]
        transition_targets = (
            identifier_mask & previous_identifier
        )[response_mask].float()
        transition_gate_logits = result[
            "transition_gate_logits"
        ].squeeze(-1)
        transition_gate_loss = F.binary_cross_entropy_with_logits(
            transition_gate_logits[response_mask],
            transition_targets,
        )
        loss = response_loss + alignment_loss + transition_gate_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        value = float(loss.item())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if step == 1 or step % 100 == 0 or step == steps:
            pointer_top1 = result["pointer_scores"][
                identifier_mask
            ].argmax(dim=-1)
            transition_probabilities = torch.softmax(
                model.copy_transition_logits.detach(), dim=0
            )
            record = {
                "step": step,
                "combined_objective": value,
                "response_nll": float(response_loss.item()),
                "pointer_alignment_nll": float(alignment_loss.item()),
                "pointer_position_accuracy": float(
                    (
                        pointer_top1
                        == pointer_labels[identifier_mask]
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "transition_gate_binary_cross_entropy": float(
                    transition_gate_loss.item()
                ),
                "transition_offset_probabilities": [
                    float(value)
                    for value in transition_probabilities.tolist()
                ],
                "gradient_norm_before_clip": float(gradient_norm),
                "cpu_wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    if best_state is None:
        raise RuntimeError("transition-pointer continuation produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    wall = time.perf_counter() - started
    spec = PortableDomainSpec(
        domain_id=parent_spec.domain_id,
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer_transition",
        embedding_width=parent_spec.embedding_width,
        pointer_width=parent_spec.pointer_width,
    )
    artifact = build_portable_artifact(
        model,
        spec,
        training={
            **initial.get("training", {}),
            "transition_pointer_protocol": protocol_path.relative_to(
                ROOT
            ).as_posix(),
            "transition_pointer_protocol_sha256": _sha256(protocol_path),
            "transition_pointer_seed": seed,
            "transition_pointer_steps": steps,
            "best_transition_pointer_objective": best_loss,
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    evidence = {
        "format": "layercake-phase4-portable-transition-pointer-training/1",
        "status": "TRAINED",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "parent_artifact": initial_artifact_path.relative_to(ROOT).as_posix(),
        "parent_artifact_file_sha256": _sha256(initial_artifact_path),
        "parent_payload_hash": initial["payload_hash"],
        "device": "cpu",
        "torch_threads": int(settings["torch_threads"]),
        "seed": seed,
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "trainable_parameters": trainable_count,
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "best_combined_objective": best_loss,
        "cpu_wall_seconds": wall,
        "peak_process_resident_memory_bytes": peak_rss,
        "transition_offsets": protocol["single_bounded_change"][
            "transition_offsets"
        ],
        "final_transition_offset_probabilities": [
            float(value)
            for value in torch.softmax(
                model.copy_transition_logits.detach(), dim=0
            ).tolist()
        ],
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
        "history": history,
        "validation_split_accessed_during_training": False,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def convert_self_gated_transition(
    protocol_path: Path,
    initial_artifact_path: Path,
    artifact_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["status"]
        != "PREREGISTERED_BEFORE_IMPLEMENTATION_AND_CONVERSION"
    ):
        raise ValueError("self-gated transition conversion is not preregistered")
    parent = protocol["parent_artifact"]
    if _sha256(initial_artifact_path) != parent["file_sha256"]:
        raise ValueError("self-gated transition parent artifact hash mismatch")
    if artifact_path.exists() or evidence_path.exists():
        raise RuntimeError("self-gated transition outputs are immutable")
    initial = torch.load(
        initial_artifact_path, map_location="cpu", weights_only=True
    )
    parent_spec, parent_model = load_portable_artifact(initial, "cpu")
    if initial["payload_hash"] != parent["payload_hash"]:
        raise ValueError("self-gated transition parent payload hash mismatch")
    model = PortableDomainDecoder(
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer_self_transition",
        embedding_width=parent_spec.embedding_width,
        pointer_width=parent_spec.pointer_width,
    )
    model.load_state_dict(parent_model.state_dict(), strict=True)
    model.eval()
    parent_state_hash = state_dict_hash(parent_model.state_dict())
    converted_state_hash = state_dict_hash(model.state_dict())
    if parent_state_hash != converted_state_hash:
        raise ValueError("self-gated conversion changed parameter tensors")
    if model.parameter_count() != int(parent["parameters"]):
        raise ValueError("self-gated conversion changed parameter count")
    spec = PortableDomainSpec(
        domain_id=parent_spec.domain_id,
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer_self_transition",
        embedding_width=parent_spec.embedding_width,
        pointer_width=parent_spec.pointer_width,
    )
    artifact = build_portable_artifact(
        model,
        spec,
        training={
            **initial.get("training", {}),
            "self_gated_transition_protocol": protocol_path.relative_to(
                ROOT
            ).as_posix(),
            "self_gated_transition_protocol_sha256": _sha256(protocol_path),
            "conversion_training_rows_accessed": 0,
            "conversion_optimizer_steps": 0,
            "parent_state_dict_hash": parent_state_hash,
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    evidence = {
        "format": "layercake-phase4-self-gated-transition-conversion/1",
        "status": "CONVERTED_WITH_BIT_IDENTICAL_PARAMETERS",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "parent_artifact": initial_artifact_path.relative_to(ROOT).as_posix(),
        "parent_artifact_file_sha256": _sha256(initial_artifact_path),
        "parent_payload_hash": initial["payload_hash"],
        "parent_state_dict_hash": parent_state_hash,
        "converted_state_dict_hash": converted_state_hash,
        "parameter_tensors_bit_identical": True,
        "parameters": model.parameter_count(),
        "new_parameters": 0,
        "trained_parameters": 0,
        "optimizer_steps": 0,
        "training_rows_accessed": 0,
        "validation_rows_accessed": 0,
        "test_rows_accessed": 0,
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def convert_max_projection(
    protocol_path: Path,
    initial_artifact_path: Path,
    artifact_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["status"]
        != "PREREGISTERED_BEFORE_IMPLEMENTATION_AND_CONVERSION"
    ):
        raise ValueError("max-projection conversion is not preregistered")
    parent = protocol["parent_artifact"]
    if _sha256(initial_artifact_path) != parent["file_sha256"]:
        raise ValueError("max-projection parent artifact hash mismatch")
    if artifact_path.exists() or evidence_path.exists():
        raise RuntimeError("max-projection outputs are immutable")
    initial = torch.load(
        initial_artifact_path, map_location="cpu", weights_only=True
    )
    parent_spec, parent_model = load_portable_artifact(initial, "cpu")
    if initial["payload_hash"] != parent["payload_hash"]:
        raise ValueError("max-projection parent payload hash mismatch")
    model = PortableDomainDecoder(
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer_markov_max",
        embedding_width=parent_spec.embedding_width,
        pointer_width=parent_spec.pointer_width,
    )
    model.load_state_dict(parent_model.state_dict(), strict=True)
    model.eval()
    parent_state_hash = state_dict_hash(parent_model.state_dict())
    converted_state_hash = state_dict_hash(model.state_dict())
    if parent_state_hash != converted_state_hash:
        raise ValueError("max-projection conversion changed parameter tensors")
    if model.parameter_count() != int(parent["parameters"]):
        raise ValueError("max-projection conversion changed parameter count")
    spec = PortableDomainSpec(
        domain_id=parent_spec.domain_id,
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer_markov_max",
        embedding_width=parent_spec.embedding_width,
        pointer_width=parent_spec.pointer_width,
    )
    artifact = build_portable_artifact(
        model,
        spec,
        training={
            **initial.get("training", {}),
            "max_projection_protocol": protocol_path.relative_to(
                ROOT
            ).as_posix(),
            "max_projection_protocol_sha256": _sha256(protocol_path),
            "conversion_training_rows_accessed": 0,
            "conversion_optimizer_steps": 0,
            "parent_state_dict_hash": parent_state_hash,
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    evidence = {
        "format": "layercake-phase4-max-projection-conversion/1",
        "status": "CONVERTED_WITH_BIT_IDENTICAL_PARAMETERS",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "parent_artifact": initial_artifact_path.relative_to(ROOT).as_posix(),
        "parent_artifact_file_sha256": _sha256(initial_artifact_path),
        "parent_payload_hash": initial["payload_hash"],
        "parent_state_dict_hash": parent_state_hash,
        "converted_state_dict_hash": converted_state_hash,
        "parameter_tensors_bit_identical": True,
        "parameters": model.parameter_count(),
        "new_parameters": 0,
        "trained_parameters": 0,
        "optimizer_steps": 0,
        "training_rows_accessed": 0,
        "validation_rows_accessed": 0,
        "test_rows_accessed": 0,
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def continue_markov_pointer(
    protocol_path: Path,
    initial_artifact_path: Path,
    artifact_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["status"]
        != "PREREGISTERED_BEFORE_IMPLEMENTATION_AND_TRAINING"
    ):
        raise ValueError("Markov pointer integration is not preregistered")
    parent = protocol["parent_artifact"]
    if _sha256(initial_artifact_path) != parent["file_sha256"]:
        raise ValueError("Markov pointer parent artifact hash mismatch")
    if artifact_path.exists() or evidence_path.exists():
        raise RuntimeError("Markov pointer outputs are immutable")
    data_contract = protocol["training_data"]
    data_path = ROOT / data_contract["path"]
    if _sha256(data_path) != data_contract["sha256"]:
        raise ValueError("Markov pointer training dataset hash mismatch")
    rows = [row for row in _load_rows(data_path) if row["split"] == "train"]
    if len(rows) != data_contract["rows"]:
        raise ValueError("Markov pointer training row count mismatch")
    settings = protocol["bounded_run"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(int(settings["torch_threads"]))
    initial = torch.load(
        initial_artifact_path, map_location="cpu", weights_only=True
    )
    parent_spec, parent_model = load_portable_artifact(initial, "cpu")
    if initial["payload_hash"] != parent["payload_hash"]:
        raise ValueError("Markov pointer parent payload hash mismatch")
    model = PortableDomainDecoder(
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer_markov",
        embedding_width=parent_spec.embedding_width,
        pointer_width=parent_spec.pointer_width,
    )
    missing, unexpected = model.load_state_dict(
        parent_model.state_dict(), strict=False
    )
    if set(missing) != {"copy_transition_logits"} or unexpected:
        raise ValueError("Markov pointer parent is structurally incompatible")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "copy_transition_logits")
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    expected_trainable = int(
        protocol["single_bounded_change"]["trainable_parameters"]
    )
    if trainable_count != expected_trainable:
        raise ValueError("Markov pointer trainable parameter count mismatch")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    history = []
    best_loss = float("inf")
    best_state = None
    steps = int(settings["optimizer_steps"])
    batch_size = int(settings["batch_size"])
    order = list(range(len(rows)))
    cursor = len(order)
    generator = random.Random(seed + 1)
    model.train()
    for step in range(1, steps + 1):
        if cursor + batch_size > len(order):
            generator.shuffle(order)
            cursor = 0
        indexes = order[cursor : cursor + batch_size]
        cursor += batch_size
        (
            inputs,
            targets,
            response_mask,
            identifier_mask,
            pointer_labels,
        ) = _batch(rows, indexes, maximum_sequence_bytes=512)
        result = model.pointer_forward(inputs)
        response_loss = F.nll_loss(
            result["logits"][response_mask],
            targets[response_mask],
        )
        alignment_loss = F.nll_loss(
            result["pointer_scores"][identifier_mask],
            pointer_labels[identifier_mask],
        )
        loss = response_loss + alignment_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(settings["gradient_clip_norm"])
        )
        optimizer.step()
        value = float(loss.item())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if step == 1 or step % 100 == 0 or step == steps:
            pointer_top1 = result["pointer_scores"][
                identifier_mask
            ].argmax(dim=-1)
            transition_probabilities = torch.softmax(
                model.copy_transition_logits.detach(), dim=0
            )
            record = {
                "step": step,
                "combined_objective": value,
                "response_nll": float(response_loss.item()),
                "pointer_alignment_nll": float(alignment_loss.item()),
                "pointer_position_accuracy": float(
                    (
                        pointer_top1
                        == pointer_labels[identifier_mask]
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "transition_offset_probabilities": [
                    float(item)
                    for item in transition_probabilities.tolist()
                ],
                "gradient_norm_before_clip": float(gradient_norm),
                "cpu_wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    if best_state is None:
        raise RuntimeError("Markov pointer produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    wall = time.perf_counter() - started
    spec = PortableDomainSpec(
        domain_id=parent_spec.domain_id,
        feature_width=parent_spec.feature_width,
        hidden_width=parent_spec.hidden_width,
        architecture="byte_gru_pointer_markov",
        embedding_width=parent_spec.embedding_width,
        pointer_width=parent_spec.pointer_width,
    )
    artifact = build_portable_artifact(
        model,
        spec,
        training={
            **initial.get("training", {}),
            "markov_pointer_protocol": protocol_path.relative_to(
                ROOT
            ).as_posix(),
            "markov_pointer_protocol_sha256": _sha256(protocol_path),
            "markov_pointer_seed": seed,
            "markov_pointer_steps": steps,
            "best_markov_pointer_objective": best_loss,
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    evidence = {
        "format": "layercake-phase4-portable-markov-pointer-training/1",
        "status": "TRAINED",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "parent_artifact": initial_artifact_path.relative_to(ROOT).as_posix(),
        "parent_artifact_file_sha256": _sha256(initial_artifact_path),
        "parent_payload_hash": initial["payload_hash"],
        "device": "cpu",
        "torch_threads": int(settings["torch_threads"]),
        "seed": seed,
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "trainable_parameters": trainable_count,
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "best_combined_objective": best_loss,
        "cpu_wall_seconds": wall,
        "peak_process_resident_memory_bytes": peak_rss,
        "transition_offsets": protocol["single_bounded_change"][
            "transition_offsets"
        ],
        "final_transition_offset_probabilities": [
            float(item)
            for item in torch.softmax(
                model.copy_transition_logits.detach(), dim=0
            ).tolist()
        ],
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
        "history": history,
        "functional_validation_accessed_during_training": False,
        "test_split_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


@torch.inference_mode()
def evaluate(
    protocol_path: Path,
    artifact_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validation = protocol["validation"]
    dataset = ROOT / validation["path"]
    if _sha256(dataset) != validation["sha256"]:
        raise ValueError("validation dataset hash mismatch")
    if output_path.exists():
        raise RuntimeError("functional validation evidence is immutable")
    rows = [row for row in _load_rows(dataset) if row["split"] == "validation"]
    if len(rows) != validation["distinct_prompts"]:
        raise ValueError("validation row count mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    spec, _ = load_portable_artifact(artifact, "cpu")
    runtime = LayerCakeRuntime()
    runtime.install_portable_domain(artifact, "cpu")
    process = psutil.Process()
    records = []
    maximum_new_bytes = int(validation["maximum_new_bytes"])
    for index, row in enumerate(rows):
        prompt = (row["prompt"] + "\n").encode("utf-8")
        prefill_started = time.perf_counter()
        decoder = runtime.domains[spec.domain_id][1]
        prompt_tensor = torch.tensor(list(prompt), dtype=torch.long)[None]
        state = decoder.prefill_incremental(prompt_tensor)
        first = state["next_logits"].argmax(dim=-1, keepdim=True)
        ttft = time.perf_counter() - prefill_started
        generated = [int(first.item())]
        decoder.decode_incremental(first, state)
        for _ in range(maximum_new_bytes - 1):
            next_byte = state["next_logits"].argmax(dim=-1, keepdim=True)
            generated.append(int(next_byte.item()))
            decoder.decode_incremental(next_byte, state)
        latency = time.perf_counter() - prefill_started
        raw = bytes(generated)
        text = raw.decode("utf-8", errors="replace")
        source, parse_status = _extract_function(text, row["function_name"])
        passed = False
        tests = [{"status": parse_status}]
        if source is not None:
            passed, tests = _execute_tests(
                source, row["function_name"], row["tests"]
            )
        records.append(
            {
                "id": row["id"],
                "family": row["family"],
                "expected_function": row["function_name"],
                "generated_text": text,
                "generated_text_sha256": hashlib.sha256(raw).hexdigest(),
                "extracted_source": source,
                "functional_success": passed,
                "tests": tests,
                "time_to_first_output_seconds": ttft,
                "total_latency_seconds": latency,
            }
        )
        print(
            json.dumps(
                {
                    "evaluated": index + 1,
                    "total": len(rows),
                    "successes": sum(
                        record["functional_success"] for record in records
                    ),
                }
            ),
            flush=True,
        )
    successes = sum(record["functional_success"] for record in records)
    minimum = int(validation["minimum_strict_functional_successes"])
    evidence = {
        "format": "layercake-phase4-portable-functional-validation/1",
        "status": "PASS" if successes >= minimum else "FAIL",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "payload_hash": artifact["payload_hash"],
        "dataset": validation["path"],
        "dataset_sha256": _sha256(dataset),
        "split": "validation",
        "distinct_prompts": len(records),
        "functional_successes": successes,
        "functional_failures": len(records) - successes,
        "functional_success_rate": successes / max(1, len(records)),
        "minimum_functional_successes": minimum,
        "median_time_to_first_output_seconds": statistics.median(
            record["time_to_first_output_seconds"] for record in records
        ),
        "median_total_latency_seconds": statistics.median(
            record["total_latency_seconds"] for record in records
        ),
        "resident_memory_bytes_after": int(process.memory_info().rss),
        "autonomous_neural_generation": True,
        "persistent_incremental_state": True,
        "teacher_at_inference": False,
        "test_split_accessed": False,
        "records": records,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "train",
            "continue-identifier",
            "continue-pointer",
            "continue-transition-pointer",
            "convert-self-transition",
            "continue-markov-pointer",
            "convert-max-projection",
            "evaluate",
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT
        / "moonshot"
        / "phase4_portable_decoder_functional_preregistration.json",
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--initial-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = (
        args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    )
    artifact = (
        args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.command == "train":
        result = train(protocol, artifact, output)
    elif args.command == "continue-identifier":
        if args.initial_artifact is None:
            parser.error("continue-identifier requires --initial-artifact")
        initial_artifact = (
            args.initial_artifact
            if args.initial_artifact.is_absolute()
            else ROOT / args.initial_artifact
        )
        result = continue_identifier(
            protocol, initial_artifact, artifact, output
        )
    elif args.command == "continue-pointer":
        if args.initial_artifact is None:
            parser.error("continue-pointer requires --initial-artifact")
        initial_artifact = (
            args.initial_artifact
            if args.initial_artifact.is_absolute()
            else ROOT / args.initial_artifact
        )
        result = continue_pointer(
            protocol, initial_artifact, artifact, output
        )
    elif args.command == "continue-transition-pointer":
        if args.initial_artifact is None:
            parser.error(
                "continue-transition-pointer requires --initial-artifact"
            )
        initial_artifact = (
            args.initial_artifact
            if args.initial_artifact.is_absolute()
            else ROOT / args.initial_artifact
        )
        result = continue_transition_pointer(
            protocol, initial_artifact, artifact, output
        )
    elif args.command == "convert-self-transition":
        if args.initial_artifact is None:
            parser.error(
                "convert-self-transition requires --initial-artifact"
            )
        initial_artifact = (
            args.initial_artifact
            if args.initial_artifact.is_absolute()
            else ROOT / args.initial_artifact
        )
        result = convert_self_gated_transition(
            protocol, initial_artifact, artifact, output
        )
    elif args.command == "continue-markov-pointer":
        if args.initial_artifact is None:
            parser.error(
                "continue-markov-pointer requires --initial-artifact"
            )
        initial_artifact = (
            args.initial_artifact
            if args.initial_artifact.is_absolute()
            else ROOT / args.initial_artifact
        )
        result = continue_markov_pointer(
            protocol, initial_artifact, artifact, output
        )
    elif args.command == "convert-max-projection":
        if args.initial_artifact is None:
            parser.error(
                "convert-max-projection requires --initial-artifact"
            )
        initial_artifact = (
            args.initial_artifact
            if args.initial_artifact.is_absolute()
            else ROOT / args.initial_artifact
        )
        result = convert_max_projection(
            protocol, initial_artifact, artifact, output
        )
    else:
        result = evaluate(protocol, artifact, output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in result
                if key
                in {
                    "status",
                    "seed",
                    "best_response_cross_entropy",
                    "cpu_wall_seconds",
                    "functional_successes",
                    "functional_success_rate",
                    "evidence_sha256",
                }
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
