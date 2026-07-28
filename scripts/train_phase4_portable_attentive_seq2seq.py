from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import psutil
import torch
import torch.nn.functional as F

import _common
from layercake.portable_domain import (
    PortableDomainDecoder,
    PortableDomainSpec,
    build_portable_artifact,
    load_portable_artifact,
)
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _load_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _batch(
    rows: list[dict[str, Any]],
    *,
    maximum_prompt_bytes: int,
    maximum_response_bytes: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prompts: list[bytes] = []
    responses: list[bytes] = []
    identifier_spans: list[tuple[int, int]] = []
    pointer_spans: list[tuple[int, int]] = []
    for row in rows:
        prompt = (row["prompt"] + "\n").encode("utf-8")
        response = row["response"].encode("utf-8")
        identifier = row["function_name"].encode("utf-8")
        if len(prompt) > maximum_prompt_bytes:
            raise ValueError("prompt exceeds preregistered byte limit")
        if len(response) > maximum_response_bytes:
            raise ValueError("response exceeds preregistered byte limit")
        prompt_start = prompt.find(identifier)
        response_start = response.find(identifier)
        if prompt_start < 0 or response_start < 0:
            raise ValueError("function identifier is absent")
        prompts.append(prompt)
        responses.append(response)
        identifier_spans.append(
            (response_start, response_start + len(identifier))
        )
        pointer_spans.append((prompt_start, len(identifier)))

    prompt_width = max(len(value) for value in prompts)
    response_width = max(len(value) for value in responses)
    prompt_ids = torch.zeros(
        len(rows), prompt_width, dtype=torch.long, device=device
    )
    prompt_lengths = torch.empty(
        len(rows), dtype=torch.long, device=device
    )
    response_ids = torch.zeros(
        len(rows), response_width, dtype=torch.long, device=device
    )
    response_mask = torch.zeros(
        len(rows), response_width, dtype=torch.bool, device=device
    )
    identifier_mask = torch.zeros_like(response_mask)
    pointer_labels = torch.full(
        (len(rows), response_width),
        -100,
        dtype=torch.long,
        device=device,
    )
    for index, (
        prompt,
        response,
        identifier_span,
        pointer_span,
    ) in enumerate(
        zip(prompts, responses, identifier_spans, pointer_spans)
    ):
        prompt_ids[index, : len(prompt)] = torch.tensor(
            list(prompt), dtype=torch.long, device=device
        )
        prompt_lengths[index] = len(prompt)
        response_ids[index, : len(response)] = torch.tensor(
            list(response), dtype=torch.long, device=device
        )
        response_mask[index, : len(response)] = True
        identifier_start, identifier_stop = identifier_span
        identifier_mask[index, identifier_start:identifier_stop] = True
        prompt_start, identifier_length = pointer_span
        pointer_labels[
            index, identifier_start:identifier_stop
        ] = torch.arange(
            prompt_start,
            prompt_start + identifier_length,
            dtype=torch.long,
            device=device,
        )
    return {
        "prompt_ids": prompt_ids,
        "prompt_lengths": prompt_lengths,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "identifier_mask": identifier_mask,
        "pointer_labels": pointer_labels,
    }


class _BalancedSampler:
    def __init__(
        self,
        functional: list[dict[str, Any]],
        lexical: list[dict[str, Any]],
        *,
        seed: int,
        functional_per_batch: int,
        lexical_per_batch: int,
    ):
        self.groups = (functional, lexical)
        self.counts = (functional_per_batch, lexical_per_batch)
        self.orders = [list(range(len(group))) for group in self.groups]
        self.cursors = [len(order) for order in self.orders]
        self.random = random.Random(seed)

    def _take(self, group_index: int) -> list[dict[str, Any]]:
        group = self.groups[group_index]
        count = self.counts[group_index]
        order = self.orders[group_index]
        cursor = self.cursors[group_index]
        if cursor + count > len(order):
            self.random.shuffle(order)
            cursor = 0
        selected = [group[index] for index in order[cursor : cursor + count]]
        self.cursors[group_index] = cursor + count
        return selected

    def next(self) -> list[dict[str, Any]]:
        rows = self._take(0) + self._take(1)
        self.random.shuffle(rows)
        return rows


def _losses(
    model: PortableDomainDecoder,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    result = model.seq2seq_forward(
        batch["prompt_ids"],
        batch["prompt_lengths"],
        batch["response_ids"],
    )
    response_mask = batch["response_mask"]
    identifier_mask = batch["identifier_mask"]
    response_nll = F.nll_loss(
        result["logits"][response_mask],
        batch["response_ids"][response_mask],
    )
    pointer_alignment = F.cross_entropy(
        result["pointer_scores"][identifier_mask],
        batch["pointer_labels"][identifier_mask],
    )
    gate_targets = identifier_mask[response_mask].float()
    gate_loss = F.binary_cross_entropy_with_logits(
        result["gate_logits"].squeeze(-1)[response_mask],
        gate_targets,
    )
    return response_nll + pointer_alignment + gate_loss, {
        **result,
        "response_nll": response_nll,
        "pointer_alignment": pointer_alignment,
        "gate_loss": gate_loss,
    }


def _cpu_fallback_smoke(
    model: PortableDomainDecoder,
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    smoke = PortableDomainDecoder(
        feature_width=model.feature_width,
        hidden_width=model.hidden_width,
        architecture=model.architecture,
        embedding_width=model.embedding_width,
        pointer_width=model.pointer_width,
    )
    smoke.load_state_dict(model.state_dict())
    smoke.train()
    optimizer = torch.optim.AdamW(smoke.parameters(), lr=1e-5)
    batch = _batch(
        rows,
        maximum_prompt_bytes=int(settings["maximum_prompt_bytes"]),
        maximum_response_bytes=int(settings["maximum_response_bytes"]),
        device=torch.device("cpu"),
    )
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    started = time.perf_counter()
    loss, _ = _losses(smoke, batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    wall = time.perf_counter() - started
    return {
        "status": "PASS",
        "device": "cpu",
        "rows": len(rows),
        "optimizer_steps": 1,
        "loss": float(loss.item()),
        "wall_seconds": wall,
        "resident_memory_before_bytes": rss_before,
        "resident_memory_after_bytes": int(process.memory_info().rss),
    }


def train(
    protocol_path: Path,
    artifact_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol["status"]
        != "PREREGISTERED_BEFORE_IMPLEMENTATION_AND_TRAINING"
    ):
        raise ValueError("attentive seq2seq training is not preregistered")
    if artifact_path.exists() or evidence_path.exists():
        raise RuntimeError("attentive seq2seq outputs are immutable")
    data_contract = protocol["training_data"]
    functional_path = ROOT / data_contract["functional"]["path"]
    lexical_path = ROOT / data_contract["lexical_conformance"]["path"]
    if _sha256(functional_path) != data_contract["functional"]["sha256"]:
        raise ValueError("functional training data hash mismatch")
    if _sha256(lexical_path) != data_contract["lexical_conformance"]["sha256"]:
        raise ValueError("lexical training data hash mismatch")
    functional = [
        row for row in _load_rows(functional_path) if row["split"] == "train"
    ]
    lexical = [
        row for row in _load_rows(lexical_path) if row["split"] == "train"
    ]
    if len(functional) != data_contract["functional"]["train_rows"]:
        raise ValueError("functional training row count mismatch")
    if len(lexical) != data_contract["lexical_conformance"]["train_rows"]:
        raise ValueError("lexical training row count mismatch")

    settings = protocol["gpu_first_bounded_run"]
    if not torch.cuda.is_available():
        raise RuntimeError("preregistered primary CUDA device is unavailable")
    device = torch.device(settings["primary_device"])
    device_index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )
    device_name = torch.cuda.get_device_name(device_index)
    if device_name != settings["required_device_name"]:
        raise RuntimeError("CUDA device identity does not match protocol")
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    architecture = protocol["architecture"]
    model = PortableDomainDecoder(
        feature_width=int(architecture["encoder_direction_width"]),
        hidden_width=int(architecture["decoder_width"]),
        architecture=architecture["architecture"],
        embedding_width=int(architecture["embedding_width"]),
        pointer_width=int(architecture["attention_width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    sampling = data_contract["sampling_per_batch"]
    sampler = _BalancedSampler(
        functional,
        lexical,
        seed=seed + 1,
        functional_per_batch=int(sampling["functional_rows"]),
        lexical_per_batch=int(sampling["lexical_rows"]),
    )
    batch_settings = {
        "maximum_prompt_bytes": data_contract["maximum_prompt_bytes"],
        "maximum_response_bytes": data_contract["maximum_response_bytes"],
    }
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    best_loss = float("inf")
    best_state = None
    history = []
    visible_units = 0
    raw_bytes = 0
    model.train()
    steps = int(settings["optimizer_steps"])
    for step in range(1, steps + 1):
        rows = sampler.next()
        batch = _batch(rows, device=device, **batch_settings)
        visible_units += int(
            batch["prompt_lengths"].sum().item()
            + batch["response_mask"].sum().item()
        )
        raw_bytes += sum(
            len((row["prompt"] + "\n").encode("utf-8"))
            + len(row["response"].encode("utf-8"))
            for row in rows
        )
        loss, result = _losses(model, batch)
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
            identifier_mask = batch["identifier_mask"]
            pointer_top1 = result["pointer_scores"][
                identifier_mask
            ].argmax(dim=-1)
            record = {
                "step": step,
                "combined_objective": value,
                "response_nll": float(result["response_nll"].item()),
                "pointer_alignment_cross_entropy": float(
                    result["pointer_alignment"].item()
                ),
                "pointer_position_accuracy": float(
                    (
                        pointer_top1
                        == batch["pointer_labels"][identifier_mask]
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "copy_gate_binary_cross_entropy": float(
                    result["gate_loss"].item()
                ),
                "gradient_norm_before_clip": float(gradient_norm),
                "gpu_wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    if best_state is None:
        raise RuntimeError("attentive seq2seq produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    torch.cuda.synchronize(device_index)
    wall = time.perf_counter() - started
    peak_gpu = int(torch.cuda.max_memory_allocated(device_index))
    cpu_model = model.to("cpu")
    cpu_fallback = _cpu_fallback_smoke(
        cpu_model,
        functional[:4] + lexical[:4],
        batch_settings,
    )
    spec = PortableDomainSpec(
        domain_id="python",
        feature_width=cpu_model.feature_width,
        hidden_width=cpu_model.hidden_width,
        architecture=cpu_model.architecture,
        embedding_width=cpu_model.embedding_width,
        pointer_width=cpu_model.pointer_width,
    )
    artifact = build_portable_artifact(
        cpu_model,
        spec,
        training={
            "protocol": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": _sha256(protocol_path),
            "seed": seed,
            "primary_device": str(device),
            "primary_device_name": device_name,
            "precision": settings["precision"],
            "optimizer_steps": steps,
            "best_combined_objective": best_loss,
            "functional_training_data_sha256": _sha256(functional_path),
            "lexical_training_data_sha256": _sha256(lexical_path),
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    evidence = {
        "format": "layercake-phase4-portable-attentive-seq2seq-training/1",
        "status": "TRAINED",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "source_commit": _git_head(),
        "seed": seed,
        "primary_device": str(device),
        "primary_device_name": device_name,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "precision": settings["precision"],
        "optimizer_steps": steps,
        "batch_size": int(settings["batch_size"]),
        "parameters": cpu_model.parameter_count(),
        "trainable_parameters": sum(
            parameter.numel() for parameter in cpu_model.parameters()
        ),
        "raw_utf8_training_bytes_exposed": raw_bytes,
        "model_visible_nonpadding_units": visible_units,
        "best_combined_objective": best_loss,
        "gpu_wall_seconds": wall,
        "peak_accelerator_memory_bytes": peak_gpu,
        "peak_process_resident_memory_bytes": peak_rss,
        "cpu_fallback_smoke": cpu_fallback,
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
        "history": history,
        "functional_validation_accessed_during_training": False,
        "lexical_validation_accessed_during_training": False,
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
def diagnose(
    protocol_path: Path,
    artifact_path: Path,
    dataset_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("attentive seq2seq diagnostic is immutable")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    _, model = load_portable_artifact(artifact, "cpu")
    if model.architecture != "byte_attentive_seq2seq":
        raise ValueError("diagnostic requires attentive seq2seq artifact")
    if _sha256(dataset_path) != protocol["training_data"][
        "lexical_conformance"
    ]["sha256"]:
        raise ValueError("lexical diagnostic dataset hash mismatch")
    rows = [
        row for row in _load_rows(dataset_path) if row["split"] == "validation"
    ]
    expected_rows = protocol["validation_sequence"]["first"][
        "distinct_prompts"
    ]
    if len(rows) != expected_rows:
        raise ValueError("lexical diagnostic row count mismatch")
    batch = _batch(
        rows,
        maximum_prompt_bytes=int(
            protocol["training_data"]["maximum_prompt_bytes"]
        ),
        maximum_response_bytes=int(
            protocol["training_data"]["maximum_response_bytes"]
        ),
        device=torch.device("cpu"),
    )
    model.eval()
    result = model.seq2seq_forward(
        batch["prompt_ids"],
        batch["prompt_lengths"],
        batch["response_ids"],
    )
    identifier_mask = batch["identifier_mask"]
    response_mask = batch["response_mask"]
    positions = result["pointer_scores"].argmax(dim=-1)
    pointer_bytes = torch.gather(
        batch["prompt_ids"], 1, positions
    )
    gates = torch.sigmoid(result["gate_logits"].squeeze(-1))
    evidence = {
        "format": "layercake-phase4-attentive-seq2seq-lexical-diagnostic/1",
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "payload_hash": artifact["payload_hash"],
        "dataset": dataset_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(dataset_path),
        "split": "validation",
        "distinct_prompts": len(rows),
        "teacher_forced_identifier_units": int(identifier_mask.sum()),
        "teacher_forced_pointer_position_accuracy": float(
            (
                positions[identifier_mask]
                == batch["pointer_labels"][identifier_mask]
            )
            .float()
            .mean()
        ),
        "teacher_forced_pointer_byte_accuracy": float(
            (
                pointer_bytes[identifier_mask]
                == batch["response_ids"][identifier_mask]
            )
            .float()
            .mean()
        ),
        "teacher_forced_identifier_next_byte_accuracy": float(
            (
                result["logits"][identifier_mask].argmax(dim=-1)
                == batch["response_ids"][identifier_mask]
            )
            .float()
            .mean()
        ),
        "teacher_forced_response_next_byte_accuracy": float(
            (
                result["logits"][response_mask].argmax(dim=-1)
                == batch["response_ids"][response_mask]
            )
            .float()
            .mean()
        ),
        "mean_identifier_copy_gate_probability": float(
            gates[identifier_mask].mean()
        ),
        "mean_nonidentifier_copy_gate_probability": float(
            gates[response_mask & ~identifier_mask].mean()
        ),
        "functional_promotion_credit": 0,
        "test_split_accessed": False,
    }
    thresholds = protocol["validation_sequence"]["first"]
    evidence["status"] = (
        "PASS"
        if evidence["teacher_forced_pointer_byte_accuracy"]
        >= thresholds["minimum_teacher_forced_pointer_byte_accuracy"]
        and evidence["teacher_forced_identifier_next_byte_accuracy"]
        >= thresholds["minimum_teacher_forced_identifier_next_byte_accuracy"]
        else "FAIL"
    )
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "diagnose"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    protocol = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    artifact = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.command == "train":
        result = train(protocol, artifact, output)
    else:
        if args.dataset is None:
            parser.error("diagnose requires --dataset")
        dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
        result = diagnose(protocol, artifact, dataset, output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in result
                if key
                in {
                    "status",
                    "seed",
                    "best_combined_objective",
                    "gpu_wall_seconds",
                    "parameters",
                    "teacher_forced_pointer_byte_accuracy",
                    "teacher_forced_identifier_next_byte_accuracy",
                    "evidence_sha256",
                }
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
