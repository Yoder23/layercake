"""Bounded capacity-control and shallow sparse English-core campaign."""

from __future__ import annotations

from contextlib import nullcontext
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
from typing import Any, Sequence

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from layercake.models.shallow_sparse_english import (
    ShallowSparseEnglishConfig,
    ShallowSparseEnglishCore,
)
from layercake.phase2_recertification import _output_quality
from layercake.representation_bakeoff import (
    QWEN_BPS,
    QWEN_QUALITY,
    _load_prompts,
    _select_token,
)
from layercake.training.data import sha256_file


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    Path(os.environ["USERPROFILE"])
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--distilgpt2"
    / "snapshots"
    / "2290a62682d06624634c1f46a6ad5be0f47f38aa"
)
CURRICULUM = (
    ROOT
    / "data"
    / "moonshot"
    / "phase2"
    / "instruction_curriculum_semantic_control_v2.jsonl"
)
WIKI = (
    ROOT / "data" / "moonshot" / "v2" / "wikitext103" / "train_development.bin"
)
VALIDATION = (
    ROOT / "data" / "moonshot" / "v2" / "wikitext103" / "validation.bin"
)
SELECTION = (
    ROOT
    / "data"
    / "moonshot"
    / "v2"
    / "wikitext103"
    / "architecture_selection.bin"
)
TASK_TAXONOMY = (
    "continuation",
    "explanation",
    "planning",
    "comparison",
    "instruction_following",
    "reasoning",
    "summarization",
    "question_answering",
    "repetition_control",
    "coherence_or_supplied_context",
)
TASK_INDEX = {name: index for index, name in enumerate(TASK_TAXONOMY)}
TASK_ALIASES = {
    "coherence": "coherence_or_supplied_context",
    "grounded_qa": "question_answering",
    "rewrite": "coherence_or_supplied_context",
    "email_from_notes": "instruction_following",
    "tone_and_format": "instruction_following",
    "combine_facts": "coherence_or_supplied_context",
    "entity_reference": "question_answering",
    "conversation": "coherence_or_supplied_context",
    "clarification": "question_answering",
    "abstention": "question_answering",
    "synthetic_rule": "reasoning",
    "distractor_resistance": "reasoning",
    "quoted_conflict": "instruction_following",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        json.loads(line)
        for line in CURRICULUM.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return (
        [row for row in rows if row["split"] == "train"],
        [row for row in rows if row["split"] == "instruction_validation"],
    )


def _task_id(row: dict[str, Any]) -> int:
    task = str(row["task"])
    task = TASK_ALIASES.get(task, task)
    if task not in TASK_INDEX:
        task = "coherence_or_supplied_context"
    return TASK_INDEX[task]


def _tokenizer(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1_000_000_000
    return tokenizer


def _instruction_batch(
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    max_tokens: int = 256,
    generated_prefixes: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    encoded: list[list[int]] = []
    labels: list[list[int]] = []
    prompt_lengths: list[int] = []
    task_ids: list[int] = []
    response_units = 0
    for index, row in enumerate(rows):
        prompt = tokenizer.encode(str(row["prompt"]) + "\n")
        response = tokenizer.encode(str(row["response"]))
        generated = [] if generated_prefixes is None else generated_prefixes[index]
        generated_count = min(len(generated), max(0, len(response) - 1))
        sequence = (
            prompt + generated + response[generated_count:]
        )[:max_tokens]
        target = [-100] * min(len(sequence), len(prompt) + len(generated))
        target.extend(
            response[generated_count:generated_count + max(0, len(sequence) - len(target))]
        )
        target = target[:len(sequence)]
        encoded.append(sequence)
        labels.append(target)
        prompt_lengths.append(min(len(prompt), len(sequence)))
        task_ids.append(_task_id(row))
        response_units += sum(value >= 0 for value in target)
    length = max(len(values) for values in encoded)
    input_ids = torch.full(
        (len(rows), length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.full(
        (len(rows), length),
        -100,
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros(
        (len(rows), length), dtype=torch.long, device=device
    )
    for index, (values, target) in enumerate(zip(encoded, labels)):
        input_ids[index, :len(values)] = torch.tensor(
            values, dtype=torch.long, device=device
        )
        target_ids[index, :len(target)] = torch.tensor(
            target, dtype=torch.long, device=device
        )
        attention[index, :len(values)] = 1
    return (
        input_ids,
        target_ids,
        attention,
        torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        response_units,
    ), torch.tensor(task_ids, dtype=torch.long, device=device)


def _shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].flatten(0, 1),
        labels[:, 1:].flatten(),
        ignore_index=-100,
    )


def _wiki_tokens(tokenizer) -> torch.Tensor:
    text = WIKI.read_bytes().decode("utf-8", errors="replace")
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def _wiki_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    length: int,
    generator: random.Random,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for _ in range(batch_size):
        start = generator.randrange(0, len(tokens) - length - 1)
        rows.append(tokens[start:start + length + 1])
    return torch.stack(rows).to(device)


@torch.inference_mode()
def _teacher_instruction_loss(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> float:
    losses = []
    for offset in range(0, min(64, len(rows)), 4):
        selected = rows[offset:offset + 4]
        batch, _ = _instruction_batch(
            tokenizer, selected, device=device
        )
        ids, labels, attention, _, _ = batch
        logits = model(input_ids=ids, attention_mask=attention).logits
        losses.append(float(_shifted_ce(logits, labels)))
    return statistics.mean(losses)


def train_teacher(output: Path, *, steps: int = 1200) -> dict[str, Any]:
    output = output if output.is_absolute() else (ROOT / output)
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"teacher artifact is immutable: {output}")
    torch.manual_seed(9824)
    random.seed(9824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _tokenizer(SOURCE)
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE, local_files_only=True
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.train()
    train_rows, validation_rows = _rows()
    wiki = _wiki_tokens(tokenizer)
    rng = random.Random(9824)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5.0e-5, weight_decay=0.01
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.float16))
        if use_amp
        else (lambda: nullcontext())
    )
    curves = []
    response_units = 0
    raw_bytes = 0
    successful = 0
    skipped = 0
    started = time.perf_counter()
    while successful < steps:
        selected = [train_rows[rng.randrange(len(train_rows))] for _ in range(4)]
        batch, _ = _instruction_batch(
            tokenizer, selected, device=device
        )
        ids, labels, attention, _, observed = batch
        wiki_ids = _wiki_batch(
            wiki,
            batch_size=4,
            length=255,
            generator=rng,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            instruction_logits = model(
                input_ids=ids, attention_mask=attention
            ).logits
            instruction_loss = _shifted_ce(instruction_logits, labels)
            wiki_logits = model(input_ids=wiki_ids).logits
            wiki_loss = F.cross_entropy(
                wiki_logits[:, :-1].flatten(0, 1),
                wiki_ids[:, 1:].flatten(),
            )
            loss = instruction_loss + 0.25 * wiki_loss
        before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before:
            skipped += 1
            continue
        successful += 1
        response_units += observed
        raw_bytes += sum(
            len((row["prompt"] + row["response"]).encode("utf-8"))
            for row in selected
        ) + 4 * 255
        if successful == 1 or successful % 300 == 0:
            model.eval()
            heldout = _teacher_instruction_loss(
                model,
                tokenizer,
                validation_rows,
                device=device,
            )
            model.train()
            curve = {
                "step": successful,
                "instruction_loss": float(instruction_loss.detach()),
                "wiki_loss": float(wiki_loss.detach()),
                "heldout_instruction_loss": heldout,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    model.eval()
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    checkpoint = output / "model.safetensors"
    metadata = {
        "format": "layercake-phase2-nonpromotable-capacity-control/1",
        "status": "CONTROL_ONLY",
        "model_id": "distilgpt2",
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(checkpoint),
        },
        "tokenizer": {
            "path": (output / "tokenizer.json").relative_to(ROOT).as_posix(),
            "sha256": sha256_file(output / "tokenizer.json"),
        },
        "source": {
            "snapshot": str(SOURCE),
            "model_sha256": sha256_file(SOURCE / "model.safetensors"),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training": {
            "steps": steps,
            "successful_optimizer_steps": successful,
            "skipped_amp_optimizer_steps": skipped,
            "response_tokens_seen": response_units,
            "raw_utf8_bytes_exposed": raw_bytes,
            "wikitext_regularization_weight": 0.25,
            "curves": curves,
            "wall_seconds": time.perf_counter() - started,
        },
        "test_accessed": False,
        "promotion_eligible": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def initialize_student(teacher_path: Path, output: Path) -> dict[str, Any]:
    teacher_path = (
        teacher_path if teacher_path.is_absolute() else ROOT / teacher_path
    ).resolve()
    output = (output if output.is_absolute() else ROOT / output).resolve()
    if output.exists():
        raise RuntimeError(f"student base artifact is immutable: {output}")
    torch.manual_seed(9824)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, local_files_only=True
    ).eval()
    tokenizer = _tokenizer(teacher_path)
    model = ShallowSparseEnglishCore()
    with torch.no_grad():
        model.transformer.wte.weight.copy_(teacher.transformer.wte.weight)
        model.transformer.wpe.weight.copy_(teacher.transformer.wpe.weight)
        model.transformer.ln_f.load_state_dict(
            teacher.transformer.ln_f.state_dict()
        )
        for target, source in zip((0, 1, 2), (0, 2, 5)):
            model.transformer.h[target].load_state_dict(
                teacher.transformer.h[source].state_dict()
            )
    if model.parameter_count() > 70_000_000:
        raise RuntimeError("student exceeds preregistered parameter ceiling")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(checkpoint),
    )
    tokenizer.save_pretrained(output)
    metadata = {
        "format": "layercake-shallow-sparse-english/1",
        "status": "UNTRAINED_TASK_CAKES",
        "architecture": model.config.canonical_dict(),
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(checkpoint),
        },
        "tokenizer": {
            "path": (output / "tokenizer.json").relative_to(ROOT).as_posix(),
            "sha256": sha256_file(output / "tokenizer.json"),
        },
        "initialization": {
            "teacher_checkpoint_sha256": sha256_file(
                teacher_path / "model.safetensors"
            ),
            "retained_source_blocks": [0, 2, 5],
            "source_blocks": 6,
            "student_blocks": 3,
        },
        "parameters": {
            "total": model.parameter_count(),
            "active": model.active_parameter_count(),
            "active_fraction": (
                model.active_parameter_count() / model.parameter_count()
            ),
        },
        "physical_sparsity": model.physical_sparse_contract(),
        "quality": {
            "validation": None,
            "architecture_selection": None,
            "test": None,
            "test_accessed": False,
        },
        "training": None,
        "incremental_state": {
            "implemented": True,
            "mechanism": "three-block GPT-2-compatible KV cache plus cached task route",
        },
        "test_accessed": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_student(path: Path, *, device: torch.device | str = "cpu"):
    path = (path if path.is_absolute() else ROOT / path).resolve()
    metadata = _read(path / "metadata.json")
    model = ShallowSparseEnglishCore(
        ShallowSparseEnglishConfig(**metadata["architecture"])
    )
    model.load_state_dict(
        load_file(str(path / "model.safetensors"), device=str(device)),
        strict=True,
    )
    return model.to(device).eval(), _tokenizer(path), metadata


@torch.inference_mode()
def _student_generated_prefixes(
    model: ShallowSparseEnglishCore,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    horizon: int,
    device: torch.device,
) -> list[list[int]]:
    prefixes = []
    model.eval()
    for row in rows:
        prompt = tokenizer.encode(str(row["prompt"]) + "\n")
        prompt = prompt[:240]
        route = torch.tensor([_task_id(row)], device=device)
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        result = model(
            ids,
            prompt_lengths=torch.tensor([len(prompt)], device=device),
            task_routes=route,
            use_cache=True,
        )
        state = {
            "past_key_values": result["past_key_values"],
            "task_routes": route,
            "next_logits": result["logits"][:, -1],
            "generated_ids": ids[:, :0],
        }
        for _ in range(horizon):
            _, state = model.decode_step(state)
        prefixes.append(state["generated_ids"][0].tolist())
    model.train()
    return prefixes


@torch.inference_mode()
def _student_instruction_loss(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> float:
    losses = []
    model.eval()
    for offset in range(0, min(64, len(rows)), 4):
        selected = rows[offset:offset + 4]
        batch, tasks = _instruction_batch(
            tokenizer, selected, device=device
        )
        ids, labels, attention, prompt_lengths, _ = batch
        result = model(
            ids,
            attention_mask=attention,
            prompt_lengths=prompt_lengths,
            task_routes=tasks,
        )
        losses.append(float(_shifted_ce(result["logits"], labels)))
    return statistics.mean(losses)


@torch.inference_mode()
def evaluate_bpb(
    model: ShallowSparseEnglishCore,
    tokenizer,
    corpus: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    raw = corpus.read_bytes()
    total_nll = 0.0
    total_bytes = 0
    total_tokens = 0
    correct = 0
    model.eval()
    for index in range(128):
        start = (index * 7919) % max(1, len(raw) - 256)
        payload = raw[start:start + 256]
        text = payload.decode("utf-8", errors="ignore")
        covered = len(text.encode("utf-8"))
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        result = model(
            tokens[:, :-1],
            task_routes=torch.zeros(1, dtype=torch.long, device=device),
        )
        logits = result["logits"]
        targets = tokens[:, 1:]
        total_nll += float(
            F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )
        )
        correct += int((logits.argmax(-1) == targets).sum())
        total_tokens += int(targets.numel())
        total_bytes += covered
    return {
        "bits_per_byte": total_nll / max(1, total_bytes) / __import__("math").log(2),
        "covered_raw_bytes": total_bytes,
        "evaluated_tokens": total_tokens,
        "token_accuracy": correct / max(1, total_tokens),
    }


def train_student(
    teacher_path: Path,
    base_path: Path,
    output: Path,
    *,
    steps: int = 2400,
) -> dict[str, Any]:
    teacher_path = (
        teacher_path if teacher_path.is_absolute() else ROOT / teacher_path
    ).resolve()
    base_path = (base_path if base_path.is_absolute() else ROOT / base_path).resolve()
    output = (output if output.is_absolute() else ROOT / output).resolve()
    if output.exists():
        raise RuntimeError(f"student artifact is immutable: {output}")
    torch.manual_seed(9824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, parent = load_student(base_path, device=device)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, local_files_only=True
    ).to(device).eval()
    model.train()
    train_rows, validation_rows = _rows()
    wiki = _wiki_tokens(tokenizer)
    rng = random.Random(9824)
    cake_parameters = list(model.task_classifier.parameters())
    for cake in model.task_cakes:
        cake_parameters.extend(cake.parameters())
    cake_ids = {id(parameter) for parameter in cake_parameters}
    shared = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in cake_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": shared, "lr": 2.0e-5},
            {"params": cake_parameters, "lr": 1.0e-4},
        ],
        weight_decay=0.01,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.float16))
        if use_amp
        else (lambda: nullcontext())
    )
    successful = 0
    skipped = 0
    recovery_batches = 0
    recovery_horizons = {"8": 0, "32": 0, "64": 0}
    response_units = 0
    raw_bytes = 0
    curves = []
    started = time.perf_counter()
    while successful < steps:
        selected = [train_rows[rng.randrange(len(train_rows))] for _ in range(4)]
        generated = None
        if successful >= 300 and successful % 4 == 0:
            horizon = (8, 32, 64)[recovery_batches % 3]
            generated = _student_generated_prefixes(
                model,
                tokenizer,
                selected,
                horizon=horizon,
                device=device,
            )
            recovery_batches += 1
            recovery_horizons[str(horizon)] += 1
        batch, tasks = _instruction_batch(
            tokenizer,
            selected,
            device=device,
            generated_prefixes=generated,
        )
        ids, labels, attention, prompt_lengths, observed = batch
        wiki_ids = _wiki_batch(
            wiki,
            batch_size=4,
            length=255,
            generator=rng,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            student = model(
                ids,
                attention_mask=attention,
                prompt_lengths=prompt_lengths,
                task_routes=tasks,
            )
            instruction_loss = _shifted_ce(student["logits"], labels)
            task_loss = F.cross_entropy(student["task_logits"], tasks)
            with torch.no_grad():
                teacher_logits = teacher(
                    input_ids=ids, attention_mask=attention
                ).logits
            response_mask = labels[:, 1:] >= 0
            student_response = student["logits"][:, :-1][response_mask]
            teacher_response = teacher_logits[:, :-1][response_mask]
            top_values, top_indexes = teacher_response.float().topk(
                64, dim=-1
            )
            teacher_probabilities = torch.softmax(top_values, dim=-1)
            student_log_probabilities = torch.log_softmax(
                student_response.float(), dim=-1
            ).gather(-1, top_indexes)
            distillation_loss = -(
                teacher_probabilities * student_log_probabilities
            ).sum(dim=-1).mean()
            wiki_result = model(
                wiki_ids[:, :-1],
                task_routes=torch.zeros(
                    wiki_ids.shape[0], dtype=torch.long, device=device
                ),
            )
            wiki_loss = F.cross_entropy(
                wiki_result["logits"].flatten(0, 1),
                wiki_ids[:, 1:].flatten(),
            )
            loss = (
                instruction_loss
                + 0.50 * distillation_loss
                + task_loss
                + 0.25 * wiki_loss
            )
        before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before:
            skipped += 1
            continue
        successful += 1
        response_units += observed
        raw_bytes += sum(
            len((row["prompt"] + row["response"]).encode("utf-8"))
            for row in selected
        ) + 4 * 255
        if successful == 1 or successful % 300 == 0:
            heldout = _student_instruction_loss(
                model,
                tokenizer,
                validation_rows,
                device=device,
            )
            curve = {
                "step": successful,
                "instruction_loss": float(instruction_loss.detach()),
                "distillation_loss": float(distillation_loss.detach()),
                "task_loss": float(task_loss.detach()),
                "wiki_loss": float(wiki_loss.detach()),
                "heldout_instruction_loss": heldout,
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(json.dumps(curve), flush=True)
    validation = evaluate_bpb(
        model, tokenizer, VALIDATION, device=device
    )
    selection = evaluate_bpb(
        model, tokenizer, SELECTION, device=device
    )
    model.eval()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "model.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(checkpoint),
    )
    tokenizer.save_pretrained(output)
    metadata = {
        **parent,
        "format": "layercake-shallow-sparse-english/1",
        "status": "TRAINED",
        "checkpoint": {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(checkpoint),
        },
        "tokenizer": {
            "path": (output / "tokenizer.json").relative_to(ROOT).as_posix(),
            "sha256": sha256_file(output / "tokenizer.json"),
        },
        "parent_checkpoint": parent["checkpoint"],
        "quality": {
            "validation": validation,
            "architecture_selection": selection,
            "test": None,
            "test_accessed": False,
        },
        "training": {
            "steps": steps,
            "successful_optimizer_steps": successful,
            "skipped_amp_optimizer_steps": skipped,
            "response_tokens_seen": response_units,
            "raw_utf8_bytes_exposed": raw_bytes,
            "teacher_checkpoint_sha256": sha256_file(
                teacher_path / "model.safetensors"
            ),
            "top_teacher_logits_distilled": 64,
            "wikitext_regularization_weight": 0.25,
            "self_generated_prefix_schedule": [8, 32, 64],
            "self_generated_prefix_recovery_batches": recovery_batches,
            "self_generated_prefix_horizon_batches": recovery_horizons,
            "curves": curves,
            "wall_seconds": time.perf_counter() - started,
        },
        "test_accessed": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _generate_student(
    model,
    tokenizer,
    prompt: str,
    *,
    output_bytes: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    started = time.perf_counter_ns()
    prompt_ids = tokenizer.encode(prompt + "\n")
    tokenized = time.perf_counter_ns()
    state = model.prefill(torch.tensor([prompt_ids], dtype=torch.long))
    first_ready = time.perf_counter_ns()
    generated_ids: list[int] = []
    decode_started = first_ready
    while True:
        selected = _select_token(
            state["next_logits"],
            generated_ids,
            repetition_penalty,
            no_repeat_ngram_size,
        )
        token_id = int(selected.item())
        generated_ids.append(token_id)
        _, state = model.decode_step(state, next_token=selected)
        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        payload = text.encode("utf-8")
        if len(payload) >= output_bytes:
            break
        if len(prompt_ids) + len(generated_ids) >= model.config.max_tokens:
            raise RuntimeError("student context ended before byte target")
    completed = time.perf_counter_ns()
    return {
        "generated_bytes": payload,
        "generated_ids": generated_ids,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_ids),
        "tokenization_seconds": (tokenized - started) / 1e9,
        "prefill_seconds": (first_ready - tokenized) / 1e9,
        "decode_seconds": (completed - decode_started) / 1e9,
        "total_latency_seconds": (completed - started) / 1e9,
        "time_to_first_output_seconds": (first_ready - started) / 1e9,
        "bytes_per_second_decode": len(payload) / (
            (completed - decode_started) / 1e9
        ),
        "bytes_per_second_total": len(payload) / (
            (completed - started) / 1e9
        ),
        "resident_memory_bytes_before": rss_before,
        "resident_memory_bytes_after": int(process.memory_info().rss),
        "cake_route": int(state["task_routes"].item()),
    }


def _generate_teacher(
    model,
    tokenizer,
    prompt: str,
    *,
    output_bytes: int,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    prompt_ids = tokenizer.encode(prompt + "\n")
    tokenized = time.perf_counter_ns()
    ids = torch.tensor(
        [prompt_ids], dtype=torch.long, device=next(model.parameters()).device
    )
    result = model(input_ids=ids, use_cache=True, return_dict=True)
    past = result.past_key_values
    next_logits = result.logits[:, -1]
    first_ready = time.perf_counter_ns()
    generated_ids: list[int] = []
    while True:
        selected = _select_token(
            next_logits,
            generated_ids,
            1.15,
            4,
        )
        generated_ids.append(int(selected.item()))
        result = model(
            input_ids=selected[:, None],
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = result.past_key_values
        next_logits = result.logits[:, -1]
        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        payload = text.encode("utf-8")
        if len(payload) >= output_bytes:
            break
        if len(prompt_ids) + len(generated_ids) >= 1024:
            raise RuntimeError("teacher context ended before byte target")
    completed = time.perf_counter_ns()
    return {
        "generated_bytes": payload,
        "generated_ids": generated_ids,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_ids),
        "tokenization_seconds": (tokenized - started) / 1e9,
        "prefill_seconds": (first_ready - tokenized) / 1e9,
        "decode_seconds": (completed - first_ready) / 1e9,
        "total_latency_seconds": (completed - started) / 1e9,
    }


def screen_teacher(
    checkpoint: Path,
    output: Path,
    *,
    output_bytes: int = 640,
) -> dict[str, Any]:
    checkpoint = (
        checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    ).resolve()
    output = (output if output.is_absolute() else ROOT / output).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = _tokenizer(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True
    ).to(device).eval()
    metadata = _read(checkpoint / "metadata.json")
    manifest_path, prompts = _load_prompts(ROOT)
    records = []
    for prompt in prompts:
        generated = _generate_teacher(
            model,
            tokenizer,
            prompt["text"],
            output_bytes=output_bytes,
        )
        payload = generated.pop("generated_bytes")
        generated_ids = generated.pop("generated_ids")
        records.append(
            {
                "prompt_id": prompt["id"],
                "prompt_sha256": prompt["sha256"],
                "category": prompt["category"],
                "generated_hex": payload.hex(),
                "generated_sha256": hashlib.sha256(payload).hexdigest(),
                "generated_token_ids_sha256": _canonical_sha(generated_ids),
                "metrics": _output_quality(payload),
                "timing": generated,
                "execution": {
                    "prompt_tokens": generated["prompt_tokens"],
                    "generated_tokens": generated["generated_tokens"],
                    "maximum_active_experts_per_generated_token": 0,
                    "external_path_counters": {
                        "planner_calls": 0,
                        "retrieval_calls": 0,
                        "stored_answer_calls": 0,
                        "template_calls": 0,
                        "forced_token_calls": 0,
                    },
                },
            }
        )
    aggregates = {
        name: statistics.mean(
            float(record["metrics"][name]) for record in records
        )
        for name in records[0]["metrics"]
    }
    document = {
        "format": "layercake-nonpromotable-capacity-control-screen/1",
        "status": "PASS",
        "candidate": checkpoint.name,
        "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "tokenizer_sha256": metadata["tokenizer"]["sha256"],
        "prompt_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "prompt_manifest_sha256": sha256_file(manifest_path),
        "distinct_prompts": len(prompts),
        "output_bytes_per_prompt_minimum": output_bytes,
        "device": str(device),
        "promotion_eligible": False,
        "test_accessed": False,
        "records": records,
        "aggregates": aggregates,
    }
    document["screen_sha256"] = _canonical_sha(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "checkpoint_sha256": document["checkpoint_sha256"],
        "screen_sha256": document["screen_sha256"],
        "aggregates": aggregates,
        "promotion_eligible": False,
    }


def screen_student(
    checkpoint: Path,
    output: Path,
    *,
    output_bytes: int = 640,
    threads: int = 14,
) -> dict[str, Any]:
    checkpoint = (
        checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
    ).resolve()
    output = (output if output.is_absolute() else ROOT / output).resolve()
    torch.set_num_threads(threads)
    model, tokenizer, metadata = load_student(checkpoint)
    manifest_path, prompts = _load_prompts(ROOT)
    warm = model.prefill(
        torch.tensor(
            [tokenizer.encode("Warm autonomous LayerCake generation.")],
            dtype=torch.long,
        )
    )
    model.decode_step(warm)
    records = []
    for prompt in prompts:
        result = _generate_student(
            model,
            tokenizer,
            prompt["text"],
            output_bytes=output_bytes,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
        )
        payload = result.pop("generated_bytes")
        generated_ids = result.pop("generated_ids")
        records.append(
            {
                "prompt_id": prompt["id"],
                "prompt_sha256": prompt["sha256"],
                "category": prompt["category"],
                "generated_hex": payload.hex(),
                "generated_sha256": hashlib.sha256(payload).hexdigest(),
                "generated_token_ids_sha256": _canonical_sha(generated_ids),
                "metrics": _output_quality(payload),
                "timing": {
                    key: result[key]
                    for key in (
                        "tokenization_seconds",
                        "prefill_seconds",
                        "decode_seconds",
                        "total_latency_seconds",
                        "time_to_first_output_seconds",
                        "bytes_per_second_decode",
                        "bytes_per_second_total",
                        "resident_memory_bytes_before",
                        "resident_memory_bytes_after",
                    )
                },
                "execution": {
                    "prompt_tokens": result["prompt_tokens"],
                    "generated_tokens": result["generated_tokens"],
                    "sparse_decode_steps": result["generated_tokens"],
                    "maximum_active_experts_per_generated_token": 1,
                    "task_cake_route": result["cake_route"],
                    "external_path_counters": {
                        "planner_calls": 0,
                        "retrieval_calls": 0,
                        "stored_answer_calls": 0,
                        "template_calls": 0,
                        "forced_token_calls": 0,
                    },
                },
            }
        )
    aggregates = {
        name: statistics.mean(
            float(record["metrics"][name]) for record in records
        )
        for name in records[0]["metrics"]
    }
    for name in (
        "tokenization_seconds",
        "time_to_first_output_seconds",
        "total_latency_seconds",
        "bytes_per_second_decode",
        "bytes_per_second_total",
        "resident_memory_bytes_after",
    ):
        aggregates[f"median_{name}"] = statistics.median(
            float(record["timing"][name]) for record in records
        )
    qwen = _read(ROOT / QWEN_QUALITY)[
        "systems"
    ]["qwen25-05b-cpu"]["aggregates"]
    comparison = {
        "repetition_rate_delta_layercake_minus_qwen": (
            aggregates["repetition_rate"] - qwen["repetition_rate"]
        ),
        "word_diversity_delta_layercake_minus_qwen": (
            aggregates["word_diversity"] - qwen["word_diversity"]
        ),
        "valid_utf8_delta_layercake_minus_qwen": (
            aggregates["valid_utf8"] - qwen["valid_utf8"]
        ),
        "transformer_relative_median_decode_throughput": (
            aggregates["median_bytes_per_second_decode"] / QWEN_BPS
        ),
        "product_surface_noninferiority_pass": (
            aggregates["repetition_rate"] <= qwen["repetition_rate"] + 0.02
            and aggregates["word_diversity"] >= qwen["word_diversity"] - 0.02
            and aggregates["invalid_output"] <= qwen["invalid_output"]
        ),
    }
    document = {
        "format": "layercake-representation-functional-screen/1",
        "status": "PASS",
        "representation": "gpt2_byte_fallback_bpe",
        "candidate": checkpoint.parent.name + "/" + checkpoint.name,
        "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "tokenizer_sha256": metadata["tokenizer"]["sha256"],
        "architecture": metadata["architecture"],
        "parameters": metadata["parameters"],
        "quality": metadata["quality"],
        "training": metadata["training"],
        "prompt_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "prompt_manifest_sha256": sha256_file(manifest_path),
        "distinct_prompts": len(prompts),
        "output_bytes_per_prompt_minimum": output_bytes,
        "decoding": {
            "mode": "deterministic_logit_control",
            "source": "checkpoint neural token logits",
            "external_override": False,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
        },
        "threads": threads,
        "test_accessed": False,
        "exact_command": " ".join(sys.argv),
        "records": records,
        "aggregates": aggregates,
        "qwen_product_reference_aggregates": qwen,
        "comparison": comparison,
    }
    document["screen_sha256"] = _canonical_sha(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "checkpoint_sha256": document["checkpoint_sha256"],
        "screen_sha256": document["screen_sha256"],
        "aggregates": aggregates,
        "comparison": comparison,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    teacher = sub.add_parser("train-teacher")
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--steps", type=int, default=1200)
    initialize = sub.add_parser("initialize-student")
    initialize.add_argument("--teacher", type=Path, required=True)
    initialize.add_argument("--output", type=Path, required=True)
    student = sub.add_parser("train-student")
    student.add_argument("--teacher", type=Path, required=True)
    student.add_argument("--base", type=Path, required=True)
    student.add_argument("--output", type=Path, required=True)
    student.add_argument("--steps", type=int, default=2400)
    screen = sub.add_parser("screen-student")
    screen.add_argument("--checkpoint", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--output-bytes", type=int, default=640)
    screen.add_argument("--threads", type=int, default=14)
    teacher_screen = sub.add_parser("screen-teacher")
    teacher_screen.add_argument("--checkpoint", type=Path, required=True)
    teacher_screen.add_argument("--output", type=Path, required=True)
    teacher_screen.add_argument("--output-bytes", type=int, default=640)
    args = parser.parse_args(argv)
    if args.command == "train-teacher":
        result = train_teacher(args.output, steps=args.steps)
    elif args.command == "initialize-student":
        result = initialize_student(args.teacher, args.output)
    elif args.command == "train-student":
        result = train_student(
            args.teacher, args.base, args.output, steps=args.steps
        )
    elif args.command == "screen-student":
        result = screen_student(
            args.checkpoint,
            args.output,
            output_bytes=args.output_bytes,
            threads=args.threads,
        )
    else:
        result = screen_teacher(
            args.checkpoint,
            args.output,
            output_bytes=args.output_bytes,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
