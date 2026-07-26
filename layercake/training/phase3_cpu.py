"""CPU-only Phase 3 training and profiler for the sealed Phase 2 architecture.

The promoted path never loads pretrained weights.  It uses the Phase 2
tokenizer only as a reversible representation and trains the exact three-block
core plus physically selected instruction cakes from random initialization.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Iterator, Sequence

import psutil
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from layercake.models.shallow_sparse_english import ShallowSparseEnglishCore
from layercake.training.phase2_shallow_sparse import (
    CURRICULUM,
    ROOT,
    TASK_INDEX,
    TASK_TAXONOMY,
    TASK_ALIASES,
    VALIDATION,
    WIKI,
)


PHASE2_TOKENIZER = (
    ROOT
    / "artifacts"
    / "moonshot"
    / "phase2_shallow_sparse_pretrained"
    / "student2400-seed-9824"
)
PROTOCOL = ROOT / "moonshot" / "phase3_training_efficiency_lock.json"
EXPECTED_ARCHITECTURE_HASH = (
    "b8020040300df16d627895685b584a1b615492275bda59ed42fca52762e12ad0"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _task_id(row: dict[str, Any]) -> int:
    task = TASK_ALIASES.get(str(row["task"]), str(row["task"]))
    return TASK_INDEX.get(task, TASK_INDEX["coherence_or_supplied_context"])


def _load_instruction_rows(tokenizer) -> dict[int, list[dict[str, Any]]]:
    grouped = {index: [] for index in range(len(TASK_TAXONOMY))}
    for line in CURRICULUM.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["split"] != "train":
            continue
        prompt = tokenizer.encode(str(row["prompt"]) + "\n")
        response = tokenizer.encode(str(row["response"]))
        grouped[_task_id(row)].append(
            {
                "id": row["id"],
                "prompt": prompt,
                "response": response,
                "raw_bytes": len(
                    (str(row["prompt"]) + "\n" + str(row["response"])).encode(
                        "utf-8"
                    )
                ),
            }
        )
    if not all(grouped.values()):
        missing = [TASK_TAXONOMY[key] for key, rows in grouped.items() if not rows]
        raise RuntimeError(f"curriculum has no train rows for routes: {missing}")
    return grouped


@contextmanager
def _peak_rss_monitor() -> Iterator[dict[str, int]]:
    process = psutil.Process()
    state = {"peak": int(process.memory_info().rss)}
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.002):
            state["peak"] = max(state["peak"], int(process.memory_info().rss))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        stop.set()
        thread.join()
        state["peak"] = max(state["peak"], int(process.memory_info().rss))


class TrainingOnlyHorizonHeads(nn.Module):
    """Small projections removed before inference; horizon one is identity."""

    def __init__(self, width: int, horizons: Sequence[int]):
        super().__init__()
        self.horizons = tuple(int(value) for value in horizons)
        self.projections = nn.ModuleDict()
        for horizon in self.horizons:
            if horizon == 1:
                continue
            projection = nn.Linear(width, width, bias=False)
            nn.init.eye_(projection.weight)
            self.projections[str(horizon)] = projection

    def project(self, hidden: torch.Tensor, horizon: int) -> torch.Tensor:
        if horizon == 1:
            return hidden
        return self.projections[str(horizon)](hidden)


def _baseline() -> GPT2LMHeadModel:
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=768,
            n_layer=6,
            n_head=12,
            n_inner=3072,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
    )


def _weight(model: nn.Module) -> torch.Tensor:
    if isinstance(model, ShallowSparseEnglishCore):
        return model.output_weight
    if isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    raise TypeError(type(model))


def _hidden(
    model: nn.Module,
    input_ids: torch.Tensor,
    routes: torch.Tensor,
    prompt_lengths: torch.Tensor | None,
    attention_mask: torch.Tensor,
    *,
    sparse_vocabulary_gradient: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = _weight(model)
    # Sampled steps keep the tied vocabulary gradient row-sparse. Exact
    # normalization steps deliberately use a dense vocabulary gradient because
    # every output row contributes probability mass to the normalizer.
    embeds = F.embedding(
        input_ids,
        weight,
        sparse=sparse_vocabulary_gradient,
    )
    if isinstance(model, ShallowSparseEnglishCore):
        hidden = model.transformer(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        summary = model._prompt_summary(
            hidden,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
        )
        task_logits = model.task_classifier(summary)
        return model._dispatch(hidden, routes), task_logits
    hidden = model.transformer(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    return hidden, None


def _candidate_vocabulary(
    targets: Sequence[torch.Tensor],
    *,
    negatives: int,
    vocabulary_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    required = torch.cat([target.flatten() for target in targets])
    required = required[required >= 0]
    sampled = torch.randint(
        0, vocabulary_size, (negatives,), generator=generator, dtype=torch.long
    )
    return torch.unique(torch.cat((required.cpu(), sampled)), sorted=True)


def _sampled_multihorizon_loss(
    hidden: torch.Tensor,
    targets: dict[int, torch.Tensor],
    weight: torch.Tensor,
    heads: TrainingOnlyHorizonHeads,
    *,
    negatives: int,
    generator: torch.Generator,
    sparse_vocabulary_gradient: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    candidates = _candidate_vocabulary(
        list(targets.values()),
        negatives=negatives,
        vocabulary_size=weight.shape[0],
        generator=generator,
    ).to(hidden.device)
    output_embeddings = F.embedding(
        candidates,
        weight,
        sparse=sparse_vocabulary_gradient,
    )
    losses = []
    visible_targets = 0
    for horizon in heads.horizons:
        target = targets[horizon]
        valid = target >= 0
        projected = heads.project(hidden, horizon)[valid]
        mapped = torch.searchsorted(candidates, target[valid])
        logits = projected @ output_embeddings.transpose(0, 1)
        losses.append(F.cross_entropy(logits.float(), mapped))
        visible_targets += int(mapped.numel())
    loss = sum(value / (2 ** index) for index, value in enumerate(losses))
    return loss, {
        "candidate_vocabulary_rows": int(candidates.numel()),
        "supervised_target_units": visible_targets,
        "horizon_losses": [float(value.detach()) for value in losses],
    }


def _chunked_exact_next_token_loss(
    hidden: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    maximum_positions: int,
    vocabulary_chunk_rows: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Compute exact full-vocabulary CE without materializing full logits.

    The selected positions are deterministic and span the valid target range.
    Chunking changes peak temporary-logit memory only; the result and gradient
    are the exact softmax objective for those positions.
    """

    if maximum_positions <= 0:
        raise ValueError("maximum_positions must be positive")
    if vocabulary_chunk_rows <= 0:
        raise ValueError("vocabulary_chunk_rows must be positive")
    valid = target >= 0
    selected_hidden = hidden[valid]
    selected_target = target[valid]
    if selected_target.numel() == 0:
        raise ValueError("exact softmax requires at least one valid target")
    if selected_target.numel() > maximum_positions:
        indices = torch.linspace(
            0,
            selected_target.numel() - 1,
            steps=maximum_positions,
            device=selected_target.device,
        ).round().to(torch.long)
        selected_hidden = selected_hidden.index_select(0, indices)
        selected_target = selected_target.index_select(0, indices)
    target_embeddings = F.embedding(
        selected_target,
        weight,
        sparse=False,
    )
    target_logits = (selected_hidden * target_embeddings).sum(dim=-1).float()
    log_normalizer = None
    chunks = 0
    for start in range(0, weight.shape[0], vocabulary_chunk_rows):
        stop = min(start + vocabulary_chunk_rows, weight.shape[0])
        logits = selected_hidden @ weight[start:stop].transpose(0, 1)
        chunk_normalizer = torch.logsumexp(logits.float(), dim=-1)
        log_normalizer = (
            chunk_normalizer
            if log_normalizer is None
            else torch.logaddexp(log_normalizer, chunk_normalizer)
        )
        chunks += 1
    assert log_normalizer is not None
    loss = (log_normalizer - target_logits).mean()
    return loss, {
        "exact_softmax_positions": int(selected_target.numel()),
        "exact_softmax_vocabulary_rows": int(weight.shape[0]),
        "exact_softmax_chunks": chunks,
        "exact_softmax_loss": float(loss.detach()),
    }


def _foundation_batch(
    tokens: torch.Tensor,
    *,
    tokenizer,
    batch_size: int,
    sequence: int,
    horizons: Sequence[int],
    rng: random.Random,
) -> tuple[
    torch.Tensor,
    dict[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    int,
]:
    maximum = max(horizons)
    rows = []
    raw_bytes = 0
    for _ in range(batch_size):
        start = rng.randrange(0, len(tokens) - sequence - maximum)
        row = tokens[start : start + sequence + maximum]
        rows.append(row)
        raw_bytes += len(
            tokenizer.decode(
                row.tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).encode("utf-8")
        )
    full = torch.stack(rows)
    inputs = full[:, :sequence]
    targets = {
        horizon: full[:, horizon : horizon + sequence] for horizon in horizons
    }
    routes = torch.zeros(batch_size, dtype=torch.long)
    return inputs, targets, routes, torch.ones_like(inputs), raw_bytes


def _instruction_batch(
    rows: list[dict[str, Any]],
    *,
    tokenizer,
    batch_size: int,
    sequence: int,
    horizons: Sequence[int],
    rng: random.Random,
    route: int,
) -> tuple[
    torch.Tensor,
    dict[int, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    maximum = max(horizons)
    chosen = [rows[rng.randrange(len(rows))] for _ in range(batch_size)]
    encoded = []
    lengths = []
    prompt_lengths = []
    raw_bytes = 0
    for row in chosen:
        values = (row["prompt"] + row["response"])[: sequence + maximum]
        encoded.append(values)
        lengths.append(len(values))
        prompt_lengths.append(min(len(row["prompt"]), sequence))
        raw_bytes += len(
            tokenizer.decode(
                values,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).encode("utf-8")
        )
    full = torch.zeros(
        batch_size, sequence + maximum, dtype=torch.long
    )
    for index, values in enumerate(encoded):
        full[index, : len(values)] = torch.tensor(values)
    inputs = full[:, :sequence]
    attention = torch.zeros_like(inputs)
    for index, length in enumerate(lengths):
        attention[index, : min(length, sequence)] = 1
    targets = {}
    positions = torch.arange(sequence)[None]
    for horizon in horizons:
        target = full[:, horizon : horizon + sequence].clone()
        valid_length = torch.tensor(lengths)[:, None] - horizon
        response_start = torch.tensor(prompt_lengths)[:, None] - horizon
        valid = (positions < valid_length) & (positions >= response_start)
        target[~valid] = -100
        targets[horizon] = target
    return (
        inputs,
        targets,
        torch.full((batch_size,), route, dtype=torch.long),
        torch.tensor(prompt_lengths, dtype=torch.long),
        attention,
        raw_bytes,
    )


def _dense_parameters(
    model: nn.Module, heads: TrainingOnlyHorizonHeads
) -> list[nn.Parameter]:
    vocabulary_id = id(_weight(model))
    return [
        parameter
        for parameter in list(model.parameters()) + list(heads.parameters())
        if id(parameter) != vocabulary_id
    ]


def _sparse_sgd_step(weight: torch.Tensor, learning_rate: float) -> int:
    gradient = weight.grad
    if gradient is None:
        return 0
    if gradient.is_sparse:
        gradient = gradient.coalesce()
        rows = gradient.indices()[0]
        values = gradient.values()
        with torch.no_grad():
            weight.index_add_(0, rows, values, alpha=-learning_rate)
        updated_rows = int(torch.unique(rows).numel())
    else:
        # Exact full-vocabulary normalization necessarily touches every output
        # row. Keep the update stateless so it does not allocate dense Adam
        # moments for the 50,257 x 768 tied table.
        with torch.no_grad():
            weight.add_(gradient, alpha=-learning_rate)
        updated_rows = int(weight.shape[0])
    weight.grad = None
    return updated_rows


class RowWiseAdagradVocabulary:
    """Physically sparse scalar second-moment state for sampled vocab rows."""

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        learning_rate: float,
        dense_exact_learning_rate: float,
        epsilon: float = 1.0e-12,
    ):
        if learning_rate <= 0:
            raise ValueError("row-wise learning rate must be positive")
        self.weight = weight
        self.learning_rate = float(learning_rate)
        self.dense_exact_learning_rate = float(dense_exact_learning_rate)
        self.epsilon = float(epsilon)
        # A dict ensures optimizer state is allocated only for rows physically
        # encountered by sampled training; no vocabulary-sized moment tensor is
        # allocated merely because the vocabulary is installed.
        self.accumulator: dict[int, float] = {}

    def step(self) -> tuple[int, dict[str, float | int | str]]:
        gradient = self.weight.grad
        if gradient is None:
            return 0, {
                "kind": "none",
                "state_rows": len(self.accumulator),
                "logical_state_bytes": len(self.accumulator) * 12,
            }
        if not gradient.is_sparse:
            with torch.no_grad():
                self.weight.add_(
                    gradient,
                    alpha=-self.dense_exact_learning_rate,
                )
            self.weight.grad = None
            return int(self.weight.shape[0]), {
                "kind": "stateless_dense_exact_sgd",
                "state_rows": len(self.accumulator),
                "logical_state_bytes": len(self.accumulator) * 12,
                "learning_rate": self.dense_exact_learning_rate,
            }
        gradient = gradient.coalesce()
        rows = gradient.indices()[0]
        values = gradient.values()
        row_ids = [int(value) for value in rows.tolist()]
        row_mean_squares = values.square().mean(dim=1).detach().cpu().tolist()
        accumulated = []
        for row, mean_square in zip(row_ids, row_mean_squares):
            value = self.accumulator.get(row, 0.0) + float(mean_square)
            self.accumulator[row] = value
            accumulated.append(value)
        denominators = (
            torch.tensor(
                accumulated,
                dtype=values.dtype,
                device=values.device,
            ).sqrt()
            + self.epsilon
        )
        normalized = values / denominators[:, None]
        with torch.no_grad():
            self.weight.index_add_(
                0,
                rows,
                normalized,
                alpha=-self.learning_rate,
            )
        self.weight.grad = None
        update_rms = (
            normalized.square().mean(dim=1).sqrt() * self.learning_rate
        )
        return int(torch.unique(rows).numel()), {
            "kind": "physically_sparse_rowwise_adagrad",
            "state_rows": len(self.accumulator),
            "logical_state_bytes": len(self.accumulator) * 12,
            "learning_rate": self.learning_rate,
            "median_update_row_rms": float(update_rms.median()),
            "maximum_update_row_rms": float(update_rms.max()),
        }

    def state_tensors(self) -> dict[str, torch.Tensor]:
        rows = sorted(self.accumulator)
        return {
            "rows": torch.tensor(rows, dtype=torch.int64),
            "accumulator": torch.tensor(
                [self.accumulator[row] for row in rows],
                dtype=torch.float32,
            ),
        }

    def load_state_tensors(self, state: dict[str, torch.Tensor]) -> None:
        rows = state["rows"].tolist()
        values = state["accumulator"].tolist()
        if len(rows) != len(values) or len(set(rows)) != len(rows):
            raise RuntimeError("invalid row-wise vocabulary optimizer state")
        self.accumulator = {
            int(row): float(value) for row, value in zip(rows, values)
        }


def _estimated_training_operations(
    *,
    layers: int,
    batch: int,
    sequence: int,
    width: int,
    candidates: int,
    horizons: Sequence[int],
    active_cake: bool,
) -> int:
    # Multiply-accumulate accounting for forward and its two backward matrix
    # products.  This is an architecture-derived count, not a marketing FLOP.
    transformer_forward = layers * (
        12 * batch * sequence * width * width
        + 2 * batch * sequence * sequence * width
    )
    output_forward = (
        len(horizons) * batch * sequence * width * candidates
    )
    auxiliary_forward = (
        max(0, len(horizons) - 1)
        * batch
        * sequence
        * width
        * width
    )
    cake_forward = (
        2 * batch * sequence * width * 64 if active_cake else 0
    )
    return 3 * (
        transformer_forward
        + output_forward
        + auxiliary_forward
        + cake_forward
    )


def _estimated_exact_softmax_operations(
    *, positions: int, width: int, vocabulary: int
) -> int:
    # One forward matrix product and two backward matrix products.
    return 3 * positions * width * vocabulary


@torch.inference_mode()
def _evaluate_bpb(
    model: nn.Module,
    tokenizer,
    *,
    samples: int,
) -> dict[str, float | int]:
    raw = VALIDATION.read_bytes()
    total_nll = 0.0
    total_bytes = 0
    total_tokens = 0
    correct = 0
    model.eval()
    for index in range(samples):
        start = (index * 7919) % max(1, len(raw) - 256)
        payload = raw[start : start + 256]
        text = payload.decode("utf-8", errors="ignore")
        covered = len(text.encode("utf-8"))
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        tokens = torch.tensor([ids], dtype=torch.long)
        if isinstance(model, ShallowSparseEnglishCore):
            logits = model(
                tokens[:, :-1],
                task_routes=torch.zeros(1, dtype=torch.long),
            )["logits"]
        else:
            logits = model(
                input_ids=tokens[:, :-1],
                use_cache=False,
                return_dict=True,
            ).logits
        target = tokens[:, 1:]
        total_nll += float(
            F.cross_entropy(
                logits.flatten(0, 1), target.flatten(), reduction="sum"
            )
        )
        correct += int((logits.argmax(-1) == target).sum())
        total_tokens += int(target.numel())
        total_bytes += covered
    model.train()
    return {
        "bits_per_byte": total_nll / max(1, total_bytes) / math.log(2),
        "covered_raw_bytes": total_bytes,
        "evaluated_tokens": total_tokens,
        "token_accuracy": correct / max(1, total_tokens),
        "samples": samples,
    }


def _save_diagnostic_checkpoint(
    model: nn.Module,
    heads: TrainingOnlyHorizonHeads,
    directory: Path,
    metadata: dict[str, Any],
    vocabulary_optimizer_state: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if directory.exists():
        raise RuntimeError(f"diagnostic checkpoint is immutable: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    if isinstance(model, ShallowSparseEnglishCore):
        model_path = directory / "model.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            },
            str(model_path),
        )
    else:
        model.save_pretrained(directory, safe_serialization=True)
        model_path = directory / "model.safetensors"
    heads_path = directory / "training_only_horizon_heads.safetensors"
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in heads.state_dict().items()
        },
        str(heads_path),
    )
    metadata = {
        **metadata,
        "model": {
            "path": model_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(model_path),
            "bytes": model_path.stat().st_size,
        },
        "training_only_horizon_heads": {
            "path": heads_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(heads_path),
            "bytes": heads_path.stat().st_size,
            "included_in_final_inference": False,
        },
    }
    if vocabulary_optimizer_state is not None:
        optimizer_path = directory / "vocabulary_optimizer_state.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in vocabulary_optimizer_state.items()
            },
            str(optimizer_path),
        )
        metadata["training_only_vocabulary_optimizer_state"] = {
            "path": optimizer_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(optimizer_path),
            "bytes": optimizer_path.stat().st_size,
            "included_in_final_inference": False,
        }
    _write(directory / "metadata.json", metadata)
    return metadata


def _validate_lock() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "LOCKED_BEFORE_PROMOTED_CPU_TRAINING":
        raise RuntimeError("Phase 3 protocol is not locked")
    if (
        protocol["phase2_parent"]["architecture_sha256"]
        != EXPECTED_ARCHITECTURE_HASH
    ):
        raise RuntimeError("Phase 2 architecture lineage changed")
    for entry in protocol["frozen_data"].values():
        path = ROOT / entry["path"]
        if not path.is_file() or _sha(path) != entry["sha256"]:
            raise RuntimeError(f"frozen Phase 3 input is stale: {path}")
    return protocol


def profile_training(
    *,
    system: str,
    output: Path,
    seed: int,
    steps: int,
    warmup_steps: int,
    batch_size: int,
    sequence: int,
    negatives: int,
    horizons: Sequence[int],
    threads: int,
    evaluation_samples: int = 0,
    checkpoint_directory: Path | None = None,
    target_units: int | None = None,
    progress_every: int = 0,
    resume_directory: Path | None = None,
    instruction_period: int = 4,
    exact_softmax_period: int = 0,
    exact_softmax_positions: int = 64,
    exact_softmax_chunk_rows: int = 4096,
    exact_softmax_weight: float = 1.0,
    vocabulary_optimizer_mode: str = "stateless_sgd",
    rowwise_adagrad_learning_rate: float = 0.0,
) -> dict[str, Any]:
    protocol = _validate_lock()
    if system not in {"layercake_complete", "dense_transformer"}:
        raise ValueError(f"unsupported profiled system: {system}")
    if output.exists():
        raise RuntimeError(f"profile evidence is immutable: {output}")
    if exact_softmax_period < 0:
        raise ValueError("exact_softmax_period cannot be negative")
    if exact_softmax_positions <= 0:
        raise ValueError("exact_softmax_positions must be positive")
    if exact_softmax_chunk_rows <= 0:
        raise ValueError("exact_softmax_chunk_rows must be positive")
    if exact_softmax_weight < 0:
        raise ValueError("exact_softmax_weight cannot be negative")
    if vocabulary_optimizer_mode not in {
        "stateless_sgd",
        "rowwise_adagrad_sparse",
    }:
        raise ValueError(
            f"unsupported vocabulary optimizer: {vocabulary_optimizer_mode}"
        )
    if (
        vocabulary_optimizer_mode == "rowwise_adagrad_sparse"
        and rowwise_adagrad_learning_rate <= 0
    ):
        raise ValueError(
            "row-wise Adagrad requires a positive measured learning rate"
        )
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.memory_allocated() != 0:
        raise RuntimeError("CUDA was already allocated in the CPU-only process")
    process = psutil.Process()
    cpu_start = process.cpu_times()
    end_to_end_started = time.perf_counter()
    tokenization_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        PHASE2_TOKENIZER, local_files_only=True
    )
    wiki_text = WIKI.read_bytes().decode("utf-8", errors="replace")
    wiki_tokens = torch.tensor(tokenizer.encode(wiki_text), dtype=torch.long)
    instruction_rows = _load_instruction_rows(tokenizer)
    tokenization_seconds = time.perf_counter() - tokenization_started
    initialization_started = time.perf_counter()
    model: nn.Module = (
        ShallowSparseEnglishCore()
        if system == "layercake_complete"
        else _baseline()
    )
    model.train()
    heads = TrainingOnlyHorizonHeads(768, horizons)
    parent_checkpoint = None
    if resume_directory is not None:
        parent_metadata_path = resume_directory / "metadata.json"
        parent_metadata = json.loads(
            parent_metadata_path.read_text(encoding="utf-8")
        )
        if (
            parent_metadata.get("random_initialization") is not True
            or parent_metadata.get("pretrained_weights_loaded") is not False
            or parent_metadata.get("system") != system
        ):
            raise RuntimeError("resume parent violates random-init lineage")
        model_path = resume_directory / "model.safetensors"
        model.load_state_dict(load_file(str(model_path)), strict=True)
        heads.load_state_dict(
            load_file(
                str(
                    resume_directory
                    / "training_only_horizon_heads.safetensors"
                )
            ),
            strict=True,
        )
        parent_checkpoint = {
            "path": resume_directory.relative_to(ROOT).as_posix(),
            "metadata_sha256": _sha(parent_metadata_path),
            "model_sha256": _sha(model_path),
            "optimizer_state_reset": True,
        }
    dense = _dense_parameters(model, heads)
    optimizer = torch.optim.AdamW(
        dense, lr=3.0e-4, betas=(0.9, 0.95), weight_decay=0.1
    )
    vocabulary_optimizer = (
        RowWiseAdagradVocabulary(
            _weight(model),
            learning_rate=rowwise_adagrad_learning_rate,
            dense_exact_learning_rate=1.0e-2,
        )
        if vocabulary_optimizer_mode == "rowwise_adagrad_sparse"
        else None
    )
    vocabulary_optimizer_state_loaded = False
    if vocabulary_optimizer is not None and resume_directory is not None:
        state_path = (
            resume_directory / "vocabulary_optimizer_state.safetensors"
        )
        if state_path.is_file():
            vocabulary_optimizer.load_state_tensors(
                load_file(str(state_path))
            )
            vocabulary_optimizer_state_loaded = True
    initialization_seconds = time.perf_counter() - initialization_started
    rng = random.Random(seed)
    negative_generator = torch.Generator().manual_seed(seed ^ 0xBAD5EED)
    route_losses = {route: 10.0 for route in range(10)}
    step_records = []
    raw_bytes = 0
    model_visible_units = 0
    supervised_units = 0
    operation_count = 0
    vocabulary_rows_updated = set()
    training_started = time.perf_counter()
    with _peak_rss_monitor() as memory:
        for step in range(1, warmup_steps + steps + 1):
            # The first frozen 5M-unit interval uses a deterministic shared
            # trace.  Later intervals may adapt the *shared* trace from paired
            # validation deficits, but may never let each model choose its own
            # examples because that would break paired data order.
            instruction = step % instruction_period == 0
            exact_softmax_step = (
                exact_softmax_period > 0
                and step % exact_softmax_period == 0
                and not instruction
            )
            prompt_lengths = None
            if instruction:
                route = (
                    (step // instruction_period) - 1
                ) % len(TASK_TAXONOMY)
                (
                    inputs,
                    targets,
                    routes,
                    prompt_lengths,
                    attention_mask,
                    exposed,
                ) = _instruction_batch(
                    instruction_rows[route],
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    sequence=sequence,
                    horizons=horizons,
                    rng=rng,
                    route=route,
                )
            else:
                route = 0
                (
                    inputs,
                    targets,
                    routes,
                    attention_mask,
                    exposed,
                ) = _foundation_batch(
                    wiki_tokens,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                    sequence=sequence,
                    horizons=horizons,
                    rng=rng,
                )
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            hidden, task_logits = _hidden(
                model,
                inputs,
                routes,
                prompt_lengths,
                attention_mask,
                sparse_vocabulary_gradient=not exact_softmax_step,
            )
            loss, loss_metrics = _sampled_multihorizon_loss(
                hidden,
                targets,
                _weight(model),
                heads,
                negatives=negatives,
                generator=negative_generator,
                sparse_vocabulary_gradient=not exact_softmax_step,
            )
            exact_metrics = None
            if exact_softmax_step:
                exact_loss, exact_metrics = _chunked_exact_next_token_loss(
                    hidden,
                    targets[1],
                    _weight(model),
                    maximum_positions=exact_softmax_positions,
                    vocabulary_chunk_rows=exact_softmax_chunk_rows,
                )
                loss = loss + exact_softmax_weight * exact_loss
            if task_logits is not None and instruction:
                loss = loss + F.cross_entropy(task_logits, routes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense, 1.0)
            optimizer.step()
            if vocabulary_optimizer is None:
                sparse_rows = _sparse_sgd_step(_weight(model), 1.0e-2)
                vocabulary_step = {
                    "kind": (
                        "stateless_dense_exact_sgd"
                        if exact_softmax_step
                        else "stateless_row_sparse_sgd"
                    ),
                    "state_rows": 0,
                    "logical_state_bytes": 0,
                    "learning_rate": 1.0e-2,
                }
            else:
                sparse_rows, vocabulary_step = vocabulary_optimizer.step()
            elapsed = time.perf_counter() - started
            if instruction:
                route_losses[route] = (
                    0.9 * route_losses[route] + 0.1 * float(loss.detach())
                )
            if step > warmup_steps:
                raw_bytes += exposed
                model_visible_units += int(attention_mask.sum())
                supervised_units += int(loss_metrics["supervised_target_units"])
                candidates = int(loss_metrics["candidate_vocabulary_rows"])
                operation_count += _estimated_training_operations(
                    layers=3 if system == "layercake_complete" else 6,
                    batch=batch_size,
                    sequence=sequence,
                    width=768,
                    candidates=candidates,
                    horizons=horizons,
                    active_cake=system == "layercake_complete",
                )
                if exact_metrics is not None:
                    operation_count += _estimated_exact_softmax_operations(
                        positions=int(
                            exact_metrics["exact_softmax_positions"]
                        ),
                        width=768,
                        vocabulary=int(
                            exact_metrics["exact_softmax_vocabulary_rows"]
                        ),
                    )
                step_records.append(
                    {
                        "step": step - warmup_steps,
                        "wall_seconds": elapsed,
                        "loss": float(loss.detach()),
                        "instruction_batch": instruction,
                        "route": route,
                        "candidate_vocabulary_rows": candidates,
                        "sparse_vocabulary_rows_updated": sparse_rows,
                        "vocabulary_gradient_kind": (
                            "dense_exact_full_vocabulary"
                            if exact_metrics is not None
                            else "row_sparse_sampled"
                        ),
                        "vocabulary_optimizer_step": vocabulary_step,
                        "exact_softmax": exact_metrics,
                    }
                )
                vocabulary_rows_updated.add(sparse_rows)
                if (
                    progress_every
                    and len(step_records) % progress_every == 0
                ):
                    print(
                        json.dumps(
                            {
                                "system": system,
                                "step": len(step_records),
                                "model_visible_nonpadding_units": (
                                    model_visible_units
                                ),
                                "latest_loss": float(loss.detach()),
                                "elapsed_seconds": (
                                    time.perf_counter() - training_started
                                ),
                                "peak_rss": memory["peak"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if target_units is not None and model_visible_units >= target_units:
                    break
    training_seconds = time.perf_counter() - training_started
    evaluation_started = time.perf_counter()
    quality = (
        _evaluate_bpb(model, tokenizer, samples=evaluation_samples)
        if evaluation_samples
        else None
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    checkpoint_seconds = 0.0
    checkpoint = None
    if checkpoint_directory is not None:
        checkpoint_started = time.perf_counter()
        checkpoint = _save_diagnostic_checkpoint(
            model,
            heads,
            checkpoint_directory,
            {
                "format": "layercake-phase3-diagnostic-checkpoint/1",
                "status": "DIAGNOSTIC_NOT_PROMOTED",
                "system": system,
                "seed": seed,
                "random_initialization": True,
                "pretrained_weights_loaded": False,
                "training_parent": parent_checkpoint,
                "optimizer_state_reset_at_resume": (
                    resume_directory is not None
                ),
                "instruction_period_steps": instruction_period,
                "exact_softmax_period_steps": exact_softmax_period,
                "exact_softmax_positions": exact_softmax_positions,
                "exact_softmax_chunk_rows": exact_softmax_chunk_rows,
                "exact_softmax_weight": exact_softmax_weight,
                "vocabulary_optimizer_mode": vocabulary_optimizer_mode,
                "rowwise_adagrad_learning_rate": (
                    rowwise_adagrad_learning_rate
                ),
                "vocabulary_optimizer_state_loaded": (
                    vocabulary_optimizer_state_loaded
                ),
                "model_visible_nonpadding_units": model_visible_units,
                "raw_utf8_training_bytes_exposed": raw_bytes,
                "quality": quality,
            },
            (
                vocabulary_optimizer.state_tensors()
                if vocabulary_optimizer is not None
                else None
            ),
        )
        checkpoint_seconds = time.perf_counter() - checkpoint_started
    end_to_end_seconds = time.perf_counter() - end_to_end_started
    cpu_end = process.cpu_times()
    dense_parameters = sum(value.numel() for value in dense)
    total_parameters = sum(value.numel() for value in model.parameters())
    active_parameters = (
        model.active_parameter_count()
        if isinstance(model, ShallowSparseEnglishCore)
        else total_parameters
    )
    measured_step_seconds = sum(row["wall_seconds"] for row in step_records)
    result = {
        "format": "layercake-phase3-cpu-training-profile/1",
        "status": "PASS",
        "promotion_eligible": False,
        "profile_role": "pre-promoted systems and feasibility measurement",
        "system": system,
        "seed": seed,
        "device": "cpu",
        "accelerator_hours": 0.0,
        "cuda_memory_allocated_before_and_after": [0, torch.cuda.memory_allocated()],
        "protocol": {
            "path": PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": _sha(PROTOCOL),
            "phase2_architecture_sha256": EXPECTED_ARCHITECTURE_HASH,
        },
        "configuration": {
            "steps": steps,
            "target_model_visible_nonpadding_units": target_units,
            "warmup_steps": warmup_steps,
            "batch_size": batch_size,
            "sequence_units": sequence,
            "negative_vocabulary_rows": negatives,
            "prediction_horizons": list(horizons),
            "threads": threads,
            "random_initialization": True,
            "pretrained_weights_loaded": False,
            "resume_parent": parent_checkpoint,
            "optimizer_state_reset_at_resume": (
                resume_directory is not None
            ),
            "instruction_period_steps": instruction_period,
            "exact_softmax_period_steps": exact_softmax_period,
            "exact_softmax_positions": exact_softmax_positions,
            "exact_softmax_chunk_rows": exact_softmax_chunk_rows,
            "exact_softmax_weight": exact_softmax_weight,
            "vocabulary_optimizer_mode": vocabulary_optimizer_mode,
            "rowwise_adagrad_learning_rate": (
                rowwise_adagrad_learning_rate
            ),
            "vocabulary_optimizer_state_loaded": (
                vocabulary_optimizer_state_loaded
            ),
            "tokenizer_only_reused": True,
            "route_homogeneous_batches": True,
            "sparse_vocabulary_gradient": exact_softmax_period == 0,
            "vocabulary_gradient_mode": (
                "row-sparse sampled steps plus infrequent dense exact steps"
                if exact_softmax_period
                else "row-sparse sampled steps"
            ),
            "vocabulary_optimizer": (
                (
                    "physically sparse row-wise Adagrad on sampled rows; "
                    "stateless dense SGD on exact steps"
                )
                if vocabulary_optimizer is not None
                else (
                    "stateless hybrid sparse/dense SGD"
                    if exact_softmax_period
                    else "stateless row-sparse SGD"
                )
            ),
            "dense_optimizer": "AdamW",
        },
        "accounting": {
            "raw_utf8_training_bytes_exposed": raw_bytes,
            "model_visible_nonpadding_units": model_visible_units,
            "forward_backward_supervised_target_units": supervised_units,
            "optimizer_steps": len(step_records),
            "estimated_executed_cpu_multiply_accumulates": operation_count,
            "dense_trainable_parameters": dense_parameters,
            "total_model_parameters": total_parameters,
            "active_model_parameters": active_parameters,
            "active_parameter_seconds": active_parameters * measured_step_seconds,
            "tokenizer_training_wall_time_seconds": 0.0,
            "tokenization_wall_time_seconds": tokenization_seconds,
            "model_initialization_wall_time_seconds": initialization_seconds,
            "training_loop_wall_time_seconds_including_warmup": training_seconds,
            "measured_step_wall_time_seconds": measured_step_seconds,
            "evaluation_wall_time_seconds": evaluation_seconds,
            "checkpointing_wall_time_seconds": checkpoint_seconds,
            "end_to_end_wall_time_seconds": end_to_end_seconds,
            "median_step_wall_time_seconds": float(
                torch.tensor(
                    [row["wall_seconds"] for row in step_records]
                ).median()
            ),
            "units_per_wall_second": (
                model_visible_units / measured_step_seconds
            ),
            "peak_process_resident_memory_bytes": memory["peak"],
            "vocabulary_optimizer_state_rows": (
                len(vocabulary_optimizer.accumulator)
                if vocabulary_optimizer is not None
                else 0
            ),
            "vocabulary_optimizer_logical_state_bytes": (
                len(vocabulary_optimizer.accumulator) * 12
                if vocabulary_optimizer is not None
                else 0
            ),
            "process_user_cpu_seconds": cpu_end.user - cpu_start.user,
            "process_system_cpu_seconds": cpu_end.system - cpu_start.system,
        },
        "step_records": step_records,
        "quality": quality,
        "checkpoint": (
            {
                "path": checkpoint_directory.relative_to(ROOT).as_posix(),
                "metadata_sha256": _sha(
                    checkpoint_directory / "metadata.json"
                ),
                "model_sha256": checkpoint["model"]["sha256"],
            }
            if checkpoint is not None and checkpoint_directory is not None
            else None
        ),
        "training_parent": parent_checkpoint,
        "phase2_inference_architecture_unchanged": (
            system == "layercake_complete"
        ),
        "final_inference_auxiliaries": [],
        "hardware": protocol["declared_cpu"],
    }
    result["evidence_sha256"] = _canonical_sha(result)
    _write(output, result)
    return result


def _parse_horizons(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item)
    if not parsed or parsed[0] != 1 or sorted(set(parsed)) != list(parsed):
        raise argparse.ArgumentTypeError(
            "horizons must be sorted unique integers beginning with 1"
        )
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile exact Phase 3 CPU-only training systems"
    )
    parser.add_argument(
        "system", choices=("layercake_complete", "dense_transformer")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9824)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--negatives", type=int, default=2048)
    parser.add_argument("--horizons", type=_parse_horizons, default=(1, 2, 4))
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--evaluation-samples", type=int, default=0)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--target-units", type=int)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--resume-directory", type=Path)
    parser.add_argument("--instruction-period", type=int, default=4)
    parser.add_argument("--exact-softmax-period", type=int, default=0)
    parser.add_argument("--exact-softmax-positions", type=int, default=64)
    parser.add_argument("--exact-softmax-chunk-rows", type=int, default=4096)
    parser.add_argument("--exact-softmax-weight", type=float, default=1.0)
    parser.add_argument(
        "--vocabulary-optimizer",
        choices=("stateless_sgd", "rowwise_adagrad_sparse"),
        default="stateless_sgd",
    )
    parser.add_argument(
        "--rowwise-adagrad-learning-rate",
        type=float,
        default=0.0,
    )
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    checkpoint_directory = args.checkpoint_directory
    if checkpoint_directory is not None and not checkpoint_directory.is_absolute():
        checkpoint_directory = ROOT / checkpoint_directory
    resume_directory = args.resume_directory
    if resume_directory is not None and not resume_directory.is_absolute():
        resume_directory = ROOT / resume_directory
    result = profile_training(
        system=args.system,
        output=output,
        seed=args.seed,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        sequence=args.sequence,
        negatives=args.negatives,
        horizons=args.horizons,
        threads=args.threads,
        evaluation_samples=args.evaluation_samples,
        checkpoint_directory=checkpoint_directory,
        target_units=args.target_units,
        progress_every=args.progress_every,
        resume_directory=resume_directory,
        instruction_period=args.instruction_period,
        exact_softmax_period=args.exact_softmax_period,
        exact_softmax_positions=args.exact_softmax_positions,
        exact_softmax_chunk_rows=args.exact_softmax_chunk_rows,
        exact_softmax_weight=args.exact_softmax_weight,
        vocabulary_optimizer_mode=args.vocabulary_optimizer,
        rowwise_adagrad_learning_rate=args.rowwise_adagrad_learning_rate,
    )
    print(json.dumps(result["accounting"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
