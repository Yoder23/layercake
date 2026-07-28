"""Generic Phase 5 data, training, evaluation, and frozen-core controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
from typing import Any, Mapping

import psutil
import torch
import torch.nn.functional as F

import _common
from layercake.portable_token_plan import (
    BOS_ID,
    EOS_ID,
    LosslessLexemePointerTokenizer,
    PortableTokenPlan,
    build_token_plan_artifact,
    load_token_plan_artifact,
)
from layercake.runtime.native.shallow_sparse_onnx import (
    NativeRuntime,
    _generate as generate_native_core,
)
from layercake.training.generic_domain import (
    CONFIG_FORMAT,
    canonical_sha,
    evaluate_generated,
    load_dataset,
    render_dataset,
    sha256_file,
    write_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "moonshot/phase5_generic_multidomain_preregistration.json"
CONTRACT_SHA256 = (
    "49ae047d9a2a11e066404cd2944b43590d08ce20971928996b77e3c62b747597"
)
TRAINING_FORMAT = "layercake-phase5-generic-domain-training-protocol/1"
EVALUATION_FORMAT = "layercake-phase5-generic-domain-evaluation-protocol/1"
BASELINE_FORMAT = "layercake-phase5-frozen-core-baseline-protocol/1"


def _rooted(path: Path) -> Path:
    value = path if path.is_absolute() else ROOT / path
    value = value.resolve()
    try:
        value.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("Phase 5 path escapes repository") from error
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_parent_contract(protocol: Mapping[str, Any]) -> None:
    if sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise ValueError("Phase 5 preregistration changed")
    parent = protocol.get("parent_contract")
    if parent != {
        "path": _relative(CONTRACT),
        "sha256": CONTRACT_SHA256,
    }:
        raise ValueError("protocol does not bind the Phase 5 contract")


def generate_data(
    config_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError("Phase 5 data outputs are immutable")
    config = _read(config_path)
    if config.get("format") != CONFIG_FORMAT:
        raise ValueError("unsupported generic domain config")
    rows = render_dataset(config)
    write_dataset(output_path, rows)
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    split_copy = {
        split: {
            value
            for row in values
            for value in row["copy_lexemes"]
        }
        for split, values in split_rows.items()
    }
    split_overlap = any(
        split_copy[left] & split_copy[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )
    result = {
        "format": "layercake-phase5-generic-domain-data-freeze/1",
        "status": "PASS" if not split_overlap else "FAIL",
        "domain_id": config["domain_id"],
        "framework_commit": _git_head(),
        "config": {
            "path": _relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "dataset": {
            "path": _relative(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
        "rows": {
            split: len(values)
            for split, values in split_rows.items()
        },
        "families": sorted(
            {row["family"] for row in rows}
        ),
        "family_count": len({row["family"] for row in rows}),
        "split_copy_lexeme_overlap": split_overlap,
        "split_content_sha256": {
            split: canonical_sha(values)
            for split, values in split_rows.items()
        },
        "functional_evaluators": sorted(
            {
                row["evaluation"]["kind"]
                for row in rows
            }
        ),
        "test_accessed": False,
    }
    result["evidence_sha256"] = canonical_sha(result)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _encode_target(
    tokenizer: LosslessLexemePointerTokenizer,
    row: Mapping[str, Any],
    source_lexemes: list[bytes],
) -> list[int]:
    return tokenizer.encode_target(
        row["response"],
        copy_lexemes=row["copy_lexemes"],
        source_lexemes=source_lexemes,
    )


def _batch(
    rows: list[dict[str, Any]],
    tokenizer: LosslessLexemePointerTokenizer,
    *,
    maximum_source_lexemes: int,
    maximum_target_actions: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sources: list[list[int]] = []
    targets: list[list[int]] = []
    raw_bytes = 0
    for row in rows:
        prompt = row["prompt"] + "\n"
        source_ids, source_lexemes = tokenizer.encode_source(prompt)
        target = _encode_target(tokenizer, row, source_lexemes)
        if len(source_ids) > maximum_source_lexemes:
            raise ValueError("source exceeds protocol lexeme maximum")
        if len(target) > maximum_target_actions:
            raise ValueError("target exceeds protocol action maximum")
        if (
            tokenizer.decode_actions(target, source_lexemes)
            != row["response"].encode("utf-8")
        ):
            raise RuntimeError("target failed lossless roundtrip")
        sources.append(source_ids)
        targets.append(target)
        raw_bytes += len(prompt.encode("utf-8"))
        raw_bytes += len(row["response"].encode("utf-8"))
    source_width = max(map(len, sources))
    target_width = max(map(len, targets))
    source = torch.zeros(
        len(rows), source_width, dtype=torch.long, device=device
    )
    target = torch.full(
        (len(rows), target_width),
        -100,
        dtype=torch.long,
        device=device,
    )
    for index, (source_ids, target_ids) in enumerate(
        zip(sources, targets)
    ):
        source[index, : len(source_ids)] = torch.tensor(
            source_ids, dtype=torch.long, device=device
        )
        target[index, : len(target_ids)] = torch.tensor(
            target_ids, dtype=torch.long, device=device
        )
    return {
        "source_ids": source,
        "target_actions": target,
        "raw_bytes": torch.tensor(raw_bytes, device=device),
    }


def _loss(
    model: PortableTokenPlan,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    result = model(batch["source_ids"], batch["target_actions"])
    mask = batch["target_actions"].ge(0)
    loss = F.nll_loss(
        result["log_probs"][mask],
        batch["target_actions"][mask],
    )
    return loss, result


class _Sampler:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size > len(rows):
            raise ValueError("batch size exceeds training rows")
        self.rows = rows
        self.batch_size = batch_size
        self.random = random.Random(seed)
        self.order = list(range(len(rows)))
        self.cursor = len(rows)

    def next(self) -> list[dict[str, Any]]:
        if self.cursor + self.batch_size > len(self.order):
            self.random.shuffle(self.order)
            self.cursor = 0
        selected = self.order[
            self.cursor : self.cursor + self.batch_size
        ]
        self.cursor += self.batch_size
        return [self.rows[index] for index in selected]


def _cpu_fallback(
    model: PortableTokenPlan,
    tokenizer: LosslessLexemePointerTokenizer,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    copy = PortableTokenPlan(**model.canonical_config()).cpu()
    copy.load_state_dict(model.state_dict())
    copy.train()
    optimizer = torch.optim.AdamW(copy.parameters(), lr=1e-5)
    batch = _batch(
        rows,
        tokenizer,
        maximum_source_lexemes=copy.maximum_source_lexemes,
        maximum_target_actions=copy.maximum_target_actions,
        device=torch.device("cpu"),
    )
    started = time.perf_counter()
    loss, _ = _loss(copy, batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "status": "PASS",
        "device": "cpu",
        "optimizer_steps": 1,
        "rows": len(rows),
        "loss": float(loss.item()),
        "wall_seconds": time.perf_counter() - started,
    }


def train(
    protocol_path: Path,
    artifact_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if artifact_path.exists() or output_path.exists():
        raise RuntimeError("Phase 5 training outputs are immutable")
    protocol = _read(protocol_path)
    if (
        protocol.get("format") != TRAINING_FORMAT
        or protocol.get("status")
        != "PREREGISTERED_BEFORE_TRAINING"
    ):
        raise ValueError("Phase 5 training protocol is invalid")
    _verify_parent_contract(protocol)
    dataset_path = _rooted(Path(protocol["dataset"]["path"]))
    if sha256_file(dataset_path) != protocol["dataset"]["sha256"]:
        raise ValueError("training dataset hash mismatch")
    rows = load_dataset(dataset_path)
    domain_id = str(protocol["domain_id"])
    if {row["domain_id"] for row in rows} != {domain_id}:
        raise ValueError("dataset domain identity mismatch")
    train_rows = [row for row in rows if row["split"] == "train"]
    if len(train_rows) != protocol["dataset"]["train_rows"]:
        raise ValueError("training row count mismatch")
    tokenizer = LosslessLexemePointerTokenizer.build_generic(train_rows)
    model_config = dict(protocol["model"])
    if model_config.pop("architecture") != (
        "portable_token_plan_pointer_transformer"
    ):
        raise ValueError("unsupported Phase 5 model architecture")
    settings = protocol["training"]
    if not torch.cuda.is_available():
        raise RuntimeError("preregistered CUDA training is unavailable")
    device = torch.device(str(settings["primary_device"]))
    device_index = device.index or 0
    device_name = torch.cuda.get_device_name(device_index)
    if device_name != settings["required_device_name"]:
        raise RuntimeError("CUDA device identity mismatch")
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    model = PortableTokenPlan(
        fixed_vocab_size=tokenizer.vocab_size,
        **model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    sampler = _Sampler(
        train_rows,
        batch_size=int(settings["batch_size"]),
        seed=seed + 1,
    )
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    raw_bytes = 0
    visible_actions = 0
    steps = int(settings["optimizer_steps"])
    model.train()
    for step in range(1, steps + 1):
        batch = _batch(
            sampler.next(),
            tokenizer,
            maximum_source_lexemes=model.maximum_source_lexemes,
            maximum_target_actions=model.maximum_target_actions,
            device=device,
        )
        raw_bytes += int(batch["raw_bytes"].item())
        visible_actions += int(
            batch["target_actions"].ge(0).sum().item()
        )
        loss, result = _loss(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(settings["gradient_clip_norm"]),
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
            mask = batch["target_actions"].ge(0)
            targets = batch["target_actions"][mask]
            predicted = result["log_probs"][mask].argmax(dim=-1)
            pointer = targets.ge(tokenizer.vocab_size)
            item = {
                "step": step,
                "action_negative_log_likelihood": value,
                "action_accuracy": float(
                    predicted.eq(targets).float().mean().item()
                ),
                "pointer_action_accuracy": float(
                    predicted[pointer]
                    .eq(targets[pointer])
                    .float()
                    .mean()
                    .item()
                ),
                "gradient_norm_before_clip": float(gradient_norm),
                "gpu_wall_seconds": time.perf_counter() - started,
            }
            history.append(item)
            print(json.dumps(item), flush=True)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    torch.cuda.synchronize(device_index)
    gpu_wall = time.perf_counter() - started
    peak_gpu = int(torch.cuda.max_memory_allocated(device_index))
    cpu_model = model.cpu()
    fallback = _cpu_fallback(
        cpu_model, tokenizer, train_rows[:8]
    )
    artifact = build_token_plan_artifact(
        cpu_model,
        tokenizer,
        domain_id=domain_id,
        training={
            "protocol": _relative(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "seed": seed,
            "domain_id": domain_id,
            "dataset_sha256": sha256_file(dataset_path),
            "primary_device": str(device),
            "primary_device_name": device_name,
            "optimizer_steps": steps,
            "best_action_negative_log_likelihood": best_loss,
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    output = {
        "format": "layercake-phase5-generic-domain-training/1",
        "status": "TRAINED",
        "domain_id": domain_id,
        "seed": seed,
        "source_commit": _git_head(),
        "protocol": {
            "path": _relative(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "dataset": {
            "path": _relative(dataset_path),
            "sha256": sha256_file(dataset_path),
            "train_rows": len(train_rows),
        },
        "primary_device": str(device),
        "primary_device_name": device_name,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "precision": settings["precision"],
        "optimizer_steps": steps,
        "batch_size": int(settings["batch_size"]),
        "parameters": cpu_model.parameter_count(),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in cpu_model.parameters()
        ),
        "tokenizer_sha256": tokenizer.hash(),
        "tokenizer_format": tokenizer.format_version,
        "fixed_vocabulary_size": tokenizer.vocab_size,
        "raw_utf8_training_bytes_exposed": raw_bytes,
        "model_visible_target_actions": visible_actions,
        "best_action_negative_log_likelihood": best_loss,
        "gpu_wall_seconds": gpu_wall,
        "peak_accelerator_memory_bytes": peak_gpu,
        "peak_process_resident_memory_bytes": peak_rss,
        "cpu_fallback_smoke": fallback,
        "artifact": {
            "path": _relative(artifact_path),
            "sha256": sha256_file(artifact_path),
            "spec_hash": artifact["spec_hash"],
            "payload_hash": artifact["payload_hash"],
        },
        "history": history,
        "validation_accessed": False,
        "test_accessed": False,
    }
    output["evidence_sha256"] = canonical_sha(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


@torch.inference_mode()
def evaluate(
    protocol_path: Path,
    artifact_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("Phase 5 evaluation output is immutable")
    protocol = _read(protocol_path)
    if (
        protocol.get("format") != EVALUATION_FORMAT
        or protocol.get("status")
        != "PREREGISTERED_BEFORE_EVALUATION"
    ):
        raise ValueError("Phase 5 evaluation protocol is invalid")
    _verify_parent_contract(protocol)
    if sha256_file(artifact_path) != protocol["artifact"]["sha256"]:
        raise ValueError("evaluation artifact hash mismatch")
    dataset_path = _rooted(Path(protocol["dataset"]["path"]))
    if sha256_file(dataset_path) != protocol["dataset"]["sha256"]:
        raise ValueError("evaluation dataset hash mismatch")
    split = str(protocol["split"])
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split is unsupported")
    rows = [
        row for row in load_dataset(dataset_path)
        if row["split"] == split
    ]
    if len(rows) != protocol["distinct_prompts"]:
        raise ValueError("evaluation prompt count mismatch")
    artifact = torch.load(
        artifact_path, map_location="cpu", weights_only=True
    )
    device = torch.device(str(protocol["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("evaluation CUDA device is unavailable")
    spec, tokenizer, model = load_token_plan_artifact(artifact, device)
    if spec["domain_id"] != protocol["domain_id"]:
        raise ValueError("evaluation domain identity mismatch")
    sources = [
        tokenizer.encode_source(row["prompt"] + "\n")
        for row in rows
    ]
    width = max(len(ids) for ids, _ in sources)
    source_tensor = torch.zeros(
        len(rows), width, dtype=torch.long, device=device
    )
    for index, (ids, _) in enumerate(sources):
        source_tensor[index, : len(ids)] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
    started = time.perf_counter()
    action_rows = model.generate_actions(
        source_tensor,
        maximum_actions=int(protocol["maximum_output_actions"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device.index or 0)
    generation_wall = time.perf_counter() - started
    records = []
    for row, actions, (_, source_lexemes) in zip(
        rows, action_rows, sources
    ):
        raw = tokenizer.decode_actions(actions, source_lexemes)
        passed, checks = evaluate_generated(raw, row)
        records.append(
            {
                "id": row["id"],
                "family": row["family"],
                "output_hex": raw.hex(),
                "output_sha256": hashlib.sha256(raw).hexdigest(),
                "generated_actions": list(map(int, actions)),
                "generated_action_count": len(actions),
                "functional_success": passed,
                "checks": checks,
            }
        )
    successes = sum(row["functional_success"] for row in records)
    minimum = int(protocol["minimum_functional_successes"])
    output = {
        "format": "layercake-phase5-generic-domain-functional/1",
        "status": "PASS" if successes >= minimum else "FAIL",
        "domain_id": protocol["domain_id"],
        "seed": int(protocol["seed"]),
        "split": split,
        "protocol": {
            "path": _relative(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "artifact": {
            "path": _relative(artifact_path),
            "sha256": sha256_file(artifact_path),
            "payload_hash": artifact["payload_hash"],
        },
        "dataset": {
            "path": _relative(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "distinct_prompts": len(records),
        "functional_successes": successes,
        "functional_failures": len(records) - successes,
        "functional_success_rate": successes / len(records),
        "minimum_functional_successes": minimum,
        "families": sorted({row["family"] for row in records}),
        "median_generated_actions": statistics.median(
            row["generated_action_count"] for row in records
        ),
        "evaluation_device": str(device),
        "batched_generation_wall_seconds": generation_wall,
        "autonomous_neural_generation": True,
        "neural_fixed_and_pointer_action_selection": True,
        "templates_retrieval_stored_answers_or_output_rewrite": False,
        "teacher_at_inference": False,
        "test_accessed": split == "test",
        "records": records,
    }
    output["evidence_sha256"] = canonical_sha(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def core_baseline(
    protocol_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("Phase 5 baseline output is immutable")
    protocol = _read(protocol_path)
    if (
        protocol.get("format") != BASELINE_FORMAT
        or protocol.get("status")
        != "PREREGISTERED_BEFORE_BASELINE_EVALUATION"
    ):
        raise ValueError("Phase 5 baseline protocol is invalid")
    _verify_parent_contract(protocol)
    dataset_path = _rooted(Path(protocol["dataset"]["path"]))
    if sha256_file(dataset_path) != protocol["dataset"]["sha256"]:
        raise ValueError("baseline dataset hash mismatch")
    split = str(protocol["split"])
    rows = [
        row for row in load_dataset(dataset_path)
        if row["split"] == split
    ]
    if len(rows) != protocol["distinct_prompts"]:
        raise ValueError("baseline prompt count mismatch")
    runtime_path = _rooted(Path(protocol["runtime"]["path"]))
    runtime = NativeRuntime(
        runtime_path, threads=int(protocol["runtime"]["threads"])
    )
    records = []
    for row in rows:
        generated = generate_native_core(
            runtime,
            row["prompt"],
            output_bytes=int(protocol["output_bytes"]),
        )
        raw = generated.pop("payload")
        passed, checks = evaluate_generated(raw, row)
        records.append(
            {
                "id": row["id"],
                "family": row["family"],
                "output_sha256": hashlib.sha256(raw).hexdigest(),
                "generated_bytes": len(raw),
                "functional_success": passed,
                "checks": checks,
                "timing": generated["timing"],
            }
        )
    successes = sum(row["functional_success"] for row in records)
    output = {
        "format": "layercake-phase5-frozen-core-functional/1",
        "status": "COMPLETE",
        "system": "sealed_phase2_frozen_core",
        "domain_id": protocol["domain_id"],
        "split": split,
        "protocol": {
            "path": _relative(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "checkpoint_sha256": protocol["runtime"][
            "checkpoint_sha256"
        ],
        "runtime": {
            "path": _relative(runtime_path),
            "graph_sha256": protocol["runtime"]["graph_sha256"],
            "threads": int(protocol["runtime"]["threads"]),
        },
        "dataset": {
            "path": _relative(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "distinct_prompts": len(records),
        "functional_successes": successes,
        "functional_failures": len(records) - successes,
        "functional_success_rate": successes / len(records),
        "functional_error_rate": (
            len(records) - successes
        ) / len(records),
        "autonomous_neural_generation": True,
        "teacher_at_inference": False,
        "test_accessed": split == "test",
        "records": records,
    }
    output["evidence_sha256"] = canonical_sha(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    data = subcommands.add_parser("generate-data")
    data.add_argument("--config", type=Path, required=True)
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--manifest", type=Path, required=True)
    training = subcommands.add_parser("train")
    training.add_argument("--protocol", type=Path, required=True)
    training.add_argument("--artifact", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    evaluation = subcommands.add_parser("evaluate")
    evaluation.add_argument("--protocol", type=Path, required=True)
    evaluation.add_argument("--artifact", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    baseline = subcommands.add_parser("core-baseline")
    baseline.add_argument("--protocol", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate-data":
        result = generate_data(
            _rooted(args.config),
            _rooted(args.output),
            _rooted(args.manifest),
        )
    elif args.command == "train":
        result = train(
            _rooted(args.protocol),
            _rooted(args.artifact),
            _rooted(args.output),
        )
    elif args.command == "evaluate":
        result = evaluate(
            _rooted(args.protocol),
            _rooted(args.artifact),
            _rooted(args.output),
        )
    else:
        result = core_baseline(
            _rooted(args.protocol),
            _rooted(args.output),
        )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "domain_id",
                    "seed",
                    "functional_successes",
                    "functional_failures",
                    "evidence_sha256",
                )
                if key in result
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] in {
        "PASS",
        "TRAINED",
        "COMPLETE",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
