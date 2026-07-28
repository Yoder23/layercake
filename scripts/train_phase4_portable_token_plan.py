from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
from typing import Any

import psutil
import torch
import torch.nn.functional as F

import _common
from layercake.portable_token_plan import (
    LosslessLexemePointerTokenizer,
    PortableTokenPlan,
    build_token_plan_artifact,
    load_token_plan_artifact,
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


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class _BalancedSampler:
    def __init__(
        self,
        functional: list[dict[str, Any]],
        lexical: list[dict[str, Any]],
        *,
        seed: int,
        functional_per_batch: int,
        lexical_per_batch: int,
    ) -> None:
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


def _batch(
    rows: list[dict[str, Any]],
    tokenizer: LosslessLexemePointerTokenizer,
    *,
    maximum_source_lexemes: int,
    maximum_target_actions: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sources = []
    targets = []
    raw_bytes = 0
    for row in rows:
        prompt = row["prompt"] + "\n"
        source_ids, source_lexemes = tokenizer.encode_source(prompt)
        target = tokenizer.encode_target(
            row["response"],
            function_name=row["function_name"],
            source_lexemes=source_lexemes,
        )
        if len(source_ids) > maximum_source_lexemes:
            raise ValueError("source exceeds preregistered lexeme maximum")
        if len(target) > maximum_target_actions:
            raise ValueError("target exceeds preregistered action maximum")
        sources.append(source_ids)
        targets.append(target)
        raw_bytes += len(prompt.encode("utf-8")) + len(
            row["response"].encode("utf-8")
        )
    source_width = max(len(value) for value in sources)
    target_width = max(len(value) for value in targets)
    source_tensor = torch.zeros(
        len(rows), source_width, dtype=torch.long, device=device
    )
    target_tensor = torch.full(
        (len(rows), target_width),
        -100,
        dtype=torch.long,
        device=device,
    )
    for index, (source, target) in enumerate(zip(sources, targets)):
        source_tensor[index, : len(source)] = torch.tensor(
            source, dtype=torch.long, device=device
        )
        target_tensor[index, : len(target)] = torch.tensor(
            target, dtype=torch.long, device=device
        )
    return {
        "source_ids": source_tensor,
        "target_actions": target_tensor,
        "raw_bytes": torch.tensor(raw_bytes, device=device),
    }


def _loss(
    model: PortableTokenPlan,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    result = model(batch["source_ids"], batch["target_actions"])
    mask = batch["target_actions"].ge(0)
    loss = F.nll_loss(
        result["log_probs"][mask],
        batch["target_actions"][mask],
    )
    return loss, result


def _verify_training_tokenizer(
    rows: list[dict[str, Any]],
    tokenizer: LosslessLexemePointerTokenizer,
    *,
    maximum_source_lexemes: int,
    maximum_target_actions: int,
) -> dict[str, Any]:
    maximum_source = 0
    maximum_target = 0
    pointer_actions = 0
    for row in rows:
        prompt = row["prompt"] + "\n"
        source_ids, source_lexemes = tokenizer.encode_source(prompt)
        actions = tokenizer.encode_target(
            row["response"],
            function_name=row["function_name"],
            source_lexemes=source_lexemes,
        )
        if tokenizer.decode_actions(actions, source_lexemes) != row[
            "response"
        ].encode("utf-8"):
            raise RuntimeError("training response failed lossless roundtrip")
        pointers = [
            action
            for action in actions
            if action >= tokenizer.vocab_size
        ]
        if len(pointers) != 1:
            raise RuntimeError(
                "training response must use exactly one identifier pointer"
            )
        maximum_source = max(maximum_source, len(source_ids))
        maximum_target = max(maximum_target, len(actions))
        pointer_actions += len(pointers)
    if maximum_source > maximum_source_lexemes:
        raise ValueError("training source lexemes exceed protocol")
    if maximum_target > maximum_target_actions:
        raise ValueError("training target actions exceed protocol")
    return {
        "rows": len(rows),
        "lossless_roundtrip_rows": len(rows),
        "identifier_pointer_actions": pointer_actions,
        "maximum_source_lexemes": maximum_source,
        "maximum_target_actions": maximum_target,
        "status": "PASS",
    }


def _cpu_fallback_smoke(
    model: PortableTokenPlan,
    tokenizer: LosslessLexemePointerTokenizer,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    smoke = PortableTokenPlan(**model.canonical_config())
    smoke.load_state_dict(model.state_dict())
    smoke.train()
    optimizer = torch.optim.AdamW(smoke.parameters(), lr=1e-5)
    batch = _batch(
        rows,
        tokenizer,
        maximum_source_lexemes=smoke.maximum_source_lexemes,
        maximum_target_actions=smoke.maximum_target_actions,
        device=torch.device("cpu"),
    )
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    started = time.perf_counter()
    loss, _ = _loss(smoke, batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {
        "status": "PASS",
        "device": "cpu",
        "rows": len(rows),
        "optimizer_steps": 1,
        "loss": float(loss.item()),
        "wall_seconds": time.perf_counter() - started,
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
        raise ValueError("token-plan training is not preregistered")
    if artifact_path.exists() or evidence_path.exists():
        raise RuntimeError("token-plan outputs are immutable")
    data_contract = protocol["training_data"]
    functional_path = ROOT / data_contract["functional"]["path"]
    lexical_path = ROOT / data_contract["lexical_conformance"]["path"]
    if _sha256(functional_path) != data_contract["functional"]["sha256"]:
        raise ValueError("functional training data hash mismatch")
    if _sha256(lexical_path) != data_contract[
        "lexical_conformance"
    ]["sha256"]:
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
    training_rows = functional + lexical
    tokenizer = LosslessLexemePointerTokenizer.build(training_rows)
    tokenizer_contract = protocol["tokenizer_contract"]
    tokenizer_check = _verify_training_tokenizer(
        training_rows,
        tokenizer,
        maximum_source_lexemes=int(
            tokenizer_contract["maximum_source_lexemes"]
        ),
        maximum_target_actions=int(
            tokenizer_contract["maximum_target_actions_including_eos"]
        ),
    )
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
    model = PortableTokenPlan(
        fixed_vocab_size=tokenizer.vocab_size,
        model_width=int(architecture["model_width"]),
        attention_heads=int(architecture["attention_heads"]),
        encoder_layers=int(architecture["encoder_layers"]),
        decoder_layers=int(architecture["decoder_layers"]),
        feedforward_width=int(architecture["feedforward_width"]),
        pointer_width=int(architecture["pointer_width"]),
        dropout=float(architecture["dropout"]),
        maximum_source_lexemes=int(
            tokenizer_contract["maximum_source_lexemes"]
        ),
        maximum_target_actions=int(
            tokenizer_contract["maximum_target_actions_including_eos"]
        ),
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
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    best_loss = float("inf")
    best_state = None
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
        visible_actions += int(batch["target_actions"].ge(0).sum().item())
        loss, result = _loss(model, batch)
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
            mask = batch["target_actions"].ge(0)
            targets = batch["target_actions"][mask]
            predictions = result["log_probs"][mask].argmax(dim=-1)
            pointer_mask = targets.ge(tokenizer.vocab_size)
            record = {
                "step": step,
                "action_negative_log_likelihood": value,
                "action_accuracy": float(
                    predictions.eq(targets).float().mean().item()
                ),
                "pointer_action_accuracy": float(
                    predictions[pointer_mask]
                    .eq(targets[pointer_mask])
                    .float()
                    .mean()
                    .item()
                ),
                "gradient_norm_before_clip": float(gradient_norm),
                "gpu_wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        peak_rss = max(peak_rss, int(process.memory_info().rss))
    if best_state is None:
        raise RuntimeError("token-plan training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    torch.cuda.synchronize(device_index)
    wall = time.perf_counter() - started
    peak_gpu = int(torch.cuda.max_memory_allocated(device_index))
    cpu_model = model.to("cpu")
    cpu_fallback = _cpu_fallback_smoke(
        cpu_model,
        tokenizer,
        functional[:4] + lexical[:4],
    )
    artifact = build_token_plan_artifact(
        cpu_model,
        tokenizer,
        training={
            "protocol": protocol_path.relative_to(ROOT).as_posix(),
            "protocol_sha256": _sha256(protocol_path),
            "seed": seed,
            "primary_device": str(device),
            "primary_device_name": device_name,
            "precision": settings["precision"],
            "optimizer_steps": steps,
            "best_action_negative_log_likelihood": best_loss,
            "functional_training_data_sha256": _sha256(functional_path),
            "lexical_training_data_sha256": _sha256(lexical_path),
        },
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    evidence = {
        "format": "layercake-phase4-portable-token-plan-training/1",
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
        "fixed_vocabulary_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer.hash(),
        "tokenizer_training_check": tokenizer_check,
        "raw_utf8_training_bytes_exposed": raw_bytes,
        "model_visible_target_actions": visible_actions,
        "best_action_negative_log_likelihood": best_loss,
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


def _load_evaluation(
    protocol_path: Path,
    artifact_path: Path,
    *,
    expected_status: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    LosslessLexemePointerTokenizer,
    PortableTokenPlan,
    torch.device,
]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != expected_status:
        raise ValueError("token-plan evaluation is not preregistered")
    if _sha256(artifact_path) != protocol["artifact"]["file_sha256"]:
        raise ValueError("token-plan evaluation artifact hash mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact["payload_hash"] != protocol["artifact"]["payload_hash"]:
        raise ValueError("token-plan evaluation payload hash mismatch")
    requested = protocol["evaluation_device"]
    if requested == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("preregistered evaluation GPU is unavailable")
    device = torch.device(requested)
    _, tokenizer, model = load_token_plan_artifact(artifact, device)
    return protocol, artifact, tokenizer, model, device


@torch.inference_mode()
def evaluate_lexical(
    protocol_path: Path,
    artifact_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("token-plan lexical evidence is immutable")
    protocol, artifact, tokenizer, model, device = _load_evaluation(
        protocol_path,
        artifact_path,
        expected_status="PREREGISTERED_BEFORE_LEXICAL_EVALUATION",
    )
    dataset_path = ROOT / protocol["dataset"]["path"]
    if _sha256(dataset_path) != protocol["dataset"]["sha256"]:
        raise ValueError("token-plan lexical dataset hash mismatch")
    rows = [
        row
        for row in _load_rows(dataset_path)
        if row["split"] == protocol["dataset"]["split"]
    ]
    if len(rows) != protocol["dataset"]["distinct_prompts"]:
        raise ValueError("token-plan lexical row count mismatch")
    sources = [
        tokenizer.encode_source(row["prompt"] + "\n") for row in rows
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
    actions = model.generate_actions(
        source_tensor,
        maximum_actions=int(protocol["generation"]["maximum_actions"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device.index)
    wall = time.perf_counter() - started
    records = []
    for row, action_row, (_, source_lexemes) in zip(
        rows, actions, sources
    ):
        raw = tokenizer.decode_actions(action_row, source_lexemes)
        required = (
            b"def " + row["function_name"].encode("utf-8") + b"("
        )
        records.append(
            {
                "id": row["id"],
                "expected_prefix_utf8": required.decode("utf-8"),
                "generated_prefix_utf8": raw[: len(required)].decode(
                    "utf-8", errors="replace"
                ),
                "generated_bytes_sha256": hashlib.sha256(raw).hexdigest(),
                "exact_prefix": raw.startswith(required),
                "generated_actions": len(action_row),
            }
        )
    successes = sum(record["exact_prefix"] for record in records)
    minimum = int(protocol["gate"]["minimum_exact_prefix_successes"])
    evidence = {
        "format": "layercake-phase4-portable-token-plan-lexical/1",
        "status": "PASS" if successes >= minimum else "FAIL",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "payload_hash": artifact["payload_hash"],
        "tokenizer_sha256": tokenizer.hash(),
        "dataset": dataset_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(dataset_path),
        "split": protocol["dataset"]["split"],
        "distinct_prompts": len(records),
        "exact_prefix_successes": successes,
        "exact_prefix_failures": len(records) - successes,
        "exact_prefix_rate": successes / len(records),
        "minimum_exact_prefix_successes": minimum,
        "evaluation_device": str(device),
        "batched_generation_wall_seconds": wall,
        "autonomous_neural_generation": True,
        "dynamic_pointer_actions_neurally_selected": True,
        "forced_pointer_positions_or_output_rewrite": False,
        "functional_promotion_credit": 0,
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


@torch.inference_mode()
def evaluate_functional(
    protocol_path: Path,
    artifact_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("token-plan functional evidence is immutable")
    protocol, artifact, tokenizer, model, device = _load_evaluation(
        protocol_path,
        artifact_path,
        expected_status="PREREGISTERED_BEFORE_FUNCTIONAL_EVALUATION",
    )
    dataset_path = ROOT / protocol["validation"]["path"]
    if _sha256(dataset_path) != protocol["validation"]["sha256"]:
        raise ValueError("token-plan functional dataset hash mismatch")
    rows = [
        row
        for row in _load_rows(dataset_path)
        if row["split"] == "validation"
    ]
    if len(rows) != protocol["validation"]["distinct_prompts"]:
        raise ValueError("token-plan functional row count mismatch")
    sources = [
        tokenizer.encode_source(row["prompt"] + "\n") for row in rows
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
    actions = model.generate_actions(
        source_tensor,
        maximum_actions=int(protocol["validation"]["maximum_output_actions"]),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device.index)
    generation_wall = time.perf_counter() - started
    records = []
    for row, action_row, (_, source_lexemes) in zip(
        rows, actions, sources
    ):
        raw = tokenizer.decode_actions(action_row, source_lexemes)
        if len(raw) > int(protocol["validation"]["maximum_output_bytes"]):
            raw = raw[: int(protocol["validation"]["maximum_output_bytes"])]
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
                "generated_actions": len(action_row),
                "extracted_source": source,
                "functional_success": passed,
                "tests": tests,
            }
        )
    successes = sum(record["functional_success"] for record in records)
    minimum = int(protocol["validation"]["minimum_functional_successes"])
    evidence = {
        "format": "layercake-phase4-portable-token-plan-functional/1",
        "status": "PASS" if successes >= minimum else "FAIL",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "payload_hash": artifact["payload_hash"],
        "tokenizer_sha256": tokenizer.hash(),
        "dataset": dataset_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(dataset_path),
        "split": "validation",
        "distinct_prompts": len(records),
        "functional_successes": successes,
        "functional_failures": len(records) - successes,
        "functional_success_rate": successes / len(records),
        "minimum_functional_successes": minimum,
        "median_generated_actions": statistics.median(
            record["generated_actions"] for record in records
        ),
        "evaluation_device": str(device),
        "batched_generation_wall_seconds": generation_wall,
        "autonomous_neural_generation": True,
        "dynamic_pointer_actions_neurally_selected": True,
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
        choices=("train", "evaluate-lexical", "evaluate-functional"),
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = (
        args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    )
    artifact = (
        args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    )
    output = (
        args.output if args.output.is_absolute() else ROOT / args.output
    )
    if args.command == "train":
        result = train(protocol, artifact, output)
    elif args.command == "evaluate-lexical":
        result = evaluate_lexical(protocol, artifact, output)
    else:
        result = evaluate_functional(protocol, artifact, output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in result
                if key
                in {
                    "status",
                    "seed",
                    "parameters",
                    "fixed_vocabulary_size",
                    "best_action_negative_log_likelihood",
                    "gpu_wall_seconds",
                    "exact_prefix_successes",
                    "exact_prefix_rate",
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
