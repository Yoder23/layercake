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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = []
    response_starts = []
    for index in indexes:
        row = rows[index]
        prompt = (row["prompt"] + "\n").encode("utf-8")
        response = row["response"].encode("utf-8")
        sequence = list((prompt + response)[:maximum_sequence_bytes])
        if len(sequence) < 2 or len(prompt) >= len(sequence):
            raise ValueError("functional row has no trainable response bytes")
        sequences.append(sequence)
        response_starts.append(len(prompt) - 1)
    width = max(len(sequence) - 1 for sequence in sequences)
    inputs = torch.zeros((len(sequences), width), dtype=torch.long)
    targets = torch.zeros((len(sequences), width), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for batch_index, (sequence, response_start) in enumerate(
        zip(sequences, response_starts)
    ):
        length = len(sequence) - 1
        inputs[batch_index, :length] = torch.tensor(sequence[:-1])
        targets[batch_index, :length] = torch.tensor(sequence[1:])
        mask[batch_index, response_start:length] = True
    return inputs, targets, mask


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
        inputs, targets, mask = _batch(
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
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT
        / "moonshot"
        / "phase4_portable_decoder_functional_preregistration.json",
    )
    parser.add_argument("--artifact", type=Path, required=True)
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
