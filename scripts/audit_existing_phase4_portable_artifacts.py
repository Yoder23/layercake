from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

import _common
from layercake.portable_domain import (
    LayerCakeRuntime,
    artifact_payload_bytes,
    load_portable_artifact,
)
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _execute_tests,
    _extract_function,
    _load_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = (
    ROOT / "moonshot" / "phase4_existing_portable_audit_preregistration.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "moonshot"
    / "phase4"
    / "existing_portable_artifact_audit.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


@torch.inference_mode()
def _evaluate_artifact(
    artifact_path: Path,
    rows: list[dict[str, Any]],
    *,
    maximum_new_bytes: int,
    context_bytes: int,
) -> dict[str, Any]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    spec, decoder = load_portable_artifact(artifact, "cpu")
    runtime = LayerCakeRuntime()
    runtime.install_portable_domain(artifact, "cpu")
    records = []
    for index, row in enumerate(rows):
        prompt = row["prompt"] + "\n"
        prompt_bytes = prompt.encode("utf-8")
        started = time.perf_counter()
        generated = runtime.generate(
            prompt_bytes,
            max_new_bytes=maximum_new_bytes,
            domain_id=spec.domain_id,
            context_bytes=context_bytes,
        )
        latency = time.perf_counter() - started
        complete = bytes(generated[0].tolist())
        continuation = complete[len(prompt_bytes) :]
        text = continuation.decode("utf-8", errors="replace")
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
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "expected_function": row["function_name"],
                "generated_text": text,
                "generated_text_sha256": hashlib.sha256(
                    continuation
                ).hexdigest(),
                "extracted_source": source,
                "functional_success": passed,
                "tests": tests,
                "total_latency_seconds": latency,
            }
        )
        print(
            json.dumps(
                {
                    "artifact": artifact_path.name,
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
    return {
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "file_sha256": _sha256(artifact_path),
        "spec": spec.canonical_dict(),
        "spec_hash": artifact["spec_hash"],
        "payload_hash": artifact["payload_hash"],
        "payload_bytes": artifact_payload_bytes(artifact),
        "parameters": decoder.parameter_count(),
        "distinct_prompts": len(records),
        "functional_successes": successes,
        "functional_failures": len(records) - successes,
        "functional_success_rate": successes / max(1, len(records)),
        "median_total_latency_seconds": statistics.median(
            record["total_latency_seconds"] for record in records
        ),
        "records": records,
    }


def run(preregistration: Path, output: Path) -> dict[str, Any]:
    preregistration = preregistration.resolve()
    output = output.resolve()
    protocol = json.loads(preregistration.read_text(encoding="utf-8"))
    if protocol.get("status") != "PREREGISTERED_BEFORE_EVALUATION":
        raise ValueError("audit protocol is not preregistered")
    dataset_contract = protocol["dataset"]
    if dataset_contract["split"] != "validation":
        raise ValueError("historical artifact audit must remain validation-only")
    dataset = _resolve(dataset_contract["path"])
    if _sha256(dataset) != dataset_contract["sha256"]:
        raise ValueError("locked functional dataset hash mismatch")
    rows = [
        row
        for row in _load_rows(dataset)
        if row["split"] == dataset_contract["split"]
    ]
    if len(rows) != dataset_contract["distinct_prompts"]:
        raise ValueError("locked functional prompt count mismatch")
    generation = protocol["fixed_generation"]
    torch.set_num_threads(1)
    evaluated = []
    for declared in protocol["screened_artifacts_in_order"]:
        artifact_path = _resolve(declared["path"])
        if _sha256(artifact_path) != declared["file_sha256"]:
            raise ValueError(f"artifact file hash mismatch: {declared['path']}")
        result = _evaluate_artifact(
            artifact_path,
            rows,
            maximum_new_bytes=int(generation["maximum_new_bytes"]),
            context_bytes=int(generation["context_bytes"]),
        )
        result["declared_training_seed"] = declared["training_seed"]
        result["declared_lineage"] = declared["lineage"]
        evaluated.append(result)
    minimum = int(protocol["semantic_gate"]["minimum_functional_successes"])
    eligible = [
        result
        for result in evaluated
        if result["functional_successes"] >= minimum
    ]
    winner = None
    if eligible:
        winner = min(
            eligible,
            key=lambda result: (
                -result["functional_successes"],
                result["payload_bytes"],
                result["artifact"],
            ),
        )
    document = {
        "format": "layercake-phase4-existing-portable-artifact-audit/1",
        "status": (
            "EXISTING_ARTIFACT_SEMANTIC_GATE_PASS"
            if winner is not None
            else "NO_EXISTING_ARTIFACT_PASSES_SEMANTIC_GATE"
        ),
        "preregistration": preregistration.relative_to(ROOT).as_posix(),
        "preregistration_sha256": _sha256(preregistration),
        "dataset": dataset.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(dataset),
        "split": dataset_contract["split"],
        "test_split_accessed": False,
        "autonomous_neural_generation": True,
        "generation_policy": generation,
        "minimum_functional_successes": minimum,
        "winner": (
            {
                "artifact": winner["artifact"],
                "file_sha256": winner["file_sha256"],
                "payload_hash": winner["payload_hash"],
                "functional_successes": winner["functional_successes"],
            }
            if winner is not None
            else None
        ),
        "artifacts": evaluated,
        "excluded_historical_artifacts": protocol[
            "excluded_historical_artifacts"
        ],
        "scientific_conclusion": (
            "The existing mathematical transfer mechanism and semantic "
            "capability quality are separate gates. This audit changes no "
            "artifact and gives no test-split credit."
        ),
    }
    document["evidence_sha256"] = _canonical_sha(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.preregistration, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "winner": result["winner"],
                "evidence_sha256": result["evidence_sha256"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
