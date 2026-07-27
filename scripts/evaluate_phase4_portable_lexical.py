from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

import _common
from layercake.portable_domain import LayerCakeRuntime, load_portable_artifact
from layercake.training.phase4_python_cake import _canonical_sha, _load_rows


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = (
        args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    )
    artifact_path = (
        args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    )
    output_path = (
        args.output if args.output.is_absolute() else ROOT / args.output
    )
    if output_path.exists():
        raise RuntimeError("lexical autonomous evidence is immutable")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "PREREGISTERED_BEFORE_AUTONOMOUS_EVALUATION":
        raise ValueError("lexical autonomous evaluation is not preregistered")
    if _sha256(artifact_path) != protocol["artifact"]["file_sha256"]:
        raise ValueError("lexical autonomous artifact hash mismatch")
    dataset_path = ROOT / protocol["dataset"]["path"]
    if _sha256(dataset_path) != protocol["dataset"]["sha256"]:
        raise ValueError("lexical autonomous dataset hash mismatch")
    rows = [
        row
        for row in _load_rows(dataset_path)
        if row["split"] == protocol["dataset"]["split"]
    ]
    if len(rows) != protocol["dataset"]["distinct_prompts"]:
        raise ValueError("lexical autonomous row count mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact["payload_hash"] != protocol["artifact"]["payload_hash"]:
        raise ValueError("lexical autonomous payload hash mismatch")
    spec, _ = load_portable_artifact(artifact, "cpu")
    runtime = LayerCakeRuntime()
    runtime.install_portable_domain(artifact, "cpu")
    decoder = runtime.domains[spec.domain_id][1]
    records = []
    for index, row in enumerate(rows):
        prompt = (row["prompt"] + "\n").encode("utf-8")
        required = b"def " + row["function_name"].encode("utf-8") + b"("
        started = time.perf_counter()
        state = decoder.prefill_incremental(
            torch.tensor(list(prompt), dtype=torch.long)[None]
        )
        generated = []
        for _ in range(len(required)):
            next_byte = state["next_logits"].argmax(
                dim=-1, keepdim=True
            )
            generated.append(int(next_byte.item()))
            decoder.decode_incremental(next_byte, state)
        latency = time.perf_counter() - started
        raw = bytes(generated)
        exact = raw == required
        records.append(
            {
                "id": row["id"],
                "expected_prefix_utf8": required.decode("utf-8"),
                "generated_prefix_utf8": raw.decode(
                    "utf-8", errors="replace"
                ),
                "generated_prefix_sha256": hashlib.sha256(raw).hexdigest(),
                "exact_prefix": exact,
                "latency_seconds": latency,
            }
        )
        if (index + 1) % 32 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": index + 1,
                        "successes": sum(
                            record["exact_prefix"] for record in records
                        ),
                    }
                ),
                flush=True,
            )
    successes = sum(record["exact_prefix"] for record in records)
    minimum = int(protocol["gate"]["minimum_exact_prefix_successes"])
    evidence = {
        "format": "layercake-phase4-portable-lexical-autonomous/1",
        "status": "PASS" if successes >= minimum else "FAIL",
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": _sha256(protocol_path),
        "artifact": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_file_sha256": _sha256(artifact_path),
        "payload_hash": artifact["payload_hash"],
        "dataset": dataset_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(dataset_path),
        "split": protocol["dataset"]["split"],
        "distinct_prompts": len(records),
        "exact_prefix_successes": successes,
        "exact_prefix_failures": len(records) - successes,
        "exact_prefix_rate": successes / len(records),
        "minimum_exact_prefix_successes": minimum,
        "median_latency_seconds": statistics.median(
            record["latency_seconds"] for record in records
        ),
        "autonomous_neural_generation": True,
        "persistent_incremental_state": True,
        "forced_units_or_output_rewrite": False,
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
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "exact_prefix_successes": successes,
                "exact_prefix_rate": evidence["exact_prefix_rate"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
