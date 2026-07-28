from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

import _common
from layercake.portable_domain import load_portable_artifact
from layercake.training.phase4_python_cake import (
    _canonical_sha,
    _execute_tests,
    _extract_function,
    _load_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only causal upper bound for handing an exact generated "
            "function-name prefix to a preserved portable semantic decoder."
        )
    )
    parser.add_argument("--semantic-artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-new-bytes", type=int, default=320)
    parser.add_argument(
        "--include-opening-parenthesis",
        action="store_true",
    )
    args = parser.parse_args()
    artifact_path = (
        args.semantic_artifact
        if args.semantic_artifact.is_absolute()
        else ROOT / args.semantic_artifact
    )
    dataset_path = (
        args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    )
    output_path = (
        args.output if args.output.is_absolute() else ROOT / args.output
    )
    if output_path.exists():
        raise RuntimeError("causal handoff diagnostic evidence is immutable")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    _, decoder = load_portable_artifact(artifact, "cpu")
    decoder.eval()
    rows = [
        row for row in _load_rows(dataset_path) if row["split"] == "validation"
    ]
    records = []
    for row in rows:
        prompt = (row["prompt"] + "\n").encode("utf-8")
        forced_prefix = b"def " + row["function_name"].encode("utf-8")
        if args.include_opening_parenthesis:
            forced_prefix += b"("
        state = decoder.prefill_incremental(
            torch.tensor(list(prompt), dtype=torch.long)[None]
        )
        generated = []
        for value in forced_prefix:
            observed = torch.tensor([[value]], dtype=torch.long)
            generated.append(value)
            decoder.decode_incremental(observed, state)
        remaining = max(0, args.maximum_new_bytes - len(generated))
        for _ in range(remaining):
            next_byte = state["next_logits"].argmax(
                dim=-1, keepdim=True
            )
            generated.append(int(next_byte.item()))
            decoder.decode_incremental(next_byte, state)
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
                "forced_prefix_utf8": forced_prefix.decode("utf-8"),
                "generated_text_sha256": hashlib.sha256(raw).hexdigest(),
                "generated_text": text,
                "extracted_source": source,
                "causal_handoff_functional_success": passed,
                "tests": tests,
            }
        )
    successes = sum(
        record["causal_handoff_functional_success"] for record in records
    )
    result = {
        "format": "layercake-phase4-causal-handoff-diagnostic/1",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION_CREDIT",
        "prohibited_as_runtime_or_promotion_evidence": True,
        "intervention": (
            "force only UTF-8 bytes for 'def ' plus the expected function "
            "identifier"
            + (
                " and opening parenthesis"
                if args.include_opening_parenthesis
                else ""
            )
            + " into the semantic decoder state, then autonomously generate "
            "every remaining byte"
        ),
        "opening_parenthesis_in_forced_prefix": (
            args.include_opening_parenthesis
        ),
        "semantic_artifact": artifact_path.relative_to(ROOT).as_posix(),
        "semantic_artifact_file_sha256": _sha256(artifact_path),
        "semantic_payload_hash": artifact["payload_hash"],
        "dataset": dataset_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": _sha256(dataset_path),
        "split": "validation",
        "distinct_prompts": len(records),
        "maximum_new_bytes": args.maximum_new_bytes,
        "causal_handoff_functional_successes": successes,
        "causal_handoff_functional_success_rate": successes
        / max(1, len(records)),
        "test_split_accessed": False,
        "records": records,
    }
    result["evidence_sha256"] = _canonical_sha(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "causal_handoff_functional_successes": successes,
                "causal_handoff_functional_success_rate": result[
                    "causal_handoff_functional_success_rate"
                ],
                "evidence_sha256": result["evidence_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
