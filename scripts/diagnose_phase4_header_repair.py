from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import _common
from layercake.training.phase4_python_cake import (
    _execute_tests,
    _extract_function,
    _load_rows,
)


_FIRST_FUNCTION_HEADER = re.compile(r"def\s+[^\n(]*\(")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only upper bound after replacing the first generated "
            "function name while retaining its generated signature and body."
        )
    )
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    dataset = {
        row["id"]: row
        for row in _load_rows(args.dataset)
        if row["split"] == evaluation["split"]
    }
    records = []
    successes = 0
    parseable = 0
    for record in evaluation["records"]:
        row = dataset[record["id"]]
        match = _FIRST_FUNCTION_HEADER.search(record["generated_text"])
        repaired_text = record["generated_text"]
        if match is not None:
            repaired_text = (
                repaired_text[: match.start()]
                + f"def {row['function_name']}("
                + repaired_text[match.end() :]
            )
        source, status = _extract_function(
            repaired_text, row["function_name"]
        )
        passed = False
        if source is not None:
            parseable += 1
            passed, _ = _execute_tests(
                source, row["function_name"], row["tests"]
            )
        successes += int(passed)
        records.append(
            {
                "id": record["id"],
                "header_found": match is not None,
                "parse_status": status,
                "header_repair_functional_success": passed,
            }
        )

    result = {
        "format": "layercake-phase4-header-repair-diagnostic/1",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION_CREDIT",
        "prohibited_as_runtime_or_promotion_evidence": True,
        "repair": (
            "replace only the first generated def-name span; retain generated "
            "signature, body, and all continuation text"
        ),
        "evaluation_sha256": hashlib.sha256(
            args.evaluation.read_bytes()
        ).hexdigest(),
        "dataset_sha256": hashlib.sha256(
            args.dataset.read_bytes()
        ).hexdigest(),
        "distinct_prompts": len(records),
        "parseable_after_header_repair": parseable,
        "header_repair_functional_successes": successes,
        "header_repair_functional_success_rate": successes
        / max(1, len(records)),
        "records": records,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
