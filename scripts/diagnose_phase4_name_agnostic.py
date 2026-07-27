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


_FUNCTION_NAME = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only functional score under generated names."
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
        generated_names = _FUNCTION_NAME.findall(record["generated_text"])
        passed = False
        parsed = False
        selected_name = None
        for generated_name in generated_names:
            source, status = _extract_function(
                record["generated_text"], generated_name
            )
            if source is None:
                continue
            parsed = True
            selected_name = generated_name
            passed, _ = _execute_tests(
                source, generated_name, row["tests"]
            )
            if passed:
                break
        parseable += int(parsed)
        successes += int(passed)
        records.append(
            {
                "id": record["id"],
                "generated_function": selected_name,
                "parseable_first_function": parsed,
                "name_agnostic_functional_success": passed,
            }
        )
    result = {
        "format": "layercake-phase4-name-agnostic-diagnostic/1",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION_CREDIT",
        "evaluation_sha256": hashlib.sha256(
            args.evaluation.read_bytes()
        ).hexdigest(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "distinct_prompts": len(records),
        "parseable_first_functions": parseable,
        "name_agnostic_functional_successes": successes,
        "name_agnostic_functional_success_rate": successes
        / max(1, len(records)),
        "records": records,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
