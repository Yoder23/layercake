from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.training.data import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/moonshot/phase4/python_functional_v1.jsonl"
SOURCE_SHA256 = (
    "9f8bbaee5794b3aab388d2e4f5269c641648e98af919261463f5396128f8a8c3"
)
PREFIXES = (
    "Code only. Implement",
    "No prose. Define",
    "Output valid Python for",
    "Create only the Python callable",
)


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise RuntimeError(f"functional probe is immutable: {output}")
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("source functional dataset identity changed")
    source_rows = [
        row for row in _load_rows(SOURCE)
        if row["split"] == "validation"
    ]
    rows = []
    source_names = {row["function_name"] for row in _load_rows(SOURCE)}
    for index, source in enumerate(source_rows):
        old_name = source["function_name"]
        new_name = old_name.replace(
            "_validation_", "_parallelprobe_"
        )
        if new_name == old_name or new_name in source_names:
            raise RuntimeError("fresh functional name construction failed")
        original = source["prompt"].replace(old_name, new_name)
        marker = f"Python function named {new_name}"
        if marker not in original:
            raise RuntimeError(
                f"source prompt shape changed for {source['id']}"
            )
        suffix = original.split(marker, 1)[1]
        prompt = f"{PREFIXES[index % len(PREFIXES)]} {new_name}{suffix}"
        response = source["response"].replace(old_name, new_name)
        rows.append(
            {
                **source,
                "id": source["id"].replace(
                    "validation", "parallel-probe", 1
                ),
                "split": "parallel_probe",
                "function_name": new_name,
                "prompt": prompt,
                "response": response,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "format": "layercake-phase4-parallel-plan-functional-probe/1",
        "status": "LOCKED_FRESH_PROMOTION_SCREEN_NOT_ACCESSED",
        "dataset": output.relative_to(ROOT).as_posix(),
        "dataset_sha256": sha256_file(output),
        "source_dataset": SOURCE.relative_to(ROOT).as_posix(),
        "source_dataset_sha256": SOURCE_SHA256,
        "split": "parallel_probe",
        "rows": len(rows),
        "families": len({row["family"] for row in rows}),
        "prompt_templates": len(PREFIXES),
        "function_names_disjoint_from_source": not bool(
            {row["function_name"] for row in rows} & source_names
        ),
        "model_outputs_accessed": 0,
        "promotion_result_accessed": False,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
