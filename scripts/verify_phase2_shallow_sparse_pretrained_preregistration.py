"""Fail-closed checks for the final shallow sparse Phase 2 hypothesis."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    Path(os.environ["USERPROFILE"])
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--distilgpt2"
    / "snapshots"
    / "2290a62682d06624634c1f46a6ad5be0f47f38aa"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    contract_path = ROOT / "moonshot/phase2_shallow_sparse_pretrained_preregistration.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    hypothesis = contract["hypothesis"]
    student = contract["student_contract"]
    if contract.get("format") != "layercake-phase2-shallow-sparse-pretrained-preregistration/1":
        errors.append("unexpected contract format")
    if hypothesis["attention_blocks"] != 3:
        errors.append("student block count changed")
    if hypothesis["task_cake_count"] != 10 or hypothesis["task_cake_rank"] != 64:
        errors.append("task-cake graph changed")
    if student["retained_source_blocks"] != [0, 2, 5]:
        errors.append("retained source blocks changed")
    if student["source_control_instruction_steps"] != 1200:
        errors.append("source-control step budget changed")
    if student["student_distillation_and_instruction_steps"] != 2400:
        errors.append("student step budget changed")
    source = contract["source_control"]
    for filename, key in (
        ("config.json", "config_sha256"),
        ("model.safetensors", "model_sha256"),
        ("tokenizer.json", "tokenizer_json_sha256"),
    ):
        if sha256_file(SNAPSHOT / filename) != source[key]:
            errors.append(f"source {filename} hash differs")
    curriculum = ROOT / contract["training_contract"]["instruction_curriculum"]
    if (
        sha256_file(curriculum)
        != contract["training_contract"]["instruction_curriculum_sha256"]
    ):
        errors.append("instruction curriculum hash differs")
    decision = ROOT / contract["prior_branch_decision"]
    if not json.loads(decision.read_text(encoding="utf-8"))[
        "decision"
    ].startswith("CLOSED_NEGATIVE"):
        errors.append("prior branch is not closed negative")
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "contract": contract_path.relative_to(ROOT).as_posix(),
                "contract_sha256": sha256_file(contract_path),
                "source_model_sha256": source["model_sha256"],
                "test_accessed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
