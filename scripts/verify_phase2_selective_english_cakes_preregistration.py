"""Fail-closed checks for the selective English-cake preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    contract_path = ROOT / "moonshot/phase2_selective_english_cakes_preregistration.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if contract.get("status") != "PREREGISTERED":
        errors.append("contract is not preregistered")
    if contract["hypothesis"]["task_cake_count"] != 10:
        errors.append("task-cake count changed")
    if contract["hypothesis"]["task_cake_rank"] != 64:
        errors.append("task-cake rank changed")
    substrate = contract["substrate_contract"]
    if substrate["distinct_output_bytes"] != 200_000_000:
        errors.append("substrate byte budget changed")
    if sum(substrate["mixture"].values()) != substrate["distinct_output_bytes"]:
        errors.append("substrate mixture does not sum to the declared byte budget")
    if substrate["pretraining_steps"] != 24_414:
        errors.append("substrate step budget changed")
    training = contract["training_contract"]
    if training["instruction_initial_steps"] != 3_000:
        errors.append("instruction initial budget changed")
    if training["instruction_same_lineage_extension_steps_max"] != 6_000:
        errors.append("instruction extension budget changed")
    curriculum = ROOT / training["instruction_curriculum"]
    if sha256_file(curriculum) != training["instruction_curriculum_sha256"]:
        errors.append("instruction curriculum hash differs")
    parent = (
        ROOT
        / contract["parent_checkpoint_path"]
        / "model.safetensors"
    )
    if sha256_file(parent) != contract["parent_checkpoint_sha256"]:
        errors.append("parent checkpoint hash differs")
    decision = ROOT / contract["prior_branch_decision"]
    if json.loads(decision.read_text(encoding="utf-8"))["decision"] != "CLOSED_NEGATIVE":
        errors.append("prior factor branch is not closed negative")
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "contract": contract_path.relative_to(ROOT).as_posix(),
                "contract_sha256": sha256_file(contract_path),
                "parent_checkpoint_sha256": contract["parent_checkpoint_sha256"],
                "test_accessed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
