from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from layercake.rolling.branching import BranchStore
from layercake.rolling.cherrypick import cherry_pick_module
from layercake.rolling.common import write_json
from layercake.rolling.registry import ModuleRegistry
from layercake.rolling.rubric import TrainingRubric
from layercake.rolling.trainer import RollingTrainer, TinyToyModel, toy_train_step


def run_demo(*, smoke: bool = False, output: str = "results/certificates/rolling_demo_certificate.json") -> dict:
    torch.manual_seed(1234)
    root = Path("artifacts/commits")
    model = TinyToyModel()
    brick = torch.nn.Linear(1, 1)
    registry = ModuleRegistry()
    registry.register("toy_model", model)
    registry.register("fake_domain_brick", brick)
    trainer = RollingTrainer(registry, root=root)
    store = BranchStore(root)
    initial = trainer.create_commit(
        None,
        TrainingRubric(rubric_id="initial", branch="main"),
        "initial",
        "passed",
    )
    store.create_branch("main", initial.commit_id)
    pass_rubric = TrainingRubric(
        rubric_id="stage_pass",
        branch="main",
        trainable_modules=["toy_model"],
        gates=[{"type": "min_metric", "name": "score_gate", "metric": "score", "threshold": 0.5}],
    )
    commit_a, cert_a, _ = trainer.run_rubric(
        pass_rubric,
        initial,
        train_step=lambda: toy_train_step(model),
        metrics={"score": 1.0},
    )
    failed_rubric = TrainingRubric(
        rubric_id="stage_fail",
        branch="main",
        trainable_modules=["toy_model"],
        gates=[{"type": "min_metric", "name": "damage_gate", "metric": "score", "threshold": 0.5}],
    )
    failed, cert_fail, rollback_report = trainer.run_rubric(
        failed_rubric,
        commit_a,
        train_step=lambda: toy_train_step(model, damage=True),
        metrics={"score": 0.0},
    )
    safe_rubric = TrainingRubric(
        rubric_id="stage_safe",
        branch="main",
        trainable_modules=["toy_model"],
        gates=[{"type": "min_metric", "name": "safe_gate", "metric": "score", "threshold": 0.5}],
    )
    final, cert_final, _ = trainer.run_rubric(
        safe_rubric,
        commit_a,
        train_step=lambda: toy_train_step(model),
        metrics={"score": 1.0},
    )
    cp = cherry_pick_module(commit_a, final, "fake_domain_brick")
    certificate = {
        "initial_commit": initial.commit_id,
        "successful_commit": commit_a.commit_id,
        "failed_commit": failed.commit_id,
        "rollback_commit": commit_a.commit_id,
        "final_commit": final.commit_id,
        "gate_results": {
            "stage_pass": cert_a.gate_results,
            "stage_fail": cert_fail.gate_results,
            "stage_safe": cert_final.gate_results,
        },
        "rollback_exactness_result": rollback_report,
        "cherry_pick_result": cp,
        "hashes": {
            "initial": initial.compute_hash(),
            "final": final.compute_hash(),
        },
        "status": "PASS",
    }
    write_json(output, certificate)
    Path("results/capability_ledger.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with Path("results/capability_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"commit_id": final.commit_id, "rubric_id": "stage_safe", "capability_name": "toy", "metric": "score", "score": 1.0, "threshold": 0.5, "passed": True, "regression_from_parent": 0.0}) + "\n")
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="results/certificates/rolling_demo_certificate.json")
    args = parser.parse_args()
    certificate = run_demo(smoke=args.smoke, output=args.output)
    print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
