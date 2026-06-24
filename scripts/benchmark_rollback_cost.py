from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from layercake.rolling.registry import ModuleRegistry
from layercake.rolling.rubric import TrainingRubric
from layercake.rolling.trainer import RollingTrainer, TinyToyModel, toy_train_step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/rolling_rollback_benchmark.json")
    args = parser.parse_args()

    registry = ModuleRegistry()
    model = TinyToyModel()
    registry.register("toy_model", model)
    registry.register("fake_domain_brick", torch.nn.Linear(1, 1))
    trainer = RollingTrainer(registry, root="artifacts/commits/rollback_benchmark")
    pass_rubric = TrainingRubric(
        rubric_id="rollback_parent",
        description="Rollback parent",
        max_steps=1,
        trainable_modules=["toy_model"],
        gates=[{"type": "min_metric", "name": "score_gate", "metric": "score", "threshold": 0.5}],
    )
    parent, _, _ = trainer.run_rubric(
        pass_rubric,
        train_step=lambda: toy_train_step(model),
        metrics={"score": 1.0},
        certificate_path="results/certificates/rollback_parent.json",
    )
    fail_rubric = TrainingRubric(
        rubric_id="rollback_failure",
        description="Rollback failure",
        max_steps=1,
        trainable_modules=["toy_model"],
        gates=[{"type": "min_metric", "name": "score_gate", "metric": "score", "threshold": 0.5}],
    )
    start = time.perf_counter()
    failed, cert, rollback = trainer.run_rubric(
        fail_rubric,
        parent_commit=parent,
        train_step=lambda: toy_train_step(model, damage=True),
        metrics={"score": 0.0},
        certificate_path="results/certificates/rollback_failure.json",
    )
    elapsed = time.perf_counter() - start
    result = {
        "status": "PASS" if (not cert.passed and rollback) else "FAIL",
        "failed_commit": failed.commit_id,
        "restored_commit": rollback["restored_commit"] if rollback else None,
        "elapsed_seconds": elapsed,
        "restored_modules": rollback["restored_modules"] if rollback else [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
