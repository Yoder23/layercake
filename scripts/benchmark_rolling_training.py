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
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", default="results/rolling_training_benchmark.json")
    args = parser.parse_args()

    registry = ModuleRegistry()
    model = TinyToyModel()
    registry.register("toy_model", model)
    registry.register("fake_domain_brick", torch.nn.Linear(1, 1))
    trainer = RollingTrainer(registry, root="artifacts/commits/rolling_benchmark")
    rubric = TrainingRubric(
        rubric_id="rolling_training_benchmark",
        description="Rolling training benchmark",
        max_steps=args.steps,
        trainable_modules=["toy_model"],
        gates=[{"type": "min_metric", "name": "score_gate", "metric": "score", "threshold": 0.5}],
    )
    start = time.perf_counter()
    commit, cert, _ = trainer.run_rubric(
        rubric,
        train_step=lambda: toy_train_step(model),
        metrics={"score": 1.0},
        certificate_path="results/certificates/rolling_training_benchmark.json",
    )
    elapsed = time.perf_counter() - start
    result = {
        "status": "PASS" if cert.passed else "FAIL",
        "commit_id": commit.commit_id,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(args.steps, 1),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if cert.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
