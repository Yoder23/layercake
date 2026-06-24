import torch

from layercake.rolling.registry import ModuleRegistry
from layercake.rolling.rubric import TrainingRubric
from layercake.rolling.trainer import RollingTrainer, TinyToyModel, toy_train_step


def test_trainer_pass_fail_and_rollback(tmp_path):
    registry = ModuleRegistry()
    model = TinyToyModel()
    registry.register("toy_model", model)
    registry.register("fake_domain_brick", torch.nn.Linear(1, 1))
    trainer = RollingTrainer(registry, root=tmp_path / "commits")
    rubric = TrainingRubric(
        rubric_id="pass",
        description="pass",
        max_steps=1,
        trainable_modules=["toy_model"],
        gates=[{"type": "min_metric", "name": "score_gate", "metric": "score", "threshold": 0.5}],
    )
    parent, cert, rollback = trainer.run_rubric(
        rubric,
        train_step=lambda: toy_train_step(model),
        metrics={"score": 1.0},
        certificate_path=tmp_path / "pass_cert.json",
    )
    assert cert.passed and rollback is None and parent.status == "passed"
    failing = TrainingRubric(
        rubric_id="fail",
        description="fail",
        max_steps=1,
        trainable_modules=["toy_model"],
        gates=[{"type": "min_metric", "name": "score_gate", "metric": "score", "threshold": 0.5}],
    )
    failed, failed_cert, rollback = trainer.run_rubric(
        failing,
        parent_commit=parent,
        train_step=lambda: toy_train_step(model, damage=True),
        metrics={"score": 0.0},
        certificate_path=tmp_path / "fail_cert.json",
    )
    assert not failed_cert.passed
    assert failed.status == "failed"
    assert rollback["restored_commit"] == parent.commit_id
    assert registry.module_hashes() == parent.module_hashes
