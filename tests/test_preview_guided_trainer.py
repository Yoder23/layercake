import torch

from layercake.rolling.registry import ModuleRegistry
from layercake.rolling.rubric import TrainingRubric
from layercake.rolling.trainer import RollingTrainer, TinyToyModel


def test_preview_guided_trainer_commit_and_rollback(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello layercake", encoding="utf-8")
    model = TinyToyModel()
    registry = ModuleRegistry()
    registry.register("layercake_core", model)
    trainer = RollingTrainer(registry, root=tmp_path / "commits")
    parent = trainer.create_commit(None, TrainingRubric(rubric_id="initial"), "initial", "passed")
    rubric = TrainingRubric(rubric_id="pg", max_steps=2, trainable_modules=["layercake_core"], gates=[{"type": "min_metric", "metric": "score", "threshold": 0.5}])
    commit, cert, rollback, preview, syllabus = trainer.run_preview_guided(rubric, data, model=None, parent_commit=parent, train_step=lambda: 0.9, metrics={"score": 1.0}, certificate_path=tmp_path / "cert.json")
    assert cert.passed and commit.commit_id and preview.preview_id and syllabus.syllabus_id
    values = iter([1.0, 9.0])
    failed, failed_cert, rollback, *_ = trainer.run_preview_guided(rubric, data, model=None, parent_commit=commit, train_step=lambda: next(values), metrics={"score": 1.0}, certificate_path=tmp_path / "bad.json")
    assert not failed_cert.passed
    assert rollback["restored_commit"] == commit.commit_id
