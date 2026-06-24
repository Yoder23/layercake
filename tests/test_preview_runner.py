from pathlib import Path

from layercake.rolling.preview import RubricPreview, run_preview
from layercake.rolling.rubric import TrainingRubric


def test_preview_runner_saves_artifact(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("hello", encoding="utf-8")
    preview = run_preview(TrainingRubric(rubric_id="runner"), data, output_dir=tmp_path)
    path = tmp_path / f"{preview.preview_id}.json"
    assert path.exists()
    assert RubricPreview.load(path).preview_id == preview.preview_id
