from layercake.rolling.preview import run_preview
from layercake.rolling.rubric import TrainingRubric
from layercake.rolling.syllabus import compile_syllabus


def test_syllabus_modes_compile_and_order(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("easy\n{{{{ hard }}}}\nmedium text\n", encoding="utf-8")
    rubric = TrainingRubric(rubric_id="s", max_steps=3)
    preview = run_preview(rubric, data, output_dir=tmp_path)
    easy = compile_syllabus(rubric, preview, mode="easy_to_hard", output_dir=tmp_path)
    hard = compile_syllabus(rubric, preview, mode="hard_to_easy", output_dir=tmp_path)
    assert easy.ordered_data_buckets[0]["difficulty"] <= easy.ordered_data_buckets[-1]["difficulty"]
    assert hard.ordered_data_buckets[0]["difficulty"] >= hard.ordered_data_buckets[-1]["difficulty"]
    balanced = compile_syllabus(rubric, preview, mode="entropy_balanced", output_dir=tmp_path)
    assert len(balanced.sampling_weights) == len(balanced.ordered_data_buckets)
    rehearsal = compile_syllabus(rubric, preview, mode="rehearsal_interleaved", output_dir=tmp_path)
    assert any(bucket.get("rehearsal") for bucket in rehearsal.ordered_data_buckets[1:])
