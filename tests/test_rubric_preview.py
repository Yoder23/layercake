from layercake.rolling.preview import byte_entropy, run_preview
from layercake.rolling.rubric import TrainingRubric


def test_preview_hash_stable_and_changes_with_dataset(tmp_path):
    data = tmp_path / "a.txt"
    data.write_text("aaaa", encoding="utf-8")
    rubric = TrainingRubric(rubric_id="p")
    p1 = run_preview(rubric, data, output_dir=tmp_path)
    p2 = run_preview(rubric, data, output_dir=tmp_path)
    assert p1.compute_hash() == p2.compute_hash()
    data.write_text("abcd", encoding="utf-8")
    p3 = run_preview(rubric, data, output_dir=tmp_path)
    assert p1.dataset_manifest_hash != p3.dataset_manifest_hash


def test_byte_entropy_simple_data():
    assert byte_entropy(b"aaaa") == 0.0
    assert round(byte_entropy(b"ab"), 4) == 1.0


def test_fixed_patch_compression_measured(tmp_path):
    data = tmp_path / "patch.txt"
    data.write_text("abcd", encoding="utf-8")
    preview = run_preview(TrainingRubric(rubric_id="patch"), data, patch_size=2, output_dir=tmp_path)
    assert preview.patch_count == 2
    assert preview.patch_compression_ratio == 2.0
