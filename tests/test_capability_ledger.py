import json

from layercake.rolling.reports import append_capability_ledger


def test_capability_ledger_appends_valid_row(tmp_path):
    path = tmp_path / "ledger.jsonl"
    row = append_capability_ledger(path, commit_id="c", rubric_id="r", preview_id="p", syllabus_id="s", value=1, threshold=2, passed=True)
    loaded = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert loaded["commit_id"] == row["commit_id"] == "c"
    assert loaded["passed"] is True
