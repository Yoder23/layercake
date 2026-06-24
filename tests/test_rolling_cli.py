import json

from layercake.rolling.cli import main


def test_cli_help_and_run_rubric(tmp_path, capsys):
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps({"rubric_id": "cli", "name": "CLI", "gates": [{"name": "score_gate", "type": "min_metric", "metric": "score", "threshold": 0.5}]}),
        encoding="utf-8",
    )
    assert main(["run-rubric", str(rubric)]) == 0
    out = capsys.readouterr().out
    assert '"rubric_id": "cli"' in out
