from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.rolling.preview import run_preview
from layercake.rolling.rubric import TrainingRubric
from layercake.rolling.syllabus import compile_syllabus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/curriculum_modes_benchmark.json")
    args = parser.parse_args()
    data = Path("data/rolling_preview_smoke.txt")
    if not data.exists():
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text("aaa easy\n{{ hard_symbol }}\nunicode café\n", encoding="utf-8")
    rubric = TrainingRubric(rubric_id="curriculum_modes", max_steps=2)
    preview = run_preview(rubric, data)
    modes = ["easy_to_hard", "entropy_balanced", "rehearsal_interleaved", "hard_to_easy"]
    result = {
        "status": "PASS",
        "preview_id": preview.preview_id,
        "modes": {mode: compile_syllabus(rubric, preview, mode=mode).to_dict() for mode in modes},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
