from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.rolling.preview import preview_summary, run_preview
from layercake.rolling.rubric import TrainingRubric


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rubric")
    parser.add_argument("--data", default="data/rolling_preview_smoke.txt")
    parser.add_argument("--output-dir", default="results/previews")
    args = parser.parse_args()
    data = Path(args.data)
    if not data.exists():
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text("hello layercake\nprint('byte patch')\nmobile cpu training\n", encoding="utf-8")
    rubric = TrainingRubric.from_file(args.rubric)
    preview = run_preview(rubric, data, output_dir=args.output_dir)
    print(json.dumps({"summary": preview_summary(preview), "preview": preview.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
