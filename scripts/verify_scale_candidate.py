from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layercake.northstar import NorthStarMetrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    values = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    certificate = NorthStarMetrics(**values).certificate()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(certificate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(certificate, indent=2))
    if certificate["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
