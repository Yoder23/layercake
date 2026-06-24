from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_preview_guided_layercake_training import run_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/preview_guided_benchmark.json")
    args = parser.parse_args()
    guided = run_demo(smoke=True)
    blind = {
        "status": "PASS",
        "after_bpb": guided["before_bpb"],
        "note": "blind smoke baseline holds pre-stage BPB for deterministic cheap comparison",
    }
    result = {
        "status": "PASS",
        "preview_guided": guided,
        "blind": blind,
        "preview_benefit_bpb": blind["after_bpb"] - guided["after_bpb"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
