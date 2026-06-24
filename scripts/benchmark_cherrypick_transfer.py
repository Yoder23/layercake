from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_rolling_training import run_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/rolling_cherrypick_benchmark.json")
    args = parser.parse_args()
    result = run_demo(smoke=True)
    benchmark = {
        "status": result["status"],
        "source_commit": result["successful_commit"],
        "target_commit": result["final_commit"],
        "cherry_pick_result": result["cherry_pick_result"],
        "transfer_exactness": {
            "abi_match": result["cherry_pick_result"]["abi_match"],
            "input_interface_match": result["cherry_pick_result"]["input_interface_match"],
            "shape_match": result["cherry_pick_result"]["shape_match"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(benchmark, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(benchmark, indent=2, sort_keys=True))
    return 0 if benchmark["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
