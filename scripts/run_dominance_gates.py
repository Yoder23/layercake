from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.rolling.scaling_gates import run_dominance_suite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--output-dir", default="results/dominance")
    args = parser.parse_args()
    result = run_dominance_suite({
        "layercake_training_seconds": 1.0,
        "transformer_training_seconds": 1.0,
        "layercake_bpb": 2.0,
        "transformer_bpb": 2.0,
        "layercake_trainable_params": 10,
        "transformer_trainable_params": 10,
        "rollback_recovered": True,
        "transfer_exact": True,
    })
    output = Path(args.output_dir) / f"{args.run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
