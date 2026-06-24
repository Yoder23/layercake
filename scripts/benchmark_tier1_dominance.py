from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.rolling.tier1 import run_tier1_dominance_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", default="results/dominance/tier1_smoke.json")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-byte", type=int, default=8)
    parser.add_argument("--d-abi", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=64)
    args = parser.parse_args()
    result = run_tier1_dominance_smoke(
        steps=args.steps,
        output_path=args.output,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        d_byte=args.d_byte,
        d_abi=args.d_abi,
        max_patches=args.max_patches,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
