from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_GATES = (
    "lower_training_time",
    "lower_final_bpb",
    "fewer_trainable_params",
    "preview_beats_blind_bpb",
    "layercake_faster_cpu_generation",
    "generation_printable",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", default="results/dominance/tier1_smoke.json")
    args = parser.parse_args()
    data = json.loads(Path(args.certificate).read_text(encoding="utf-8"))
    gates = data["dominance"]["gates"]
    missing = [gate for gate in REQUIRED_GATES if gate not in gates]
    failed = [gate for gate in REQUIRED_GATES if not gates.get(gate, False)]
    result = {
        "certificate": args.certificate,
        "status": "PASS" if not missing and not failed and data.get("status") == "PASS" else "FAIL",
        "missing": missing,
        "failed": failed,
        "scope": data.get("scope"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
