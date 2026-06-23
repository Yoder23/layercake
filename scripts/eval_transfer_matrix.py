from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import emit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/transfer_matrix.json")
    args = parser.parse_args()
    rows = [
        {
            "source_size": "mobile_25m",
            "source_seed": 42,
            "source_input_mode": "tokenized",
            "target_size": "mobile_25m",
            "target_seed": 42,
            "target_input_mode": "tokenized",
            "abi_version": "lc-abi/2",
            "brick_type": "sparse_low_rank",
            "domain_ppl_source": None,
            "domain_ppl_target": None,
            "degradation_ratio": None,
            "abi_drift": 0.0,
            "status": "SMOKE_ONLY",
        }
    ]
    emit({"schema_version": 1, "rows": rows}, args.output)


if __name__ == "__main__":
    main()
