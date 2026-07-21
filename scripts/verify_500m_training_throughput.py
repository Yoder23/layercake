"""Compare 500M LayerCake training-throughput quickruns.

This verifier intentionally does not promote claims. It records whether a
candidate architecture improves measured training throughput on the local
hardware under matched quickrun settings.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dominance"


def _load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    latest = data["latest"]
    return {
        "path": str(path.relative_to(ROOT)),
        "status": data.get("status"),
        "config_name": data.get("config_name"),
        "global_block": data.get("model_config", {}).get("global_block"),
        "trainable_params": latest["trainable_params"],
        "steps_per_second": latest["steps_per_second"],
        "gib_per_hour": latest["gib_per_hour"],
        "bpb": latest["bpb"],
        "data_seconds_per_step": latest["data_seconds_per_step"],
        "forward_backward_seconds_per_step": latest[
            "forward_backward_seconds_per_step"
        ],
        "optimizer_seconds_per_step": latest["optimizer_seconds_per_step"],
    }


def main() -> None:
    dense = _load_metrics(
        ROOT / "runs_experiment" / "byte_500m_core_quickrun" / "training_metrics.json"
    )
    sparse = _load_metrics(
        ROOT
        / "runs_experiment"
        / "byte_500m_sparse_state_quickrun"
        / "training_metrics.json"
    )
    speed_ratio = sparse["steps_per_second"] / max(dense["steps_per_second"], 1e-12)
    fwd_bwd_ratio = sparse["forward_backward_seconds_per_step"] / max(
        dense["forward_backward_seconds_per_step"], 1e-12
    )
    output = {
        "status": "PASS" if speed_ratio > 1.0 else "FAIL",
        "claim": "500M sparse-state quickrun beats dense training throughput",
        "dense": dense,
        "sparse_state": sparse,
        "sparse_vs_dense_steps_per_second_ratio": speed_ratio,
        "sparse_vs_dense_forward_backward_time_ratio": fwd_bwd_ratio,
        "interpretation": (
            "Sparse-state is faster than dense under this matched quickrun."
            if speed_ratio > 1.0
            else "Sparse-state is slower than dense under this matched quickrun; do not promote it as the 500M speed path without a longer-context rematch or kernel-level optimization."
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "500m_training_throughput_comparison.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
