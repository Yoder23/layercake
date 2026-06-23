from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    layercake = load("scale15m_hier4x4_full.json")
    layercake_inference = load("scale15m_hier4x4_inference.json")
    baseline = load("scale15m_bpe_matched.json")
    baseline_inference = load("scale15m_bpe_matched_inference.json")
    patch_row = next(
        row
        for row in layercake_inference["rows"]
        if row["path"] == "patch_base"
    )
    gates = {
        "parameter_count": layercake["parameters"] <= baseline["parameters"],
        "matched_training_bytes": abs(
            layercake["estimated_total_training_bytes"]
            - baseline["estimated_total_training_bytes"]
        )
        / baseline["estimated_total_training_bytes"]
        <= 0.01,
        "training_wall_time": (
            layercake["elapsed_seconds"] <= baseline["elapsed_seconds"]
        ),
        "heldout_general_bpb": (
            layercake["general"]["bpb"] <= baseline["general"]["bpb"]
        ),
        "gpu_inference_throughput": (
            patch_row["bytes_per_second"]
            >= baseline_inference["estimated_bytes_per_second"]
        ),
    }
    result = {
        "status": "PASS" if all(gates.values()) else "OPEN",
        "required_gates": gates,
        "metrics": {
            "layercake_parameters": layercake["parameters"],
            "baseline_parameters": baseline["parameters"],
            "layercake_training_bytes": layercake[
                "estimated_total_training_bytes"
            ],
            "baseline_training_bytes": baseline[
                "estimated_total_training_bytes"
            ],
            "layercake_training_seconds": layercake["elapsed_seconds"],
            "baseline_training_seconds": baseline["elapsed_seconds"],
            "layercake_general_bpb": layercake["general"]["bpb"],
            "baseline_general_bpb": baseline["general"]["bpb"],
            "layercake_gpu_bytes_per_second": patch_row["bytes_per_second"],
            "baseline_gpu_bytes_per_second": baseline_inference[
                "estimated_bytes_per_second"
            ],
        },
        "next_required_improvement": {
            "absolute_bpb_gap": (
                layercake["general"]["bpb"] - baseline["general"]["bpb"]
            ),
            "throughput_multiplier_required": (
                baseline_inference["estimated_bytes_per_second"]
                / patch_row["bytes_per_second"]
            ),
        },
        "scope": (
            "Single seed, 14.7-14.8M parameters, approximately 10.3M sampled "
            "training bytes, 20 MB local corpus, RTX 3080 Laptop GPU."
        ),
    }
    path = RESULTS / "general_core_frontier.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
