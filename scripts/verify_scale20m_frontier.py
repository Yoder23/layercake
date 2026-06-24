from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    baseline = load("scale20m_bpe448_l7_seed6250.json")
    candidates = {
        "w32_qk": load("scale20m_lc448_w32_qk_seed6250.json"),
        "w32_qk_batch24": load(
            "scale20m_lc448_w32_qk_batch24_seed6250.json"
        ),
        "w32_qk_5g3l": load("scale20m_lc448_5g3l_w32_qk_seed6250.json"),
        "w16_qk": load("scale20m_lc448_w16_qk_seed6250.json"),
        "w16_qk_batch24": load(
            "scale20m_lc448_w16_qk_batch24_seed6250.json"
        ),
    }
    gates = {}
    for name, candidate in candidates.items():
        gates[f"{name}_smaller_model"] = (
            candidate["parameters"] < baseline["parameters"]
        )
        gates[f"{name}_better_quality"] = (
            candidate["general"]["bpb"] < baseline["general"]["bpb"]
        )
        gates[f"{name}_no_more_training_bytes"] = (
            candidate["estimated_total_training_bytes"]
            <= baseline["estimated_total_training_bytes"]
        )
        gates[f"{name}_faster_training"] = (
            candidate["elapsed_seconds"] < baseline["elapsed_seconds"]
        )
    failed = [name for name, passed in gates.items() if not passed]
    best_quality_name, best_quality = min(
        candidates.items(), key=lambda item: item[1]["general"]["bpb"]
    )
    fastest_name, fastest = min(
        candidates.items(), key=lambda item: item[1]["elapsed_seconds"]
    )
    certificate = {
        "status": "PASS" if not failed else "FAIL",
        "claim": (
            "Intermediate width-448 LayerCake scale-up versus a matched "
            "20.6M BPE transformer. This certificate is expected to fail "
            "until the general-core scale bottleneck is resolved."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "baseline_parameters": baseline["parameters"],
            "baseline_bpb": baseline["general"]["bpb"],
            "baseline_training_seconds": baseline["elapsed_seconds"],
            "best_quality_candidate": best_quality_name,
            "best_quality_bpb": best_quality["general"]["bpb"],
            "best_quality_training_seconds": best_quality[
                "elapsed_seconds"
            ],
            "fastest_candidate": fastest_name,
            "fastest_bpb": fastest["general"]["bpb"],
            "fastest_training_seconds": fastest["elapsed_seconds"],
            "candidates": {
                name: {
                    "parameters": candidate["parameters"],
                    "bpb": candidate["general"]["bpb"],
                    "training_seconds": candidate["elapsed_seconds"],
                    "training_bytes": candidate[
                        "estimated_total_training_bytes"
                    ],
                }
                for name, candidate in candidates.items()
            },
        },
        "diagnosis": (
            "Width scaling from the passing 15M architecture reduces the "
            "quality gap versus the prior 26M run but still loses to BPE. "
            "Batch compression, shorter local windows, and shifting one "
            "block between local/global depth do not close the quality or "
            "training-time gates."
        ),
    }
    output = RESULTS / "scale20m_frontier_certificate.json"
    output.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
