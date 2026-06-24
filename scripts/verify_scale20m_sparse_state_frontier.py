from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    baseline = load("scale20m_bpe448_l7_seed6250.json")
    sparse = load("scale20m_sparse_state448_lw8_seed6250.json")
    probe = load("scale20m_sparse_state448_lw8_probe.json")
    dense = load("scale20m_lc448_w32_qk_seed6250.json")
    gates = {
        "smaller_model": sparse["parameters"] < baseline["parameters"],
        "better_quality": (
            sparse["general"]["bpb"] < baseline["general"]["bpb"]
        ),
        "no_more_training_bytes": (
            sparse["estimated_total_training_bytes"]
            <= baseline["estimated_total_training_bytes"]
        ),
        "faster_training": (
            sparse["elapsed_seconds"] < baseline["elapsed_seconds"]
        ),
        "better_than_dense_layercake_quality": (
            sparse["general"]["bpb"] < dense["general"]["bpb"]
        ),
        "probe_beats_dense_probe_quality": (
            probe["general"]["bpb"] < 2.3018265881927946
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    certificate = {
        "status": "PASS" if not failed else "FAIL",
        "claim": (
            "Pure-PyTorch sparse-state global patch core at the current "
            "20M scale boundary. It improves LayerCake quality versus dense "
            "20M variants but does not yet beat the matched BPE transformer."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "baseline_parameters": baseline["parameters"],
            "baseline_bpb": baseline["general"]["bpb"],
            "baseline_training_seconds": baseline["elapsed_seconds"],
            "sparse_parameters": sparse["parameters"],
            "sparse_bpb": sparse["general"]["bpb"],
            "sparse_training_seconds": sparse["elapsed_seconds"],
            "dense_best_bpb": dense["general"]["bpb"],
            "dense_best_training_seconds": dense["elapsed_seconds"],
            "sparse_probe_bpb": probe["general"]["bpb"],
            "sparse_probe_training_seconds": probe["elapsed_seconds"],
        },
        "diagnosis": (
            "Structured sparse-state mixing improves the LayerCake 20M BPB "
            "frontier from 2.0256 to 2.0214, but pure-PyTorch gathered "
            "attention is not training-efficient enough and still trails "
            "BPE quality by 0.0060 BPB."
        ),
    }
    output = RESULTS / "scale20m_sparse_state_frontier_certificate.json"
    output.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
