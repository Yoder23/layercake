from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    receiver = load("receiver68m_4g4l_batch24_seed4242.json")
    baseline = load("receiver_bpe69m_seed4242.json")
    transfer = load("northstar_transfer_to_receiver68m_batch24.json")
    generation_cpu = load(
        "receiver68m_4g4l_batch24_generation_cpu1.json"
    )
    adapter = load("scale15m_bpe_python_adapter_r16.json")
    gates = {
        "receiver_smaller": (
            receiver["parameters"] < baseline["parameters"]
        ),
        "receiver_better_general_bpb": (
            receiver["general"]["bpb"] < baseline["general"]["bpb"]
        ),
        "receiver_faster_training": (
            receiver["elapsed_seconds"] < baseline["elapsed_seconds"]
        ),
        "receiver_faster_cpu_generation": (
            generation_cpu["speed_ratio"] > 1.0
        ),
        "lossless_transfer_ppl": transfer["ppl_ratio"] == 1.0,
        "lossless_transfer_logits": transfer["max_logit_diff"] == 0.0,
        "lossless_transfer_generation": transfer["generation"]["equal"],
        "transferred_domain_beats_transformer": (
            transfer["target"]["bpb"]
            < adapter["after"]["domain"]["bpb"]
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "receiver_parameters": receiver["parameters"],
            "baseline_parameters": baseline["parameters"],
            "receiver_general_bpb": receiver["general"]["bpb"],
            "baseline_general_bpb": baseline["general"]["bpb"],
            "receiver_training_seconds": receiver["elapsed_seconds"],
            "baseline_training_seconds": baseline["elapsed_seconds"],
            "receiver_cpu_generation_speed_ratio": generation_cpu[
                "speed_ratio"
            ],
            "receiver_cpu_generation_bytes_per_second": generation_cpu[
                "layercake"
            ]["bytes_per_second"],
            "baseline_cpu_generation_bytes_per_second": generation_cpu[
                "bpe"
            ]["bytes_per_second"],
            "transfer_ppl_ratio": transfer["ppl_ratio"],
            "transfer_max_logit_diff": transfer["max_logit_diff"],
            "receiver_transferred_domain_bpb": transfer["target"]["bpb"],
            "transformer_adapter_domain_bpb": adapter["after"]["domain"][
                "bpb"
            ],
        },
    }
    output = RESULTS / "receiver_frontier_certificate.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
