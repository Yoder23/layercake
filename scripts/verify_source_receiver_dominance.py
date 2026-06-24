from __future__ import annotations

import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    frontier = load("results/dominance/tier1_local_frontier_certificate.json")
    locked = load("results/dominance/layercake_frontier_cpu_receiver_certificate.json")
    receiver = load("results/receiver_frontier_certificate.json")
    local_all_pass = all(frontier.get("passed", {}).values())
    receiver_gates = receiver["required_gates"]
    gates = {
        "source_local_frontier_passes": local_all_pass,
        "locked_source_receiver_cpu_frontier_passes": locked["status"] == "PASS",
        "receiver_smaller_than_transformer": receiver_gates["receiver_smaller"],
        "receiver_better_general_bpb": receiver_gates["receiver_better_general_bpb"],
        "receiver_faster_training": receiver_gates["receiver_faster_training"],
        "receiver_faster_cpu_generation": receiver_gates["receiver_faster_cpu_generation"],
        "transfer_ppl_ratio_exact": receiver["metrics"]["transfer_ppl_ratio"] == 1.0,
        "transfer_max_logit_diff_exact": receiver["metrics"]["transfer_max_logit_diff"] == 0.0,
        "transfer_generation_exact": receiver_gates["lossless_transfer_generation"],
        "transferred_domain_beats_transformer": receiver_gates["transferred_domain_beats_transformer"],
    }
    result = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "Source dominance plus receiver-after-lossless-transfer dominance. Local frontier probes are smoke/local scale; locked receiver certificate carries exact transfer.",
        "gates": gates,
        "failed": [name for name, passed in gates.items() if not passed],
        "source_frontier_certificate": "results/dominance/tier1_local_frontier_certificate.json",
        "locked_cpu_receiver_certificate": "results/dominance/layercake_frontier_cpu_receiver_certificate.json",
        "receiver_certificate": "results/receiver_frontier_certificate.json",
        "metrics": {
            "source_local_passed": frontier.get("passed", {}),
            "receiver_transfer_ppl_ratio": receiver["metrics"]["transfer_ppl_ratio"],
            "receiver_transfer_max_logit_diff": receiver["metrics"]["transfer_max_logit_diff"],
            "receiver_domain_bpb": receiver["metrics"]["receiver_transferred_domain_bpb"],
            "transformer_adapter_domain_bpb": receiver["metrics"]["transformer_adapter_domain_bpb"],
        },
    }
    output = Path("results/dominance/source_receiver_transfer_dominance.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
