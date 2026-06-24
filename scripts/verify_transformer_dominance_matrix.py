from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def main() -> int:
    local = load("results/dominance/tier1_local_frontier_certificate.json")
    transition15 = load("results/scale15m_transition_frontier_certificate.json")
    receiver = load("results/receiver_frontier_certificate.json")
    source_receiver = load("results/dominance/source_receiver_transfer_dominance.json")
    bpe20 = load("results/scale20m_bpe448_l7_seed6250.json")
    transition20 = load("results/scale20m_transition_lw336_conv2_2500.json")

    receiver_gates = receiver["required_gates"]
    source_receiver_gates = source_receiver["gates"]
    twenty_m_gates = {
        "quality_beats_bpe20": transition20["general"]["bpb"] < bpe20["general"]["bpb"],
        "training_time_beats_bpe20": transition20["elapsed_seconds"] < bpe20["elapsed_seconds"],
        "params_below_bpe20": transition20["parameters"] < bpe20["parameters"],
        "training_bytes_no_more_than_bpe20": (
            transition20["estimated_total_training_bytes"]
            <= bpe20["estimated_total_training_bytes"]
        ),
    }
    promoted_tiers = {
        "local_methodology_ladder": {
            "status": "PASS" if all(local.get("passed", {}).values()) else "FAIL",
            "scope": local["scope"],
            "required_gates": local.get("passed", {}),
            "failed": [name for name, passed in local.get("passed", {}).items() if not passed],
        },
        "full_corpus_15m_source_and_transfer": {
            "status": transition15["status"],
            "scope": transition15["scope"],
            "required_gates": transition15["required_gates"],
            "failed": transition15["failed_required"],
        },
        "receiver_after_lossless_transfer": {
            "status": "PASS" if all(source_receiver_gates.values()) else "FAIL",
            "scope": source_receiver["scope"],
            "required_gates": {
                "receiver_smaller_than_transformer": receiver_gates["receiver_smaller"],
                "receiver_better_general_bpb": receiver_gates["receiver_better_general_bpb"],
                "receiver_faster_training": receiver_gates["receiver_faster_training"],
                "receiver_faster_cpu_generation": receiver_gates["receiver_faster_cpu_generation"],
                "transfer_ppl_ratio_exact": source_receiver_gates["transfer_ppl_ratio_exact"],
                "transfer_max_logit_diff_exact": source_receiver_gates["transfer_max_logit_diff_exact"],
                "transfer_generation_exact": source_receiver_gates["transfer_generation_exact"],
                "transferred_domain_beats_transformer": source_receiver_gates["transferred_domain_beats_transformer"],
            },
            "failed": _failed({
                "receiver_smaller_than_transformer": receiver_gates["receiver_smaller"],
                "receiver_better_general_bpb": receiver_gates["receiver_better_general_bpb"],
                "receiver_faster_training": receiver_gates["receiver_faster_training"],
                "receiver_faster_cpu_generation": receiver_gates["receiver_faster_cpu_generation"],
                "transfer_ppl_ratio_exact": source_receiver_gates["transfer_ppl_ratio_exact"],
                "transfer_max_logit_diff_exact": source_receiver_gates["transfer_max_logit_diff_exact"],
                "transfer_generation_exact": source_receiver_gates["transfer_generation_exact"],
                "transferred_domain_beats_transformer": source_receiver_gates["transferred_domain_beats_transformer"],
            }),
        },
    }
    unpromoted_tiers = {
        "full_corpus_20m_source": {
            "status": "OPEN",
            "reason": (
                "Quality, parameter, and training-byte gates pass, but training-time "
                "dominance does not. This tier is intentionally excluded from promoted "
                "dominance claims until every required gate passes."
            ),
            "artifact": "results/scale20m_transition_lw336_conv2_2500.json",
            "required_gates": twenty_m_gates,
            "failed": _failed(twenty_m_gates),
            "metrics": {
                "layercake_bpb": transition20["general"]["bpb"],
                "bpe_bpb": bpe20["general"]["bpb"],
                "layercake_training_seconds": transition20["elapsed_seconds"],
                "bpe_training_seconds": bpe20["elapsed_seconds"],
                "layercake_parameters": transition20["parameters"],
                "bpe_parameters": bpe20["parameters"],
            },
        }
    }
    promoted_failures = {
        name: tier["failed"]
        for name, tier in promoted_tiers.items()
        if tier["failed"] or tier["status"] != "PASS"
    }
    result = {
        "status": "PASS" if not promoted_failures else "FAIL",
        "claim_boundary": (
            "PASS means every promoted dominance tier has required transformer gates. "
            "OPEN tiers are retained evidence and cannot be marketed as wins."
        ),
        "promoted_tiers": promoted_tiers,
        "unpromoted_tiers": unpromoted_tiers,
        "failed_promoted_tiers": promoted_failures,
    }
    output = RESULTS / "transformer_dominance_matrix.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
