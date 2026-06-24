from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _printable_rate(text: str) -> float:
    return sum(ch.isprintable() or ch.isspace() for ch in text) / max(len(text), 1)


def _alpha_space_rate(text: str) -> float:
    return sum(ch.isalpha() or ch.isspace() for ch in text) / max(len(text), 1)


def _distinct_trigram_rate(text: str) -> float:
    trigrams = [text[index : index + 3] for index in range(max(len(text) - 2, 0))]
    return len(set(trigrams)) / max(len(trigrams), 1)


def _has_repeated_ngram(text: str, width: int) -> bool:
    grams = [text[index : index + width] for index in range(max(len(text) - width + 1, 0))]
    return len(grams) != len(set(grams))


def main() -> int:
    bpe = load("scale15m_bpe_matched.json")
    source = load("scale15m_transition_lw280_2300_noprofile.json")
    generation = load("scale15m_transition_lw280_2300_generation_cpu1_norepeat4.json")
    transfer = load("lossless_domain_transition15m_2300_to_5m.json")
    adapter = load("scale15m_bpe_python_adapter_r16.json")

    layercake_utf8 = generation["layercake"]["utf8"]
    parameter_ratio = source["parameters"] / bpe["parameters"]
    bpb_ratio = source["general"]["bpb"] / bpe["general"]["bpb"]
    training_time_ratio = source["elapsed_seconds"] / bpe["elapsed_seconds"]
    training_byte_ratio = (
        source["estimated_total_training_bytes"]
        / bpe["estimated_total_training_bytes"]
    )
    domain_bpb_ratio = transfer["target"]["bpb"] / adapter["after"]["domain"]["bpb"]
    gates = {
        "source_at_least_2pct_smaller_than_bpe": parameter_ratio <= 0.98,
        "source_at_least_0_5pct_better_general_bpb": bpb_ratio <= 0.995,
        "source_at_least_1pct_faster_training": training_time_ratio <= 0.99,
        "source_uses_no_more_training_bytes": training_byte_ratio <= 1.0,
        "source_at_least_10pct_faster_cpu_generation": generation["speed_ratio"] >= 1.10,
        "source_generation_printable": _printable_rate(layercake_utf8) >= 0.95,
        "source_generation_alpha_space": _alpha_space_rate(layercake_utf8) >= 0.85,
        "source_generation_distinct_trigrams": (
            _distinct_trigram_rate(layercake_utf8) >= 0.80
        ),
        "source_generation_no_repeated_8gram": (
            not _has_repeated_ngram(layercake_utf8, 8)
        ),
        "source_generation_no_repeated_4gram": (
            not _has_repeated_ngram(layercake_utf8, 4)
        ),
        "transfer_ppl_ratio_exact": transfer["ppl_ratio"] == 1.0,
        "transfer_max_logit_diff_exact": transfer["max_logit_diff"] == 0.0,
        "transfer_generation_exact": transfer["generation"]["equal"],
        "transferred_domain_at_least_10pct_better_than_adapter": domain_bpb_ratio <= 0.90,
    }
    failed = [name for name, passed in gates.items() if not passed]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "15M-class transition-head LayerCake source/core rematch plus "
            "lossless transfer into the existing 5.40M receiver. This is not "
            "a 20M BPE win; the 20M time gate remains open."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "artifacts": {
            "source": "results/scale15m_transition_lw280_2300_noprofile.json",
            "generation": (
                "results/scale15m_transition_lw280_2300_generation_cpu1_norepeat4.json"
            ),
            "transfer": "results/lossless_domain_transition15m_2300_to_5m.json",
            "bpe": "results/scale15m_bpe_matched.json",
            "adapter": "results/scale15m_bpe_python_adapter_r16.json",
        },
        "metrics": {
            "layercake_parameters": source["parameters"],
            "bpe_parameters": bpe["parameters"],
            "layercake_general_bpb": source["general"]["bpb"],
            "bpe_general_bpb": bpe["general"]["bpb"],
            "layercake_training_seconds": source["elapsed_seconds"],
            "bpe_training_seconds": bpe["elapsed_seconds"],
            "layercake_training_bytes": source[
                "estimated_total_training_bytes"
            ],
            "bpe_training_bytes": bpe["estimated_total_training_bytes"],
            "parameter_ratio": parameter_ratio,
            "general_bpb_ratio": bpb_ratio,
            "training_time_ratio": training_time_ratio,
            "training_byte_ratio": training_byte_ratio,
            "cpu_generation_speed_ratio": generation["speed_ratio"],
            "generation_printable_rate": _printable_rate(layercake_utf8),
            "generation_alpha_space_rate": _alpha_space_rate(layercake_utf8),
            "generation_distinct_trigram_rate": _distinct_trigram_rate(layercake_utf8),
            "generation_has_repeated_8gram": _has_repeated_ngram(layercake_utf8, 8),
            "layercake_generation_utf8": generation["layercake"]["utf8"],
            "bpe_generation_utf8": generation["bpe"]["utf8"],
            "transfer_ppl_ratio": transfer["ppl_ratio"],
            "transfer_max_logit_diff": transfer["max_logit_diff"],
            "transfer_generation_equal": transfer["generation"]["equal"],
            "transferred_domain_bpb": transfer["target"]["bpb"],
            "transformer_adapter_domain_bpb": adapter["after"]["domain"]["bpb"],
            "transferred_domain_bpb_ratio": domain_bpb_ratio,
        },
    }
    output = RESULTS / "scale15m_transition_frontier_certificate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
