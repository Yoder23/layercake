from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DOMAINS = {
    "game_dialogue": {
        "train": "results/game_dialogue_smoke_gru_train.json",
        "quantized": "results/game_dialogue_smoke_gru_int8.json",
        "transfer": "results/game_dialogue_smoke_gru_lossless_transfer.json",
    },
    "game_lore": {
        "train": "results/game_lore_smoke_gru_train.json",
        "quantized": "results/game_lore_smoke_gru_int8.json",
        "transfer": "results/game_lore_smoke_gru_lossless_transfer.json",
    },
    "game_quest_state": {
        "train": "results/game_quest_state_smoke_gru_train.json",
        "quantized": "results/game_quest_state_smoke_gru_int8.json",
        "transfer": "results/game_quest_state_smoke_gru_lossless_transfer.json",
    },
    "technical_text": {
        "train": "results/technical_text_smoke_gru_train.json",
        "quantized": "results/technical_text_smoke_gru_int8.json",
        "transfer": "results/technical_text_smoke_gru_lossless_transfer.json",
    },
}


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def printable(text: str) -> bool:
    return all(char.isprintable() or char in "\n\r\t" for char in text)


def main() -> int:
    per_domain = {}
    for name, paths in DOMAINS.items():
        train = load(paths["train"])
        quantized = load(paths["quantized"])
        transfer = load(paths["transfer"])
        gates = {
            "trained": train["status"] == "TRAINED",
            "custom_file_used": bool(train["domain_files"]),
            "int8_quantized": quantized["status"] == "PASS"
            and quantized["quantization"] == "int8_symmetric_per_tensor",
            "payload_compressed": quantized["compression_ratio"] < 0.30,
            "transfer_exact_status": transfer["status"] == "PASS",
            "ppl_ratio_exact": transfer["ppl_ratio"] == 1.0,
            "max_logit_diff_exact": transfer["max_logit_diff"] == 0.0,
            "generation_equal": transfer["generation"]["equal"],
            "bpb_under_3": transfer["source"]["bpb"] < 3.0,
            "top1_accuracy_over_50pct": (
                transfer["source"]["top1_byte_accuracy"] > 0.50
            ),
            "generation_printable": printable(transfer["generation"]["predicted_utf8"]),
        }
        per_domain[name] = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "required_gates": gates,
            "failed": [gate for gate, passed in gates.items() if not passed],
            "metrics": {
                "train_bytes": train["train_bytes"],
                "eval_bytes": train["eval_bytes"],
                "training_seconds": train["elapsed_seconds"],
                "quantized_payload_bytes": quantized["quantized_payload_bytes"],
                "compression_ratio": quantized["compression_ratio"],
                "bpb": transfer["source"]["bpb"],
                "ppl": transfer["source"]["ppl"],
                "top1_byte_accuracy": transfer["source"]["top1_byte_accuracy"],
                "transfer_ppl_ratio": transfer["ppl_ratio"],
                "max_logit_diff": transfer["max_logit_diff"],
                "predicted_utf8": transfer["generation"]["predicted_utf8"],
            },
        }
    failed_domains = [
        name for name, item in per_domain.items() if item["status"] != "PASS"
    ]
    result = {
        "status": "PASS" if not failed_domains else "FAIL",
        "scope": (
            "Cross-domain portable payload smoke across dialogue, lore, "
            "quest/state, and technical text. This tests workflow robustness "
            "and exact transfer across varied local text styles. It is not "
            "a universal all-corpora dominance claim."
        ),
        "domains": per_domain,
        "failed_domains": failed_domains,
        "required_gates": {
            "four_domains_tested": len(per_domain) == 4,
            "all_domains_pass": not failed_domains,
            "all_transfers_exact": all(
                item["required_gates"]["ppl_ratio_exact"]
                and item["required_gates"]["max_logit_diff_exact"]
                and item["required_gates"]["generation_equal"]
                for item in per_domain.values()
            ),
            "all_quality_smoke_gates_pass": all(
                item["required_gates"]["bpb_under_3"]
                and item["required_gates"]["top1_accuracy_over_50pct"]
                and item["required_gates"]["generation_printable"]
                for item in per_domain.values()
            ),
        },
        "metrics": {
            "mean_bpb": sum(
                item["metrics"]["bpb"] for item in per_domain.values()
            )
            / len(per_domain),
            "min_top1_byte_accuracy": min(
                item["metrics"]["top1_byte_accuracy"] for item in per_domain.values()
            ),
            "max_transfer_logit_diff": max(
                item["metrics"]["max_logit_diff"] for item in per_domain.values()
            ),
        },
        "open_requirements": {
            "large_external_corpora": False,
            "matched_transformer_adapter_per_domain": False,
            "task_level_quality_per_domain": False,
            "multi_seed_per_domain": False,
        },
    }
    output = RESULTS / "cross_domain_smoke_frontier_certificate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
