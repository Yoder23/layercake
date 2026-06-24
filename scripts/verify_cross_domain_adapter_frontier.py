from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DOMAINS = {
    "game_dialogue": {
        "train": "results/game_dialogue_smoke_gru_100_train.json",
        "quantized": "results/game_dialogue_smoke_gru_100_int8.json",
        "transfer": "results/game_dialogue_smoke_gru_100_lossless_transfer.json",
        "adapter": "results/bpe_adapter_game_dialogue_smoke.json",
    },
    "game_lore": {
        "train": "results/game_lore_smoke_gru_train.json",
        "quantized": "results/game_lore_smoke_gru_int8.json",
        "transfer": "results/game_lore_smoke_gru_lossless_transfer.json",
        "adapter": "results/bpe_adapter_game_lore_smoke.json",
    },
    "game_quest_state": {
        "train": "results/game_quest_state_smoke_gru_train.json",
        "quantized": "results/game_quest_state_smoke_gru_int8.json",
        "transfer": "results/game_quest_state_smoke_gru_lossless_transfer.json",
        "adapter": "results/bpe_adapter_game_quest_state_smoke.json",
    },
    "technical_text": {
        "train": "results/technical_text_smoke_gru_lr1e3_300_train.json",
        "quantized": "results/technical_text_smoke_gru_lr1e3_300_int8.json",
        "transfer": "results/technical_text_smoke_gru_lr1e3_300_lossless_transfer.json",
        "adapter": "results/bpe_adapter_technical_text_smoke.json",
    },
}


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> int:
    per_domain = {}
    for name, paths in DOMAINS.items():
        train = load(paths["train"])
        quantized = load(paths["quantized"])
        transfer = load(paths["transfer"])
        adapter = load(paths["adapter"])
        lc_bpb = transfer["source"]["bpb"]
        adapter_bpb = adapter["after"]["domain"]["bpb"]
        gates = {
            "layercake_lower_domain_bpb": lc_bpb < adapter_bpb,
            "layercake_faster_domain_training": (
                train["elapsed_seconds"] < adapter["elapsed_seconds"]
            ),
            "layercake_smaller_payload": (
                quantized["quantized_payload_bytes"] < adapter["artifact_bytes_fp32"]
            ),
            "layercake_transfer_exact": (
                transfer["status"] == "PASS"
                and transfer["ppl_ratio"] == 1.0
                and transfer["max_logit_diff"] == 0.0
                and transfer["generation"]["equal"]
            ),
            "adapter_trained": adapter["status"] == "TRAINED",
            "adapter_had_same_domain_file": bool(adapter["domain_files"]),
        }
        per_domain[name] = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "required_gates": gates,
            "failed": [gate for gate, passed in gates.items() if not passed],
            "metrics": {
                "layercake_domain_bpb": lc_bpb,
                "adapter_domain_bpb": adapter_bpb,
                "domain_bpb_delta": lc_bpb - adapter_bpb,
                "layercake_training_seconds": train["elapsed_seconds"],
                "adapter_training_seconds": adapter["elapsed_seconds"],
                "training_speed_ratio_adapter_over_layercake": (
                    adapter["elapsed_seconds"] / train["elapsed_seconds"]
                ),
                "layercake_payload_bytes": quantized["quantized_payload_bytes"],
                "adapter_payload_bytes": adapter["artifact_bytes_fp32"],
                "payload_ratio": (
                    quantized["quantized_payload_bytes"]
                    / adapter["artifact_bytes_fp32"]
                ),
                "transfer_ppl_ratio": transfer["ppl_ratio"],
                "transfer_max_logit_diff": transfer["max_logit_diff"],
            },
        }
    failed_domains = [
        name for name, item in per_domain.items() if item["status"] != "PASS"
    ]
    result = {
        "status": "PASS" if not failed_domains else "FAIL",
        "scope": (
            "Cross-domain smoke comparison against matched BPE residual adapters. "
            "Each LayerCake portable payload must beat the adapter on domain BPB, "
            "training seconds, payload size, and exact source/receiver transfer. "
            "This is a small-fixture smoke gate, not a large external-corpus claim."
        ),
        "domains": per_domain,
        "failed_domains": failed_domains,
        "required_gates": {
            "four_domains_tested": len(per_domain) == 4,
            "all_domains_beat_adapter": not failed_domains,
        },
        "metrics": {
            "max_domain_bpb_delta": max(
                item["metrics"]["domain_bpb_delta"] for item in per_domain.values()
            ),
            "min_training_speed_ratio_adapter_over_layercake": min(
                item["metrics"]["training_speed_ratio_adapter_over_layercake"]
                for item in per_domain.values()
            ),
            "max_payload_ratio": max(
                item["metrics"]["payload_ratio"] for item in per_domain.values()
            ),
        },
        "open_requirements": {
            "large_external_corpora": False,
            "multi_seed_per_domain": False,
            "task_level_quality_per_domain": False,
            "real_mobile_adapter_runtime_per_domain": False,
        },
    }
    output = RESULTS / "cross_domain_adapter_frontier_certificate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
