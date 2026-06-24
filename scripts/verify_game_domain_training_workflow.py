from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def main() -> int:
    train = load("results/game_dialogue_smoke_gru_train.json")
    quantized = load("results/game_dialogue_smoke_gru_int8.json")
    transfer = load("results/game_dialogue_smoke_gru_lossless_transfer.json")
    predicted = transfer["generation"]["predicted_utf8"]
    gates = {
        "custom_game_domain_file_was_used": bool(train["domain_files"]),
        "training_completed": train["status"] == "TRAINED",
        "training_records_byte_counts": train["train_bytes"] > 0
        and train["eval_bytes"] > 0,
        "int8_quantized_payload_created": quantized["status"] == "PASS"
        and quantized["quantization"] == "int8_symmetric_per_tensor",
        "quantized_payload_smaller_than_fp32": quantized["compression_ratio"] < 0.30,
        "source_receiver_transfer_exact": transfer["status"] == "PASS",
        "transfer_ppl_ratio_exact": transfer["ppl_ratio"] == 1.0,
        "transfer_max_logit_diff_exact": transfer["max_logit_diff"] == 0.0,
        "transfer_generation_exact": transfer["generation"]["equal"],
        "smoke_domain_bpb_under_3": transfer["source"]["bpb"] < 3.0,
        "smoke_domain_top1_accuracy_over_50pct": (
            transfer["source"]["top1_byte_accuracy"] > 0.50
        ),
        "smoke_generation_printable": all(
            char.isprintable() or char in "\n\r\t" for char in predicted
        ),
    }
    result = {
        "status": "PASS" if not failed(gates) else "FAIL",
        "scope": (
            "Game-domain training/deployment workflow smoke. This proves a "
            "custom game-style text file can train a portable domain payload, "
            "quantize it, install it into separate LayerCake runtimes, and "
            "migrate it exactly. It is not a production game-dialogue quality "
            "claim."
        ),
        "required_gates": gates,
        "failed_required": failed(gates),
        "artifacts": {
            "train": "results/game_dialogue_smoke_gru_train.json",
            "quantized": "results/game_dialogue_smoke_gru_int8.json",
            "transfer": "results/game_dialogue_smoke_gru_lossless_transfer.json",
        },
        "metrics": {
            "train_bytes": train["train_bytes"],
            "eval_bytes": train["eval_bytes"],
            "training_seconds": train["elapsed_seconds"],
            "parameters": train["parameters"],
            "quantized_payload_bytes": quantized["quantized_payload_bytes"],
            "compression_ratio": quantized["compression_ratio"],
            "domain_bpb": transfer["source"]["bpb"],
            "domain_ppl": transfer["source"]["ppl"],
            "domain_top1_byte_accuracy": transfer["source"]["top1_byte_accuracy"],
            "transfer_ppl_ratio": transfer["ppl_ratio"],
            "transfer_max_logit_diff": transfer["max_logit_diff"],
            "generation_equal_after_transfer": transfer["generation"]["equal"],
            "predicted_utf8": predicted,
        },
        "open_requirements_for_production_game_domain": {
            "user_game_corpus": False,
            "large_heldout_game_eval": False,
            "npc_dialogue_task_rubric": False,
            "multi_domain_routing_policy": False,
            "human_or_model_judged_dialogue_quality": False,
        },
    }
    output = RESULTS / "game_domain_training_workflow_certificate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
