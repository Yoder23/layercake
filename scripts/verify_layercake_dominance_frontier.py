from __future__ import annotations

import json
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    source = _load("results/northstar_mobile_certificate.json")
    receiver = _load("results/receiver_frontier_certificate.json")
    source_gates = source.get("required_gates", {})
    receiver_gates = receiver.get("required_gates", {})
    cpu_mobile_required = {
        "source_all_required": all(source_gates.values()),
        "receiver_all_required": all(receiver_gates.values()),
        "source_smaller": source["metrics"]["layercake_parameters"] < source["metrics"]["bpe_parameters"],
        "source_better_bpb_all_seeds": all(
            bpb < source["metrics"]["bpe_general_bpb"]
            for bpb in source["metrics"]["layercake_general_bpb"]
        ),
        "source_mean_faster_training": source["metrics"]["layercake_mean_training_seconds"] < source["metrics"]["bpe_training_seconds"],
        "source_faster_cpu_generation_all_seeds": all(
            ratio > 1.0 for ratio in source["metrics"]["cpu_generation_speed_ratio"]
        ),
        "source_coherent_norepeat8": source_gates.get("coherent_cpu_generation_norepeat8", False),
        "source_lossless_transfer_all_seeds": all(
            ratio == 1.0 for ratio in source["metrics"]["migration_ppl_ratio"]
        )
        and all(diff == 0.0 for diff in source["metrics"]["migration_max_logit_diff"]),
        "receiver_smaller": receiver_gates.get("receiver_smaller", False),
        "receiver_better_general_bpb": receiver_gates.get("receiver_better_general_bpb", False),
        "receiver_faster_training": receiver_gates.get("receiver_faster_training", False),
        "receiver_faster_cpu_generation": receiver_gates.get("receiver_faster_cpu_generation", False),
        "receiver_lossless_transfer": receiver_gates.get("lossless_transfer_ppl", False)
        and receiver_gates.get("lossless_transfer_logits", False)
        and receiver_gates.get("lossless_transfer_generation", False),
        "receiver_transferred_domain_beats_transformer": receiver_gates.get("transferred_domain_beats_transformer", False),
    }
    accelerator_gates = {
        "source_gpu_generation_faster_than_bpe": source["metrics"].get("gpu_generation_speed_ratio", 0.0) > 1.0,
    }
    result = {
        "status": "PASS" if all(cpu_mobile_required.values()) else "FAIL",
        "claim_scope": "CPU/mobile-proxy source and receiver dominance under existing locked certificates. This is not an all-accelerator win.",
        "cpu_mobile_required": cpu_mobile_required,
        "failed_cpu_mobile_required": [k for k, v in cpu_mobile_required.items() if not v],
        "accelerator_gates": accelerator_gates,
        "failed_accelerator_gates": [k for k, v in accelerator_gates.items() if not v],
        "source_certificate": "results/northstar_mobile_certificate.json",
        "receiver_certificate": "results/receiver_frontier_certificate.json",
        "metrics": {
            "source_layercake_params": source["metrics"]["layercake_parameters"],
            "source_bpe_params": source["metrics"]["bpe_parameters"],
            "source_layercake_bpb": source["metrics"]["layercake_general_bpb"],
            "source_bpe_bpb": source["metrics"]["bpe_general_bpb"],
            "source_cpu_generation_speed_ratio": source["metrics"]["cpu_generation_speed_ratio"],
            "source_gpu_generation_speed_ratio": source["metrics"].get("gpu_generation_speed_ratio"),
            "receiver_params": receiver["metrics"]["receiver_parameters"],
            "receiver_baseline_params": receiver["metrics"]["baseline_parameters"],
            "receiver_general_bpb": receiver["metrics"]["receiver_general_bpb"],
            "receiver_baseline_bpb": receiver["metrics"]["baseline_general_bpb"],
            "receiver_cpu_generation_speed_ratio": receiver["metrics"]["receiver_cpu_generation_speed_ratio"],
            "receiver_transfer_ppl_ratio": receiver["metrics"]["transfer_ppl_ratio"],
            "receiver_transfer_max_logit_diff": receiver["metrics"]["transfer_max_logit_diff"],
        },
    }
    output = Path("results/dominance/layercake_frontier_cpu_receiver_certificate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
