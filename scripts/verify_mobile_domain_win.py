from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    portable_train = load("portable_python_gru148k.json")
    portable_int8 = load("portable_python_gru148k_int8.json")
    portable_transfer = load("lossless_domain_scale15m_to_5m_int8.json")
    portable_cpu = load("portable_domain_cpu_matched.json")
    portable_gpu = load("portable_domain_gpu.json")
    adapter = load("scale15m_bpe_python_adapter_r16.json")
    adapter_cpu = load("scale15m_bpe_python_adapter_cpu_matched.json")
    adapter_gpu = load("scale15m_bpe_python_adapter_inference.json")
    portable_transfer_seed2 = load(
        "lossless_domain_seed6061_scale15m_to_5m_int8.json"
    )
    adapter_seed2 = load("scale15m_bpe_python_adapter_r16_seed6263.json")
    general = load("scale15m_gate_certificate.json")

    portable_bpb = portable_transfer["source"]["bpb"]
    adapter_bpb = adapter["after"]["domain"]["bpb"]
    gates = {
        "better_domain_bpb": portable_bpb < adapter_bpb,
        "lower_training_wall_time": (
            portable_train["evaluation"]["bpb"] < adapter_bpb
            and portable_train["history"]
            and portable_train.get("status") == "TRAINED"
            and portable_train.get("mode") == "core_independent_lossless"
        ),
        "smaller_deployment_artifact": (
            portable_int8["quantized_payload_bytes"]
            < adapter["artifact_bytes_fp32"]
        ),
        "faster_single_thread_cpu": (
            portable_cpu["forward"]["bytes_per_second"]
            > adapter_cpu["estimated_bytes_per_second"]
        ),
        "exact_cross_host_transfer": (
            portable_transfer["status"] == "PASS"
            and portable_transfer["ppl_ratio"] == 1.0
            and portable_transfer["max_logit_diff"] == 0.0
            and portable_transfer["generation"]["equal"]
        ),
        "second_seed_quality_replication": (
            portable_transfer_seed2["source"]["bpb"]
            < adapter_seed2["after"]["domain"]["bpb"]
            and portable_transfer_seed2["ppl_ratio"] == 1.0
            and portable_transfer_seed2["max_logit_diff"] == 0.0
        ),
    }
    # Training elapsed is stored in the training result; enforce separately for a
    # clear failure message while retaining compatibility with the result schema.
    portable_elapsed = portable_train["elapsed_seconds"]
    adapter_elapsed = adapter["elapsed_seconds"]
    gates["lower_training_wall_time"] = portable_elapsed < adapter_elapsed
    failed = [name for name, passed in gates.items() if not passed]
    certificate = {
        "status": "PASS" if not failed else "FAIL",
        "claim": (
            "LayerCake portable domains beat a matched low-rank tokenizer-"
            "transformer adapter for this mobile CPU/domain-deployment protocol."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "layercake_domain_bpb": portable_bpb,
            "transformer_adapter_domain_bpb": adapter_bpb,
            "layercake_training_seconds": portable_elapsed,
            "transformer_adapter_training_seconds": adapter_elapsed,
            "layercake_artifact_bytes": portable_int8[
                "quantized_payload_bytes"
            ],
            "transformer_adapter_artifact_bytes": adapter[
                "artifact_bytes_fp32"
            ],
            "layercake_cpu_bytes_per_second": portable_cpu["forward"][
                "bytes_per_second"
            ],
            "transformer_adapter_cpu_bytes_per_second": adapter_cpu[
                "estimated_bytes_per_second"
            ],
            "layercake_gpu_bytes_per_second": portable_gpu["forward"][
                "bytes_per_second"
            ],
            "transformer_adapter_gpu_bytes_per_second": adapter_gpu[
                "estimated_bytes_per_second"
            ],
            "transformer_adapter_active_general_bpb_regression": adapter[
                "general_bpb_regression"
            ],
            "layercake_seed2_domain_bpb": portable_transfer_seed2["source"][
                "bpb"
            ],
            "transformer_adapter_seed2_domain_bpb": adapter_seed2["after"][
                "domain"
            ]["bpb"],
        },
        "research_targets": {
            "general_core_bpb_parity": general["research_targets"][
                "general_bpb_parity_with_byte"
            ],
            "gpu_prefill_win": (
                portable_gpu["forward"]["bytes_per_second"]
                > adapter_gpu["estimated_bytes_per_second"]
            ),
            "task_level_code_generation": False,
            "real_mobile_hardware": False,
        },
        "scope": (
            "Single Python domain with two independent adaptation seeds. Isolated "
            "timing is from the first seed. Hardware is a desktop x86 one-thread "
            "CPU proxy and RTX 3080 Laptop GPU. The baseline is a residual rank-16 "
            "adapter, not every possible PEFT method."
        ),
    }
    path = RESULTS / "mobile_domain_win_certificate.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
