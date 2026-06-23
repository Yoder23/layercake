from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    bpe = load("scale15m_bpe_matched.json")
    bpe_prefill = load("scale15m_bpe_matched_inference_batch1.json")
    layer1 = load("scale15m_northstar_arpatch2_seed6250.json")
    layer2 = load("scale15m_northstar_arpatch2_seed6263.json")
    layer_prefill = load("scale15m_northstar_arpatch2_inference_batch1.json")
    generation1 = load(
        "scale15m_northstar_arpatch2_cached_generation_quality_exact.json"
    )
    generation2 = load(
        "scale15m_northstar_arpatch2_seed6263_cached_generation_quality.json"
    )
    speed1 = load(
        "scale15m_northstar_arpatch2_stateful_cached_generation_cpu1_exact.json"
    )
    speed2 = load(
        "scale15m_northstar_arpatch2_seed6263_generation_cpu1.json"
    )
    gpu_speed = load(
        "scale15m_northstar_arpatch2_stateful_cached_generation_gpu_exact.json"
    )
    transfer1 = load("northstar_lossless_domain_scale15m_to_5m_int8.json")
    transfer2 = load("northstar_lossless_domain_seed6263_to_5m_int8.json")
    domain_train = load("portable_python_gru148k.json")
    domain_quantized = load("portable_python_gru148k_int8.json")
    domain_cpu = load("northstar_portable_domain_cpu1.json")
    adapter1 = load("scale15m_bpe_python_adapter_r16.json")
    adapter2 = load("scale15m_bpe_python_adapter_r16_seed6263.json")
    adapter_cpu = load("northstar_bpe_python_adapter_cpu1.json")

    patch_row = next(
        row for row in layer_prefill["rows"] if row["path"] == "patch_base"
    )
    layer_prefill_ms = patch_row["seconds"] * 1000 / layer_prefill["iterations"]
    mean_layer_training = (
        layer1["elapsed_seconds"] + layer2["elapsed_seconds"]
    ) / 2
    gates = {
        "core_quality_seed6250": (
            layer1["general"]["bpb"] < bpe["general"]["bpb"]
        ),
        "core_quality_seed6263": (
            layer2["general"]["bpb"] < bpe["general"]["bpb"]
        ),
        "mean_training_time": mean_layer_training < bpe["elapsed_seconds"],
        "equal_or_lower_training_bytes": (
            layer1["estimated_total_training_bytes"]
            <= bpe["estimated_total_training_bytes"]
            and layer2["estimated_total_training_bytes"]
            <= bpe["estimated_total_training_bytes"]
        ),
        "smaller_model": layer1["parameters"] < bpe["parameters"],
        "faster_batch1_prefill": (
            layer_prefill_ms < bpe_prefill["median_ms"]
        ),
        "cached_generation_quality_seed6250": (
            generation1["status"] == "PASS"
        ),
        "cached_generation_quality_seed6263": (
            generation2["status"] == "PASS"
        ),
        "faster_cpu_generation_seed6250": (
            speed1["speed_ratio"] > 1.0
        ),
        "faster_cpu_generation_seed6263": (
            speed2["speed_ratio"] > 1.0
        ),
        "lossless_migration_seed6250": (
            transfer1["status"] == "PASS"
            and transfer1["max_logit_diff"] == 0.0
            and transfer1["ppl_ratio"] == 1.0
            and transfer1["generation"]["equal"]
        ),
        "lossless_migration_seed6263": (
            transfer2["status"] == "PASS"
            and transfer2["max_logit_diff"] == 0.0
            and transfer2["ppl_ratio"] == 1.0
            and transfer2["generation"]["equal"]
        ),
        "migrated_domain_quality_seed6250": (
            transfer1["target"]["bpb"] < adapter1["after"]["domain"]["bpb"]
        ),
        "migrated_domain_quality_seed6263": (
            transfer2["target"]["bpb"] < adapter2["after"]["domain"]["bpb"]
        ),
        "lower_domain_training_time": (
            domain_train["elapsed_seconds"] < adapter1["elapsed_seconds"]
        ),
        "smaller_domain_payload": (
            domain_quantized["quantized_payload_bytes"]
            < adapter1["artifact_bytes_fp32"]
        ),
        "faster_domain_cpu": (
            domain_cpu["forward"]["bytes_per_second"]
            > adapter_cpu["estimated_bytes_per_second"]
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    certificate = {
        "status": "PASS" if not failed else "FAIL",
        "claim": (
            "At ~15M parameters, LayerCake beats the matched tokenizer "
            "transformer on heldout BPB, mean fixed-budget training time, "
            "batch-1 prefill latency, and one-thread cached generation; an "
            "unchanged portable Python domain payload then migrates exactly "
            "into an independent ~5M LayerCake host and beats the matched "
            "transformer domain adapter."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "bpe_parameters": bpe["parameters"],
            "layercake_parameters": layer1["parameters"],
            "bpe_general_bpb": bpe["general"]["bpb"],
            "layercake_general_bpb": [
                layer1["general"]["bpb"],
                layer2["general"]["bpb"],
            ],
            "bpe_training_seconds": bpe["elapsed_seconds"],
            "layercake_training_seconds": [
                layer1["elapsed_seconds"],
                layer2["elapsed_seconds"],
            ],
            "layercake_mean_training_seconds": mean_layer_training,
            "bpe_batch1_prefill_ms": bpe_prefill["median_ms"],
            "layercake_batch1_prefill_ms": layer_prefill_ms,
            "cached_generation_bpb": [
                generation1["bpb"],
                generation2["bpb"],
            ],
            "cpu_generation_speed_ratio": [
                speed1["speed_ratio"],
                speed2["speed_ratio"],
            ],
            "gpu_generation_speed_ratio": gpu_speed["speed_ratio"],
            "migrated_domain_bpb": [
                transfer1["target"]["bpb"],
                transfer2["target"]["bpb"],
            ],
            "transformer_adapter_domain_bpb": [
                adapter1["after"]["domain"]["bpb"],
                adapter2["after"]["domain"]["bpb"],
            ],
            "migration_max_logit_diff": [
                transfer1["max_logit_diff"],
                transfer2["max_logit_diff"],
            ],
            "migration_ppl_ratio": [
                transfer1["ppl_ratio"],
                transfer2["ppl_ratio"],
            ],
            "domain_training_seconds": domain_train["elapsed_seconds"],
            "adapter_training_seconds": adapter1["elapsed_seconds"],
            "domain_payload_bytes": domain_quantized[
                "quantized_payload_bytes"
            ],
            "adapter_payload_bytes": adapter1["artifact_bytes_fp32"],
            "domain_cpu_bytes_per_second": domain_cpu["forward"][
                "bytes_per_second"
            ],
            "adapter_cpu_bytes_per_second": adapter_cpu[
                "estimated_bytes_per_second"
            ],
        },
        "limitations": {
            "gpu_generation_win": False,
            "gpu_generation_speed_ratio": gpu_speed["speed_ratio"],
            "real_phone_measurement": False,
            "domains_tested": ["python"],
            "scale": "approximately 15M source and 5M target parameters",
        },
    }
    path = RESULTS / "northstar_mobile_certificate.json"
    path.write_text(
        json.dumps(certificate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(certificate, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
