from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    bpe = load("bpe_bytefallback_2048_benchmark.json")
    brick = load("sparse_brick_continuous2028_r16_p2.json")
    seed = load("final_transfer_seed314.json")
    size = load("final_transfer_large2718.json")
    quant = load("final_transfer_seed314_int8.json")
    inference = load("final_inference_benchmark.json")

    rows = {row["path"]: row for row in inference["rows"]}
    gates = {
        "general_bpb_vs_bpe": (
            brick["patch_base_general"]["bpb"] <= bpe["general"]["bpb"]
        ),
        "core_smaller_than_bpe": (
            inference["patch_parameters"] < bpe["parameters"]
        ),
        "core_smaller_than_byte": (
            inference["patch_parameters"] < inference["byte_parameters"]
        ),
        "active_brick_faster_than_byte_base": (
            rows["patch_brick"]["bytes_per_second"]
            >= rows["byte_base"]["bytes_per_second"]
        ),
        "source_domain_improves": (
            brick["patch_target_domain"]["ppl"]
            < brick["patch_base_domain"]["ppl"]
        ),
        "source_general_bounded": (
            brick["patch_target_general"]["ppl"]
            / brick["patch_base_general"]["ppl"]
            <= 1.05
        ),
        "cross_seed_bounded": seed["status"] == "PASS",
        "cross_size_bounded": size["status"] == "PASS",
        "int8_bounded": quant["status"] == "PASS",
        "top_k_sparse": (
            brick["brick_config"]["top_k"] < brick["brick_config"]["num_experts"]
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    certificate = {
        "status": "PASS" if not failed else "FAIL",
        "gates": gates,
        "failed": failed,
        "metrics": {
            "byte_patch_general_bpb": brick["patch_base_general"]["bpb"],
            "bpe_general_bpb": bpe["general"]["bpb"],
            "byte_patch_parameters": inference["patch_parameters"],
            "bpe_parameters": bpe["parameters"],
            "byte_parameters": inference["byte_parameters"],
            "patch_brick_bytes_per_second": rows["patch_brick"]["bytes_per_second"],
            "byte_base_bytes_per_second": rows["byte_base"]["bytes_per_second"],
            "source_domain_ppl_before": brick["patch_base_domain"]["ppl"],
            "source_domain_ppl_after": brick["patch_target_domain"]["ppl"],
            "cross_seed_domain_ratio": seed["domain_ratio"],
            "cross_seed_general_ratio": seed["general_ratio"],
            "cross_size_domain_ratio": size["domain_ratio"],
            "cross_size_general_ratio": size["general_ratio"],
            "int8_domain_ratio": quant["domain_ratio"],
            "int8_general_ratio": quant["general_ratio"],
        },
        "scope": (
            "Small-scale local-corpus evidence. This certificate does not prove "
            "large-scale or universal tokenizer dominance."
        ),
    }
    output = RESULTS / "research_gate_certificate.json"
    output.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
