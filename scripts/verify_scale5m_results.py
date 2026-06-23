from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    core = load("scale5m_seed4242_continued.json")
    brick = load("scale5m_sparse_seed4242_continued.json")
    bpe = load("scale5m_bpe_baseline.json")
    transfer = load("scale5m_to_scale2m_transfer.json")
    quant = load("scale5m_to_scale2m_transfer_int8.json")
    inference = load("scale5m_inference_with_brick.json")
    rows = {row["path"]: row for row in inference["rows"]}

    required = {
        "patch_smaller_than_bpe": (
            core["patch_parameters"] < bpe["parameters"]
        ),
        "patch_smaller_than_byte": (
            core["patch_parameters"] < core["byte_parameters"]
        ),
        "patch_base_faster_than_byte": (
            rows["patch_base"]["bytes_per_second"]
            > rows["byte_base"]["bytes_per_second"]
        ),
        "patch_brick_faster_than_byte": (
            rows["patch_brick"]["bytes_per_second"]
            > rows["byte_base"]["bytes_per_second"]
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
        "cross_size_seed_transfer": transfer["status"] == "PASS",
        "int8_transfer": quant["status"] == "PASS",
    }
    targets = {
        "general_bpb_parity_with_bpe": (
            core["patch_general"]["bpb"] <= bpe["general"]["bpb"]
        )
    }
    failed_required = [key for key, value in required.items() if not value]
    certificate = {
        "status": "PASS" if not failed_required else "FAIL",
        "required_gates": required,
        "research_targets": targets,
        "failed_required": failed_required,
        "metrics": {
            "patch_parameters": core["patch_parameters"],
            "byte_parameters": core["byte_parameters"],
            "bpe_parameters": bpe["parameters"],
            "patch_general_bpb": core["patch_general"]["bpb"],
            "bpe_general_bpb": bpe["general"]["bpb"],
            "patch_base_bytes_per_second": rows["patch_base"]["bytes_per_second"],
            "patch_brick_bytes_per_second": rows["patch_brick"]["bytes_per_second"],
            "byte_base_bytes_per_second": rows["byte_base"]["bytes_per_second"],
            "source_domain_ppl_before": brick["patch_base_domain"]["ppl"],
            "source_domain_ppl_after": brick["patch_target_domain"]["ppl"],
            "cross_size_domain_ratio": transfer["domain_ratio"],
            "cross_size_general_ratio": transfer["general_ratio"],
            "int8_domain_ratio": quant["domain_ratio"],
            "int8_general_ratio": quant["general_ratio"],
        },
        "scope": (
            "5.40M patch core on 20 MB local general data. Architecture gates "
            "pass; matched-update BPE quality parity remains open."
        ),
    }
    path = RESULTS / "scale5m_gate_certificate.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2))
    if failed_required:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
