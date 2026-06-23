from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    core = load("scale15m_seed6250_5000.json")
    inference = load("scale15m_inference.json")
    transfer = load("lossless_domain_scale15m_to_5m_int8.json")
    external = load("lossless_domain_external_python_int8.json")
    rows = {row["path"]: row for row in inference["rows"]}
    gates = {
        "patch_smaller_than_byte": (
            core["patch_parameters"] < core["byte_parameters"]
        ),
        "patch_faster_than_byte": (
            rows["patch_base"]["bytes_per_second"]
            > rows["byte_base"]["bytes_per_second"]
        ),
        "repository_transfer_exact": (
            transfer["status"] == "PASS"
            and transfer["ppl_ratio"] == 1.0
            and transfer["max_logit_diff"] == 0.0
            and transfer["generation"]["equal"]
        ),
        "external_transfer_exact": (
            external["status"] == "PASS"
            and external["ppl_ratio"] == 1.0
            and external["max_logit_diff"] == 0.0
            and external["generation"]["equal"]
        ),
        "external_domain_prediction_useful": (
            external["source"]["ppl"] <= 6.0
            and external["source"]["top1_byte_accuracy"] >= 0.55
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    certificate = {
        "status": "PASS" if not failed else "FAIL",
        "required_gates": gates,
        "failed_required": failed,
        "research_targets": {
            "general_bpb_parity_with_byte": (
                core["patch_general"]["bpb"] <= core["byte_general"]["bpb"]
            ),
            "task_level_code_generation": False,
            "native_mobile_int8_execution": False,
        },
        "metrics": {
            "total_paired_steps": core["total_paired_steps"],
            "patch_parameters": core["patch_parameters"],
            "byte_parameters": core["byte_parameters"],
            "patch_general_bpb": core["patch_general"]["bpb"],
            "byte_general_bpb": core["byte_general"]["bpb"],
            "patch_bytes_per_second": rows["patch_base"]["bytes_per_second"],
            "byte_bytes_per_second": rows["byte_base"]["bytes_per_second"],
            "repository_domain_ppl": transfer["source"]["ppl"],
            "external_domain_ppl": external["source"]["ppl"],
            "external_top1_byte_accuracy": external["source"][
                "top1_byte_accuracy"
            ],
        },
        "scope": (
            "Single-seed 20 MB local-corpus checkpoint and x86/CUDA measurements. "
            "The external Python result is filesystem-disjoint, not a contamination "
            "audit. Mobile hardware and task-level coding remain open."
        ),
    }
    path = RESULTS / "scale15m_gate_certificate.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
