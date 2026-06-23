from __future__ import annotations

import json
from pathlib import Path

import _common


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    names = [
        "lossless_domain_small.json",
        "lossless_domain_scale5m_to_2m.json",
        "lossless_domain_scale15m_to_5m.json",
        "lossless_domain_scale15m_to_5m_int8.json",
    ]
    checked = {}
    for name in names:
        result = json.loads((root / "results" / name).read_text(encoding="utf-8"))
        passed = (
            result["status"] == "PASS"
            and result["contract"]["unchanged_decoder_payload"]
            and not result["contract"]["core_logits_used"]
            and result["max_logit_diff"] == 0.0
            and result["ppl_ratio"] == 1.0
            and result["source"]["ppl"] == result["target"]["ppl"]
            and result["generation"]["equal"]
        )
        if not passed:
            raise SystemExit(f"strict lossless gate failed: {name}")
        checked[name] = {
            "ppl": result["source"]["ppl"],
            "top1_byte_accuracy": result["source"]["top1_byte_accuracy"],
            "ratio": result["ppl_ratio"],
            "max_logit_diff": result["max_logit_diff"],
        }
    fp32 = checked["lossless_domain_scale15m_to_5m.json"]
    int8 = checked["lossless_domain_scale15m_to_5m_int8.json"]
    int8["ppl_degradation_ratio_vs_fp32"] = int8["ppl"] / fp32["ppl"]
    if int8["ppl_degradation_ratio_vs_fp32"] > 1.01:
        raise SystemExit("int8 portable-domain PPL degradation exceeds 1%")
    print(json.dumps({"status": "PASS", "results": checked}, indent=2))


if __name__ == "__main__":
    main()
