from __future__ import annotations

import argparse
from pathlib import Path
import json

from _common import emit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/transfer_matrix.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    transfer_artifacts = sorted(
        list(results_dir.glob("lossless_domain*.json"))
        + list(results_dir.glob("northstar_lossless_domain*.json"))
        + list(results_dir.glob("scale*m_lossless_domain*.json"))
    )

    rows = []
    for artifact in transfer_artifacts:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        source = payload.get("source", {})
        target = payload.get("target", {})
        source_bpb = source.get("bpb")
        target_bpb = target.get("bpb")
        ppl_ratio = payload.get("ppl_ratio")
        max_logit_diff = payload.get("max_logit_diff")
        generation_equal = bool(payload.get("generation", {}).get("equal", False))
        no_damage = (
            isinstance(source_bpb, (int, float))
            and isinstance(target_bpb, (int, float))
            and abs(float(source_bpb) - float(target_bpb)) <= 1e-12
        )
        exact = (
            ppl_ratio == 1.0
            and max_logit_diff == 0.0
            and generation_equal
            and no_damage
        )
        rows.append(
            {
                "source_size": "unknown",
                "source_seed": payload.get("source_seed"),
                "source_input_mode": "bytes",
                "target_size": "unknown",
                "target_seed": payload.get("target_seed"),
                "target_input_mode": "bytes",
                "abi_version": payload.get("contract", {}).get("abi_version", "lc-abi/2"),
                "brick_type": "portable_domain_decoder",
                "domain_ppl_source": source.get("ppl"),
                "domain_ppl_target": target.get("ppl"),
                "degradation_ratio": ppl_ratio,
                "abi_drift": max_logit_diff,
                "status": "PASS" if exact else "FAIL",
                "artifact": str(artifact.relative_to(root)).replace("\\", "/"),
                "generation_equal": generation_equal,
                "source_bpb": source_bpb,
                "target_bpb": target_bpb,
                "no_damage": no_damage,
            }
        )

    summary = {
        "schema_version": 2,
        "rows": rows,
        "summary": {
            "artifact_count": len(rows),
            "pass_count": sum(1 for row in rows if row["status"] == "PASS"),
            "fail_count": sum(1 for row in rows if row["status"] == "FAIL"),
            "all_exact": all(row["status"] == "PASS" for row in rows) if rows else False,
        },
    }
    emit(summary, args.output)


if __name__ == "__main__":
    main()
