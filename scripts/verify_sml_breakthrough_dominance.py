from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OG = ROOT.parent / "layercakeogwithdecoder"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _failed(gates: dict[str, bool]) -> list[str]:
    return [name for name, passed in gates.items() if not passed]


def _fair_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("name") == name:
            return row
    return None


def _tier_from_fair(row: dict[str, Any] | None, tier: str) -> dict[str, Any]:
    if row is None:
        gates = {
            "artifact_exists": False,
            "quality_advantage": False,
            "training_speed_advantage": False,
            "parameter_match_within_1pct": False,
            "hellaswag_advantage_or_missing": False,
        }
        return {
            "tier": tier,
            "status": "FAIL",
            "gates": gates,
            "failed": _failed(gates),
            "metrics": {},
        }

    lc_hs = row.get("lc_hellaswag")
    bl_hs = row.get("bl_hellaswag")
    hs_gate = True
    if isinstance(lc_hs, (int, float)) and isinstance(bl_hs, (int, float)):
        hs_gate = lc_hs > bl_hs

    gates = {
        "artifact_exists": True,
        "quality_advantage": float(row.get("c4_advantage_pct", -999.0)) > 0.0,
        "training_speed_advantage": float(row.get("bl_time_s", 0.0)) > float(row.get("lc_time_s", 1e18)),
        "parameter_match_within_1pct": float(row.get("param_diff_pct", 999.0)) <= 1.0,
        "hellaswag_advantage_or_missing": hs_gate,
    }
    return {
        "tier": tier,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "failed": _failed(gates),
        "metrics": {
            "lc_c4_ppl": row.get("lc_c4_ppl"),
            "bl_c4_ppl": row.get("bl_c4_ppl"),
            "c4_advantage_pct": row.get("c4_advantage_pct"),
            "lc_time_s": row.get("lc_time_s"),
            "bl_time_s": row.get("bl_time_s"),
            "param_diff_pct": row.get("param_diff_pct"),
            "lc_hellaswag": row.get("lc_hellaswag"),
            "bl_hellaswag": row.get("bl_hellaswag"),
        },
    }


def main() -> int:
    small = _load(RESULTS / "dominance_up_to_25m_research_certificate.json")
    transfer = _load(RESULTS / "transfer_matrix_v2.json")
    fair = _load(OG / "fair_scaling_results" / "fair_scaling_results.json")

    fair_rows: list[dict[str, Any]] = fair if isinstance(fair, list) else []

    small_status = bool(small and small.get("status") == "PASS")
    medium_tier = _tier_from_fair(_fair_row(fair_rows, "48M"), "medium_48m")
    large_150 = _tier_from_fair(_fair_row(fair_rows, "150M"), "large_150m")
    large_350 = _tier_from_fair(_fair_row(fair_rows, "350M"), "large_350m")

    transfer_rows = transfer.get("rows", []) if transfer else []
    transfer_gate = bool(transfer and transfer.get("summary", {}).get("all_exact", False))
    transfer_large_hint = any(
        "scale26m" in str(row.get("artifact", "")) for row in transfer_rows
    )

    gates = {
        "small_strict_dominance": small_status,
        "medium_48m_dominance": medium_tier["status"] == "PASS",
        "large_150m_dominance": large_150["status"] == "PASS",
        "large_350m_dominance": large_350["status"] == "PASS",
        "transfer_exactness_matrix": transfer_gate,
        "transfer_has_large_scale_evidence": transfer_large_hint,
    }

    status = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "status": status,
        "scope": "Small/Medium/Large breakthrough dominance gate over matched transformers.",
        "gates": gates,
        "failed": _failed(gates),
        "tiers": {
            "small": {
                "status": "PASS" if small_status else "FAIL",
                "artifact": "results/dominance_up_to_25m_research_certificate.json",
                "failed": small.get("failed", []) if small else ["artifact_missing"],
            },
            "medium_48m": medium_tier,
            "large_150m": large_150,
            "large_350m": large_350,
        },
        "transfer": {
            "status": "PASS" if transfer_gate else "FAIL",
            "artifact": "results/transfer_matrix_v2.json",
            "all_exact": transfer.get("summary", {}).get("all_exact") if transfer else None,
            "artifact_count": transfer.get("summary", {}).get("artifact_count") if transfer else 0,
            "has_large_scale_evidence": transfer_large_hint,
        },
        "next_commands": [
            "python scripts/run_groundbreaking_research_loop.py --max-iterations 10 --execute-repairs",
            "python ../layercakeogwithdecoder/FAIR_SCALING_EXPERIMENT.py --scale 48M --steps 20000",
            "python ../layercakeogwithdecoder/FAIR_SCALING_EXPERIMENT.py --scale 150M --steps 20000",
            "python ../layercakeogwithdecoder/FAIR_SCALING_EXPERIMENT.py --scale 350M --steps 20000",
        ],
    }

    out = RESULTS / "sml_breakthrough_certificate.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
