"""Mutation-based hostile checks for the Phase 2 r3 typed verifier."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layercake.evaluation.phase2_r3_evidence import (  # noqa: E402
    Phase2R3EvidenceError,
    validate_native_benchmark_document,
    validate_quality_document,
)


PHASE = ROOT / "results/moonshot/phase2_recertification"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    quality = read(PHASE / "raw_runs/quality_seeds.json")
    short = read(PHASE / "raw_runs/native_benchmark_128.json")
    checkpoint = quality["records"][0]["checkpoint_sha256"]
    cases = []

    def native_case(name, mutate):
        value = copy.deepcopy(short)
        mutate(value)
        try:
            validate_native_benchmark_document(
                value, output_bytes=128, checkpoint_sha256=checkpoint
            )
        except (Phase2R3EvidenceError, KeyError, TypeError, ValueError):
            cases.append({"name": name, "detected": True})
        else:
            cases.append({"name": name, "detected": False})

    def quality_case(name, mutate):
        value = copy.deepcopy(quality)
        mutate(value)
        try:
            validate_quality_document(ROOT, value)
        except (Phase2R3EvidenceError, KeyError, TypeError, ValueError):
            cases.append({"name": name, "detected": True})
        else:
            cases.append({"name": name, "detected": False})

    native_case("truncated_observation_set", lambda x: x["records"].pop())
    native_case(
        "forged_checkpoint",
        lambda x: x["records"][0].__setitem__("checkpoint_sha256", "0" * 64),
    )
    native_case(
        "forged_output",
        lambda x: x["records"][0].__setitem__("output_hex", "00"),
    )
    native_case(
        "forged_throughput",
        lambda x: x["records"][0].__setitem__("bytes_per_second", 1e9),
    )
    native_case(
        "duplicate_run_id",
        lambda x: x["records"][1].__setitem__(
            "run_id", x["records"][0]["run_id"]
        ),
    )
    native_case(
        "missing_prompt",
        lambda x: [
            row.__setitem__("prompt_id", "one")
            for row in x["records"]
        ],
    )
    native_case(
        "forged_cache",
        lambda x: x["records"][0]["persistent_state"].__setitem__(
            "cached_tokens_per_layer", [0, 0, 0]
        ),
    )
    native_case(
        "planner_injection",
        lambda x: x["records"][0]["external_path_counters"].__setitem__(
            "planner_calls", 1
        ),
    )
    native_case(
        "inactive_cake_call",
        lambda x: x["records"][0]["sparse_execution"].__setitem__(
            "inactive_cake_forward_calls", 1
        ),
    )
    native_case(
        "rss_over_limit",
        lambda x: x["records"][0].__setitem__(
            "process_resident_bytes", 214_990_848
        ),
    )
    native_case(
        "forged_aggregate_ratio",
        lambda x: x["aggregates"].__setitem__(
            "median_throughput_ratio", 99.0
        ),
    )
    native_case(
        "failed_subgate_hidden",
        lambda x: x["aggregates"]["gates"].__setitem__(
            "persistent_cache_exact", False
        ),
    )
    quality_case(
        "missing_seed", lambda x: x["records"].pop()
    )
    quality_case(
        "duplicate_checkpoint",
        lambda x: x["records"][1].__setitem__(
            "checkpoint_sha256", x["records"][0]["checkpoint_sha256"]
        ),
    )
    quality_case(
        "stale_screen_hash",
        lambda x: x["records"][0]["screen"].__setitem__(
            "sha256", "0" * 64
        ),
    )
    quality_case(
        "test_access_concealed",
        lambda x: x["records"][0]["final_quality"].__setitem__(
            "path",
            x["records"][0]["screen"]["path"],
        ),
    )
    result = {
        "format": "layercake-phase2-r3-adversarial-checks/1",
        "status": "PASS" if all(row["detected"] for row in cases) else "FAIL",
        "detected": sum(row["detected"] for row in cases),
        "attempted": len(cases),
        "cases": cases,
    }
    result["evidence_sha256"] = canonical(result)
    write(PHASE / "adversarial_checks.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
