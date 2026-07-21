from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/breakthrough_equal/local_transformer_runtime_comparison.json"
REQUIRED_SPEED_RATIO = 5.0
REQUIRED_RUNTIME_FIELDS = [
    "model",
    "runtime",
    "runtime_version",
    "quantization",
    "prompt_set",
    "hardware",
    "cpu_threads",
    "gpu_settings",
]


def _read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_value(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    return value is not None and value != "" and value != []


def build_comparison(
    evidence: dict[str, Any] | None,
    *,
    output_path: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    evidence = evidence or {}
    metadata = evidence.get("runtime_metadata", {})
    ratios = evidence.get("ratios", {})
    quality = evidence.get("quality", {})
    artifacts = evidence.get("artifacts", {})

    cpu_ratio = _as_float(ratios.get("cpu_generation_speed_ratio"))
    gpu_ratio = _as_float(ratios.get("gpu_generation_speed_ratio"))
    gates = {
        "evidence_present": bool(evidence),
        "pinned_runtime_metadata": all(
            _has_value(metadata, field) for field in REQUIRED_RUNTIME_FIELDS
        ),
        "same_prompt_pack": bool(evidence.get("same_prompt_pack")),
        "raw_generations_present": bool(artifacts.get("layercake_raw_generations"))
        and bool(artifacts.get("transformer_raw_generations")),
        "cpu_generation_5x": cpu_ratio >= REQUIRED_SPEED_RATIO,
        "gpu_generation_5x": gpu_ratio >= REQUIRED_SPEED_RATIO,
        "quality_noninferior": bool(quality.get("noninferior_or_better")),
    }
    failed = [name for name, passed in gates.items() if not passed]
    status = "PASS" if not failed else ("OPEN" if not evidence else "FAIL")
    return {
        "status": status,
        "scope": (
            "Pinned practical local-transformer runtime comparison. This is the "
            "product-runtime track, not the same-data fair-neural proof. PASS "
            "requires exact runtime metadata, identical prompt packs, raw "
            "generation artifacts, noninferior quality, and 5x CPU/GPU generation."
        ),
        "claim_boundary": (
            "A missing or weak local runtime comparison remains a blocker for "
            "public product claims even when matched-transformer science gates pass."
        ),
        "required_speed_ratio": REQUIRED_SPEED_RATIO,
        "required_runtime_fields": REQUIRED_RUNTIME_FIELDS,
        "gates": gates,
        "failed_required": failed,
        "ratios": {
            "cpu_generation_speed_ratio": cpu_ratio,
            "gpu_generation_speed_ratio": gpu_ratio,
        },
        "runtime_metadata": metadata,
        "quality": quality,
        "artifacts": artifacts,
        "same_prompt_pack": bool(evidence.get("same_prompt_pack")),
        "output": str(
            output_path.relative_to(root)
            if output_path.is_relative_to(root)
            else output_path
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a pinned local transformer runtime comparison artifact."
    )
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evidence_path = (
        args.evidence
        if args.evidence is None or args.evidence.is_absolute()
        else ROOT / args.evidence
    )
    result = build_comparison(
        _read_optional_json(evidence_path),
        output_path=args.output,
        root=ROOT,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
