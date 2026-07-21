from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/breakthrough_equal/tiny_real_train_sweep_smoke.json"
DEFAULT_OUTPUT = ROOT / "results/breakthrough_equal/tiny_real_train_sweep_smoke_integrity.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _get(row: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = row
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def verify(report: dict[str, Any], *, output_path: Path, root: Path = ROOT) -> dict[str, Any]:
    scales = report.get("scales", [])
    split = report.get("data_split", {})
    scale_count = len(scales)
    layercake_training = bool(scales) and all(
        _positive(_get(row, "layercake.train.elapsed_seconds"))
        and _positive(_get(row, "layercake.train.steps_per_second"))
        for row in scales
    )
    transformer_training = bool(scales) and all(
        _positive(_get(row, "baseline.train.elapsed_seconds"))
        and _positive(_get(row, "baseline.train.steps_per_second"))
        for row in scales
    )
    layercake_eval = bool(scales) and all(
        _positive(_get(row, "layercake.general_bpb")) for row in scales
    )
    transformer_eval = bool(scales) and all(
        _positive(_get(row, "baseline.general_bpb")) for row in scales
    )
    raw_samples = bool(scales) and all(
        bool(row.get("qa_samples"))
        and all(
            isinstance(_get(sample, "layercake.text"), str)
            and isinstance(_get(sample, "baseline.text"), str)
            for sample in row.get("qa_samples", [])
        )
        for row in scales
    )
    split_declared = bool(
        split.get("disjoint_by_construction") is True
        and split.get("train_sha256")
        and split.get("eval_sha256")
        and split.get("train_sha256") != split.get("eval_sha256")
    )
    gates = {
        "artifact_has_scales": scale_count > 0,
        "fresh_training_steps_positive": int(report.get("steps", 0) or 0) > 0,
        "train_eval_bytes_positive": _positive(report.get("train_bytes"))
        and _positive(report.get("eval_bytes")),
        "same_train_eval_split_declared": split_declared,
        "layercake_training_observed": layercake_training,
        "transformer_training_observed": transformer_training,
        "layercake_eval_observed": layercake_eval,
        "transformer_eval_observed": transformer_eval,
        "raw_generation_samples_present": raw_samples,
        "dominance_result_retained": report.get("status") in {"PASS", "FAIL"}
        and isinstance(report.get("summary_gates"), dict),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "Evidence-hygiene certificate for the tiny real paired train/eval smoke. "
            "PASS means both LayerCake and BPE transformer training/eval machinery "
            "ran and retained raw failures. It does not prove the breakthrough claim."
        ),
        "claim_boundary": (
            "The source benchmark may FAIL domination gates; this certificate only "
            "proves the sweep can execute fresh paired training and preserve results."
        ),
        "source_status": report.get("status", "UNKNOWN"),
        "source_scope": report.get("scope"),
        "device": report.get("device"),
        "scale_count": scale_count,
        "gates": gates,
        "failed_required": failed,
        "data_split": split,
        "summary_gates": report.get("summary_gates", {}),
        "output": str(
            output_path.relative_to(root)
            if output_path.is_relative_to(root)
            else output_path
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify tiny real paired LayerCake/BPE train-eval smoke integrity."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    result = verify(_read(input_path), output_path=args.output, root=ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
