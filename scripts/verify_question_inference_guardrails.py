from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUESTION_RELEVANCE_TERMS = {
    "xml_json_schema": ("item", "id", "42", "ok"),
    "screen_edit_action": ("save", "top", "right", "button"),
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _sample_relevance(name: str, text: str) -> float:
    terms = QUESTION_RELEVANCE_TERMS.get(name)
    if not terms:
        return 1.0
    normalized = text.lower()
    hits = sum(1 for term in terms if term in normalized)
    return hits / len(terms)


def _metrics(row: dict[str, Any]) -> dict[str, float]:
    samples = row.get("samples", [])
    lc = [sample["layercake"] for sample in samples]
    tx = [sample["transformer"] for sample in samples]
    return {
        "speed_ratio": float(row["summary"]["mean_speed_ratio_layercake_over_transformer"]),
        "layercake_printable": _mean([float(item["printable_ratio"]) for item in lc]),
        "transformer_printable": _mean([float(item["printable_ratio"]) for item in tx]),
        "layercake_distinct_trigram": _mean(
            [float(item["distinct_word_trigram"]) for item in lc]
        ),
        "transformer_distinct_trigram": _mean(
            [float(item["distinct_word_trigram"]) for item in tx]
        ),
        "layercake_max_repeat_8gram": _mean(
            [float(item["max_repeat_8gram"]) for item in lc]
        ),
        "transformer_max_repeat_8gram": _mean(
            [float(item["max_repeat_8gram"]) for item in tx]
        ),
        "layercake_relevance": _mean(
            [
                _sample_relevance(
                    str(sample.get("name", "")),
                    str(sample["layercake"].get("text", "")),
                )
                for sample in samples
            ]
        ),
    }


def verify(
    *,
    cpu: dict[str, Any],
    gpu: dict[str, Any],
    min_cpu_speed_ratio: float,
    min_gpu_speed_ratio: float,
    min_printable: float,
) -> dict[str, Any]:
    cpu_metrics = _metrics(cpu)
    gpu_metrics = _metrics(gpu)
    gates = {
        "cpu_speed_ratio_met": cpu_metrics["speed_ratio"] >= min_cpu_speed_ratio,
        "gpu_speed_ratio_met": gpu_metrics["speed_ratio"] >= min_gpu_speed_ratio,
        "cpu_printable_met": cpu_metrics["layercake_printable"] >= min_printable,
        "gpu_printable_met": gpu_metrics["layercake_printable"] >= min_printable,
        "cpu_distinct_trigram_noninferior": (
            cpu_metrics["layercake_distinct_trigram"]
            >= cpu_metrics["transformer_distinct_trigram"]
        ),
        "gpu_distinct_trigram_noninferior": (
            gpu_metrics["layercake_distinct_trigram"]
            >= gpu_metrics["transformer_distinct_trigram"]
        ),
        "cpu_repetition_no_worse": (
            cpu_metrics["layercake_max_repeat_8gram"]
            <= cpu_metrics["transformer_max_repeat_8gram"]
        ),
        "gpu_repetition_no_worse": (
            gpu_metrics["layercake_max_repeat_8gram"]
            <= gpu_metrics["transformer_max_repeat_8gram"]
        ),
        "cpu_question_relevance_met": cpu_metrics["layercake_relevance"] >= 0.75,
        "gpu_question_relevance_met": gpu_metrics["layercake_relevance"] >= 0.75,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "status": "PASS" if not blockers else "FAIL",
        "scope": (
            "Question-prompt inference guardrail for the current LayerCake "
            "architecture. CPU and GPU comparison artifacts must use the same "
            "trained models and same prompts; quality gates are simple "
            "printability, diversity, and repetition diagnostics. Named "
            "question prompts also require minimal answer relevance."
        ),
        "blockers": blockers,
        "gates": gates,
        "metrics": {
            "cpu": cpu_metrics,
            "gpu": gpu_metrics,
        },
        "thresholds": {
            "min_cpu_speed_ratio": min_cpu_speed_ratio,
            "min_gpu_speed_ratio": min_gpu_speed_ratio,
            "min_printable": min_printable,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-comparison", required=True, type=Path)
    parser.add_argument("--gpu-comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-cpu-speed-ratio", type=float, default=3.0)
    parser.add_argument("--min-gpu-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-printable", type=float, default=0.95)
    args = parser.parse_args()
    result = verify(
        cpu=_read(args.cpu_comparison),
        gpu=_read(args.gpu_comparison),
        min_cpu_speed_ratio=args.min_cpu_speed_ratio,
        min_gpu_speed_ratio=args.min_gpu_speed_ratio,
        min_printable=args.min_printable,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
