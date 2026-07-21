from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get(row: dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        cur: Any = row
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def _ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def _category_rate(row: dict[str, Any], category: str, metric: str) -> float:
    category_metrics = _get(row, "metrics.category_metrics", default={}) or {}
    return float((category_metrics.get(category) or {}).get(metric, 0.0))


def _generation_metrics(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples", [])
    nonempty = all(bool(str(sample.get("text", "")).strip()) for sample in samples)
    printable = all(float(sample.get("printable_ratio", 0.0)) >= 0.95 for sample in samples)
    alpha = all(float(sample.get("alpha_space_ratio", 0.0)) >= 0.75 for sample in samples)
    no_repeat = all(float(sample.get("max_repeat_8gram", 999.0)) <= 4.0 for sample in samples)
    lexical = all(
        float(sample.get("unique_word_count", 0.0)) >= 8.0
        and float(sample.get("distinct_word_ratio", 0.0)) >= 0.35
        and float(sample.get("one_char_word_ratio", 1.0)) <= 0.35
        and float(sample.get("unique_alpha_char_count", 0.0)) >= 10.0
        for sample in samples
    )
    relevance = all(bool(sample.get("relevance_pass", False)) for sample in samples)
    return {
        "device": str(row.get("device", "unknown")).lower(),
        "generation_bytes_per_second": float(_get(row, "metrics.generation_bytes_per_second", default=0.0)),
        "quality_score": float(_get(row, "metrics.quality_score", default=0.0)),
        "relevance_rate": float(_get(row, "metrics.relevance_rate", default=0.0)),
        "alias_match_rate": float(_get(row, "metrics.alias_match_rate", default=0.0)),
        "exact_relevance_rate": _category_rate(row, "exact", "relevance_rate"),
        "paraphrase_relevance_rate": _category_rate(row, "paraphrase", "relevance_rate"),
        "exact_alias_match_rate": _category_rate(row, "exact", "alias_match_rate"),
        "paraphrase_alias_match_rate": _category_rate(row, "paraphrase", "alias_match_rate"),
        "samples_nonempty": bool(samples) and nonempty,
        "samples_printable": printable,
        "samples_alpha": alpha,
        "samples_no_repeat_8": no_repeat,
        "samples_lexically_diverse": bool(samples) and lexical,
        "samples_relevant": bool(samples) and relevance,
    }


def verify(
    *,
    layercake_cpu_generation: dict[str, Any],
    transformer_cpu_generation: dict[str, Any],
    layercake_gpu_generation: dict[str, Any],
    transformer_gpu_generation: dict[str, Any],
    source_certificate: dict[str, Any] | None,
    transfer_certificate: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    lc_cpu = _generation_metrics(layercake_cpu_generation)
    tx_cpu = _generation_metrics(transformer_cpu_generation)
    lc_gpu = _generation_metrics(layercake_gpu_generation)
    tx_gpu = _generation_metrics(transformer_gpu_generation)
    ratios = {
        "cpu_generation_speed_ratio": _ratio(
            lc_cpu["generation_bytes_per_second"],
            tx_cpu["generation_bytes_per_second"],
        ),
        "gpu_generation_speed_ratio": _ratio(
            lc_gpu["generation_bytes_per_second"],
            tx_gpu["generation_bytes_per_second"],
        ),
        "cpu_quality_ratio": _ratio(lc_cpu["quality_score"], tx_cpu["quality_score"]),
        "gpu_quality_ratio": _ratio(lc_gpu["quality_score"], tx_gpu["quality_score"]),
        "cpu_relevance_ratio": _ratio(lc_cpu["relevance_rate"], tx_cpu["relevance_rate"]),
        "gpu_relevance_ratio": _ratio(lc_gpu["relevance_rate"], tx_gpu["relevance_rate"]),
        "cpu_paraphrase_relevance_ratio": _ratio(
            lc_cpu["paraphrase_relevance_rate"],
            tx_cpu["paraphrase_relevance_rate"],
        ),
        "gpu_paraphrase_relevance_ratio": _ratio(
            lc_gpu["paraphrase_relevance_rate"],
            tx_gpu["paraphrase_relevance_rate"],
        ),
    }
    layercake_sample_gates = {
        "cpu_samples_nonempty": lc_cpu["samples_nonempty"],
        "cpu_samples_printable": lc_cpu["samples_printable"],
        "cpu_samples_alpha": lc_cpu["samples_alpha"],
        "cpu_samples_no_repeat_8": lc_cpu["samples_no_repeat_8"],
        "cpu_samples_lexically_diverse": lc_cpu["samples_lexically_diverse"],
        "cpu_samples_relevant": lc_cpu["samples_relevant"],
        "gpu_samples_nonempty": lc_gpu["samples_nonempty"],
        "gpu_samples_printable": lc_gpu["samples_printable"],
        "gpu_samples_alpha": lc_gpu["samples_alpha"],
        "gpu_samples_no_repeat_8": lc_gpu["samples_no_repeat_8"],
        "gpu_samples_lexically_diverse": lc_gpu["samples_lexically_diverse"],
        "gpu_samples_relevant": lc_gpu["samples_relevant"],
    }
    gates = {
        "source_certificate_pass": (
            True if source_certificate is None else source_certificate.get("status") == "PASS"
        ),
        "transfer_certificate_pass": (
            True if transfer_certificate is None else transfer_certificate.get("status") == "PASS"
        ),
        "cpu_devices": lc_cpu["device"] == "cpu" and tx_cpu["device"] == "cpu",
        "gpu_devices": lc_gpu["device"] == "cuda" and tx_gpu["device"] == "cuda",
        "cpu_generation_5x_met": ratios["cpu_generation_speed_ratio"] >= args.min_cpu_generation_speed_ratio,
        "gpu_generation_noninferior": ratios["gpu_generation_speed_ratio"] >= args.min_gpu_generation_speed_ratio,
        "cpu_quality_noninferior": ratios["cpu_quality_ratio"] >= args.min_quality_ratio,
        "gpu_quality_noninferior": ratios["gpu_quality_ratio"] >= args.min_quality_ratio,
        "cpu_relevance_noninferior": ratios["cpu_relevance_ratio"] >= args.min_relevance_ratio,
        "gpu_relevance_noninferior": ratios["gpu_relevance_ratio"] >= args.min_relevance_ratio,
        "cpu_layercake_exact_relevance_full": lc_cpu["exact_relevance_rate"] >= args.min_layercake_category_relevance,
        "cpu_layercake_paraphrase_relevance_full": lc_cpu["paraphrase_relevance_rate"] >= args.min_layercake_category_relevance,
        "gpu_layercake_exact_relevance_full": lc_gpu["exact_relevance_rate"] >= args.min_layercake_category_relevance,
        "gpu_layercake_paraphrase_relevance_full": lc_gpu["paraphrase_relevance_rate"] >= args.min_layercake_category_relevance,
        "cpu_layercake_paraphrase_alias_full": lc_cpu["paraphrase_alias_match_rate"] >= args.min_layercake_paraphrase_alias_rate,
        "gpu_layercake_paraphrase_alias_full": lc_gpu["paraphrase_alias_match_rate"] >= args.min_layercake_paraphrase_alias_rate,
        **layercake_sample_gates,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": (
            "Instruction-domain exact and paraphrase generation dominance certificate. "
            "This is a domain-runtime gate, not a universal open-domain claim."
        ),
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake_cpu_generation": lc_cpu,
            "transformer_cpu_generation": tx_cpu,
            "layercake_gpu_generation": lc_gpu,
            "transformer_gpu_generation": tx_gpu,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify LayerCake instruction-domain exact/paraphrase dominance"
    )
    parser.add_argument("--layercake-cpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-cpu-generation", required=True, type=Path)
    parser.add_argument("--layercake-gpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-gpu-generation", required=True, type=Path)
    parser.add_argument("--source-certificate", type=Path)
    parser.add_argument("--transfer-certificate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-cpu-generation-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-gpu-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    parser.add_argument("--min-relevance-ratio", type=float, default=1.0)
    parser.add_argument("--min-layercake-category-relevance", type=float, default=1.0)
    parser.add_argument("--min-layercake-paraphrase-alias-rate", type=float, default=1.0)
    args = parser.parse_args()
    result = verify(
        layercake_cpu_generation=_read(args.layercake_cpu_generation),
        transformer_cpu_generation=_read(args.transformer_cpu_generation),
        layercake_gpu_generation=_read(args.layercake_gpu_generation),
        transformer_gpu_generation=_read(args.transformer_gpu_generation),
        source_certificate=_read(args.source_certificate) if args.source_certificate else None,
        transfer_certificate=_read(args.transfer_certificate) if args.transfer_certificate else None,
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
