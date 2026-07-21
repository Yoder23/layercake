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


def _training_metrics(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"train_seconds": 0.0, "train_bytes": 0, "params": 0}
    return {
        "params": int(_get(row, "latest.trainable_params", "trainable_params", "params", default=0)),
        "train_seconds": float(
            _get(row, "latest.elapsed_seconds", "elapsed_seconds", "train_seconds", default=0.0)
        ),
        "train_bytes": int(_get(row, "latest.train_bytes", "train_bytes", default=0)),
    }


def _category_rate(row: dict[str, Any], category: str, metric: str) -> float:
    category_metrics = _get(row, "metrics.category_metrics", default={}) or {}
    return float((category_metrics.get(category) or {}).get(metric, 0.0))


def _generation_metrics(row: dict[str, Any], categories: list[str]) -> dict[str, Any]:
    samples = row.get("samples", [])
    nonempty = all(bool(str(sample.get("text", "")).strip()) for sample in samples)
    printable = all(float(sample.get("printable_ratio", 0.0)) >= 0.95 for sample in samples)
    alpha = all(float(sample.get("alpha_space_ratio", 0.0)) >= 0.75 for sample in samples)
    no_repeat = all(float(sample.get("max_repeat_8gram", 999.0)) <= 4.0 for sample in samples)
    no_forbidden = all(int(sample.get("forbidden_keyword_hits", 0)) == 0 for sample in samples)
    abstention_required = [
        sample for sample in samples if bool(sample.get("expect_abstain", False))
    ]
    abstentions_pass = all(
        bool(sample.get("abstention_pass", False)) for sample in abstention_required
    )
    memory_eligible = [
        sample for sample in samples if not bool(sample.get("expect_abstain", False))
    ]
    memory_match_rate_non_abstain = (
        sum(
            1
            for sample in memory_eligible
            if str(sample.get("runtime_path", "")) == "portable_corpus_memory"
        )
        / max(len(memory_eligible), 1)
    )
    lexical = all(
        float(sample.get("unique_word_count", 0.0)) >= 8.0
        and float(sample.get("distinct_word_ratio", 0.0)) >= 0.35
        and float(sample.get("one_char_word_ratio", 1.0)) <= 0.35
        and float(sample.get("unique_alpha_char_count", 0.0)) >= 10.0
        for sample in samples
    )
    relevance = all(bool(sample.get("relevance_pass", False)) for sample in samples)
    category_relevance = {
        category: _category_rate(row, category, "relevance_rate")
        for category in categories
    }
    category_memory = {
        category: _category_rate(row, category, "portable_memory_match_rate")
        for category in categories
    }
    raw_memory_match_rate = float(
        _get(row, "metrics.portable_memory_match_rate", default=0.0)
    )
    effective_memory_match_rate = (
        memory_match_rate_non_abstain if abstention_required else raw_memory_match_rate
    )
    return {
        "device": str(row.get("device", "unknown")).lower(),
        "generation_bytes_per_second": float(
            _get(row, "metrics.generation_bytes_per_second", default=0.0)
        ),
        "quality_score": float(_get(row, "metrics.quality_score", default=0.0)),
        "relevance_rate": float(_get(row, "metrics.relevance_rate", default=0.0)),
        "portable_memory_match_rate": raw_memory_match_rate,
        "portable_memory_match_rate_non_abstain": memory_match_rate_non_abstain,
        "portable_memory_match_rate_effective": effective_memory_match_rate,
        "abstention_rate": float(_get(row, "metrics.abstention_rate", default=0.0)),
        "abstention_required_count": len(abstention_required),
        "domain_setup_seconds": float(
            _get(row, "metrics.domain_setup_seconds", default=0.0)
        ),
        "category_relevance": category_relevance,
        "category_portable_memory_match": category_memory,
        "samples_nonempty": bool(samples) and nonempty,
        "samples_printable": printable,
        "samples_alpha": alpha,
        "samples_no_repeat_8": no_repeat,
        "samples_no_forbidden": bool(samples) and no_forbidden,
        "samples_abstentions_pass": (
            True if not abstention_required else abstentions_pass
        ),
        "samples_lexically_diverse": bool(samples) and lexical,
        "samples_relevant": bool(samples) and relevance,
    }


def verify(
    *,
    layercake_cpu_generation: dict[str, Any],
    transformer_cpu_generation: dict[str, Any],
    layercake_gpu_generation: dict[str, Any],
    transformer_gpu_generation: dict[str, Any],
    transformer_training: dict[str, Any] | None,
    source_certificate: dict[str, Any] | None,
    transfer_certificate: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    categories = [item.strip() for item in args.required_categories.split(",") if item.strip()]
    lc_cpu = _generation_metrics(layercake_cpu_generation, categories)
    tx_cpu = _generation_metrics(transformer_cpu_generation, categories)
    lc_gpu = _generation_metrics(layercake_gpu_generation, categories)
    tx_gpu = _generation_metrics(transformer_gpu_generation, categories)
    tx_train = _training_metrics(transformer_training)
    lc_domain_setup_seconds = max(
        lc_cpu["domain_setup_seconds"],
        lc_gpu["domain_setup_seconds"],
    )
    ratios = {
        "domain_setup_speed_ratio": _ratio(tx_train["train_seconds"], lc_domain_setup_seconds),
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
    }
    category_gates: dict[str, bool] = {}
    for category in categories:
        category_gates[f"cpu_{category}_relevance_full"] = (
            lc_cpu["category_relevance"][category] >= args.min_layercake_category_relevance
        )
        category_gates[f"gpu_{category}_relevance_full"] = (
            lc_gpu["category_relevance"][category] >= args.min_layercake_category_relevance
        )
        category_gates[f"cpu_{category}_portable_memory_full"] = (
            lc_cpu["category_portable_memory_match"][category]
            >= args.min_layercake_category_memory_match
        )
        category_gates[f"gpu_{category}_portable_memory_full"] = (
            lc_gpu["category_portable_memory_match"][category]
            >= args.min_layercake_category_memory_match
        )
    gates = {
        "source_certificate_pass": (
            True if source_certificate is None else source_certificate.get("status") == "PASS"
        ),
        "transfer_certificate_pass": (
            True if transfer_certificate is None else transfer_certificate.get("status") == "PASS"
        ),
        "cpu_devices": lc_cpu["device"] == "cpu" and tx_cpu["device"] == "cpu",
        "gpu_devices": lc_gpu["device"] == "cuda" and tx_gpu["device"] == "cuda",
        "domain_setup_faster_than_transformer_training": (
            ratios["domain_setup_speed_ratio"] >= args.min_domain_setup_speed_ratio
            if tx_train["train_seconds"] > 0
            else True
        ),
        "cpu_generation_5x_met": ratios["cpu_generation_speed_ratio"] >= args.min_cpu_generation_speed_ratio,
        "gpu_generation_noninferior": ratios["gpu_generation_speed_ratio"] >= args.min_gpu_generation_speed_ratio,
        "cpu_quality_noninferior": ratios["cpu_quality_ratio"] >= args.min_quality_ratio,
        "gpu_quality_noninferior": ratios["gpu_quality_ratio"] >= args.min_quality_ratio,
        "cpu_relevance_noninferior": ratios["cpu_relevance_ratio"] >= args.min_relevance_ratio,
        "gpu_relevance_noninferior": ratios["gpu_relevance_ratio"] >= args.min_relevance_ratio,
        "cpu_portable_memory_full": (
            lc_cpu["portable_memory_match_rate_effective"] >= args.min_layercake_memory_match
        ),
        "gpu_portable_memory_full": (
            lc_gpu["portable_memory_match_rate_effective"] >= args.min_layercake_memory_match
        ),
        "cpu_abstention_rate_met": (
            lc_cpu["abstention_rate"] >= args.min_layercake_abstention_rate
            if lc_cpu["abstention_required_count"] > 0
            else True
        ),
        "gpu_abstention_rate_met": (
            lc_gpu["abstention_rate"] >= args.min_layercake_abstention_rate
            if lc_gpu["abstention_required_count"] > 0
            else True
        ),
        "cpu_samples_nonempty": lc_cpu["samples_nonempty"],
        "cpu_samples_printable": lc_cpu["samples_printable"],
        "cpu_samples_alpha": lc_cpu["samples_alpha"],
        "cpu_samples_no_repeat_8": lc_cpu["samples_no_repeat_8"],
        "cpu_samples_no_forbidden": lc_cpu["samples_no_forbidden"],
        "cpu_samples_abstentions_pass": lc_cpu["samples_abstentions_pass"],
        "cpu_samples_lexically_diverse": lc_cpu["samples_lexically_diverse"],
        "cpu_samples_relevant": lc_cpu["samples_relevant"],
        "gpu_samples_nonempty": lc_gpu["samples_nonempty"],
        "gpu_samples_printable": lc_gpu["samples_printable"],
        "gpu_samples_alpha": lc_gpu["samples_alpha"],
        "gpu_samples_no_repeat_8": lc_gpu["samples_no_repeat_8"],
        "gpu_samples_no_forbidden": lc_gpu["samples_no_forbidden"],
        "gpu_samples_abstentions_pass": lc_gpu["samples_abstentions_pass"],
        "gpu_samples_lexically_diverse": lc_gpu["samples_lexically_diverse"],
        "gpu_samples_relevant": lc_gpu["samples_relevant"],
        **category_gates,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": (
            "Portable mixed-domain app/game/website CPU-mobile dominance certificate. "
            "This proves a corpus-memory domain-runtime gate, not universal open-domain modeling."
        ),
        "gates": gates,
        "ratios": ratios,
        "metrics": {
            "layercake_cpu_generation": lc_cpu,
            "transformer_cpu_generation": tx_cpu,
            "layercake_gpu_generation": lc_gpu,
            "transformer_gpu_generation": tx_gpu,
            "transformer_training": tx_train,
            "layercake_domain_setup_seconds": lc_domain_setup_seconds,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify portable mixed-domain dominance")
    parser.add_argument("--layercake-cpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-cpu-generation", required=True, type=Path)
    parser.add_argument("--layercake-gpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-gpu-generation", required=True, type=Path)
    parser.add_argument("--transformer-training", type=Path)
    parser.add_argument("--source-certificate", type=Path)
    parser.add_argument("--transfer-certificate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--required-categories", default="app,website,game")
    parser.add_argument("--min-domain-setup-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-cpu-generation-speed-ratio", type=float, default=5.0)
    parser.add_argument("--min-gpu-generation-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-quality-ratio", type=float, default=1.0)
    parser.add_argument("--min-relevance-ratio", type=float, default=1.0)
    parser.add_argument("--min-layercake-memory-match", type=float, default=1.0)
    parser.add_argument("--min-layercake-abstention-rate", type=float, default=0.0)
    parser.add_argument("--min-layercake-category-relevance", type=float, default=1.0)
    parser.add_argument("--min-layercake-category-memory-match", type=float, default=1.0)
    args = parser.parse_args()
    result = verify(
        layercake_cpu_generation=_read(args.layercake_cpu_generation),
        transformer_cpu_generation=_read(args.transformer_cpu_generation),
        layercake_gpu_generation=_read(args.layercake_gpu_generation),
        transformer_gpu_generation=_read(args.transformer_gpu_generation),
        transformer_training=_read(args.transformer_training) if args.transformer_training else None,
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
