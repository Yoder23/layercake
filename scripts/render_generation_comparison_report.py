from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1e-12)


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def _md(text: Any) -> str:
    rendered = str(text if text is not None else "")
    rendered = rendered.replace("\r\n", "\n").replace("\r", "\n")
    rendered = rendered.replace("|", "\\|")
    rendered = rendered.replace("\n", "<br>")
    return rendered


def _sample_key(sample: dict[str, Any]) -> str:
    return str(sample.get("prompt", "")).strip()


def _samples_by_prompt(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {_sample_key(sample): sample for sample in row.get("samples", [])}


def _sample_relevance(sample: dict[str, Any]) -> str:
    if "relevance_pass" in sample:
        return _fmt(bool(sample.get("relevance_pass")))
    return "-"


def _keyword_score(sample: dict[str, Any]) -> str:
    if "keyword_hits" not in sample and "min_keyword_hits" not in sample:
        return "-"
    hits = int(sample.get("keyword_hits", 0))
    required = int(sample.get("min_keyword_hits", 0))
    return f"{hits}/{required}"


def _forbidden_score(sample: dict[str, Any]) -> str:
    if "forbidden_keyword_hits" not in sample:
        return "-"
    return str(int(sample.get("forbidden_keyword_hits", 0)))


def build_report(
    *,
    layercake: dict[str, Any],
    transformer: dict[str, Any],
    title: str,
    certificate: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    lc_metrics = layercake.get("metrics", {})
    tx_metrics = transformer.get("metrics", {})
    lc_bps = float(lc_metrics.get("generation_bytes_per_second", 0.0))
    tx_bps = float(tx_metrics.get("generation_bytes_per_second", 0.0))
    lc_quality = float(lc_metrics.get("quality_score", 0.0))
    tx_quality = float(tx_metrics.get("quality_score", 0.0))
    lc_relevance = lc_metrics.get("relevance_rate")
    tx_relevance = tx_metrics.get("relevance_rate")
    speed_ratio = _ratio(lc_bps, tx_bps)
    quality_ratio = _ratio(lc_quality, tx_quality)
    relevance_ratio = (
        _ratio(float(lc_relevance), float(tx_relevance))
        if lc_relevance is not None and tx_relevance is not None
        else None
    )

    lc_samples = _samples_by_prompt(layercake)
    tx_samples = _samples_by_prompt(transformer)
    prompts = list(lc_samples)
    for prompt in tx_samples:
        if prompt not in lc_samples:
            prompts.append(prompt)

    comparisons: list[dict[str, Any]] = []
    for prompt in prompts:
        lc_sample = lc_samples.get(prompt, {})
        tx_sample = tx_samples.get(prompt, {})
        comparisons.append(
            {
                "prompt": prompt,
                "category": lc_sample.get("category", tx_sample.get("category", "")),
                "layercake": {
                    "text": lc_sample.get("text", ""),
                    "relevance": lc_sample.get("relevance_pass"),
                    "keyword_score": _keyword_score(lc_sample),
                    "forbidden_hits": _forbidden_score(lc_sample),
                    "quality_score": lc_sample.get("quality_score"),
                    "bytes_per_second": lc_sample.get("bytes_per_second"),
                    "max_repeat_8gram": lc_sample.get("max_repeat_8gram"),
                    "runtime_path": lc_sample.get("runtime_path", layercake.get("model_kind")),
                    "hit_keywords": lc_sample.get("hit_keywords", []),
                    "expected_keywords": lc_sample.get(
                        "expected_keywords", tx_sample.get("expected_keywords", [])
                    ),
                },
                "transformer": {
                    "text": tx_sample.get("text", ""),
                    "relevance": tx_sample.get("relevance_pass"),
                    "keyword_score": _keyword_score(tx_sample),
                    "forbidden_hits": _forbidden_score(tx_sample),
                    "quality_score": tx_sample.get("quality_score"),
                    "bytes_per_second": tx_sample.get("bytes_per_second"),
                    "max_repeat_8gram": tx_sample.get("max_repeat_8gram"),
                    "runtime_path": tx_sample.get("runtime_path", transformer.get("model_kind")),
                    "hit_keywords": tx_sample.get("hit_keywords", []),
                },
            }
        )

    summary = {
        "title": title,
        "status": certificate.get("status") if certificate else None,
        "layercake_checkpoint": layercake.get("checkpoint"),
        "transformer_checkpoint": transformer.get("checkpoint"),
        "device": layercake.get("device"),
        "layercake_bps": lc_bps,
        "transformer_bps": tx_bps,
        "speed_ratio": speed_ratio,
        "layercake_quality": lc_quality,
        "transformer_quality": tx_quality,
        "quality_ratio": quality_ratio,
        "layercake_relevance": lc_relevance,
        "transformer_relevance": tx_relevance,
        "relevance_ratio": relevance_ratio,
        "sample_count": len(comparisons),
    }
    result = {"summary": summary, "comparisons": comparisons}

    lines = [
        f"# {title}",
        "",
        "This is a human-review report generated from saved benchmark artifacts. It is intended to make coherence, factual keyword coverage, and failure modes inspectable without reading raw JSON.",
        "",
        "## Summary",
        "",
        "| Metric | LayerCake | BPE transformer | Ratio / status |",
        "| --- | ---: | ---: | ---: |",
        f"| Device | {_md(layercake.get('device', '-'))} | {_md(transformer.get('device', '-'))} | - |",
        f"| Generation throughput | {_fmt(lc_bps)} B/s | {_fmt(tx_bps)} B/s | {_fmt(speed_ratio)}x |",
        f"| Quality heuristic | {_fmt(lc_quality)} | {_fmt(tx_quality)} | {_fmt(quality_ratio)}x |",
    ]
    if lc_relevance is not None or tx_relevance is not None:
        lines.append(
            f"| Relevance rate | {_fmt(lc_relevance)} | {_fmt(tx_relevance)} | {_fmt(relevance_ratio)}x |"
        )
    if certificate:
        lines.append(f"| Certificate status | - | - | {_md(certificate.get('status', '-'))} |")
    lines.extend(
        [
            f"| Samples compared | {len(comparisons)} | {len(comparisons)} | - |",
            "",
            "## Per-prompt review table",
            "",
            "| # | Category | Prompt | LC relevance | BPE relevance | LC keyword hits | BPE keyword hits | LC forbidden | BPE forbidden |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, row in enumerate(comparisons, start=1):
        lc = row["layercake"]
        tx = row["transformer"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _md(row["category"]),
                    _md(row["prompt"]),
                    _sample_relevance({"relevance_pass": lc["relevance"]})
                    if lc["relevance"] is not None
                    else "-",
                    _sample_relevance({"relevance_pass": tx["relevance"]})
                    if tx["relevance"] is not None
                    else "-",
                    _md(lc["keyword_score"]),
                    _md(tx["keyword_score"]),
                    _md(lc["forbidden_hits"]),
                    _md(tx["forbidden_hits"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Full side-by-side generations", ""])
    for index, row in enumerate(comparisons, start=1):
        lc = row["layercake"]
        tx = row["transformer"]
        expected_keywords = lc.get("expected_keywords", [])
        lines.extend(
            [
                f"### {index}. {_md(row['prompt'])}",
                "",
                f"- Category: `{_md(row['category'])}`",
                f"- Expected keywords: {_md(', '.join(expected_keywords) if expected_keywords else '-')}",
                f"- LayerCake runtime path: `{_md(lc.get('runtime_path'))}`",
                f"- BPE runtime path: `{_md(tx.get('runtime_path'))}`",
                "",
                "| Model | Relevance | Keyword hits | Hit keywords | Forbidden hits | Quality | Repeat-8 max | B/s |",
                "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
                f"| LayerCake | {_fmt(lc['relevance'])} | {_md(lc['keyword_score'])} | {_md(', '.join(lc.get('hit_keywords') or []))} | {_md(lc['forbidden_hits'])} | {_fmt(lc['quality_score'])} | {_fmt(lc['max_repeat_8gram'])} | {_fmt(lc['bytes_per_second'])} |",
                f"| BPE transformer | {_fmt(tx['relevance'])} | {_md(tx['keyword_score'])} | {_md(', '.join(tx.get('hit_keywords') or []))} | {_md(tx['forbidden_hits'])} | {_fmt(tx['quality_score'])} | {_fmt(tx['max_repeat_8gram'])} | {_fmt(tx['bytes_per_second'])} |",
                "",
                "**LayerCake generation**",
                "",
                f"> {_md(lc.get('text', ''))}",
                "",
                "**BPE transformer generation**",
                "",
                f"> {_md(tx.get('text', ''))}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n", result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render human-readable side-by-side generation comparison reports."
    )
    parser.add_argument("--layercake-generation", required=True, type=Path)
    parser.add_argument("--transformer-generation", required=True, type=Path)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    certificate = _read(args.certificate) if args.certificate else None
    markdown, result = build_report(
        layercake=_read(args.layercake_generation),
        transformer=_read(args.transformer_generation),
        certificate=certificate,
        title=args.title,
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(str(args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
