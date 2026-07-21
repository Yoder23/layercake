from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(eval_data: dict[str, Any], split: str) -> dict[str, Any]:
    summary = eval_data["splits"][split]["summary"]
    return {
        "layercake_exact": summary["layercake"]["exact_json_accuracy"],
        "layercake_parseable": summary["layercake"]["parseable_json_rate"],
        "layercake_similarity": summary["layercake"]["mean_char_similarity"],
        "transformer_exact": summary["transformer"]["exact_json_accuracy"],
        "transformer_parseable": summary["transformer"]["parseable_json_rate"],
        "transformer_similarity": summary["transformer"]["mean_char_similarity"],
        "speed_ratio": summary["mean_speed_ratio_layercake_over_transformer"],
        "layercake_latency_ms": summary["layercake"]["mean_latency_per_answer_seconds"] * 1000.0,
        "transformer_latency_ms": summary["transformer"]["mean_latency_per_answer_seconds"] * 1000.0,
    }


def _passes(cpu: dict[str, Any], gpu: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for device_name, data in [("cpu", cpu), ("gpu", gpu)]:
        if data.get("benchmark_mode") != "fair_neural":
            failures.append(f"{device_name}: benchmark_mode is not fair_neural")
        if data.get("layercake_structured_schema_head"):
            failures.append(f"{device_name}: structured schema head enabled")
        if data.get("layercake_direct_domain_cache"):
            failures.append(f"{device_name}: direct domain cache enabled")
        for split in ["seen", "heldout"]:
            summary = _summary(data, split)
            if summary["speed_ratio"] < 5.0:
                failures.append(f"{device_name}/{split}: speed ratio below 5x")
            if summary["layercake_exact"] < summary["transformer_exact"]:
                failures.append(f"{device_name}/{split}: LayerCake exact accuracy below transformer")
            if summary["layercake_parseable"] < summary["transformer_parseable"]:
                failures.append(f"{device_name}/{split}: LayerCake parseable rate below transformer")
    return not failures, failures


def _render_examples(eval_data: dict[str, Any], out_path: Path, *, limit: int) -> None:
    lines: list[str] = [
        "# XML Fix Generation Examples",
        "",
        f"Source eval: `{out_path.name}` companion raw JSON is referenced in the proof report.",
        "",
    ]
    for split in ["seen", "heldout"]:
        lines.extend([f"## {split.title()}", ""])
        for sample in eval_data["splits"][split]["samples"][:limit]:
            lines.extend(
                [
                    f"### {sample['name']}",
                    "",
                    "**Prompt**",
                    "```text",
                    sample["prompt"],
                    "```",
                    "",
                    "**Expected**",
                    "```json",
                    sample["expected"],
                    "```",
                    "",
                    f"**LayerCake** exact={sample['layercake']['exact_json_match']} parseable={sample['layercake']['parseable_json']} latency_ms={sample['layercake']['latency_per_answer_seconds'] * 1000.0:.3f}",
                    "```json",
                    sample["layercake"]["raw_text"],
                    "```",
                    "",
                    f"**Transformer** exact={sample['transformer']['exact_json_match']} parseable={sample['transformer']['parseable_json']} latency_ms={sample['transformer']['latency_per_answer_seconds'] * 1000.0:.3f}",
                    "```json",
                    sample["transformer"]["raw_text"].strip(),
                    "```",
                    "",
                ]
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=8)
    args = parser.parse_args()

    cpu = _load(args.cpu)
    gpu = _load(args.gpu)
    passed, failures = _passes(cpu, gpu)
    report = {
        "status": "PASS" if passed else "FAIL",
        "task": "xml_schema_fix_suggestion",
        "mode": "fair_neural",
        "candidate": "xml_fix_layercake_span128_nocopy_3m_seq512_step9000",
        "baseline": "xml_fix_bpe_1p3m",
        "artifacts": {
            "cpu_eval": str(args.cpu),
            "gpu_eval": str(args.gpu),
            "examples": str(args.examples),
            "layercake_checkpoint": "runs_experiment/xml_fix_layercake_span128_nocopy_3m_seq512/latest.pt",
            "transformer_checkpoint": "runs_experiment/xml_fix_bpe_1p3m/latest.pt",
        },
        "gates": {
            "cpu_seen": _summary(cpu, "seen"),
            "cpu_heldout": _summary(cpu, "heldout"),
            "gpu_seen": _summary(gpu, "seen"),
            "gpu_heldout": _summary(gpu, "heldout"),
        },
        "failures": failures,
        "short_verdict": (
            "Fast and strong on seen XML fixes, but not reliable on heldout copy/path fixes."
            if failures
            else "LayerCake passed speed and quality gates."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _render_examples(cpu, args.examples, limit=args.example_limit)


if __name__ == "__main__":
    main()
