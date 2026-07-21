from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(doc: dict[str, Any], split: str) -> dict[str, float]:
    summary = doc["splits"][split]["summary"]
    return {
        "runtime_exact": float(summary["runtime_exact"]),
        "runtime_parseable": float(summary["runtime_parseable"]),
        "runtime_similarity": float(summary["runtime_similarity"]),
        "runtime_latency_ms": float(summary["runtime_latency_ms"]),
        "transformer_exact": float(summary["transformer_exact"]),
        "transformer_parseable": float(summary["transformer_parseable"]),
        "transformer_similarity": float(summary["transformer_similarity"]),
        "transformer_latency_ms": float(summary["transformer_latency_ms"]),
        "speed_ratio": float(summary["speed_ratio"]),
    }


def _gate(cpu: dict[str, Any], gpu: dict[str, Any]) -> tuple[str, dict[str, bool], list[str]]:
    gates: dict[str, bool] = {}
    failures: list[str] = []
    for device, doc in [("cpu", cpu), ("gpu", gpu)]:
        gates[f"{device}_mode_structured_tool"] = doc.get("mode") == "structured_tool"
        for split in ["seen", "heldout"]:
            summary = _summary(doc, split)
            gates[f"{device}_{split}_5x_speed"] = summary["speed_ratio"] >= 5.0
            gates[f"{device}_{split}_exact_noninferior"] = (
                summary["runtime_exact"] >= summary["transformer_exact"]
            )
            gates[f"{device}_{split}_parse_noninferior"] = (
                summary["runtime_parseable"] >= summary["transformer_parseable"]
            )
            gates[f"{device}_{split}_perfect_runtime_exact"] = summary["runtime_exact"] == 1.0
    for name, passed in gates.items():
        if not passed:
            failures.append(name)
    return ("PASS" if not failures else "FAIL"), gates, failures


def _render_examples(cpu: dict[str, Any], out_path: Path, *, limit: int) -> None:
    lines: list[str] = [
        "# XML Structured Transducer Examples",
        "",
        "Mode: `structured_tool`. This is a product/runtime XML transduction proof, not the fair-neural byte-generation dominance claim.",
        "",
    ]
    for split in ["seen", "heldout"]:
        lines.extend([f"## {split.title()}", ""])
        for sample in cpu["splits"][split]["samples"][:limit]:
            lines.extend(
                [
                    f"### {sample['name']}",
                    "",
                    "```text",
                    sample["prompt"],
                    "```",
                    "",
                    "**Expected**",
                    "```json",
                    sample["expected"],
                    "```",
                    "",
                    f"**Structured Runtime** exact={sample['runtime']['exact_json_match']} latency_ms={sample['runtime']['latency_ms']:.6f}",
                    "```json",
                    sample["runtime"]["raw_text"],
                    "```",
                    "",
                    f"**Transformer** exact={sample['transformer']['exact_json_match']} latency_ms={sample['transformer']['latency_ms']:.3f}",
                    "```json",
                    sample["transformer"]["raw_text"],
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
    status, gates, failures = _gate(cpu, gpu)
    report = {
        "status": status,
        "task": "xml_schema_fix_transduction",
        "mode": "structured_tool",
        "claim_scope": "product_runtime_xml_transduction_not_fair_neural_generation",
        "candidate": "xml_fix_structured_transducer_runtime",
        "baseline": "trained_bpe_transformer_xml_fix_1p3m",
        "artifacts": {
            "cpu_eval": str(args.cpu),
            "gpu_eval": str(args.gpu),
            "examples": str(args.examples),
        },
        "gates": gates,
        "failures": failures,
        "summaries": {
            "cpu_seen": _summary(cpu, "seen"),
            "cpu_heldout": _summary(cpu, "heldout"),
            "gpu_seen": _summary(gpu, "seen"),
            "gpu_heldout": _summary(gpu, "heldout"),
        },
        "verdict": (
            "Structured XML transduction runtime beats the trained transformer on exact quality and speed."
            if status == "PASS"
            else "Structured XML transduction runtime did not pass all gates."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _render_examples(cpu, args.examples, limit=args.example_limit)


if __name__ == "__main__":
    main()
