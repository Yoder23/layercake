from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "breakthrough_equal"


def main() -> None:
    artifacts = {
        "cpu": RESULTS
        / "schema_action_span64_parallel_copy_3m_syntaxft2_parallel_oneshot_cpu.json",
        "gpu": RESULTS
        / "schema_action_span64_parallel_copy_3m_syntaxft2_parallel_oneshot_gpu.json",
    }
    lines = [
        "# Schema/Action Fair Neural Generation Examples",
        "",
        "Candidate: span64_parallel_copy_3m_syntaxft2_oneshot",
        "Mode: fair_neural, span_parallel_oneshot, no structured schema head, no direct cache, no domain cache",
        "",
    ]
    for device, path in artifacts.items():
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        lines += [f"## {device.upper()} Summary", ""]
        for split, split_doc in doc["splits"].items():
            summary = split_doc["summary"]
            layercake = summary["layercake"]
            transformer = summary["transformer"]
            ratio = summary["mean_speed_ratio_layercake_over_transformer"]
            lines.append(
                f"- {split}: speed {ratio:.3f}x; "
                f"LayerCake exact {layercake['exact_json_accuracy']:.4f} vs "
                f"transformer {transformer['exact_json_accuracy']:.4f}; "
                f"parse {layercake['parseable_json_rate']:.4f} vs "
                f"{transformer['parseable_json_rate']:.4f}; "
                f"similarity {layercake['mean_char_similarity']:.4f} vs "
                f"{transformer['mean_char_similarity']:.4f}"
            )
        lines.append("")
        for split, split_doc in doc["splits"].items():
            lines += [f"## {device.upper()} {split} Examples", ""]
            for index, row in enumerate(split_doc["samples"], 1):
                layercake = row["layercake"]
                transformer = row["transformer"]
                lines += [
                    f"### {index}. {row['name']}",
                    "",
                    f"Prompt: `{row['prompt']}`",
                    "",
                    "Expected:",
                    "```json",
                    row["expected"],
                    "```",
                    "",
                    (
                        "LayerCake raw "
                        f"({layercake['seconds']:.6f}s, "
                        f"{layercake['bytes_per_second']:.1f} B/s, "
                        f"exact={layercake['exact_json_match']}, "
                        f"parseable={layercake['parseable_json']}):"
                    ),
                    "```text",
                    layercake["raw_text"],
                    "```",
                    "",
                    (
                        "Transformer raw "
                        f"({transformer['seconds']:.6f}s, "
                        f"{transformer['bytes_per_second']:.1f} B/s, "
                        f"exact={transformer['exact_json_match']}, "
                        f"parseable={transformer['parseable_json']}):"
                    ),
                    "```text",
                    transformer["raw_text"],
                    "```",
                    "",
                ]
    output = RESULTS / "schema_action_fair_neural_generation_examples.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
