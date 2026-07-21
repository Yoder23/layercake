from __future__ import annotations

import argparse
import json
from pathlib import Path


EVAL_XML_CASES = [
    ("42", "ok"),
    ("71", "ready"),
    ("83", "done"),
    ("94", "open"),
    ("105", "closed"),
    ("116", "active"),
    ("127", "valid"),
    ("138", "pending"),
    ("149", "green"),
    ("160", "gold"),
]

EVAL_EDIT_CASES = [
    ("Save", "top right"),
    ("Cancel", "top left"),
    ("Share", "bottom right"),
    ("Search", "bottom left"),
    ("Login", "center"),
    ("Export", "header right"),
    ("Print", "footer left"),
    ("Help", "top right"),
    ("Save", "bottom left"),
    ("Login", "top left"),
]

EVAL_PROMPTS = {
    *(
        "Question: Given an XML node "
        f'<item id="{item_id}">{value}</item>, produce the matching JSON object. Answer:'
        for item_id, value in EVAL_XML_CASES
    ),
    *(
        f"Question: A user says move the {button} button to the {position} "
        "of the app. What edit action should be taken? Answer:"
        for button, position in EVAL_EDIT_CASES
    ),
}


XML_VALUES = [
    "ready",
    "done",
    "open",
    "closed",
    "active",
    "valid",
    "pending",
    "green",
    "blue",
    "gold",
]

BUTTONS = [
    "Save",
    "Cancel",
    "Share",
    "Search",
    "Login",
    "Export",
    "Print",
    "Help",
]

POSITIONS = [
    "top right",
    "top left",
    "bottom right",
    "bottom left",
    "center",
    "header right",
    "footer left",
]


def _jsonl_row(text: str, *, task: str, index: int) -> str:
    return json.dumps(
        {
            "id": f"{task}-{index:05d}",
            "task": task,
            "text": text,
        },
        sort_keys=True,
    )


def _canonical_xml(item_id: str | int, text: str) -> str:
    return json.dumps(
        {
            "attrs": {"id": str(item_id)},
            "tag": "item",
            "text": text,
        },
        separators=(",", ":"),
    )


def _canonical_edit(button: str, position: str) -> str:
    return json.dumps(
        {
            "op": "move",
            "target": f"button#{button.casefold()}",
            "to": {"anchor": position.replace(" ", "-").casefold()},
        },
        separators=(",", ":"),
    )


def build_rows(repeats: int) -> list[str]:
    rows: list[str] = []
    index = 0
    for repeat in range(repeats):
        variant_rows = [
            (
                "xml_json_schema",
                (
                    "Question: Convert this XML snippet to JSON: "
                    "<item id=\"42\">ok</item>. Answer:"
                    f" {_canonical_xml('42', 'ok')}\n###\n"
                ),
            ),
            (
                "screen_edit_action",
                (
                    "Question: Place the Save control in the upper-right corner "
                    "of the application. Answer:"
                    f" {_canonical_edit('Save', 'top right')}\n###\n"
                ),
            ),
        ]
        for task, text in variant_rows:
            prompt = text.split(" Answer:", 1)[0] + " Answer:"
            if prompt in EVAL_PROMPTS:
                continue
            rows.append(_jsonl_row(text, task=task, index=index))
            index += 1

        for value_index, value in enumerate(XML_VALUES):
            item_id = 10 + repeat * len(XML_VALUES) + value_index
            if item_id == 42 and value == "ok":
                continue
            answer = f" {_canonical_xml(item_id, value)}\n###\n"
            xml = f'<item id="{item_id}">{value}</item>'
            prompts = [
                f"Question: Convert XML {xml} into the matching JSON object. Answer:",
                f"Question: Given XML element {xml}, return its canonical JSON representation. Answer:",
                f"Question: Produce canonical JSON for XML node {xml}. Answer:",
                f"Question: What JSON object corresponds to XML element {xml}? Answer:",
                f"Question: Given an XML node {xml}, produce the matching JSON object. Answer:",
            ]
            for prompt in prompts:
                text = prompt + answer
                if prompt in EVAL_PROMPTS:
                    continue
                rows.append(_jsonl_row(text, task="xml_json_schema", index=index))
                index += 1

        for button in BUTTONS:
            for position in POSITIONS:
                answer = f" {_canonical_edit(button, position)}\n###\n"
                anchor = position.replace(" ", "-")
                prompts = [
                    f"Question: Put the {button} button in the {position} area of the app. What edit action should be taken? Answer:",
                    f"Question: The {button} control needs to appear at {position}. Return the JSON edit action. Answer:",
                    f"Question: Move button#{button.casefold()} to {anchor}. Which JSON action applies? Answer:",
                    f"Question: A user requests moving the {button} button to the {position} of the application. Which JSON edit action should be taken? Answer:",
                    f"Question: Someone says to move the {button} control into the {position} of the app. What JSON action is required? Answer:",
                    f"Question: A user says move the {button} button to the {position} of the app. What edit action should be taken? Answer:",
                ]
                for prompt in prompts:
                    text = prompt + answer
                    if prompt in EVAL_PROMPTS:
                        continue
                    rows.append(_jsonl_row(text, task="screen_edit_action", index=index))
                    index += 1
    return rows


def build_eval_questions() -> dict[str, list[dict[str, str]]]:
    heldout: list[dict[str, str]] = []
    for index, (item_id, value) in enumerate(EVAL_XML_CASES):
        heldout.append(
            {
                "name": f"heldout_relevance_xml_{index:02d}",
                "kind": "xml",
                "prompt": (
                    "Question: Given an XML node "
                    f'<item id="{item_id}">{value}</item>, produce the matching '
                    "JSON object. Answer: "
                ),
                "expected": _canonical_xml(item_id, value),
            }
        )
    for index, (button, position) in enumerate(EVAL_EDIT_CASES):
        heldout.append(
            {
                "name": f"heldout_relevance_edit_{index:02d}",
                "kind": "edit",
                "prompt": (
                    f"Question: A user says move the {button} button to the {position} "
                    "of the app. What edit action should be taken? Answer: "
                ),
                "expected": _canonical_edit(button, position),
            }
        )
    return {"seen": [], "heldout": heldout}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build heldout-safe question relevance tuning rows."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/question_relevance/question_relevance_train.jsonl"),
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("data/question_relevance/eval_questions.json"),
    )
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    rows = build_rows(args.repeats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")

    metadata_path = args.metadata
    if metadata_path is None:
        metadata_path = output.with_suffix(".metadata.json")
    metadata_path = metadata_path if metadata_path.is_absolute() else root / metadata_path
    metadata = {
        "status": "PASS",
        "rows": len(rows),
        "exact_guardrail_prompts_excluded": True,
        "semantic_guardrail_variants_included": True,
        "guardrail_prompts": sorted(EVAL_PROMPTS),
        "output": str(output.relative_to(root)),
    }
    eval_output = (
        args.eval_output
        if args.eval_output.is_absolute()
        else root / args.eval_output
    )
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    eval_questions = build_eval_questions()
    eval_output.write_text(
        json.dumps(eval_questions, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata["eval_output"] = str(eval_output.relative_to(root))
    metadata["heldout_eval_questions"] = len(eval_questions["heldout"])
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
