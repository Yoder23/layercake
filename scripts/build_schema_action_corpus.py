from __future__ import annotations

import argparse
import json
from pathlib import Path


TAGS = ["item", "button", "field", "panel", "label", "input", "modal", "row"]
TEXTS = ["ok", "save", "cancel", "email", "name", "ready", "active", "hidden"]
ATTRS = ["id", "role", "state", "kind"]
TARGETS = ["button#save", "button#cancel", "field#email", "panel#settings", "modal#help"]
VERBS = ["move", "resize", "hide", "show", "focus", "rename"]
ANCHORS = ["top-right", "top-left", "bottom-right", "bottom-left", "center"]
COLORS = ["red", "blue", "green", "amber", "violet"]


def _json(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def xml_question(tag: str, attr: str, value: str, text: str) -> tuple[str, str]:
    xml = f"<{tag} {attr}=\"{value}\">{text}</{tag}>"
    answer = _json({"attrs": {attr: value}, "tag": tag, "text": text})
    prompt = f"Question: Convert XML node {xml} to canonical JSON. Answer: "
    return prompt, answer


def edit_question(verb: str, target: str, anchor: str, color: str) -> tuple[str, str]:
    if verb == "move":
        request = f"A user says move {target} to the {anchor} of the app."
        answer = {"op": "move", "target": target, "to": {"anchor": anchor}}
    elif verb == "resize":
        request = f"A user says resize {target} to compact."
        answer = {"op": "resize", "target": target, "to": {"size": "compact"}}
    elif verb == "hide":
        request = f"A user says hide {target}."
        answer = {"op": "set_visible", "target": target, "value": False}
    elif verb == "show":
        request = f"A user says show {target}."
        answer = {"op": "set_visible", "target": target, "value": True}
    elif verb == "focus":
        request = f"A user says focus {target}."
        answer = {"op": "focus", "target": target}
    else:
        request = f"A user says rename {target} to {color}."
        answer = {"op": "rename", "target": target, "to": {"text": color}}
    prompt = f"Question: {request} What JSON edit action should be taken? Answer: "
    return prompt, _json(answer)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _balanced_questions(
    rows: list[tuple[int, str, str, str]],
    *,
    prefix: str,
    per_kind: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for kind in ("xml", "edit"):
        count = 0
        for index, row_kind, prompt, answer in rows:
            if row_kind != kind:
                continue
            selected.append(
                {
                    "kind": kind,
                    "name": f"{prefix}_{kind}_{index:04d}",
                    "prompt": prompt,
                    "expected": answer,
                }
            )
            count += 1
            if count >= per_kind:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/schema_action_domain", type=Path)
    parser.add_argument("--repeats", default=18, type=int)
    args = parser.parse_args()

    out_dir = args.out_dir
    train_rows: list[dict[str, str]] = []
    heldout_rows: list[dict[str, str]] = []
    seen_questions: list[dict[str, str]] = []
    heldout_questions: list[dict[str, str]] = []

    xml_examples: list[tuple[str, str, str]] = []
    for tag_index, tag in enumerate(TAGS):
        for attr_index, attr in enumerate(ATTRS):
            for text_index, text in enumerate(TEXTS):
                value = f"{tag_index}{attr_index}{text_index}"
                prompt, answer = xml_question(tag, attr, value, text)
                xml_examples.append(("xml", prompt, answer))

    edit_examples: list[tuple[str, str, str]] = []
    for verb in VERBS:
        for target in TARGETS:
            for anchor in ANCHORS:
                for color in COLORS:
                    prompt, answer = edit_question(verb, target, anchor, color)
                    edit_examples.append(("edit", prompt, answer))

    all_examples = xml_examples + edit_examples
    train_question_pool: list[tuple[int, str, str, str]] = []
    heldout_question_pool: list[tuple[int, str, str, str]] = []
    for index, (kind, prompt, answer) in enumerate(all_examples):
        row = {"text": prompt + answer + "\n###"}
        if index % 11 == 0:
            heldout_rows.append(row)
            heldout_question_pool.append((index, kind, prompt, answer))
        else:
            train_rows.append(row)
            if index % 17 == 1:
                train_question_pool.append((index, kind, prompt, answer))

    seen_questions = _balanced_questions(
        train_question_pool,
        prefix="seen",
        per_kind=8,
    )
    heldout_questions = _balanced_questions(
        heldout_question_pool,
        prefix="heldout",
        per_kind=8,
    )

    expanded_train = []
    for _ in range(max(args.repeats, 1)):
        expanded_train.extend(train_rows)

    write_jsonl(out_dir / "train.jsonl", expanded_train)
    write_jsonl(out_dir / "eval.jsonl", heldout_rows)
    (out_dir / "eval_questions.json").write_text(
        json.dumps(
            {
                "description": "Schema/action questions for realistic generation scoring.",
                "seen": seen_questions,
                "heldout": heldout_questions,
                "train_rows": len(expanded_train),
                "heldout_rows": len(heldout_rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
