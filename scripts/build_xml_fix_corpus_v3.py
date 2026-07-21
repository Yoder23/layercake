from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TAGS = ["screen", "panel", "dialog", "form"]
BUTTONS = ["save", "cancel", "delete", "help", "submit", "back"]
FIELDS = ["email", "name", "age", "status", "phone", "zip"]
KINDS = ["primary", "secondary", "danger"]
TYPES = ["text", "email", "number", "select"]


def canonical(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def xml_doc(
    index: int,
    *,
    tag: str,
    button: str,
    field: str,
    kind: str,
    ftype: str,
) -> str:
    return (
        f'<{tag} id="{tag}-{index:04d}">'
        f'<header><button id="{button}-btn" action="{button}" kind="{kind}" label="{button.title()}"/></header>'
        f'<body><field id="{field}-field" type="{ftype}" required="true" label="{field.title()}"/></body>'
        f'</{tag}>'
    )


def make_case(index: int, op_kind: str, rng: random.Random) -> dict[str, str]:
    tag = rng.choice(TAGS)
    button = rng.choice(BUTTONS)
    field = rng.choice(FIELDS)
    kind = rng.choice(KINDS)
    ftype = rng.choice(TYPES)
    valid = xml_doc(index, tag=tag, button=button, field=field, kind=kind, ftype=ftype)

    if op_kind == "missing_attr":
        xml = valid.replace(f'id="{button}-btn" ', "")
        expected = {
            "op": "set_attr",
            "path": f"/{tag}/header/button",
            "attr": "id",
            "value": f"{button}-btn",
        }
        expectation = "every button must have an id ending in -btn."
    elif op_kind == "wrong_button_action":
        wrong = rng.choice([item for item in BUTTONS if item != button])
        xml = valid.replace(f'action="{button}"', f'action="{wrong}"')
        expected = {
            "op": "set_attr",
            "path": f"/{tag}/header/button[@id='{button}-btn']",
            "attr": "action",
            "value": button,
        }
        expectation = f"button {button}-btn should trigger {button}."
    elif op_kind == "invalid_kind":
        xml = valid.replace(f'kind="{kind}"', 'kind="loud"')
        expected = {
            "op": "set_attr",
            "path": f"/{tag}/header/button[@id='{button}-btn']",
            "attr": "kind",
            "value": kind,
        }
        expectation = f"button {button}-btn should use kind {kind}."
    elif op_kind == "wrong_field_type":
        wrong = rng.choice([item for item in TYPES if item != ftype])
        xml = valid.replace(f'type="{ftype}"', f'type="{wrong}"')
        expected = {
            "op": "set_attr",
            "path": f"/{tag}/body/field[@id='{field}-field']",
            "attr": "type",
            "value": ftype,
        }
        expectation = f"field {field}-field should use type {ftype}."
    elif op_kind == "rename_field_label":
        new_label = f"Customer {field.title()}"
        xml = valid
        expected = {
            "op": "set_attr",
            "path": f"/{tag}/body/field[@id='{field}-field']",
            "attr": "label",
            "value": new_label,
        }
        expectation = f"change the visible field label to {new_label}."
    elif op_kind == "misplaced_node":
        xml = (
            f'<{tag} id="{tag}-{index:04d}">'
            f'<header></header><body>'
            f'<button id="{button}-btn" action="{button}" kind="{kind}" label="{button.title()}"/>'
            f'<field id="{field}-field" type="{ftype}" required="true" label="{field.title()}"/>'
            f'</body></{tag}>'
        )
        expected = {
            "op": "move",
            "path": f"/{tag}/body/button[@id='{button}-btn']",
            "to": f"/{tag}/header",
        }
        expectation = "buttons belong under header before body fields."
    else:
        raise ValueError(f"unknown op kind: {op_kind}")

    prompt = (
        "XML schema: UIFlow.v3\n"
        f"XML: {xml}\n"
        f"Expectation: {expectation}\n"
        "Fix: "
    )
    return {
        "name": f"{op_kind}_{index:05d}",
        "kind": op_kind,
        "prompt": prompt,
        "expected": canonical(expected),
    }


def build_rows(count: int, *, seed: int, start_index: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    kinds = [
        "missing_attr",
        "wrong_button_action",
        "invalid_kind",
        "wrong_field_type",
        "rename_field_label",
        "misplaced_node",
    ]
    rows = []
    for offset in range(count):
        op_kind = kinds[offset % len(kinds)]
        rows.append(make_case(start_index + offset, op_kind, rng))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/xml_fix_domain_v3", type=Path)
    parser.add_argument("--train-count", type=int, default=4096)
    parser.add_argument("--heldout-count", type=int, default=512)
    parser.add_argument("--question-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=72026)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train = build_rows(args.train_count, seed=args.seed, start_index=0)
    heldout = build_rows(args.heldout_count, seed=args.seed + 1, start_index=100000)
    seen_questions = train[: args.question_count]
    heldout_questions = heldout[: args.question_count]

    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in train:
            handle.write(json.dumps({"text": row["prompt"] + row["expected"] + "\n###"}) + "\n")
    with (out_dir / "eval.jsonl").open("w", encoding="utf-8") as handle:
        for row in heldout:
            handle.write(json.dumps({"text": row["prompt"] + row["expected"] + "\n###"}) + "\n")
    (out_dir / "eval_questions.json").write_text(
        json.dumps({"seen": seen_questions, "heldout": heldout_questions}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "schema.json").write_text(
        json.dumps(
            {
                "schema": "UIFlow.v3",
                "allowed_button_kind": KINDS,
                "allowed_field_type": TYPES,
                "operations": [
                    "set_attr:id",
                    "set_attr:action",
                    "set_attr:kind",
                    "set_attr:type",
                    "set_attr:label",
                    "move:button_to_header",
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "heldout": len(heldout),
                "out_dir": str(out_dir),
                "question_count": args.question_count,
                "train": len(train),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
