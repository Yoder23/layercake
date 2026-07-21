from __future__ import annotations

import argparse
import json
from pathlib import Path


TAGS = ["screen", "panel", "dialog", "form"]
BUTTONS = ["save", "cancel", "delete", "help"]
FIELDS = ["email", "name", "age", "status"]
KINDS = ["primary", "secondary", "danger"]
TYPES = ["text", "email", "number", "select"]


def canonical(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def xml_doc(index: int, *, tag: str, button: str, field: str, kind: str, ftype: str) -> str:
    return (
        f'<{tag} id="{tag}-{index:03d}">'
        f'<header><button id="{button}-btn" action="{button}" kind="{kind}" label="{button.title()}"/></header>'
        f'<body><field id="{field}-field" type="{ftype}" required="true" label="{field.title()}"/></body>'
        f'</{tag}>'
    )


def cases() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = 0
    for tag in TAGS:
        for button in BUTTONS:
            for field in FIELDS:
                kind = KINDS[index % len(KINDS)]
                ftype = TYPES[index % len(TYPES)]
                valid = xml_doc(index, tag=tag, button=button, field=field, kind=kind, ftype=ftype)
                without_button_id = valid.replace(f'id="{button}-btn" ', "")
                wrong_action = valid.replace(f'action="{button}"', 'action="submit"')
                invalid_kind = valid.replace(f'kind="{kind}"', 'kind="loud"')
                wrong_type = valid.replace(f'type="{ftype}"', 'type="blob"')
                misplaced = (
                    f'<{tag} id="{tag}-{index:03d}">'
                    f'<header></header><body>'
                    f'<button id="{button}-btn" action="{button}" kind="{kind}" label="{button.title()}"/>'
                    f'<field id="{field}-field" type="{ftype}" required="true" label="{field.title()}"/>'
                    f'</body></{tag}>'
                )
                rows.extend(
                    [
                        {
                            "name": f"missing_button_id_{index:04d}",
                            "kind": "missing_attr",
                            "prompt": (
                                "XML schema: UIFlow.v1\n"
                                f"XML: {without_button_id}\n"
                                "Expectation: every button must have an id ending in -btn.\n"
                                "Fix: "
                            ),
                            "expected": canonical(
                                {
                                    "op": "set_attr",
                                    "path": f"/{tag}/header/button",
                                    "attr": "id",
                                    "value": f"{button}-btn",
                                }
                            ),
                        },
                        {
                            "name": f"wrong_button_action_{index:04d}",
                            "kind": "wrong_attr",
                            "prompt": (
                                "XML schema: UIFlow.v1\n"
                                f"XML: {wrong_action}\n"
                                f"Expectation: button {button}-btn should trigger {button}.\n"
                                "Fix: "
                            ),
                            "expected": canonical(
                                {
                                    "op": "set_attr",
                                    "path": f"/{tag}/header/button[@id='{button}-btn']",
                                    "attr": "action",
                                    "value": button,
                                }
                            ),
                        },
                        {
                            "name": f"invalid_kind_{index:04d}",
                            "kind": "enum",
                            "prompt": (
                                "XML schema: UIFlow.v1\n"
                                f"XML: {invalid_kind}\n"
                                "Expectation: button kind must be primary, secondary, or danger.\n"
                                "Fix: "
                            ),
                            "expected": canonical(
                                {
                                    "op": "set_attr",
                                    "path": f"/{tag}/header/button[@id='{button}-btn']",
                                    "attr": "kind",
                                    "value": kind,
                                }
                            ),
                        },
                        {
                            "name": f"wrong_field_type_{index:04d}",
                            "kind": "wrong_attr",
                            "prompt": (
                                "XML schema: UIFlow.v1\n"
                                f"XML: {wrong_type}\n"
                                f"Expectation: field {field}-field should use type {ftype}.\n"
                                "Fix: "
                            ),
                            "expected": canonical(
                                {
                                    "op": "set_attr",
                                    "path": f"/{tag}/body/field[@id='{field}-field']",
                                    "attr": "type",
                                    "value": ftype,
                                }
                            ),
                        },
                        {
                            "name": f"rename_field_label_{index:04d}",
                            "kind": "change_request",
                            "prompt": (
                                "XML schema: UIFlow.v1\n"
                                f"XML: {valid}\n"
                                f"Expectation: change the visible field label to Customer {field.title()}.\n"
                                "Fix: "
                            ),
                            "expected": canonical(
                                {
                                    "op": "set_attr",
                                    "path": f"/{tag}/body/field[@id='{field}-field']",
                                    "attr": "label",
                                    "value": f"Customer {field.title()}",
                                }
                            ),
                        },
                        {
                            "name": f"move_button_to_body_{index:04d}",
                            "kind": "misplaced_node",
                            "prompt": (
                                "XML schema: UIFlow.v1\n"
                                f"XML: {misplaced}\n"
                                "Expectation: buttons belong under header before body fields.\n"
                                "Fix: "
                            ),
                            "expected": canonical(
                                {
                                    "op": "move",
                                    "path": f"/{tag}/body/button[@id='{button}-btn']",
                                    "to": f"/{tag}/header",
                                }
                            ),
                        },
                    ]
                )
                index += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/xml_fix_domain", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_cases = cases()
    train = [row for idx, row in enumerate(all_cases) if idx % 5 != 0]
    heldout = [row for idx, row in enumerate(all_cases) if idx % 5 == 0]
    seen_questions = train[:24]
    heldout_questions = heldout[:24]
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
                "schema": "UIFlow.v1",
                "allowed_button_kind": KINDS,
                "allowed_field_type": TYPES,
                "required": {
                    "button": ["id", "action", "kind", "label"],
                    "field": ["id", "type", "required", "label"],
                },
                "layout": "root/header/button before root/body/field",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"train": len(train), "heldout": len(heldout), "out_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
