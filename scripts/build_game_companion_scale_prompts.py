from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _words(text: str, *, limit: int = 5) -> list[str]:
    tokens = []
    for raw in text.replace(".", " ").replace(",", " ").replace(";", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum()).strip()
        if len(token) >= 4 and token.casefold() not in {
            "when",
            "with",
            "from",
            "that",
            "this",
            "while",
            "into",
            "uses",
            "nearby",
        }:
            tokens.append(token)
    deduped = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped[:limit]


def _row(
    *,
    category: str,
    prompt: str,
    expected_keywords: list[str],
    min_keyword_hits: int = 2,
) -> dict[str, Any]:
    return {
        "category": category,
        "prompt": f"Question: {prompt} Answer:",
        "expected_keywords": expected_keywords,
        "forbidden_keywords": [],
        "min_keyword_hits": min_keyword_hits,
    }


def build(rulebook: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in rulebook.get("actions", []):
        action_id = str(action["id"])
        role = str(action.get("role", ""))
        rows.append(
            _row(
                category="action_choice",
                prompt=(
                    f"When should Pip use {action_id}, and what should the "
                    "companion avoid?"
                ),
                expected_keywords=[action_id, role, *_words(str(action.get("when", ""))), *_words(str(action.get("avoid", "")), limit=3)],
                min_keyword_hits=3,
            )
        )
    for enemy in rulebook.get("enemy_archetypes", []):
        enemy_id = str(enemy["id"])
        rows.append(
            _row(
                category="enemy_counterplay",
                prompt=f"What is the safest companion response to a {enemy_id}?",
                expected_keywords=[enemy_id, *_words(str(enemy.get("threat", "")), limit=3), *_words(str(enemy.get("response", "")), limit=5)],
                min_keyword_hits=3,
            )
        )
    for priority in rulebook.get("priorities", []):
        rank = int(priority.get("rank", 0))
        rule = str(priority.get("rule", ""))
        rows.append(
            _row(
                category="priority_policy",
                prompt=f"What should Pip prioritize for priority rule {rank}?",
                expected_keywords=_words(rule, limit=8),
                min_keyword_hits=3,
            )
        )
    for example in rulebook.get("examples", []):
        situation = str(example.get("situation", ""))
        decision = example.get("decision", {}) or {}
        rows.append(
            _row(
                category="situation_decision",
                prompt=f"In this situation, what action and tactic should Pip choose? {situation}",
                expected_keywords=[
                    str(decision.get("action", "")),
                    str(decision.get("tactic", "")),
                    *_words(str(decision.get("reason", "")), limit=5),
                ],
                min_keyword_hits=3,
            )
        )
    rows.extend(
        [
            _row(
                category="companion_style",
                prompt="The player is nervous before a boss fight. Give a calm, useful companion response.",
                expected_keywords=["Breathe", "boss", "health", "telegraph", "safe", "guard"],
                min_keyword_hits=2,
            ),
            _row(
                category="game_recovery",
                prompt="I got ambushed and lost health. What should I do now?",
                expected_keywords=["Retreat", "Guard", "stabilize", "health", "threat"],
                min_keyword_hits=2,
            ),
            _row(
                category="game_tactics",
                prompt="An archer is hiding behind a brute. What is the correct response?",
                expected_keywords=["flank", "archer", "brute", "lane", "pressure"],
                min_keyword_hits=2,
            ),
        ]
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rulebook", type=Path, default=Path("data/game_domains/ember-road.rulebook.json"))
    parser.add_argument("--output", type=Path, default=Path("data/game_domains/ember-road.scale_prompts.json"))
    args = parser.parse_args()
    rulebook = json.loads(args.rulebook.read_text(encoding="utf-8"))
    rows = build(rulebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "prompts": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
