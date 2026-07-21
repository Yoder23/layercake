from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _sentence(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def build(rulebook: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for action in rulebook.get("actions", []):
        action_id = str(action["id"])
        rows.append(
            "Question: When should Pip use "
            f"{action_id}, and what should the companion avoid? Answer: "
            f"Use {action_id} for {action.get('role', 'support')}: "
            f"{_sentence(action.get('when', ''))} Avoid this when {_sentence(action.get('avoid', ''))}\n"
        )
    for enemy in rulebook.get("enemy_archetypes", []):
        enemy_id = str(enemy["id"])
        rows.append(
            f"Question: What is the safest companion response to a {enemy_id}? Answer: "
            f"{enemy_id} threat: {_sentence(enemy.get('threat', ''))} "
            f"Response: {_sentence(enemy.get('response', ''))}\n"
        )
    for priority in rulebook.get("priorities", []):
        rank = int(priority.get("rank", 0))
        rows.append(
            f"Question: What should Pip prioritize for priority rule {rank}? Answer: "
            f"Priority {rank}: {_sentence(priority.get('rule', ''))}\n"
        )
    for example in rulebook.get("examples", []):
        situation = _sentence(example.get("situation", ""))
        decision = example.get("decision", {}) or {}
        rows.append(
            "Question: In this situation, what action and tactic should Pip choose? "
            f"{situation} Answer: Choose {decision.get('action', '')} and "
            f"{decision.get('tactic', '')}. Reason: {_sentence(decision.get('reason', ''))}\n"
        )
    rows.extend(
        [
            (
                "Question: The player is nervous before a boss fight. Give a calm, useful companion response. "
                "Answer: Breathe, check health and cooldowns, watch the boss telegraph, keep a safe lane, and guard before overcommitting.\n"
            ),
            (
                "Question: I got ambushed and lost health. What should I do now? "
                "Answer: Retreat or Guard first, create space, stabilize health, identify the nearest threat, then recover tempo.\n"
            ),
            (
                "Question: An archer is hiding behind a brute. What is the correct response? "
                "Answer: Flank across lanes to pressure the archer, avoid the brute melee lane, and stop the archer from firing freely.\n"
            ),
        ]
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rulebook", type=Path, default=Path("data/game_domains/ember-road.rulebook.json"))
    parser.add_argument("--output", type=Path, default=Path("data/game_domains/ember-road.answer_layer.txt"))
    args = parser.parse_args()
    rulebook = json.loads(args.rulebook.read_text(encoding="utf-8"))
    rows = build(rulebook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
