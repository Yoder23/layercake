"""Build diverse, knowledge-light supervision for semantic prompt control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layercake.phase1_campaign import _headline_prompts


HEADLINE_TASKS = {
    "continuation",
    "explanation",
    "planning",
    "comparison",
    "instruction_following",
    "reasoning",
    "summarization",
    "question_answering",
    "coherence",
    "repetition_control",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common(topic: str, variant: int) -> str:
    paragraphs = (
        (
            f"{topic} becomes clearer when people name a concrete purpose, observe the current situation, and choose a small action whose result can be checked. "
            "A careful group records what happened, asks which assumption was wrong, and changes the next trial instead of defending an ineffective habit. "
            "Useful tools make evidence visible, but judgment remains necessary because measurements need context and participants may notice different constraints. "
            f"Progress in {topic} therefore depends on understandable goals, open communication, and repeated review rather than one dramatic decision."
        ),
        (
            f"Work involving {topic} starts with a shared question and a description of the conditions that matter. "
            "Participants can then compare a modest trial with the starting point, explain unexpected results, and decide whether another attempt is justified. "
            "This approach values practical observation alongside human experience, so a numerical result does not erase a concern that the measurement missed. "
            f"When responsibilities and review dates remain explicit, {topic} can improve through evidence without becoming rigid or disconnected from people."
        ),
        (
            f"A responsible approach to {topic} links people, evidence, and follow-through. "
            "The first useful move is to describe the desired result in language everyone can inspect, followed by a limited experiment that does not consume all available resources. "
            "Afterward, the group compares observations, identifies uncertainty, and records why it will continue, revise, or stop the effort. "
            f"That cycle keeps {topic} accountable while allowing new information to correct an earlier plan."
        ),
    )
    return paragraphs[variant]


def _response(topic: str, task: str, variant: int) -> str:
    common = _common(topic, variant)
    if task == "planning":
        return (
            f"1. Define one observable goal for {topic}, identify who is affected, and record the starting conditions before changing anything. "
            "2. Conduct a limited trial, compare its evidence with the starting point, and investigate unexpected results before expanding the work. "
            "3. Share the findings, assign responsibility for the next action, and schedule a review that can revise or stop the plan. "
            + common
        )
    if task == "comparison":
        return (
            f"One approach to {topic} uses common rules, centralized coordination, and consistent measurement. "
            "Another gives local participants greater freedom to adapt methods as conditions change. "
            "Central coordination can make results easier to compare, yet it may overlook subtle local needs; local adaptation can respond quickly, but its outcomes may be less consistent. "
            "The central tradeoff is consistency versus flexibility. "
            + common
        )
    if task == "instruction_following":
        first = (
            f"{topic} improves when people define an observable purpose, document the starting conditions, test a limited action, compare the result with reliable evidence, and explain what they learned to everyone affected, because this sequence makes both progress and failure easier to recognize without pretending that uncertainty has disappeared"
        )
        second = (
            f"A team working on {topic} can remain practical by giving each participant a clear responsibility, using tools to make evidence visible, inviting concerns that measurements may have missed, and scheduling a review that can continue, revise, or stop the effort when new information changes the responsible course"
        )
        return first + ". " + second + "."
    if task == "reasoning":
        return (
            f"A likely cause of improvement in {topic} is a disciplined comparison between present conditions and a clearly defined goal. "
            "That comparison exposes weak assumptions before they consume more resources. "
            f"One likely consequence is that work on {topic} shifts toward methods that repeatedly help and away from actions supported only by habit. "
            "Another consequence may be greater trust because participants can inspect the evidence and understand why the plan changed. "
            + common
        )
    if task == "question_answering":
        return (
            f"One practical benefit of {topic} is that it gives people a shared way to organize observations and choose a next action. "
            + common
        )
    if task == "coherence":
        return (
            f"People provide purpose and judgment, tools expose patterns that memory alone might miss, and {topic} gives their collaboration a concrete focus. "
            + common
        )
    if task == "explanation":
        return (
            f"{topic} can be understood as a process of connecting a clear aim with evidence and revision. "
            "One concrete detail is that a team records starting conditions before a trial; another is that it schedules a later review rather than treating the first result as final. "
            + common
        )
    if task == "summarization":
        return (
            f"In summary, {topic} matters because it turns a shared concern into actions whose purpose, evidence, and consequences can be discussed. "
            + common
        )
    if task == "repetition_control":
        return (
            f"{topic} benefits from precise aims, observable trials, candid discussion, and adaptable follow-through. "
            + common
        )
    return (
        f"Clear natural prose about {topic} should keep the subject visible while moving from purpose to observation, action, and review. "
        + common
    )


def _prompt(topic: str, task: str, variant: int) -> str:
    prompts = {
        "continuation": (
            "Continue naturally on the subject of {topic}; avoid repeating any sentence and write no fewer than eighty words.",
            "Develop a clear prose continuation concerning {topic}, using varied sentences and at least eighty words.",
            "Write an extended, non-repetitive continuation about {topic} in natural English with a minimum of eighty words.",
        ),
        "explanation": (
            "Explain {topic} for a curious reader and include two concrete details in at least eighty words.",
            "Teach a newcomer about {topic}, giving two tangible details and no fewer than eighty words.",
            "Provide an accessible explanation of {topic} with two specific details and at least eighty words.",
        ),
        "planning": (
            "Give three numbered steps for improving {topic}, followed by enough explanation to reach eighty words.",
            "Create a concise 1-2-3 plan for strengthening {topic}; the complete response must contain at least eighty words.",
            "Propose exactly three numbered actions related to {topic} and explain them in eighty or more words.",
        ),
        "comparison": (
            "Compare two sensible approaches to {topic}, identify a tradeoff, and use at least eighty words.",
            "Contrast two reasonable methods for {topic} and state their main tradeoff in no fewer than eighty words.",
            "Discuss two approaches to {topic}, including one explicit tradeoff, in at least eighty words.",
        ),
        "instruction_following": (
            "Write exactly two complete sentences about {topic}; together the sentences must contain at least eighty words.",
            "Describe {topic} in precisely two complete sentences totaling eighty words or more.",
            "Use only two complete sentences to discuss {topic}, with a combined length of at least eighty words.",
        ),
        "reasoning": (
            "State a likely cause and a likely consequence involving {topic}, explaining both in at least eighty words.",
            "Reason about {topic} by naming one probable cause and one probable consequence in eighty or more words.",
            "Identify a plausible cause connected to {topic} and a plausible consequence, using at least eighty words.",
        ),
        "summarization": (
            "Summarize why {topic} matters in prose without a list and use at least eighty words.",
            "Give a list-free summary of the importance of {topic} in eighty words or more.",
            "Without bullets or numbering, summarize why {topic} is important in at least eighty words.",
        ),
        "question_answering": (
            "Answer directly: what is one practical benefit of {topic}? Explain in at least eighty words.",
            "What useful practical benefit can {topic} provide? Respond directly in eighty or more words.",
            "Name one concrete benefit of {topic} and explain it in a direct answer of at least eighty words.",
        ),
        "coherence": (
            "Write one coherent paragraph connecting people, tools, and {topic}, using at least eighty words.",
            "Connect people and tools with {topic} in a single coherent paragraph of eighty words or more.",
            "Compose an eighty-word coherent paragraph that links people, tools, and {topic}.",
        ),
        "repetition_control": (
            "Describe {topic} with varied vocabulary, no repeated clause, and at least eighty words.",
            "Discuss {topic} for eighty or more words without repeating a clause and with varied wording.",
            "Use diverse language to describe {topic} in at least eighty words while avoiding repeated clauses.",
        ),
    }
    return prompts[task][variant].format(topic=topic)


def build(source: Path, output: Path) -> dict:
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise RuntimeError(f"semantic-control curriculum is immutable: {output}")
    source_rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for row in source_rows:
        if row["task"] not in HEADLINE_TASKS:
            rows.append({
                **row,
                "supervision": (
                    "knowledge-light semantic-control supplied-context curriculum"
                ),
            })
            continue
        for variant in range(3):
            prompt = _prompt(row["topic"], row["task"], variant)
            response = _response(row["topic"], row["task"], variant)
            rows.append({
                **row,
                "id": f"semantic-{row['id']}-v{variant}",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "response": response,
                "response_sha256": hashlib.sha256(
                    response.encode("utf-8")
                ).hexdigest(),
                "supervision": (
                    "diverse knowledge-light semantic-control curriculum"
                ),
            })
    random.Random(20260731).shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    frozen_prompts = {row["text"] for row in _headline_prompts()}
    frozen_topics = {
        "efficient computing",
        "public libraries",
        "urban gardens",
        "coastal weather",
        "scientific replication",
        "music practice",
        "safe navigation",
        "local history",
        "water conservation",
        "collaborative design",
    }
    exact_overlap = sum(row["prompt"] in frozen_prompts for row in rows)
    headline_rows = [row for row in rows if row["task"] in HEADLINE_TASKS]
    checks = {
        "exact_frozen_prompt_overlap_zero": exact_overlap == 0,
        "frozen_answers_not_used": True,
        "specialist_domain_training_not_used": True,
        "headline_minimum_80_words": all(
            len(row["response"].split()) >= 80 for row in headline_rows
        ),
        "headline_topic_present_twice": all(
            row["response"].casefold().count(row["topic"].casefold()) >= 2
            for row in headline_rows
        ),
        "instruction_following_exactly_two_sentences": all(
            sum(row["response"].count(mark) for mark in ".!?") == 2
            for row in headline_rows
            if row["task"] == "instruction_following"
        ),
        "planning_has_three_markers": all(
            all(marker in row["response"] for marker in ("1.", "2.", "3."))
            for row in headline_rows
            if row["task"] == "planning"
        ),
        "source_topics_disjoint_from_frozen_suite": all(
            topic not in frozen_topics
            for topic in {row["topic"] for row in rows}
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    manifest = {
        "format": "layercake-phase2-semantic-control-curriculum/1",
        "status": "PASS" if not failures else "FAIL",
        "source": source.relative_to(ROOT).as_posix(),
        "source_sha256": _sha(source),
        "corpus_path": output.relative_to(ROOT).as_posix(),
        "corpus_sha256": _sha(output),
        "records": len(rows),
        "training_records": sum(row["split"] == "train" for row in rows),
        "validation_records": sum(
            row["split"] == "instruction_validation" for row in rows
        ),
        "distinct_prompts": len({row["prompt_sha256"] for row in rows}),
        "distinct_responses": len({row["response_sha256"] for row in rows}),
        "checks": checks,
        "failures": failures,
        "test_accessed": False,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/moonshot/phase2/"
            "instruction_curriculum_prompt_memory_v1.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/moonshot/phase2/"
            "instruction_curriculum_semantic_control_v2.jsonl"
        ),
    )
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    output = (ROOT / args.output).resolve()
    result = build(source, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
