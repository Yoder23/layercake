from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.orchestration import LayerCakeOrchestrator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/breakthrough_equal/domain_orchestrator_certificate.json"


MODELS = [
    {
        "id": "lc-python-cpu-small",
        "domains": ["python"],
        "keywords": ["debug", "function", "traceback", "import"],
        "cost": 1.0,
        "capacity": 1.0,
        "abi_version": "lc-abi/2",
        "top_k": 1,
        "input_mode": "byte_patch",
    },
    {
        "id": "lc-game-cpu-small",
        "domains": ["game"],
        "keywords": ["npc", "quest", "inventory", "combat"],
        "cost": 1.0,
        "capacity": 1.0,
        "abi_version": "lc-abi/2",
        "top_k": 1,
        "input_mode": "byte_patch",
    },
    {
        "id": "lc-app-cpu-small",
        "domains": ["app"],
        "keywords": ["button", "screen", "layout", "save"],
        "cost": 1.0,
        "capacity": 1.0,
        "abi_version": "lc-abi/2",
        "top_k": 1,
        "input_mode": "byte_patch",
    },
    {
        "id": "lc-general-cpu-large",
        "domains": ["general"],
        "keywords": ["explain", "summarize", "question"],
        "cost": 6.0,
        "capacity": 6.0,
        "abi_version": "lc-abi/2",
        "top_k": 2,
        "input_mode": "byte_patch",
    },
]


BRICKS = [
    {
        "id": "python-domain-px",
        "domain": "python",
        "keywords": ["debug", "function", "traceback", "import"],
        "abi_version": "lc-abi/2",
    },
    {
        "id": "game-domain-px",
        "domain": "game",
        "keywords": ["npc", "quest", "inventory", "combat"],
        "abi_version": "lc-abi/2",
    },
    {
        "id": "app-domain-px",
        "domain": "app",
        "keywords": ["button", "screen", "layout", "save"],
        "abi_version": "lc-abi/2",
    },
]


PROMPTS = [
    {
        "name": "python_debug",
        "task": "Debug this Python traceback from an import failure.",
        "expected_model": "lc-python-cpu-small",
        "expected_domain": "python",
    },
    {
        "name": "game_npc",
        "task": "The NPC asks about the next quest after combat.",
        "expected_model": "lc-game-cpu-small",
        "expected_domain": "game",
    },
    {
        "name": "app_edit",
        "task": "Move the Save button on the app screen layout.",
        "expected_model": "lc-app-cpu-small",
        "expected_domain": "app",
    },
    {
        "name": "uncertain_general",
        "task": "Explain this vague question and summarize the options.",
        "expected_model": "lc-general-cpu-large",
        "expected_domain": "general",
        "uncertainty": 0.85,
    },
]


def verify(
    *,
    models: list[dict] | None = None,
    bricks: list[dict] | None = None,
    prompts: list[dict] | None = None,
    escalation_threshold: float = 0.6,
) -> dict:
    models = models or MODELS
    bricks = bricks or BRICKS
    prompts = prompts or PROMPTS
    orchestrator = LayerCakeOrchestrator(escalation_threshold=escalation_threshold)
    rows = []
    start = time.perf_counter()
    for prompt in prompts:
        uncertainty = float(prompt.get("uncertainty", 0.2))
        decision = orchestrator.route(prompt["task"], models, bricks, uncertainty)
        packet = orchestrator.handoff_packet(prompt["task"], models, bricks, uncertainty)
        rows.append(
            {
                "name": prompt["name"],
                "expected_model": prompt["expected_model"],
                "expected_domain": prompt["expected_domain"],
                "decision": decision,
                "packet_hash": packet.compute_hash(),
                "model_correct": decision["model_id"] == prompt["expected_model"],
                "domain_correct": decision["selected_domain"] == prompt["expected_domain"],
                "active_compute_bounded": (
                    decision["active_model_count"] == 1
                    and len(decision["active_bricks"])
                    <= next(
                        model.get("top_k", 1)
                        for model in models
                        if model["id"] == decision["model_id"]
                    )
                ),
            }
        )
    seconds = time.perf_counter() - start
    gates = {
        "all_prompts_routed_to_expected_model": all(row["model_correct"] for row in rows),
        "all_prompts_routed_to_expected_domain": all(row["domain_correct"] for row in rows),
        "active_compute_bounded": all(row["active_compute_bounded"] for row in rows),
        "single_active_model_per_prompt": all(
            row["decision"]["active_model_count"] == 1 for row in rows
        ),
        "handoff_packets_hashable": all(row["packet_hash"] for row in rows),
    }
    failed = [name for name, passed in gates.items() if not passed]
    routing_accuracy = sum(row["model_correct"] and row["domain_correct"] for row in rows) / max(len(rows), 1)
    return {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "Deterministic multi-LayerCake domain-orchestrator gate. This verifies "
            "routing, bounded active compute, and hashable handoff packets. It is "
            "not a substitute for trained specialist-model quality evaluation."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "prompt_count": len(rows),
            "routing_accuracy": routing_accuracy,
            "max_active_model_count": max(row["decision"]["active_model_count"] for row in rows),
            "max_active_bricks": max(len(row["decision"]["active_bricks"]) for row in rows),
            "routing_seconds_total": seconds,
            "routing_seconds_per_prompt": seconds / max(len(rows), 1),
        },
        "rows": rows,
        "claim_boundary": (
            "The orchestrator can route to specialized LayerCake domains with bounded "
            "active compute. Real domination still requires per-domain trained payloads "
            "and downstream task-quality certificates."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify deterministic LayerCake domain orchestration.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
