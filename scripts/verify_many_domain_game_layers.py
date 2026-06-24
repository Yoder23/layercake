from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifact_utils import build_models
from layercake.portable_domain import (
    LayerCakeRuntime,
    PortableDomainSpec,
    build_portable_artifact,
    load_portable_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load_torch(path: str) -> dict:
    return torch.load(ROOT / path, map_location="cpu")


def _game_artifact(base_artifact: dict, domain_id: str) -> dict:
    spec, model = load_portable_artifact(copy.deepcopy(base_artifact), "cpu")
    renamed = PortableDomainSpec(
        **{**spec.canonical_dict(), "domain_id": domain_id}
    )
    return build_portable_artifact(
        model,
        renamed,
        training={
            **base_artifact.get("training", {}),
            "proxy_domain_note": (
                "Renamed copy of the current portable payload. This verifies "
                "multi-domain install/migration mechanics, not game-domain training."
            ),
        },
        evaluation=base_artifact.get("evaluation", {}),
    )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_artifact = _load_torch("runs_experiment/portable_python_gru148k_v1.pt")
    source_artifact = _load_torch(
        "runs_experiment/scale15m_transition_lw280_2300_noprofile.pt"
    )
    receiver_artifact = _load_torch("runs_experiment/scale5m_seed4242_continued.pt")
    _, source_core = build_models(source_artifact, device)
    _, receiver_core = build_models(receiver_artifact, device)
    source_runtime = LayerCakeRuntime(source_core)
    receiver_runtime = LayerCakeRuntime(receiver_core)
    domain_ids = ["game_dialogue", "game_lore", "game_quest_state"]
    artifacts = [_game_artifact(base_artifact, domain_id) for domain_id in domain_ids]

    prompt = (
        "NPC: The gate is locked until the player finds the brass key.\n"
        "Player: I found the brass key. What now?\n"
        "NPC:"
    ).encode("utf-8")
    prompt_tensor = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(0)
    single_domain_logits = {}
    for artifact in artifacts:
        domain_id = artifact["spec"]["domain_id"]
        single = LayerCakeRuntime(source_core)
        single.install_portable_domain(artifact, device)
        single_domain_logits[domain_id] = single.logits(
            prompt_tensor, domain_id=domain_id
        ).detach().cpu()

    for artifact in artifacts:
        source_runtime.install_portable_domain(artifact, device)
        receiver_runtime.install_portable_domain(artifact, device)

    per_domain = {}
    max_cross_domain_interference = 0.0
    for artifact in artifacts:
        domain_id = artifact["spec"]["domain_id"]
        source_logits = source_runtime.logits(prompt_tensor, domain_id=domain_id)
        receiver_logits = receiver_runtime.logits(prompt_tensor, domain_id=domain_id)
        source_after_all = source_logits.detach().cpu()
        max_cross_domain_interference = max(
            max_cross_domain_interference,
            (source_after_all - single_domain_logits[domain_id]).abs().max().item(),
        )
        source_generation = source_runtime.generate(
            prompt_tensor,
            max_new_bytes=64,
            domain_id=domain_id,
            context_bytes=256,
        )
        receiver_generation = receiver_runtime.generate(
            prompt_tensor,
            max_new_bytes=64,
            domain_id=domain_id,
            context_bytes=256,
        )
        max_logit_diff = (source_logits - receiver_logits).abs().max().item()
        generation_equal = torch.equal(source_generation, receiver_generation)
        generated = source_generation[0, prompt_tensor.shape[1] :].detach().cpu()
        per_domain[domain_id] = {
            "max_logit_diff": max_logit_diff,
            "generation_equal": generation_equal,
            "generated_utf8": bytes(generated.tolist()).decode(
                "utf-8", errors="replace"
            ),
            "payload_hash": artifact["payload_hash"],
            "spec_hash": artifact["spec_hash"],
        }

    payload_hashes = {item["payload_hash"] for item in artifacts}
    spec_hashes = {item["spec_hash"] for item in artifacts}
    gates = {
        "three_domains_installed": (
            len(source_runtime.domains) == 3 and len(receiver_runtime.domains) == 3
        ),
        "domain_specs_are_distinct": len(spec_hashes) == len(domain_ids),
        "payload_function_reused": len(payload_hashes) == 1,
        "all_domains_transfer_exact_logits": all(
            item["max_logit_diff"] == 0.0 for item in per_domain.values()
        ),
        "all_domains_generation_exact_after_transfer": all(
            item["generation_equal"] for item in per_domain.values()
        ),
        "installing_other_domains_does_not_change_selected_domain": (
            max_cross_domain_interference == 0.0
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "Many-domain game-layer proxy. This verifies install/migration/isolation "
            "mechanics for multiple game-named domains using renamed copies of the "
            "current portable payload. It does not prove game-dialogue quality until "
            "a real game corpus payload is trained and evaluated."
        ),
        "required_gates": gates,
        "failed_required": failed,
        "domain_ids": domain_ids,
        "per_domain": per_domain,
        "metrics": {
            "installed_domains": len(source_runtime.domains),
            "max_cross_domain_interference": max_cross_domain_interference,
            "unique_payload_hashes": len(payload_hashes),
            "unique_spec_hashes": len(spec_hashes),
        },
        "open_requirements_for_real_game_domains": {
            "trained_game_dialogue_payload": False,
            "trained_game_lore_payload": False,
            "trained_game_quest_state_payload": False,
            "task_level_npc_eval": False,
            "domain_routing_policy_eval": False,
        },
    }
    output = RESULTS / "many_domain_game_layers_certificate.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
