from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from artifact_utils import build_models
from layercake.portable_domain import (
    LayerCakeRuntime,
    PortableDomainSpec,
    build_portable_artifact,
    load_portable_artifact,
)


def _load_artifact(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _clone_with_domain_id(artifact: dict, domain_id: str) -> dict:
    spec, model = load_portable_artifact(copy.deepcopy(artifact), "cpu")
    cloned_spec = PortableDomainSpec(**{**spec.canonical_dict(), "domain_id": domain_id})
    return build_portable_artifact(
        model,
        cloned_spec,
        training={
            **artifact.get("training", {}),
            "clone_note": "renamed payload clone for 10-domain runtime invariance certification",
        },
        evaluation=artifact.get("evaluation", {}),
    )


def _to_tensor(raw: bytes, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(raw), dtype=torch.long, device=device).unsqueeze(0)


def _measure(runtime: LayerCakeRuntime, prompt: torch.Tensor, domain_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    logits = runtime.logits(prompt, domain_id=domain_id).detach().cpu()
    generated = runtime.generate(
        prompt,
        max_new_bytes=32,
        domain_id=domain_id,
        context_bytes=128,
    ).detach().cpu()
    return logits, generated


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify 1-domain vs 10-domain invariance and source/receiver exactness "
            "for portable domains."
        )
    )
    parser.add_argument(
        "--source-core",
        default="runs_experiment/scale15m_transition_lw280_2300_noprofile.pt",
    )
    parser.add_argument(
        "--target-core",
        default="runs_experiment/scale5m_seed4242.pt",
    )
    parser.add_argument(
        "--output",
        default="results/moonshot_suite/ten_domain_source_receiver_invariance.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_paths = [
        ROOT / "runs_experiment/game_dialogue_smoke_gru.pt",
        ROOT / "runs_experiment/game_lore_smoke_gru.pt",
        ROOT / "runs_experiment/game_quest_state_smoke_gru.pt",
        ROOT / "runs_experiment/technical_text_smoke_gru.pt",
        ROOT / "runs_experiment/portable_python_gru148k_seed6061.pt",
    ]
    base_artifacts = [_load_artifact(path) for path in base_paths]
    target_artifact = base_artifacts[0]
    target_domain_id = target_artifact["spec"]["domain_id"]

    cloned = [
        _clone_with_domain_id(base_artifacts[index % len(base_artifacts)], f"clone_domain_{index + 1:02d}")
        for index in range(5)
    ]
    domain_artifacts = base_artifacts + cloned

    prompt_bytes = (
        "Guard: The north gate is sealed until the moon bell rings.\n"
        "Merchant: Bring me three wolf pelts and I will trade you a lantern.\n"
        "Healer:"
    ).encode("utf-8")
    prompt = _to_tensor(prompt_bytes, device)

    source_state = _load_artifact(ROOT / args.source_core)
    target_state = _load_artifact(ROOT / args.target_core)
    _, source_core = build_models(source_state, device)
    _, target_core = build_models(target_state, device)

    source_single = LayerCakeRuntime(source_core)
    source_single.install_portable_domain(target_artifact, device)
    source_single_logits, source_single_gen = _measure(
        source_single,
        prompt,
        target_domain_id,
    )

    target_single = LayerCakeRuntime(target_core)
    target_single.install_portable_domain(target_artifact, device)
    target_single_logits, target_single_gen = _measure(
        target_single,
        prompt,
        target_domain_id,
    )

    source_multi = LayerCakeRuntime(source_core)
    target_multi = LayerCakeRuntime(target_core)
    installed_ids = []
    for artifact in domain_artifacts:
        installed_ids.append(source_multi.install_portable_domain(artifact, device))
        target_multi.install_portable_domain(artifact, device)

    source_multi_logits, source_multi_gen = _measure(
        source_multi,
        prompt,
        target_domain_id,
    )
    target_multi_logits, target_multi_gen = _measure(
        target_multi,
        prompt,
        target_domain_id,
    )

    per_domain = {}
    max_source_target_logit_diff = 0.0
    all_source_target_gen_equal = True
    for domain_id in installed_ids:
        src_logits, src_gen = _measure(source_multi, prompt, domain_id)
        tgt_logits, tgt_gen = _measure(target_multi, prompt, domain_id)
        max_diff = float((src_logits - tgt_logits).abs().max().item())
        gen_equal = bool(torch.equal(src_gen, tgt_gen))
        max_source_target_logit_diff = max(max_source_target_logit_diff, max_diff)
        all_source_target_gen_equal = all_source_target_gen_equal and gen_equal
        per_domain[domain_id] = {
            "max_source_target_logit_diff": max_diff,
            "source_target_generation_equal": gen_equal,
        }

    source_single_vs_multi_logit = float((source_single_logits - source_multi_logits).abs().max().item())
    target_single_vs_multi_logit = float((target_single_logits - target_multi_logits).abs().max().item())
    source_single_vs_multi_gen = bool(torch.equal(source_single_gen, source_multi_gen))
    target_single_vs_multi_gen = bool(torch.equal(target_single_gen, target_multi_gen))

    gates = {
        "exactly_ten_domains_installed": len(installed_ids) == 10,
        "source_one_vs_ten_logits_exact": source_single_vs_multi_logit == 0.0,
        "source_one_vs_ten_generation_exact": source_single_vs_multi_gen,
        "target_one_vs_ten_logits_exact": target_single_vs_multi_logit == 0.0,
        "target_one_vs_ten_generation_exact": target_single_vs_multi_gen,
        "source_target_all_domains_logits_exact": max_source_target_logit_diff == 0.0,
        "source_target_all_domains_generation_exact": all_source_target_gen_equal,
    }
    failed = [name for name, passed in gates.items() if not passed]

    result = {
        "status": "PASS" if not failed else "FAIL",
        "scope": (
            "1-domain vs 10-domain runtime invariance plus source/receiver exact transfer "
            "for every installed portable domain."
        ),
        "device": str(device),
        "target_domain_id": target_domain_id,
        "installed_domain_ids": installed_ids,
        "required_gates": gates,
        "failed_required": failed,
        "metrics": {
            "source_one_vs_ten_max_logit_diff": source_single_vs_multi_logit,
            "target_one_vs_ten_max_logit_diff": target_single_vs_multi_logit,
            "max_source_target_logit_diff_all_domains": max_source_target_logit_diff,
            "source_one_vs_ten_generation_equal": source_single_vs_multi_gen,
            "target_one_vs_ten_generation_equal": target_single_vs_multi_gen,
            "source_target_generation_equal_all_domains": all_source_target_gen_equal,
        },
        "per_domain": per_domain,
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
