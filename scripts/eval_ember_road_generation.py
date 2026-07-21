from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layercake.causal_byte_models import CausalBytePatchLM
from layercake.portable_domain import LayerCakeRuntime


def _decode_bytes(t: torch.Tensor) -> str:
    return bytes(t.tolist()).decode("utf-8", errors="replace")


def _generate_core(model: CausalBytePatchLM, prompt: str, max_new_bytes: int = 220, no_repeat_ngram: int = 8) -> str:
    model.eval()
    device = next(model.parameters()).device
    prompt_bytes = list(prompt.encode("utf-8", errors="replace"))
    min_prompt = max(int(getattr(model, "local_window", 64)), int(getattr(model, "patch_size", 2)))
    while len(prompt_bytes) < min_prompt:
        prompt_bytes.append(ord(" "))
    patch_size = int(getattr(model, "patch_size", 2))
    if len(prompt_bytes) % patch_size:
        prompt_bytes.extend([ord(" ")] * (patch_size - (len(prompt_bytes) % patch_size)))

    prompt_ids = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(0)
    state = model.begin_cached_generation(prompt_ids)

    generated = []
    while len(generated) < max_new_bytes:
        patch = model.cached_generation_step(state, no_repeat_ngram=no_repeat_ngram)
        bytes_out = patch[0].detach().cpu().tolist()
        generated.extend(bytes_out)

    return bytes(generated[:max_new_bytes]).decode("utf-8", errors="replace")


def _generate_domain(runtime: LayerCakeRuntime, domain_id: str, prompt: str, max_new_bytes: int = 220) -> str:
    prompt_ids = torch.tensor(list(prompt.encode("utf-8", errors="replace")), dtype=torch.long).unsqueeze(0)
    out = runtime.generate(prompt_ids, max_new_bytes=max_new_bytes, domain_id=domain_id, context_bytes=256)
    continuation = out[0, prompt_ids.shape[1] :].cpu()
    return _decode_bytes(continuation)


def _printable_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="replace")
    printable = sum(1 for b in raw if b in (9, 10, 13) or 32 <= b <= 126)
    return printable / max(len(raw), 1)


def _rulebook_score(text: str, keywords: list[str]) -> float:
    lower = text.lower()
    matched = sum(1 for k in keywords if k.lower() in lower)
    return matched / max(len(keywords), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate conversational fluency and Ember Road rulebook knowledge")
    parser.add_argument("--core-checkpoint", required=True)
    parser.add_argument("--domain-artifact", required=False, default=None)
    parser.add_argument("--domain-id", required=False, default="ember_road")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.core_checkpoint, map_location="cpu", weights_only=True)
    model_cfg = ckpt["model_config"]
    model = CausalBytePatchLM(**model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    runtime = None
    domain_id = args.domain_id
    if args.domain_artifact:
        artifact = torch.load(args.domain_artifact, map_location="cpu", weights_only=True)
        domain_id = artifact["spec"]["domain_id"]
        runtime = LayerCakeRuntime(model)
        runtime.install_portable_domain(artifact, device=device)

    conversational_prompts = [
        "Player: Hey Pip, I just got back to camp. What should I focus on before the next room?\\nPip:",
        "Player: I keep getting overwhelmed when two lanes collapse at once. Give me a calm plan.\\nPip:",
        "Player: My health is shaky but I still want tempo. What is the safest aggressive play?\\nPip:",
    ]

    rulebook_prompts = [
        "Question: In Ember Road, when does a run actually end?\\nAnswer:",
        "Question: If the player is downed, what should Pip prioritize first?\\nAnswer:",
        "Question: Against shamans and archers, who should be pressured first and why?\\nAnswer:",
        "Question: When should guardLink and rallyCry be used?\\nAnswer:",
        "Question: What is the win condition against the Ancient Treant?\\nAnswer:",
    ]

    rulebook_keywords = {
        0: ["both", "down", "run"],
        1: ["survive", "clear", "wave"],
        2: ["shaman", "archer", "priority"],
        3: ["guardlink", "rallycry", "damage", "cards"],
        4: ["ancient treant", "defeat"],
    }

    conversational = []
    for prompt in conversational_prompts:
        core_text = _generate_core(model, prompt)
        conversational.append(
            {
                "prompt": prompt,
                "core_response": core_text,
                "printable_ratio": _printable_ratio(core_text),
            }
        )

    knowledge = []
    for idx, prompt in enumerate(rulebook_prompts):
        core_text = _generate_core(model, prompt)
        domain_text = None
        domain_score = None
        if runtime is not None:
            domain_text = _generate_domain(runtime, domain_id, prompt)
            domain_score = _rulebook_score(domain_text, rulebook_keywords[idx])
        knowledge.append(
            {
                "prompt": prompt,
                "core_response": core_text,
                "domain_response": domain_text,
                "core_rulebook_score": _rulebook_score(core_text, rulebook_keywords[idx]),
                "domain_rulebook_score": domain_score,
            }
        )

    domain_scores = [x["domain_rulebook_score"] for x in knowledge if x["domain_rulebook_score"] is not None]

    result = {
        "status": "COMPLETE",
        "device": str(device),
        "core_checkpoint": args.core_checkpoint,
        "domain_artifact": args.domain_artifact,
        "conversational": conversational,
        "rulebook": knowledge,
        "summary": {
            "avg_conversational_printable": sum(x["printable_ratio"] for x in conversational) / len(conversational),
            "avg_core_rulebook_score": sum(x["core_rulebook_score"] for x in knowledge) / len(knowledge),
            "avg_domain_rulebook_score": (sum(domain_scores) / len(domain_scores)) if domain_scores else None,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
