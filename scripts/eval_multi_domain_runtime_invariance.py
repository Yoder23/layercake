from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layercake.portable_domain import LayerCakeRuntime


def _load_prompt(path: Path, prompt_bytes: int) -> bytes:
    payload = path.read_bytes()
    if len(payload) < prompt_bytes:
        repeats = prompt_bytes // max(len(payload), 1) + 1
        payload = payload * repeats
    return payload[:prompt_bytes]


def _hash_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that installing many portable domains does not change the "
            "behavior of any given installed domain."
        )
    )
    parser.add_argument("--target", required=True, help="Portable domain artifact under test")
    parser.add_argument(
        "--others",
        action="append",
        default=[],
        help="Additional portable domain artifacts to install alongside the target",
    )
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--prompt-bytes", type=int, default=128)
    parser.add_argument("--generation-bytes", type=int, default=32)
    parser.add_argument("--context-bytes", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    target_path = Path(args.target)
    other_paths = [Path(item) for item in args.others]
    prompt = _load_prompt(Path(args.prompt_file), args.prompt_bytes)

    target_artifact = torch.load(target_path, map_location="cpu")
    target_id = target_artifact["spec"]["domain_id"]

    solo = LayerCakeRuntime()
    solo.install_portable_domain(target_artifact, device)

    multi = LayerCakeRuntime()
    multi.install_portable_domain(target_artifact, device)
    installed_ids = [target_id]
    for path in other_paths:
        artifact = torch.load(path, map_location="cpu")
        installed_ids.append(multi.install_portable_domain(artifact, device))

    prompt_tensor = torch.tensor(list(prompt), dtype=torch.long).unsqueeze(0)
    solo_logits = solo.logits(prompt_tensor.to(device), domain_id=target_id)
    multi_logits = multi.logits(prompt_tensor.to(device), domain_id=target_id)
    max_logit_diff = float((solo_logits - multi_logits).abs().max().item())

    solo_generation = solo.generate(
        prompt,
        max_new_bytes=args.generation_bytes,
        domain_id=target_id,
        context_bytes=args.context_bytes,
    )
    multi_generation = multi.generate(
        prompt,
        max_new_bytes=args.generation_bytes,
        domain_id=target_id,
        context_bytes=args.context_bytes,
    )
    generation_equal = torch.equal(solo_generation, multi_generation)

    result = {
        "status": "PASS" if max_logit_diff == 0.0 and generation_equal else "FAIL",
        "device": str(device),
        "target_domain_id": target_id,
        "installed_domain_ids": installed_ids,
        "installed_domain_count": len(installed_ids),
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "max_logit_diff": max_logit_diff,
        "generation_equal": generation_equal,
        "solo_generation_sha256": _hash_tensor(solo_generation),
        "multi_generation_sha256": _hash_tensor(multi_generation),
        "solo_generation_utf8": bytes(solo_generation[0].cpu().tolist()).decode("utf-8", errors="replace"),
        "multi_generation_utf8": bytes(multi_generation[0].cpu().tolist()).decode("utf-8", errors="replace"),
        "scope": (
            "Portable-domain runtime invariance: a domain installed alone must behave "
            "identically when installed alongside many other domains."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
