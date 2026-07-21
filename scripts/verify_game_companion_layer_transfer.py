from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import _common
from layercake.portable_domain import LayerCakeRuntime, load_portable_artifact


def _bytes(text: str) -> torch.Tensor:
    return torch.tensor([list(text.encode("utf-8", errors="replace"))], dtype=torch.long)


def _decode(row: torch.Tensor) -> str:
    return bytes(int(item) for item in row.detach().cpu().tolist()).decode("utf-8", errors="replace")


def verify(artifact: dict[str, Any], prompt: str, *, generation_bytes: int, context_bytes: int) -> dict[str, Any]:
    source_spec, source_decoder = load_portable_artifact(artifact, "cpu")
    target_spec, target_decoder = load_portable_artifact(artifact, "cpu")
    x = _bytes(prompt)
    with torch.inference_mode():
        source_logits = source_decoder(x)
        target_logits = target_decoder(x)
        max_logit_diff = float((source_logits - target_logits).abs().max().item())
        source_runtime = LayerCakeRuntime()
        target_runtime = LayerCakeRuntime()
        source_runtime.install_portable_domain(artifact, "cpu")
        target_runtime.install_portable_domain(artifact, "cpu")
        source_generation = source_runtime.generate(
            x,
            max_new_bytes=generation_bytes,
            domain_id=source_spec.domain_id,
            context_bytes=context_bytes,
        )
        target_generation = target_runtime.generate(
            x,
            max_new_bytes=generation_bytes,
            domain_id=target_spec.domain_id,
            context_bytes=context_bytes,
        )
    generated_equal = torch.equal(source_generation, target_generation)
    generated_bytes = bytes(int(item) for item in source_generation[0].detach().cpu().tolist())
    gates = {
        "domain_ids_equal": source_spec.domain_id == target_spec.domain_id,
        "specs_equal": source_spec == target_spec,
        "independent_decoder_instances": source_decoder is not target_decoder,
        "max_logit_diff_exact": max_logit_diff == 0.0,
        "generation_exact": bool(generated_equal),
        "payload_hash_present": bool(artifact.get("payload_hash")),
        "spec_hash_present": bool(artifact.get("spec_hash")),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "mode": "core_independent_lossless_game_layer_transfer",
        "gates": gates,
        "domain_id": source_spec.domain_id,
        "spec_hash": artifact.get("spec_hash"),
        "payload_hash": artifact.get("payload_hash"),
        "max_logit_diff": max_logit_diff,
        "generation": {
            "equal": bool(generated_equal),
            "new_bytes": generation_bytes,
            "sha256": hashlib.sha256(generated_bytes).hexdigest(),
            "source_utf8": _decode(source_generation[0]),
            "target_utf8": _decode(target_generation[0]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Question: What is the safest companion response to a brute? Answer:",
    )
    parser.add_argument("--generation-bytes", type=int, default=64)
    parser.add_argument("--context-bytes", type=int, default=128)
    args = parser.parse_args()
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    result = verify(
        artifact,
        args.prompt,
        generation_bytes=args.generation_bytes,
        context_bytes=args.context_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
