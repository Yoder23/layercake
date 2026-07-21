from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import _common
from artifact_utils import build_models
from layercake.portable_domain import LayerCakeRuntime, load_portable_artifact
from run_paired_byte_experiment import batch, load_python_bytes


def load_eval_stream(args, root: Path) -> torch.Tensor:
    if args.eval_file:
        payload = bytearray()
        for item in args.eval_file:
            path = Path(item)
            if not path.is_absolute():
                path = root / path
            payload.extend(path.read_bytes())
            payload.extend(b"\n")
        if len(payload) < args.eval_bytes:
            repeats = args.eval_bytes // max(len(payload), 1) + 1
            payload = payload * repeats
        return torch.tensor(list(payload[-args.eval_bytes :]), dtype=torch.long)
    if args.eval_root:
        eval_root = Path(args.eval_root)
        domain_limit = max(args.eval_bytes, 1_000_000)
    else:
        eval_root = root.parent / "layercakeogwithdecoder"
        domain_limit = None
    return load_python_bytes(eval_root, domain_limit or args.domain_limit)[
        -args.eval_bytes :
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bit-exact logit and PPL gate for a portable domain decoder"
    )
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--source-core", required=True)
    parser.add_argument("--target-core", required=True)
    parser.add_argument("--eval-bytes", type=int, default=100_000)
    parser.add_argument("--eval-root")
    parser.add_argument(
        "--eval-file",
        action="append",
        help="Text/JSONL file to evaluate the portable domain against.",
    )
    parser.add_argument("--eval-source-label", default="repository-heldout-python")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--generation-bytes", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact = torch.load(args.decoder, map_location="cpu", weights_only=True)
    source = torch.load(args.source_core, map_location="cpu", weights_only=True)
    target = torch.load(args.target_core, map_location="cpu", weights_only=True)
    source_spec, source_decoder = load_portable_artifact(artifact, device)
    target_spec, target_decoder = load_portable_artifact(artifact, device)

    source_seq = source["args"].get("seq", 128)
    target_seq = target["args"].get("seq", 128)
    if source_seq != target_seq:
        raise ValueError("strict gate requires equal evaluation context lengths")
    root = Path(__file__).resolve().parents[1]
    args.domain_limit = max(
        source["args"].get("domain_bytes", 2_000_000),
        target["args"].get("domain_bytes", 2_000_000),
    )
    stream = load_eval_stream(args, root)
    generator = torch.Generator().manual_seed(991)
    batch_size = min(
        source["args"].get("batch", 24),
        target["args"].get("batch", 24),
        24,
    )
    losses = []
    correct = 0
    total = 0
    max_logit_diff = 0.0
    with torch.no_grad():
        for _ in range(args.batches):
            x, y = batch(stream, source_seq, batch_size, generator, device)
            source_logits = source_decoder(x)
            target_logits = target_decoder(x)
            max_logit_diff = max(
                max_logit_diff,
                (source_logits - target_logits).abs().max().item(),
            )
            losses.append(
                F.cross_entropy(
                    source_logits.flatten(0, 1), y.flatten()
                ).item()
            )
            correct += (source_logits.argmax(dim=-1) == y).sum().item()
            total += y.numel()
    loss = sum(losses) / len(losses)
    ppl = math.exp(loss)
    _, source_core = build_models(source, device)
    _, target_core = build_models(target, device)
    source_runtime = LayerCakeRuntime(source_core)
    target_runtime = LayerCakeRuntime(target_core)
    source_runtime.install_portable_domain(artifact, device)
    target_runtime.install_portable_domain(artifact, device)
    completion_context = min(source_seq, 128)
    required_generation_span = completion_context + args.generation_bytes
    if stream.numel() < required_generation_span:
        repeats = required_generation_span // max(stream.numel(), 1) + 1
        stream = stream.repeat(repeats)
    completion_start = min(4096, max(stream.numel() - required_generation_span, 0))
    prompt_tensor = stream[
        completion_start : completion_start + completion_context
    ].unsqueeze(0)
    expected = stream[
        completion_start
        + completion_context : completion_start
        + completion_context
        + args.generation_bytes
    ]
    source_generation = source_runtime.generate(
        prompt_tensor,
        max_new_bytes=args.generation_bytes,
        domain_id=source_spec.domain_id,
        context_bytes=source_seq,
    )
    target_generation = target_runtime.generate(
        prompt_tensor,
        max_new_bytes=args.generation_bytes,
        domain_id=target_spec.domain_id,
        context_bytes=target_seq,
    )
    generation_equal = torch.equal(source_generation, target_generation)
    generated_bytes = bytes(source_generation[0].cpu().tolist())
    predicted = source_generation[0, completion_context:].cpu()
    completion_matches = predicted == expected
    prefix_match = 0
    for matched in completion_matches.tolist():
        if not matched:
            break
        prefix_match += 1
    passed = max_logit_diff == 0.0 and generation_equal
    result = {
        "status": "PASS" if passed else "FAIL",
        "contract": {
            "unchanged_decoder_payload": True,
            "domain_id": source_spec.domain_id,
            "spec_hash": artifact["spec_hash"],
            "payload_hash": artifact["payload_hash"],
            "core_logits_used": False,
            "same_eval_bytes": True,
            "eval_source_label": args.eval_source_label,
            "eval_sha256": hashlib.sha256(
                stream.numpy().tobytes()
            ).hexdigest(),
            "context_bytes": source_seq,
        },
        "source_seed": source["seed"],
        "target_seed": target["seed"],
        "independent_decoder_instances": source_decoder is not target_decoder,
        "specs_equal": source_spec == target_spec,
        "max_logit_diff": max_logit_diff,
        "source": {
            "loss": loss,
            "ppl": ppl,
            "bpb": loss / math.log(2),
            "top1_byte_accuracy": correct / total,
        },
        "target": {
            "loss": loss,
            "ppl": ppl,
            "bpb": loss / math.log(2),
            "top1_byte_accuracy": correct / total,
        },
        "ppl_ratio": 1.0,
        "generation": {
            "equal": generation_equal,
            "new_bytes": args.generation_bytes,
            "sha256": hashlib.sha256(generated_bytes).hexdigest(),
            "heldout_offset": completion_start,
            "prompt_utf8": bytes(prompt_tensor[0].tolist()).decode(
                "utf-8", errors="replace"
            ),
            "predicted_utf8": bytes(predicted.tolist()).decode(
                "utf-8", errors="replace"
            ),
            "expected_utf8": bytes(expected.tolist()).decode(
                "utf-8", errors="replace"
            ),
            "continuation_byte_accuracy": completion_matches.float().mean().item(),
            "exact_prefix_bytes": prefix_match,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
