"""Construct verifier for the v16 precision-conformant routed host."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from layercake.routed_sparse_rank768_progressive_core import RoutedSparseRank768ProgressiveCore
from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
from tests.models.test_decoder_direct_neural_core import _doc


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(kind):
    tokenizer = DecoderAwareExternalTokenizer(_doc())
    model = kind(fixed_vocab_size=tokenizer.vocab_size, full_width=24, bottleneck_width=8, attention_heads=2, replacement_layers=2, intermediate_size=16, residual_rank=16, sparse_width=8, maximum_source_actions=16, maximum_target_actions=8, maximum_sequence_actions=24).bind_tokenizer(tokenizer)
    return model, tokenizer


def execute(root: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-routed-sparse-rank768-fp16-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise RuntimeError("v16 construct governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"v16 construct binding changed: {name}")
    torch.manual_seed(16)
    source, _ = _model(RoutedSparseRank768ProgressiveCore)
    target, tokenizer = _model(PrecisionConformantRoutedSparseRank768ProgressiveCore)
    fp16_cpu = {name: value.half() for name, value in source.state_dict().items()}
    incompatible = target.load_state_dict(fp16_cpu, strict=True, assign=True)
    ids, _ = tokenizer.encode_source("hello world")
    cpu_route = target._select_route(torch.tensor([ids]))
    cuda_checks = {"available": torch.cuda.is_available(), "prefill": False, "incremental": False, "finite": False, "route_match": False}
    if torch.cuda.is_available():
        cuda_model, _ = _model(PrecisionConformantRoutedSparseRank768ProgressiveCore)
        cuda_model.load_state_dict({name: value.cuda() for name, value in fp16_cpu.items()}, strict=True, assign=True)
        cuda_model = cuda_model.cuda().eval()
        state = cuda_model.prefill_ids(ids, tokenizer.encode_source("hello world")[1])
        cuda_checks["prefill"] = True
        cuda_checks["finite"] = bool(torch.isfinite(state.next_logits).all())
        cuda_checks["route_match"] = state.route_index == cpu_route
        state.next_logits.zero_(); state.next_logits[0, 4] = 1
        _, state = cuda_model.decode_step(state)
        cuda_checks["incremental"] = state.sequence_length == len(ids) + 1 and bool(torch.isfinite(state.next_logits).all())
    checks = {
        "v15_state_dict_compatible": not incompatible.missing_keys and not incompatible.unexpected_keys and set(target.state_dict()) == set(source.state_dict()),
        "true_fp16_parameters": all(value.dtype == torch.float16 for value in target.state_dict().values()),
        "cpu_fp16_router": cpu_route in range(3),
        "cuda_available": cuda_checks["available"],
        "cuda_fp16_prefill": cuda_checks["prefill"],
        "cuda_fp16_incremental": cuda_checks["incremental"],
        "cuda_finite": cuda_checks["finite"],
        "cpu_cuda_route_identity": cuda_checks["route_match"],
        "production_parameter_count": PrecisionConformantRoutedSparseRank768ProgressiveCore.parameter_count_for_config(fixed_vocab_size=32015, full_width=3072, bottleneck_width=192, replacement_layers=32, intermediate_size=768, residual_rank=768, sparse_width=384) == 536758275,
        "zero_source_blocks": protocol["source_transformer_blocks"] == 0,
        "zero_receiver_learning": protocol["receiver_training_steps"] == 0,
    }
    result = {
        "format": "layercake-postrelease-routed-sparse-rank768-fp16-result/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": {"path": protocol_path.name, "sha256": sha(protocol_path)},
        "checks": checks,
        "cuda": cuda_checks,
        "target_parameters": 536758275,
        "external_artifact_used": False,
        "english_quality_tested": False,
        "performance_tested": False,
        "claim_boundary": "Generic precision-conformant routed host construct only; no ABI acquisition, English quality, runtime performance, or superiority claim."
    }
    result["evidence_sha256"] = hashlib.sha256((json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve(); result = execute(root, root / args.protocol); output = root / args.output
    if args.command == "execute":
        if output.exists(): raise RuntimeError("output exists")
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result:
        raise RuntimeError("stored result differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
