"""Construct verifier for the hard-top-1 sparse residual host v13."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import torch
from layercake.sparse_residual_progressive_core import SparseResidualProgressiveCore, SparseResidualProgressiveLayer
from layercake_extensions.sparse_residual_progressive_core import SPARSE_RESIDUAL_ABI_SHA256, SPARSE_RESIDUAL_ABI_VERSION, SparseResidualProgressiveCoreHost


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _top1_check() -> tuple[bool, tuple[int, ...]]:
    layer = SparseResidualProgressiveLayer(8, 2, 1, 4, residual_experts=4, rms_epsilon=1e-5, rope_theta=10000.0)
    with torch.no_grad():
        layer.post_attention_norm.weight.fill_(1); layer.residual_router.weight.zero_(); layer.residual_router.weight[2].fill_(1)
        layer.expert_coefficient_weights.fill_(float("nan")); layer.expert_output_bases.fill_(float("nan")); layer.expert_residual_means.fill_(float("nan"))
        layer.expert_coefficient_weights[2].zero_(); layer.expert_output_bases[2].zero_(); layer.expert_residual_means[2].fill_(1)
    output = layer._mlp_delta(torch.ones(1, 3, 8))
    return bool(torch.isfinite(output).all() and torch.equal(output, torch.ones_like(output))), layer.last_active_expert_counts


def execute(root: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != "layercake-postrelease-sparse-residual-progressive-construct/1" or protocol.get("status") != "PREREGISTERED_CONSTRUCT_ONLY":
        raise RuntimeError("governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"binding changed: {name}")
    spec = importlib.util.spec_from_file_location("fixture", root / "tests/models/test_sparse_residual_progressive_core.py")
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    with tempfile.TemporaryDirectory(prefix="lc-v13-") as directory:
        temp = Path(directory); package, public, key_id, state_hash, tokenizer = fixture._package(temp / "source")
        archive = sha(package); cpu = SparseResidualProgressiveCoreHost(temp / "cpu", trust_store={key_id: public})
        active = cpu.activate(package); ids = tokenizer.encode_source("hello world")[0]; state = cpu.prefill("hello world")
        first = torch.allclose(state.next_logits, cpu.module(torch.tensor([ids]))[:, -1], atol=1e-5, rtol=1e-5)
        before = tuple(value.shape[2] for value in state.layer_keys); state.next_logits.zero_(); state.next_logits[0, 4] = 1
        action, state = cpu.decode_step(state); second = torch.allclose(state.next_logits, cpu.module(torch.tensor([ids + [action]]))[:, -1], atol=1e-5, rtol=1e-5)
        after = tuple(value.shape[2] for value in state.layer_keys); verified = cpu.verify(); cpu.remove(); reinstall = cpu.activate(package)
        cuda = None
        if torch.cuda.is_available():
            cuda = SparseResidualProgressiveCoreHost(temp / "cuda", trust_store={key_id: public}, device="cuda").activate(package)
        count = SparseResidualProgressiveCore.parameter_count_for_config(
            fixed_vocab_size=32015, full_width=3072, bottleneck_width=192,
            replacement_layers=32, intermediate_size=768, residual_experts=4,
        )
        physical_top1, active_counts = _top1_check()
        checks = {
            "abi": SPARSE_RESIDUAL_ABI_VERSION == "lc-direct-neural-core/13" and SPARSE_RESIDUAL_ABI_SHA256 == protocol["interface_sha256"],
            "abi_hash": sha(root / protocol["canonical_abi"]) == SPARSE_RESIDUAL_ABI_SHA256,
            "parameters": count == 391_154_688,
            "four_experts": all(layer.residual_experts == 4 for layer in cpu.module.layers),
            "physical_top1_poison_test": physical_top1 and active_counts == (0, 0, 3, 0),
            "first": bool(first), "second": bool(second),
            "cache": all(right == left + 1 for left, right in zip(before, after)),
            "cpu": active["state_dict_hash"] == state_hash and verified["status"] == "PASS",
            "zero_learning": active["receiver_training_steps"] == active["receiver_calibration_runs"] == 0,
            "lifecycle": sha(package) == archive and reinstall["archive_hash"] == active["archive_hash"],
            "cuda": cuda is not None and cuda["archive_hash"] == active["archive_hash"] and cuda["state_dict_hash"] == state_hash,
            "zero_source": protocol["source_transformer_blocks"] == 0,
        }
    result = {
        "format": "layercake-postrelease-sparse-residual-progressive-result/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": {"path": protocol_path.name, "sha256": sha(protocol_path)},
        "checks": checks, "target_parameters": count,
        "package": {"archive_hash": archive, "state_dict_hash": state_hash},
        "external_artifact_used": False, "english_quality_tested": False, "performance_tested": False,
        "claim_boundary": "Generic sparse-residual host construct only; no ABI acquisition, English quality, speed, memory, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256((json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("execute", "verify")); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); root = Path.cwd().resolve(); result = execute(root, root / args.protocol); output = root / args.output
    if args.command == "execute":
        if output.exists(): raise RuntimeError("output exists")
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result:
        raise RuntimeError("stored result differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2)); return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
