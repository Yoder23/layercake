"""Reporter-only replay of the v8 runtime-vocabulary accounting repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from layercake.progressive_replacement_core import ProgressiveReplacementCore
from layercake.evaluation.progressive_replacement_core_construct import execute as execute_base


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != "layercake-postrelease-progressive-replacement-core-construct-repair2/1"
        or protocol.get("status") != "PREREGISTERED_EXACT_REPORTER_REPLAY"
    ):
        raise RuntimeError("progressive replacement reporter repair governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"progressive replacement reporter repair binding changed: {relative}")
    base_protocol = json.loads((root / protocol["base_repair_protocol"]).read_text(encoding="utf-8"))
    base = execute_base(root, root / base_protocol["base_protocol"])
    runtime_vocabulary = int(base_protocol["corrected_target_geometry"]["runtime_vocabulary"])
    target_parameters = ProgressiveReplacementCore.parameter_count_for_config(
        fixed_vocab_size=runtime_vocabulary,
        full_width=3_072,
        bottleneck_width=192,
        replacement_layers=32,
        intermediate_size=768,
    )
    checks = {
        "base_generic_construct_pass": base["status"] == "PASS",
        "external_actions_plus_host_specials": runtime_vocabulary == 32_011 + 4,
        "corrected_target_parameter_count": target_parameters == 253_535_232,
        "zero_source_transformer_blocks": base["checks"]["zero_source_transformer_blocks"],
        "same_package_cpu_cuda": base["checks"]["same_package_cuda"],
        "incremental_identity": base["checks"]["full_incremental_first_step_identity"]
        and base["checks"]["full_incremental_second_step_identity"],
        "zero_receiver_learning": base["receiver_training_steps"] == 0
        and base["receiver_calibration_runs"] == 0,
    }
    result = {
        "format": "layercake-postrelease-progressive-replacement-core-construct-repair-result/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": {"path": protocol_path.name, "sha256": sha(protocol_path)},
        "checks": checks,
        "runtime_vocabulary": runtime_vocabulary,
        "target_parameters": target_parameters,
        "preserved_reporter_failure": protocol["preserved_failure"],
        "supersedes_exact_target_only": "moonshot/postrelease_progressive_replacement_core_decision_v19.json",
        "generic_host_implementation_changed": False,
        "english_quality_tested": False,
        "performance_tested": False,
        "claim_boundary": "Corrected construct accounting and tokenizer compatibility only; no acquisition, quality, or performance claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(
        (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol)
    output = root / args.output
    if args.command == "execute":
        if output.exists():
            raise RuntimeError("progressive replacement repair result immutable")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result:
        raise RuntimeError("stored progressive replacement repair differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
