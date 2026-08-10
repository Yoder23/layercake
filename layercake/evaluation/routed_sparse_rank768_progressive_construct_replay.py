"""Deterministic replay envelope for the v15 construct verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import routed_sparse_rank768_progressive_construct as original


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(result: dict, replay_protocol_path: Path) -> dict:
    stable = {
        "format": "layercake-postrelease-routed-sparse-rank768-progressive-replay-result/1",
        "status": result["status"],
        "replay_protocol": {
            "path": replay_protocol_path.name,
            "sha256": sha(replay_protocol_path),
        },
        "original_protocol": result["protocol"],
        "checks": result["checks"],
        "target_parameters": result["target_parameters"],
        "random_package_identifiers_excluded": True,
        "external_artifact_used": False,
        "english_quality_tested": False,
        "performance_tested": False,
        "claim_boundary": "Deterministic routed sparse rank768 host construct replay only; no ABI acquisition, quality, runtime, or superiority claim.",
    }
    stable["evidence_sha256"] = hashlib.sha256(
        (json.dumps(stable, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return stable


def execute(root: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format")
        != "layercake-postrelease-routed-sparse-rank768-progressive-replay/1"
        or protocol.get("status") != "PREREGISTERED_DETERMINISTIC_REPLAY"
    ):
        raise RuntimeError("replay governance changed")
    for name, expected in protocol["bindings"].items():
        target = root / name
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError(f"replay binding changed: {name}")
    raw = original.execute(root, root / protocol["original_protocol"])
    return _stable(raw, protocol_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol)
    output = root / args.output
    if args.command == "execute":
        if output.exists():
            raise RuntimeError("output exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif json.loads(output.read_text(encoding="utf-8")) != result:
        raise RuntimeError("stored deterministic replay differs")
    print(json.dumps({"status": result["status"], "evidence_sha256": result["evidence_sha256"]}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
