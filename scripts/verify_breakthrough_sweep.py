from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "breakthrough_sweep.yaml"
DEFAULT_OUTPUT = ROOT / "results/breakthrough_equal/layercake_breakthrough_sweep_certificate.json"


def read_json_yaml(path: Path) -> dict[str, Any]:
    """Read the locked sweep manifest.

    The repo intentionally stores this as JSON-compatible YAML so the verifier
    does not need a PyYAML runtime dependency.
    """
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def get_path(row: Any, dotted: str) -> Any:
    cur = row
    for part in dotted.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(dotted)
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError) as exc:
                raise KeyError(dotted) from exc
        else:
            raise KeyError(dotted)
    return cur


def compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op in {">=", ">", "<=", "<"}:
        left = float(actual)
        right = float(expected)
        if op == ">=":
            return left >= right
        if op == ">":
            return left > right
        if op == "<=":
            return left <= right
        return left < right
    raise ValueError(f"unsupported check op: {op}")


def evaluate_check(artifact: dict[str, Any], check: dict[str, Any]) -> dict[str, Any]:
    path = str(check["path"])
    op = str(check["op"])
    expected = check["value"]
    try:
        actual = get_path(artifact, path)
        passed = compare(actual, op, expected)
        error = None
    except Exception as exc:  # fail closed on malformed artifacts
        actual = None
        passed = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": path,
        "op": op,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "error": error,
    }


def evaluate_gate(gate: dict[str, Any], *, root: Path) -> dict[str, Any]:
    artifact_path = root / str(gate["artifact"])
    if not artifact_path.exists():
        return {
            "artifact": str(gate["artifact"]),
            "artifact_status": "MISSING",
            "passed": False,
            "checks": [],
        }

    artifact = read_artifact(artifact_path)
    checks = [
        evaluate_check(artifact, check)
        for check in gate.get("checks", [])
    ]
    return {
        "artifact": str(gate["artifact"]),
        "artifact_status": str(artifact.get("status", "UNKNOWN")),
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "checks": checks,
    }


def verify(manifest: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    gate_results = {}
    hygiene_results = {}
    blockers = []
    track_status: dict[str, bool] = {
        name: True for name in manifest.get("tracks", {})
    }
    hygiene_status = True

    for gate in manifest.get("promoted_gates", []):
        name = str(gate["name"])
        track = str(gate["track"])
        result = evaluate_gate(gate, root=root)
        passed = bool(result["passed"])
        if not passed:
            blockers.append(name)
        gate_results[name] = {
            "track": track,
            **result,
        }
        track_status[track] = track_status.get(track, True) and passed

    for gate in manifest.get("evidence_hygiene_gates", []):
        name = str(gate["name"])
        result = evaluate_gate(gate, root=root)
        passed = bool(result["passed"])
        if not passed:
            blockers.append(name)
        hygiene_status = hygiene_status and passed
        hygiene_results[name] = result

    status = "PASS" if not blockers else "FAIL"
    return {
        "status": status,
        "campaign": manifest.get("campaign"),
        "claim": manifest.get("claim"),
        "scope": (
            "Top-level LayerCake breakthrough sweep. This certificate passes only "
            "when every promoted fair-neural and product-runtime gate in the locked "
            "manifest passes. Partial wins remain blockers."
        ),
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_claim_policy": manifest.get("claim_policy", {}),
        "hardware_target": manifest.get("hardware_target", {}),
        "baselines": manifest.get("baselines", {}),
        "track_status": {
            track: "PASS" if passed else "FAIL"
            for track, passed in sorted(track_status.items())
        },
        "evidence_hygiene_status": "PASS" if hygiene_status else "FAIL",
        "blockers": blockers,
        "promoted_gates": gate_results,
        "evidence_hygiene_gates": hygiene_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the locked LayerCake breakthrough sweep.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = read_json_yaml(args.manifest)
    result = verify(manifest, root=ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
