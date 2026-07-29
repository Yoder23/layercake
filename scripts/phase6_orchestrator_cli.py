"""External JSON boundary for the Phase 6 direct-cake orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import _common
from layercake.routing import DirectCakeOrchestrator, load_archive_bound_profiles


ROOT = Path(__file__).resolve().parents[1]


def _path(value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    if not path.is_file() and not path.parent.is_dir():
        raise ValueError(f"external orchestration path does not exist: {path}")
    return path


def execute_request(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("format") != "layercake-phase6-external-request/1":
        raise ValueError("external request identity is invalid")
    trust_store = {
        str(key_id): _path(str(path))
        for key_id, path in document.get("trust_store", {}).items()
    }
    orchestrator = DirectCakeOrchestrator(
        Path(str(document["registry_root"])),
        abi_version=str(document["abi_version"]),
        abi_hash=str(document["abi_hash"]),
        trust_store=trust_store,
        profiles=load_archive_bound_profiles(_path(str(document["profiles"]))),
        device=str(document.get("device", "cpu")),
    )
    installs = [
        orchestrator.install(_path(str(path)))
        for path in document.get("install", [])
    ]
    result = orchestrator.execute(
        str(document.get("prompt", "")),
        mode=str(document.get("mode", "automatic_top1")),
        manual=tuple(str(value) for value in document.get("manual", [])) or None,
        subrequests=tuple(str(value) for value in document.get("subrequests", [])) or None,
        core_handler=lambda prompt: f"CORE:{prompt}",
    )
    return {
        "format": "layercake-phase6-external-response/1",
        "status": "PASS",
        "installed": [
            {
                "cake_id": value["cake_id"],
                "archive_hash": value["archive_hash"],
                "signed": value["signed"],
            }
            for value in installs
        ],
        "result": result.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        request = json.loads(arguments.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("external request must be a JSON object")
        response = execute_request(request)
    except Exception as error:
        response = {
            "format": "layercake-phase6-external-response/1",
            "status": "ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        print(json.dumps(response, sort_keys=True))
        return 1
    print(json.dumps(response, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
