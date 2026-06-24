from __future__ import annotations

import json
from pathlib import Path


PROBES = {
    "276k": "results/dominance/tier1_local_276k_probe.json",
    "474k": "results/dominance/tier1_local_474k_probe.json",
    "735k": "results/dominance/tier1_local_735k_probe.json",
    "1m": "results/dominance/tier1_local_1m_probe.json",
}


def _load_optional(path: str) -> dict | None:
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def main() -> int:
    loaded = {name: _load_optional(path) for name, path in PROBES.items()}
    available = {name: data for name, data in loaded.items() if data is not None}
    passed = {name: data["status"] == "PASS" for name, data in available.items()}
    failures = {
        name: [
            gate for gate, ok in data.get("dominance", {}).get("gates", {}).items()
            if not ok
        ]
        for name, data in available.items()
        if data["status"] != "PASS"
    }
    result = {
        "status": "PASS" if passed.get("276k", False) else "FAIL",
        "scope": "Local Tier-1-style scaling frontier. 276k currently passes; larger probes are retained as scaling blockers when present.",
        "available_probes": sorted(available),
        "passed": passed,
        "failures": failures,
        "next_blocker": "Improve quality/printable generation while preserving CPU speed and trainable-parameter advantage at 474k+.",
    }
    output = Path("results/dominance/tier1_local_frontier_certificate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
