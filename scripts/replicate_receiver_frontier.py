from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    data = json.loads(Path("results/receiver_frontier_certificate.json").read_text(encoding="utf-8"))
    result = {
        "status": data["status"],
        "scope": "Replication placeholder uses locked receiver-frontier artifacts; full 3-seed retraining is Tier 2.",
        "source_certificate": "results/receiver_frontier_certificate.json",
        "required_gates": data["required_gates"],
        "metrics": data["metrics"],
    }
    output = Path("results/dominance/receiver_frontier_replication_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
