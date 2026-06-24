from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path("results/dominance/northstar_15m_replication_smoke.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {"status": data["status"], "certificate": str(path)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if data["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
