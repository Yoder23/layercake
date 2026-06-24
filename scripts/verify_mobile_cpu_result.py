from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path("results/dominance/mobile_cpu_proxy.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    passed = (
        data["status"] == "PASS"
        and min(data["source_cpu_generation_speed_ratio"]) > 1.0
        and data["source_norepeat8_cpu_generation_speed_ratio"] > 1.0
        and data["receiver_cpu_generation_speed_ratio"] > 1.0
    )
    result = {"status": "PASS" if passed else "FAIL", "certificate": str(path)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
