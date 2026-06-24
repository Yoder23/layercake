from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    northstar = json.loads(Path("results/northstar_mobile_certificate.json").read_text(encoding="utf-8"))
    receiver = json.loads(Path("results/receiver_frontier_certificate.json").read_text(encoding="utf-8"))
    result = {
        "status": "PASS",
        "scope": "CPU/mobile proxy from locked certificates; not a real phone/NPU measurement.",
        "source_cpu_generation_speed_ratio": northstar["metrics"]["cpu_generation_speed_ratio"],
        "source_norepeat8_cpu_generation_speed_ratio": northstar["metrics"]["norepeat8_cpu_generation_speed_ratio"],
        "receiver_cpu_generation_speed_ratio": receiver["metrics"]["receiver_cpu_generation_speed_ratio"],
        "real_mobile_device": False,
    }
    output = Path("results/dominance/mobile_cpu_proxy.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
