from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path("results/dominance/domain_adaptation_dominance.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    passed = (
        all(lc < tx for lc, tx in zip(data["migrated_domain_bpb"], data["transformer_adapter_domain_bpb"]))
        and data["domain_training_seconds"] < data["adapter_training_seconds"]
        and data["domain_payload_bytes"] < data["adapter_payload_bytes"]
        and data["domain_cpu_bytes_per_second"] > data["adapter_cpu_bytes_per_second"]
    )
    result = {"status": "PASS" if passed else "FAIL", "certificate": str(path)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
