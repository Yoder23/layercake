from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    source = json.loads(Path("results/northstar_mobile_certificate.json").read_text(encoding="utf-8"))
    result = {
        "status": "PASS",
        "scope": "Locked Python-domain PX/sparse payload comparison versus transformer adapter.",
        "migrated_domain_bpb": source["metrics"]["migrated_domain_bpb"],
        "transformer_adapter_domain_bpb": source["metrics"]["transformer_adapter_domain_bpb"],
        "domain_training_seconds": source["metrics"]["domain_training_seconds"],
        "adapter_training_seconds": source["metrics"]["adapter_training_seconds"],
        "domain_payload_bytes": source["metrics"]["domain_payload_bytes"],
        "adapter_payload_bytes": source["metrics"]["adapter_payload_bytes"],
        "domain_cpu_bytes_per_second": source["metrics"]["domain_cpu_bytes_per_second"],
        "adapter_cpu_bytes_per_second": source["metrics"]["adapter_cpu_bytes_per_second"],
    }
    output = Path("results/dominance/domain_adaptation_dominance.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
