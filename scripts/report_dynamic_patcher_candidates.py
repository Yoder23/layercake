from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    result = {
        "status": "STUBBED_WITH_GATES",
        "candidates": ["entropy", "surprisal", "learned_boundary_stub", "utf8_safe", "code_aware"],
        "promotion_gates": ["bpb", "training_time", "patch_compression", "cpu_generation", "gpu_generation", "abi_stability", "domain_transfer_stability"],
        "current_promotion": "none",
        "reason": "Fixed two-byte ABI remains the promoted path until a candidate beats frozen gates.",
    }
    output = Path("results/dominance/dynamic_patcher_candidates.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
