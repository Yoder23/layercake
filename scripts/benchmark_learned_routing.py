from _common import ROOT

import json

from layercake.evaluation.routing_v2 import train_and_benchmark_router


result = train_and_benchmark_router(
    ROOT / "results/moonshot/v2/routing_evidence.json",
    model_path=ROOT / "artifacts/router/semantic-router.safetensors",
)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "PASS" else 1)

