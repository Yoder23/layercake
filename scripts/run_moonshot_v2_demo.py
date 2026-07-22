from _common import ROOT

import argparse
import json

from layercake.evaluation.orchestration_v2 import run_orchestration_demo


parser = argparse.ArgumentParser()
parser.add_argument("--host", action="append", default=[])
parser.add_argument("--cake", default="artifacts/cakes/python.cake")
parser.add_argument("--public-key", default="artifacts/cakes/python.public.pem")
parser.add_argument("--router", default="artifacts/router/semantic-router.safetensors")
parser.add_argument("--domain-test", default="data/moonshot/v2/python/python_test.bin")
parser.add_argument("--registry-root", default="artifacts/demo/registries")
parser.add_argument("--output", default="results/moonshot/v2/orchestration_evidence.json")
args = parser.parse_args()
hosts = args.host or [
    "artifacts/cores/english-core-a",
    "artifacts/cores/english-core-b",
    "artifacts/cores/english-core-c",
]
result = run_orchestration_demo(
    host_dirs=[ROOT / value for value in hosts],
    cake_path=ROOT / args.cake,
    public_key_path=ROOT / args.public_key,
    router_path=ROOT / args.router,
    domain_test_path=ROOT / args.domain_test,
    registry_root=ROOT / args.registry_root,
    output_path=ROOT / args.output,
)
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["status"] == "PASS" else 1)
