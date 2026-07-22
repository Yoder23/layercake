from _common import ROOT

import argparse
import json

from layercake.evaluation.true_portability import verify_true_cross_host_portability


parser = argparse.ArgumentParser()
parser.add_argument("--cake", default="artifacts/cakes/python.cake")
parser.add_argument("--public-key", default="artifacts/cakes/python.public.pem")
parser.add_argument("--host", action="append", default=[])
parser.add_argument("--domain-test", default="data/moonshot/v2/python/python_test.bin")
parser.add_argument("--output", default="results/moonshot/v2/portability_evidence.json")
args = parser.parse_args()
hosts = args.host or [
    "artifacts/cores/english-core-a",
    "artifacts/cores/english-core-b",
    "artifacts/cores/english-core-c",
]
result = verify_true_cross_host_portability(
    cake_path=ROOT / args.cake,
    public_key_path=ROOT / args.public_key,
    host_dirs=[ROOT / path for path in hosts],
    domain_test_path=ROOT / args.domain_test,
    output_path=ROOT / args.output,
)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "PASS" else 1)

