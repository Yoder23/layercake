from _common import ROOT

import argparse
import json

from layercake.evaluation.incremental import benchmark_incremental_generation


parser = argparse.ArgumentParser()
parser.add_argument("--core", default="artifacts/cores/english-core-a")
parser.add_argument("--corpus", default="data/moonshot/v2/wikitext103/test.bin")
parser.add_argument("--output", default="results/moonshot/v2/incremental_benchmark.json")
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
result = benchmark_incremental_generation(
    ROOT / args.core, ROOT / args.corpus, ROOT / args.output, repeats=args.repeats
)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "PASS" else 1)

