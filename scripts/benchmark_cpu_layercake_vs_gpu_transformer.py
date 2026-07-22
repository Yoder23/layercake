from _common import ROOT

import argparse
import json

from layercake.evaluation.cpu_vs_gpu import benchmark_cpu_vs_gpu


parser = argparse.ArgumentParser()
parser.add_argument("--layercake-core", default="artifacts/cores/english-core-a")
parser.add_argument("--cake", default="artifacts/cakes/python.cake")
parser.add_argument("--public-key", default="artifacts/cakes/python.public.pem")
parser.add_argument("--transformer", default="artifacts/baselines/transformer-mixed")
parser.add_argument("--router", default="artifacts/router/semantic-router.safetensors")
parser.add_argument("--suite", default="configs/eval/matched_quality_mixed_workload.yaml")
parser.add_argument("--output", default="results/moonshot/v2/cpu_vs_gpu_evidence.json")
args = parser.parse_args()
result = benchmark_cpu_vs_gpu(
    ROOT / args.suite, core_dir=ROOT / args.layercake_core,
    cake_path=ROOT / args.cake, public_key_path=ROOT / args.public_key,
    transformer_dir=ROOT / args.transformer, router_path=ROOT / args.router,
    output_path=ROOT / args.output,
)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "PASS" else 1)

