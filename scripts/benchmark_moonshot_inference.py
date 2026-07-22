import argparse, json
import _common
from layercake.moonshot import benchmark_inference

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/integration_five_seed.json")
args = parser.parse_args()
print(json.dumps(benchmark_inference(args.config), indent=2, sort_keys=True))
