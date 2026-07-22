import argparse, json
import _common
from layercake.moonshot import run_suite

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/integration_five_seed.json")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
config = "configs/moonshot/smoke.json" if args.smoke else args.config
print(json.dumps(run_suite(config)["certificate"], indent=2, sort_keys=True))
