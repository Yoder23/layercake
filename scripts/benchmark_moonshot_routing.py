import argparse, json
import _common
from layercake.moonshot import build_ecosystem

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/integration_five_seed.json")
args = parser.parse_args()
print(json.dumps(build_ecosystem(args.config)["routing"], indent=2, sort_keys=True))
