"""Run the complete paired path: loading, preprocessing, training, and evaluation."""
import argparse, json
import _common
from layercake.moonshot import train_paired

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/integration_five_seed.json")
args = parser.parse_args()
print(json.dumps(train_paired(args.config), indent=2, sort_keys=True))
