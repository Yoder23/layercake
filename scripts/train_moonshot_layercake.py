"""Train the paired experiment and emit the LayerCake safe checkpoint."""
import argparse, json
import _common
from layercake.moonshot import train_paired

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/integration_five_seed.json")
args = parser.parse_args()
result = train_paired(args.config)
print(json.dumps(result["checkpoints"]["layercake"], indent=2, sort_keys=True))
