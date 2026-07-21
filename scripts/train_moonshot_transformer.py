"""Train the fair paired experiment and emit the transformer safe checkpoint."""
import argparse, json
import _common
from layercake.moonshot import train_paired

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/full.json")
args = parser.parse_args()
result = train_paired(args.config)
print(json.dumps(result["checkpoints"]["transformer"], indent=2, sort_keys=True))
