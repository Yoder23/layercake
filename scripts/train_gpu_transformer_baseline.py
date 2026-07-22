from _common import ROOT

import argparse
import json

from layercake.training.baseline import train_bpe_transformer


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/dev/transformer_same_scale.yaml")
parser.add_argument("--output", default="artifacts/baselines/transformer")
args = parser.parse_args()
print(json.dumps(train_bpe_transformer(ROOT / args.config, ROOT / args.output), indent=2))

