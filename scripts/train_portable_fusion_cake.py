from _common import ROOT

import argparse
import json

from layercake.training.cake import train_portable_fusion_cake


parser = argparse.ArgumentParser()
parser.add_argument("--core", default="artifacts/cores/english-core-a")
parser.add_argument("--config", default="configs/moonshot/dev/cake_python.yaml")
parser.add_argument("--output", default="artifacts/cakes/python.cake")
args = parser.parse_args()
print(json.dumps(train_portable_fusion_cake(ROOT / args.core, ROOT / args.config, ROOT / args.output), indent=2))

