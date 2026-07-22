from _common import ROOT

import argparse
import json

from layercake.training.foundation import train_english_core


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/dev/core_english_development.yaml")
parser.add_argument("--output", default="artifacts/cores/english-core-a")
args = parser.parse_args()
print(json.dumps(train_english_core(ROOT / args.config, ROOT / args.output), indent=2))

