from _common import ROOT

import argparse
import json

from layercake.training.baseline import adapt_transformer_mixed_domain


parser = argparse.ArgumentParser()
parser.add_argument("--baseline", default="artifacts/baselines/transformer")
parser.add_argument("--config", default="configs/moonshot/dev/transformer_mixed_adaptation.yaml")
parser.add_argument("--output", default="artifacts/baselines/transformer-mixed")
args = parser.parse_args()
print(json.dumps(adapt_transformer_mixed_domain(
    ROOT / args.baseline, ROOT / args.config, ROOT / args.output
), indent=2))

