from _common import ROOT

import argparse
import json

from layercake.training.architecture_search import run_architecture_search


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/search/architecture_search.yaml")
parser.add_argument("--output", default="results/moonshot/v2/architecture_search.json")
args = parser.parse_args()
print(json.dumps(run_architecture_search(ROOT / args.config, ROOT / args.output), indent=2))

