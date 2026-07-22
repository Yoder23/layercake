from _common import ROOT

import argparse
import json

from layercake.evaluation.moonshot_verifier import verify_moonshot_v2


parser = argparse.ArgumentParser()
parser.add_argument("--evidence", default="results/moonshot/v2")
args = parser.parse_args()
certificate = verify_moonshot_v2(ROOT, ROOT / args.evidence)
print(json.dumps(certificate, indent=2, sort_keys=True))
raise SystemExit(0 if certificate["moonshot_proven"] else 1)
