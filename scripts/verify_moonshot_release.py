import argparse, json
import _common
from layercake.moonshot import verify_release

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/moonshot/full.json")
args = parser.parse_args()
certificate = verify_release(args.config)
print(json.dumps(certificate, indent=2, sort_keys=True))
raise SystemExit(0 if certificate["moonshot_proven"] else 1)
