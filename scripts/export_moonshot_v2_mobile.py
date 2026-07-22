from _common import ROOT

import argparse
import json

import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import load_package
from layercake.models.portable_decoder import load_cake_module
from layercake.runtime.mobile_export import export_mobile_runtime


parser = argparse.ArgumentParser()
parser.add_argument("--cake", default="artifacts/cakes/python.cake")
parser.add_argument("--public-key", default="artifacts/cakes/python.public.pem")
parser.add_argument("--output", default="artifacts/mobile/python-fusion-v2.pt")
parser.add_argument("--evidence", default="results/moonshot/v2/mobile_export_evidence.json")
args = parser.parse_args()

import zipfile
with zipfile.ZipFile(ROOT / args.cake) as archive:
    key_id = CakeManifest.from_json(archive.read("manifest.json")).signature["key_id"]
package = load_package(
    ROOT / args.cake, trust_store={key_id: ROOT / args.public_key}, require_signature=True
)
module = load_cake_module(package).cpu().eval()
example = (
    torch.randn(1, 16, 256),
    torch.randn(1, 16, 64),
    torch.arange(16, dtype=torch.long)[None],
)
result = export_mobile_runtime(module, example, ROOT / args.output)
evidence = ROOT / args.evidence
evidence.parent.mkdir(parents=True, exist_ok=True)
evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["overall_status"] == "PASS" else 1)
