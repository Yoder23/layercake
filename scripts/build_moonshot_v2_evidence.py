from _common import ROOT

import json

from layercake.evaluation.development import build_development_evidence


result = build_development_evidence(ROOT, ROOT / "results/moonshot/v2/development_evidence.json")
print(json.dumps(result, indent=2, sort_keys=True))
