from _common import ROOT

import json

from layercake.evaluation.english_quality import evaluate_english_quality


result = evaluate_english_quality(
    ROOT / "artifacts/cores/english-core-a",
    ROOT / "data/moonshot/v2/wikitext103/test.bin",
    ROOT / "results/moonshot/v2/english_quality_evidence.json",
)
print(json.dumps(result, indent=2, sort_keys=True))
