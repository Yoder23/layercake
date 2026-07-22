from _common import ROOT

import json

from layercake.training.foundation import train_english_core


runs = []
for config, output in (
    ("configs/moonshot/dev/core_english_receiver_b.yaml", "artifacts/cores/english-core-b"),
    ("configs/moonshot/dev/core_english_receiver_c.yaml", "artifacts/cores/english-core-c"),
):
    runs.append(train_english_core(ROOT / config, ROOT / output))
print(json.dumps(runs, indent=2))

