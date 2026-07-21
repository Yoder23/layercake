from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import _common
from layercake.moonshot import ROOT, load_config, source_tree_hash


parser = argparse.ArgumentParser(description="Run and record the full regression suite for a moonshot certificate")
parser.add_argument("--config", default="configs/moonshot/smoke.json")
args = parser.parse_args()
config, _ = load_config(args.config)
started = time.perf_counter()
completed = subprocess.run(
    [sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, check=False
)
summary = completed.stdout + completed.stderr
match = re.search(r"(\d+) passed", summary)
security_path = ROOT / "tests" / "cake" / "test_package_security.py"
evidence = {
    "format": "layercake-pytest-evidence/1",
    "status": "PASS" if completed.returncode == 0 else "FAIL",
    "exit_code": completed.returncode,
    "passed": int(match.group(1)) if match else None,
    "seconds": time.perf_counter() - started,
    "command": [sys.executable, "-m", "pytest", "-q"],
    "package_security_test_sha256": hashlib.sha256(security_path.read_bytes()).hexdigest(),
    "source_tree_sha256": source_tree_hash(),
    "summary_tail": summary[-4000:],
}
output = ROOT / config["output"] / "pytest_evidence.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(evidence, indent=2, sort_keys=True))
raise SystemExit(completed.returncode)
